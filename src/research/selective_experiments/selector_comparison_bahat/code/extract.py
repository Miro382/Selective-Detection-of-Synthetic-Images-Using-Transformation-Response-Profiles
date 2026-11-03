"""Voliteľné doplnenie skutočných logitov a transformácií z Bahat et al. 2018."""
from pathlib import Path
import io
import json
import os
import sys
import time
import numpy as np
from PIL import Image, ImageEnhance, ImageOps
from .common import canonical, digest, read, write, sha, load_data, load_module, check_receipt


def ours(image):
    yield image
    for quality in (90,75,60):
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=quality, subsampling=0, optimize=False)
        buffer.seek(0)
        with Image.open(buffer) as decoded:
            yield decoded.convert("RGB").copy()
    for scale in (.75,.5):
        small = tuple(max(1, round(s*scale)) for s in image.size)
        yield image.resize(small, Image.Resampling.LANCZOS).resize(image.size, Image.Resampling.LANCZOS)


def bahat(image):
    """Originál + päť transformácií. Hraničné/kvantizačné konvencie sú naše."""
    yield image
    yield ImageOps.mirror(image)
    pixels = np.asarray(image, dtype=np.float64)
    padded = np.pad(pixels, ((0,0),(1,1),(0,0)), mode="edge")
    blur = (padded[:,:-2] + padded[:,1:-1] + padded[:,2:])/3.
    yield Image.fromarray(np.clip(np.rint(blur),0,255).astype(np.uint8))
    gray = np.rint(pixels @ np.array([.299,.587,.114])).clip(0,255).astype(np.uint8)
    yield Image.fromarray(np.repeat(gray[:,:,None],3,axis=2))
    yield ImageEnhance.Contrast(image).enhance(1.3)
    gamma = np.rint(255*(pixels/255.)**.85).clip(0,255).astype(np.uint8)
    yield Image.fromarray(gamma)


def check_logits(row, plan, cached, protocol_sha):
    if row["sample_id"] != plan["sample_id"] or row["image_sha256"] != plan["sha256"] or row["protocol_sha256"] != protocol_sha:
        raise ValueError("Logity patria inému obrázku alebo protokolu.")
    a, b = np.asarray(row["ours_logits"]), np.asarray(row["bahat_logits"])
    if a.shape != (6,2) or b.shape != (6,2) or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Neplatné logity.")
    # Pôvodný runtime odčíta dve FP32 hodnoty ešte na GPU.
    margins = (a[:,1].astype(np.float32)-a[:,0].astype(np.float32)).astype(float)
    if not np.allclose(margins, cached, rtol=1e-6, atol=2e-5):
        raise ValueError("Nové logity sa nezhodujú s cache. Skontrolujte prostredie/predspracovanie, nepokračujte s iným modelom.")
    if (margins[0] >= 0) != (cached[0] >= 0) or not np.array_equal(a[0], b[0]):
        raise ValueError("Zmenené pôvodné rozhodnutie.")


def scan(path, plans, margins, protocol_sha, split, repair=True):
    rows = []; previous = digest([protocol_sha, split, "logits-v1"])
    path = Path(path)
    if not path.exists():
        return rows, previous
    with path.open("r+b" if repair else "rb") as f:
        while True:
            start = f.tell(); line = f.readline()
            if not line:
                break
            if not line.endswith(b"\n"):
                if repair:
                    f.truncate(start)
                    break
                raise ValueError("Nedopísaný posledný záznam.")
            row = json.loads(line)
            if len(rows) >= len(plans):
                raise ValueError("V žurnáli je priveľa záznamov.")
            expected = digest([previous, {k:v for k,v in row.items() if k != "chain"}])
            if row.get("chain") != expected:
                raise ValueError("Poškodený úplný záznam; automaticky sa nevymazáva.")
            check_logits(row, plans[len(rows)], margins[len(rows)], protocol_sha)
            previous = expected; rows.append(row)
    return rows, previous


def runtime(run, protocol):
    import torch
    source = Path(protocol["source"])
    sp = read(Path(run) / "source_protocol.json")
    code = Path(run) / "detector_runtime"
    sys.path.insert(0, str(code))
    for name in ("v4_data_sources", "prepare_data_v4", "sampler_v4", "detector_training_v4", "selector_augmentation_v4"):
        load_module(code / (name + ".py"), name)
    detector = sys.modules["detector_training_v4"]
    augmentation = sys.modules["selector_augmentation_v4"]
    model, transform, info = detector.load_frozen_detector(Path(sp["detector_run"]), device="cuda")
    if info["model_sha256"] != protocol["model_sha256"] or info["temperature"] != protocol["temperature"]:
        raise ValueError("Načítaný detektor sa nezhoduje s cache.")
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    def predict(image):
        with torch.inference_mode():
            return model(transform(image.convert("RGB")).unsqueeze(0).to("cuda"))[0].cpu().numpy().astype(float).tolist()
    return predict, augmentation.apply_spec


def extract(run, protocol, max_new=None):
    run = Path(run); folder = run / "logits"; folder.mkdir(exist_ok=True)
    psha = sha(run / "protocol.json"); new = 0; predictor = None; started = time.monotonic()
    for split in ("train", "selection", "external"):
        receipt = split + ".complete.json"
        if (folder / receipt).exists():
            check_receipt(folder, receipt, psha)
            print("Logity už sú hotové: " + split, flush=True)
            continue
        data, plans = load_data(run, split)
        journal = folder / (split + ".jsonl")
        rows, chain = scan(journal, plans, data["raw_margins"], psha, split)
        print(f"{split}: pokračujem od {len(rows)}/{len(plans)}; detektor zostáva zmrazený.", flush=True)
        if len(rows) < len(plans) and predictor is None:
            predictor, apply_spec = runtime(run, protocol)
        with journal.open("ab") as f:
            for i in range(len(rows), len(plans)):
                if max_new is not None and new >= max_new:
                    print("Dosiahnutý --max-new. Hotové logity ostávajú uložené; rovnaký príkaz nadviaže.", flush=True)
                    return
                plan = plans[i]; path = Path(plan["path"])
                if sha(path) != plan["sha256"]:
                    raise ValueError("Obrázok sa zmenil: " + str(path))
                with Image.open(path) as image:
                    image.load(); base = apply_spec(image.convert("RGB"), plan["input_spec"])
                our_pairs = [predictor(image) for image in ours(base)]
                other_images = iter(bahat(base)); next(other_images)
                other_pairs = [our_pairs[0]] + [predictor(image) for image in other_images]
                if sha(path) != plan["sha256"]:
                    raise ValueError("Obrázok sa zmenil počas výpočtu.")
                row = {"sample_id": plan["sample_id"], "image_sha256": plan["sha256"], "protocol_sha256": psha,
                       "ours_logits": our_pairs, "bahat_logits": other_pairs}
                check_logits(row, plan, data["raw_margins"][i], psha)
                row["chain"] = digest([chain,row]); chain = row["chain"]
                f.write(canonical(row) + b"\n"); f.flush()
                rows.append(row); new += 1
                if new % 25 == 0:
                    os.fsync(f.fileno())
                    print(f"  {split}: {i+1}/{len(plans)}, {new} nových, {(time.monotonic()-started)/max(new,1):.2f} s/profil", flush=True)
            os.fsync(f.fileno())
        target = folder / (split + ".npz")
        temp = folder / (split + ".tmp.npz")
        np.savez_compressed(temp, ours_logits=np.asarray([r["ours_logits"] for r in rows]),
                            bahat_logits=np.asarray([r["bahat_logits"] for r in rows]))
        os.replace(temp, target)
        write(folder / receipt, {"protocol_sha256": psha, "count": len(rows), "chain": chain,
                                 "files": {target.name: sha(target), journal.name: sha(journal)}})
    print("Doplnené logity sú hotové. Spustite compare_selector_full.py.", flush=True)
