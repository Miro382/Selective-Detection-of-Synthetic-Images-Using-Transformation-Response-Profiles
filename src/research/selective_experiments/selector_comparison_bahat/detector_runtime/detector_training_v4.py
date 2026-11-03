"""Reproducible V4 fine-tuning; no test inputs and no selector fitting.

Training uses manifests prepared by prepare_data_v4.py. The original frozen
checkpoint remains unchanged. Epochs are replacement samples, not full passes.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import random
import shutil
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from contextlib import contextmanager

# Set before the first CUDA context is initialized.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from sampler_v4 import CappedBalancedSampler
from torchvision import models, transforms
from torchvision.transforms import InterpolationMode

from prepare_data_v4 import load_manifest, verify_dataset


VERSION = 4
CLASSES = {"real": 0, "fake": 1}
CONDITIONS = ("clean", "jpeg_q40", "resize_0p5")
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
BUILDERS = {"efficientnet_v2_s": models.efficientnet_v2_s,
            "efficientnet_v2_m": models.efficientnet_v2_m,
            "efficientnet_v2_l": models.efficientnet_v2_l}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def torch_write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def process_lock(path):
    """OS zámok sa uvoľní aj pri ukončení procesu alebo vypnutí počítača.

    Súbor s PID zostáva ako posledný diagnostický záznam, nie ako dôkaz,
    že proces ešte beží. Nerozhoduje sa podľa samotnej existencie súboru.
    """
    path = Path(path)
    handle = path.open("a+b")
    if path.stat().st_size == 0:
        handle.write(b" ")
        handle.flush()
    handle.seek(0)
    try:
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        handle.close()
        raise RuntimeError(f"Iný proces práve používa beh: {path}") from exc
    try:
        handle.seek(1)
        handle.truncate()
        handle.write(json.dumps({"pid": os.getpid(), "created_utc": utc_now()}).encode())
        handle.flush()
        yield
    finally:
        handle.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def rgb_image(image):
    # Identical to Software.inference._safe_open_image, including alpha handling.
    return image.convert("RGB")


def jpeg(image, quality, subsampling=0):
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=int(quality), subsampling=int(subsampling), optimize=False)
    buffer.seek(0)
    with Image.open(buffer) as decoded:
        decoded.load()
        return decoded.convert("RGB")


def downscale_restore(image, scale):
    original = image.size
    reduced = tuple(max(1, round(side * scale)) for side in original)
    return image.resize(reduced, Image.Resampling.LANCZOS).resize(original, Image.Resampling.LANCZOS)


def degrade(image, condition):
    # Conditions operate on the original RGB input, before model preprocessing.
    if condition == "clean":
        return image
    if condition == "jpeg_q40":
        return jpeg(image, 40)
    if condition == "resize_0p5":
        return downscale_restore(image, 0.5)
    raise ValueError(f"Neznáma podmienka: {condition}")


def eval_transform(image_size):
    return transforms.Compose([
        transforms.Resize(max(image_size + 32, int(image_size * 1.1)), interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(), transforms.Normalize(MEAN, STD),
    ])


class MildTrainingTransform:
    """Identical policy for every class/source, including a 25% clean branch."""

    def __init__(self, image_size, policy="v3"):
        if policy not in {"v3", "social"}:
            raise ValueError("Neznáma politika augmentácií.")
        self.policy = policy
        self.clean = eval_transform(image_size)
        self.spatial = transforms.Compose([
            transforms.Resize(max(image_size + 32, int(image_size * 1.1)), interpolation=InterpolationMode.BICUBIC),
            transforms.RandomResizedCrop(image_size, scale=(0.85, 1.0), ratio=(0.95, 1.05), interpolation=InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor(), transforms.Normalize(MEAN, STD),
        ])

    def __call__(self, image):
        if random.random() < 0.25:
            return self.clean(image)
        if random.random() < 0.35:
            image = downscale_restore(image, random.uniform(0.5, 0.9))
        if random.random() < 0.4:
            image = ImageEnhance.Brightness(image).enhance(random.uniform(0.8, 1.2))
            image = ImageEnhance.Contrast(image).enhance(random.uniform(0.8, 1.2))
        if random.random() < 0.1:
            image = image.filter(ImageFilter.GaussianBlur(random.uniform(0.2, 0.8)))
        if random.random() < 0.45:
            image = jpeg(image, random.randint(40, 95), random.choice([0, 2]) if self.policy == "social" else 0)
        if self.policy == "social" and random.random() < 0.3:
            image = downscale_restore(image, random.uniform(0.5, 0.9))
            image = jpeg(image, random.randint(40, 90), random.choice([0, 2]))
            if random.random() < 0.25:
                image = jpeg(image, random.randint(45, 95), random.choice([0, 2]))
        return self.spatial(image)


class ManifestDataset(Dataset):
    def __init__(self, rows, transform, condition="clean"):
        self.rows, self.transform, self.condition = list(rows), transform, condition

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        try:
            with Image.open(row["path"]) as source:
                source.load()
                image = rgb_image(source)
            image = degrade(image, self.condition)
            return self.transform(image), int(row["label"]), index
        except Exception as exc:
            # Silently skipping broken images changes the locked experiment.
            raise RuntimeError(f"Nečitateľný vstup {row['sample_id']}: {row['path']}: {exc}") from exc


def select_validation(rows, max_historical_per_class, seed):
    """Celá nová validácia a deterministicky limitovaný historický replay."""
    historical_sources = {"historical", "v3/historical"}
    selected = [row for row in rows if row["source"] not in historical_sources]
    for label in (0, 1):
        candidates = [row for row in rows if row["source"] in historical_sources and int(row["label"]) == label]
        candidates.sort(key=lambda row: hashlib.sha256(f"{seed}:val:{row['sample_id']}".encode()).hexdigest())
        selected.extend(candidates[:max_historical_per_class] if max_historical_per_class else candidates)
    selected.sort(key=lambda row: row["sample_id"])
    if len({row["sample_id"] for row in selected}) != len(selected):
        raise ValueError("Duplicitné identifikátory validačných vzoriek.")
    if {int(row["label"]) for row in selected} != {0, 1}:
        raise ValueError("Validácia potrebuje obe triedy.")
    return selected


def binary_metrics(labels, probabilities, raw_logits=None):
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if labels.shape != probabilities.shape or labels.size == 0 or not np.isfinite(probabilities).all():
        raise ValueError("Neplatné skóre alebo prázdna množina.")
    if not np.isin(labels, [0, 1]).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("Neplatná trieda alebo pravdepodobnostné skóre.")
    pred = probabilities >= 0.5
    tn = int(((labels == 0) & ~pred).sum())
    fp = int(((labels == 0) & pred).sum())
    fn = int(((labels == 1) & ~pred).sum())
    tp = int(((labels == 1) & pred).sum())
    real_accuracy = tn / (tn + fp) if tn + fp else None
    fake_accuracy = tp / (tp + fn) if tp + fn else None
    precision = tp / (tp + fp) if tp + fp else 0.0
    p = np.clip(probabilities, 1e-12, 1 - 1e-12)
    if raw_logits is not None:
        raw_logits = np.asarray(raw_logits, dtype=np.float64)
        if raw_logits.shape != labels.shape or not np.isfinite(raw_logits).all():
            raise ValueError("Neplatné logity pri vyhodnotení.")
        log_loss = float(np.mean(np.logaddexp(0.0, raw_logits) - labels * raw_logits))
    else:
        log_loss = float(-(labels * np.log(p) + (1 - labels) * np.log(1 - p)).mean())
    from sklearn.metrics import roc_auc_score
    return {"n": int(labels.size), "errors": fp + fn, "accuracy": (tn + tp) / labels.size,
            "confusion_real_fake": [[tn, fp], [fn, tp]], "real_accuracy": real_accuracy,
            "fake_recall": fake_accuracy, "fake_precision": precision,
            "fake_f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
            "balanced_accuracy": (real_accuracy + fake_accuracy) / 2 if real_accuracy is not None and fake_accuracy is not None else None,
            "log_loss": log_loss,
            "auroc": float(roc_auc_score(labels, probabilities if raw_logits is None else raw_logits)) if len(set(labels.tolist())) == 2 else None,
            "brier": float(np.mean((probabilities - labels) ** 2))}


def hierarchical_metrics(rows, probabilities, raw_logits=None):
    """Rovnaká váha tried, potom real domén/fake rodín, zdrojov a generátorov.

    Jednotriedny zdroj dostane vlastnú úspešnosť. Výberové skóre je priemer
    tried, takže FFHQ/FORLAB nemusia mať syntetický náprotivok.
    """
    probabilities = np.asarray(probabilities, dtype=np.float64)
    labels = np.array([int(row["label"]) for row in rows])
    raw_logits = np.asarray(raw_logits, dtype=np.float64) if raw_logits is not None else None
    result = binary_metrics(labels, probabilities, raw_logits)
    if set(labels.tolist()) != {0, 1}:
        raise ValueError("Výberová validácia musí mať obe triedy.")
    grouped = defaultdict(list)
    for i, row in enumerate(rows):
        level = row["domain"] if int(row["label"]) == 0 else row["family"]
        grouped[(int(row["label"]), level, row["source"], row["generator"])].append(i)
    cells, tree = {}, defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for (label, level, source, generator), indices in sorted(grouped.items()):
        indices = np.asarray(indices)
        metric = binary_metrics(labels[indices], probabilities[indices], raw_logits[indices] if raw_logits is not None else None)
        cells[f"{label}/{level}/{source}/{generator}"] = metric
        tree[label][level][source].append(metric["accuracy"])
    class_scores = {}
    for label, levels in tree.items():
        class_scores[str(label)] = float(np.mean([
            np.mean([np.mean(generators) for generators in sources.values()])
            for sources in levels.values()]))
    by_field = {}
    for field in ("source", "generator", "domain", "family"):
        by_field[field] = {}
        for value in sorted({str(row[field]) for row in rows}):
            mask = np.array([str(row[field]) == value for row in rows])
            by_field[field][value] = binary_metrics(labels[mask], probabilities[mask], raw_logits[mask] if raw_logits is not None else None)
    result.update({"by_source": by_field["source"], "by_generator": by_field["generator"],
                   "by_domain": by_field["domain"], "by_family": by_field["family"], "cells": cells,
                   "class_macro_accuracy": class_scores,
                   "hierarchical_balanced_accuracy": float(np.mean(list(class_scores.values()))),
                   "worst_cell_accuracy": min(value["accuracy"] for value in cells.values()),
                   "worst_source_accuracy": min(value["accuracy"] for value in by_field["source"].values())})
    return result


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def worker_seed(worker_id):
    seed = torch.initial_seed() % (2 ** 32)
    random.seed(seed)
    np.random.seed(seed)


def loader(dataset, args, *, sampler=None, seed=0):
    return DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, shuffle=False,
                      num_workers=args.num_workers, pin_memory=args.device == "cuda",
                      persistent_workers=False, worker_init_fn=worker_seed,
                      generator=torch.Generator().manual_seed(seed),
                      **({"prefetch_factor": 2} if args.num_workers else {}))


def build_model(checkpoint_path):
    # Trusted local checkpoint, pinned by data_protocol.json before deserialization.
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    mapping = checkpoint.get("class_to_index", config.get("class_to_index"))
    if mapping != CLASSES:
        raise ValueError(f"Nekompatibilné mapovanie tried: {mapping!r}")
    model_name, image_size, dropout = config["model"], int(config["image_size"]), float(config["dropout"])
    if model_name != "efficientnet_v2_m" or image_size != 512:
        raise ValueError(f"V4 vyžaduje EfficientNetV2-M/512, checkpoint má {model_name}/{image_size}.")
    model = BUILDERS[model_name](weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, 512),
                                     nn.SiLU(inplace=True), nn.Dropout(max(0.1, dropout / 2)), nn.Linear(512, 2))
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model, {"model": model_name, "image_size": image_size, "dropout": dropout, "class_to_index": CLASSES}


def set_training_stage(model, head_only, train_batchnorm=False):
    model.train()
    for name, parameter in model.named_parameters():
        parameter.requires_grad = not head_only or name.startswith("classifier.")
    if head_only:
        # Freeze running statistics and stochastic depth in the frozen backbone.
        model.features.eval()
    elif not train_batchnorm:
        # Accumulation does not enlarge the microbatch used by BatchNorm.
        # Keep pretrained running statistics; affine parameters stay learnable.
        for module in model.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()


def step_accumulated(model, optimizer, scaler, samples, max_grad_norm):
    if samples <= 0:
        raise ValueError("Prázdne akumulačné okno.")
    scaler.unscale_(optimizer)
    for parameter in model.parameters():
        if parameter.grad is not None:
            parameter.grad.div_(samples)
    nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm, error_if_nonfinite=not scaler.is_enabled())
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)


def train_epoch(model, data_loader, optimizer, scaler, args, epoch):
    set_training_stage(model, epoch <= args.head_warmup_epochs, args.train_batchnorm)
    optimizer.zero_grad(set_to_none=True)
    total_loss, total_correct, total_seen, window_samples, steps = 0.0, 0, 0, 0, 0
    start = last_log = time.monotonic()
    for batch_index, (images, labels, _) in enumerate(data_loader):
        images, labels = images.to(args.device, non_blocking=True), labels.to(args.device, non_blocking=True)
        with torch.amp.autocast(device_type=args.device, enabled=args.device == "cuda"):
            logits = model(images)
            loss_sum = nn.functional.cross_entropy(logits, labels, reduction="sum", label_smoothing=args.label_smoothing)
        if not torch.isfinite(loss_sum):
            raise FloatingPointError("Nefinitná tréningová strata; tréning sa zastavil bez preskočenia vzoriek.")
        scaler.scale(loss_sum).backward()
        count = labels.size(0)
        total_loss += float(loss_sum.detach())
        total_correct += int((logits.argmax(1) == labels).sum())
        total_seen += count
        window_samples += count
        if (batch_index + 1) % args.accumulation_steps == 0 or batch_index + 1 == len(data_loader):
            # Divide by actual examples, including the final short batch/window.
            step_accumulated(model, optimizer, scaler, window_samples, args.max_grad_norm)
            window_samples, steps = 0, steps + 1
        if time.monotonic() - last_log >= 30:
            print(f"  Epocha {epoch}: {total_seen}/{len(data_loader.sampler)} vzoriek, strata {total_loss/total_seen:.4f}", flush=True)
            last_log = time.monotonic()
    return {"loss": total_loss / total_seen, "accuracy": total_correct / total_seen, "n": total_seen,
            "optimizer_windows": steps, "seconds": time.monotonic() - start,
            "head_only": epoch <= args.head_warmup_epochs}


@torch.inference_mode()
def evaluate(model, rows, args, image_size, prediction_path=None):
    model.eval()
    condition_metrics, all_predictions = {}, []
    for condition in CONDITIONS:
        print(f"  Validácia {condition}: {len(rows)} obrázkov (FP32)", flush=True)
        dataset = ManifestDataset(rows, eval_transform(image_size), condition)
        probabilities, raw_logits = [], []
        started = last_log = time.monotonic()
        for images, _, _ in loader(dataset, args, seed=args.seed):
            # Match the software's FP32 evaluation, including the 0.5 decision.
            outputs = model(images.to(args.device, non_blocking=True))
            margins = outputs[:, 1] - outputs[:, 0]
            scores = torch.sigmoid(margins.double())
            probabilities.extend(scores.cpu().tolist())
            raw_logits.extend(margins.cpu().tolist())
            if time.monotonic() - last_log >= 30:
                print(f"    {len(probabilities)}/{len(rows)}", flush=True)
                last_log = time.monotonic()
        condition_metrics[condition] = hierarchical_metrics(rows, probabilities, raw_logits)
        condition_metrics[condition]["seconds"] = time.monotonic() - started
        if prediction_path is not None:
            all_predictions.extend({"sample_id": row["sample_id"], "condition": condition,
                                    "label": int(row["label"]), "source": row["source"],
                                    "generator": row["generator"], "probability_fake": float(score), "raw_logit": float(margin)}
                                   for row, score, margin in zip(rows, probabilities, raw_logits))
    if prediction_path is not None:
        with Path(prediction_path).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["sample_id", "condition", "label", "source", "generator", "probability_fake", "raw_logit"])
            writer.writeheader()
            writer.writerows(all_predictions)
    return {"selection_score": float(np.mean([value["hierarchical_balanced_accuracy"] for value in condition_metrics.values()])),
            "worst_cell_accuracy": min(value["worst_cell_accuracy"] for value in condition_metrics.values()),
            "conditions": condition_metrics}


def capture_rng():
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def source_paths():
    base = Path(__file__).resolve().parent
    names = ["detector_training_v4.py", "train_detector_v4.py", "sampler_v4.py", "calibrate_detector_v4.py",
             "prepare_data_v4.py", "v4_data_sources.py", "v4_deepshield_audit.py",
             "v4_deepshield_remote_zip.py", "v4_deepshield_index_v1.json.gz"]
    missing = [name for name in names if not (base / name).is_file()]
    if missing:
        raise FileNotFoundError("Chýba závislosť V4: " + ", ".join(missing))
    return {name: base / name for name in names}


def source_lock():
    return {name: sha256_file(path) for name, path in source_paths().items()}


def configuration(args, architecture, data_dir):
    excluded = {"output_dir", "resume", "check", "smoke", "full_hashes", "smoke_image_size"}
    values = {key: str(value.resolve()) if isinstance(value, Path) else value
              for key, value in vars(args).items() if key not in excluded}
    return {"version": VERSION, **architecture, "training": values,
            "data_protocol_sha256": sha256_file(data_dir / "data_protocol.json"),
            "source_sha256": source_lock(),
            "selection": "Mean over clean, JPEG Q40 and resize 0.5; equal classes, then real domains/fake families, sources and generators; each leaf uses its class accuracy.",
            "validation_scope": "Adaptive development validation only. The V3 initial detector and V4 use the same locked validation for comparison. Historical validation may have been seen before. DeepShield V3 errors informed V4 design; this is not external confirmation.",
            "sampler": {"policy": "hierarchical_class_domain_family_source_generator_group_cap", "max_dependency_group_repeats": args.max_group_repeats, "shorten_if_insufficient_capacity": True},
            "augmentation": {"policy": args.augmentation_policy, "social_branch": {"additional_resize_jpeg_p": 0.3, "second_jpeg_conditional_p": 0.25, "subsampling_choices": [0, 2]} if args.augmentation_policy == "social" else None, "clean_probability": 0.25, "on_augmented_branch": {
                "downscale_restore": {"p": 0.35, "scale": [0.5, 0.9], "interpolation": "LANCZOS"},
                "brightness_contrast": {"p": 0.4, "factor": [0.8, 1.2]},
                "blur": {"p": 0.1, "sigma": [0.2, 0.8]}, "jpeg": {"p": 0.45, "quality": [40, 95], "subsampling": [0, 2] if args.augmentation_policy == "social" else 0, "optimize": False},
                "crop_scale": [0.85, 1.0], "crop_ratio": [0.95, 1.05], "horizontal_flip": 0.5},
                "policy_identical_for_all_labels_and_sources": True},
            "environment": {"python": sys.version, "torch": torch.__version__, "numpy": np.__version__,
                            "torchvision": __import__("torchvision").__version__, "pillow": __import__("PIL").__version__,
                            "scikit_learn": __import__("sklearn").__version__,
                            "executable": str(Path(sys.executable).resolve()),
                            "cuda": torch.version.cuda, "device": torch.cuda.get_device_name(0) if args.device == "cuda" else "cpu"}}


def checkpoint_value(model, config, epoch, score):
    return {"model_state_dict": model.state_dict(), "config": config, "class_to_index": CLASSES,
            "epoch": epoch, "selection_score": score, "created_utc": utc_now()}


def best_checkpoint_path(run_dir, epoch):
    return run_dir / "checkpoints" / f"best_epoch_{epoch:03d}.pt"


def save_best(run_dir, model, config, epoch, score):
    target = best_checkpoint_path(run_dir, epoch)
    target.parent.mkdir(exist_ok=True)
    torch_write(target, checkpoint_value(model, config, epoch, score))
    temporary = run_dir / "best_model.pt.tmp"
    shutil.copyfile(target, temporary)
    os.replace(temporary, run_dir / "best_model.pt")


def save_resume(run_dir, model, config, epoch, score, best_epoch, bad_epochs, optimizer, scheduler, scaler, history):
    value = checkpoint_value(model, config, epoch, score)
    value.update({"best_epoch": best_epoch, "bad_epochs": bad_epochs, "optimizer_state_dict": optimizer.state_dict(),
                  "scheduler_state_dict": scheduler.state_dict(), "scaler_state_dict": scaler.state_dict(),
                  "rng_state": capture_rng(), "history": history,
                  "baseline_metrics_sha256": sha256_file(run_dir / "baseline_metrics.json"),
                  "validation_selection_sha256": sha256_file(run_dir / "validation_selection.json"),
                  "best_checkpoint_sha256": sha256_file(best_checkpoint_path(run_dir, best_epoch))})
    torch_write(run_dir / "last_training_state.pt", value)


def validate_resume(state, config, run_dir):
    if state["config"] != config:
        differing = [key for key in set(config) | set(state["config"]) if config.get(key) != state["config"].get(key)]
        raise ValueError("Obnovenie odmietnuté: zmenená konfigurácia/prostredie/dáta/kód: " + ", ".join(sorted(differing)))
    if json.loads((run_dir / "config.json").read_text(encoding="utf-8")) != config:
        raise ValueError("Obnovenie odmietnuté: config.json sa zmenil.")
    for name, expected in config["source_sha256"].items():
        if sha256_file(run_dir / "snapshot" / name) != expected:
            raise ValueError(f"Obnovenie odmietnuté: zmenený snapshot {name}.")
    for name, key in [("baseline_metrics.json", "baseline_metrics_sha256"),
                      ("validation_selection.json", "validation_selection_sha256")]:
        if sha256_file(run_dir / name) != state[key]:
            raise ValueError(f"Obnovenie odmietnuté: zmenený súbor {name}.")
    if sha256_file(best_checkpoint_path(run_dir, state["best_epoch"])) != state["best_checkpoint_sha256"]:
        raise ValueError("Obnovenie odmietnuté: zmenené váhy najlepšej dokončenej epochy.")


def verify_completed_training(run_dir):
    """Overí dokončený beh pred použitím checkpointu; nič netrénuje."""
    run_dir = Path(run_dir).resolve()
    complete_path = run_dir / "complete.json"
    if not complete_path.is_file():
        raise ValueError("Detektor V4 ešte nemá dokončený tréning (complete.json).")
    complete = json.loads(complete_path.read_text(encoding="utf-8"))
    required = {"config.json", "validation_selection.json", "baseline_metrics.json", "baseline_predictions.csv",
                "best_predictions.csv", "comparison.json", "best_model.pt", "last_training_state.pt"}
    if not required <= set(complete.get("files", {})):
        raise ValueError("Neúplný záznam dokončenia tréningu V4.")
    for name, expected in complete["files"].items():
        target = (run_dir / name).resolve()
        if not target.is_relative_to(run_dir) or sha256_file(target) != expected:
            raise ValueError(f"Zmenený výsledok tréningu: {name}")
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    if config.get("version") != VERSION or config.get("model") != "efficientnet_v2_m" or config.get("image_size") != 512:
        raise ValueError("Neplatná verzia alebo architektúra dokončeného detektora.")
    for name, expected in config["source_sha256"].items():
        if sha256_file(run_dir / "snapshot" / name) != expected:
            raise ValueError(f"Zmenená snímka kódu tréningu: {name}")
    if sha256_file(Path(__file__)) != config["source_sha256"]["detector_training_v4.py"]:
        raise ValueError("Načítavač detektora sa zmenil; použi zmrazený snapshot z tréningu.")
    return config, complete


def verify_temperature(run_dir, model_sha256):
    run_dir = Path(run_dir)
    receipt = json.loads((run_dir / "temperature.complete.json").read_text(encoding="utf-8"))
    required = {"temperature.json", "temperature_protocol.json", "temperature_logits.jsonl"}
    if not required <= set(receipt.get("files", {})):
        raise ValueError("Neúplná kalibrácia teploty.")
    for name, expected in receipt["files"].items():
        target = (run_dir / name).resolve()
        if not target.is_relative_to(run_dir.resolve()) or sha256_file(target) != expected:
            raise ValueError(f"Zmenená kalibrácia teploty: {name}")
    artifact = json.loads((run_dir / "temperature.json").read_text(encoding="utf-8"))
    temperature = float(artifact["temperature"])
    if artifact.get("model_sha256") != model_sha256 or not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Teplota nepatrí k tomuto modelu alebo je neplatná.")
    if artifact.get("detector_threshold") != 0.5:
        raise ValueError("Kalibrácia zmenila rozhodovací prah.")
    return artifact


def load_frozen_detector(run_dir, device="cuda", require_temperature=True):
    """Vráti (model, transform, info), vždy z overeného dokončeného behu V4.

    Jediný povolený necalibrovaný prístup je require_temperature=False pri
    výpočte kalibračných logitov. Nevolá sa fit ani optimalizácia.
    """
    run_dir = Path(run_dir).resolve()
    config, complete = verify_completed_training(run_dir)
    if device not in ("cpu", "cuda") or (device == "cuda" and not torch.cuda.is_available()):
        raise ValueError("Požadované výpočtové zariadenie nie je dostupné.")
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    model_sha = complete["files"]["best_model.pt"]
    temperature = verify_temperature(run_dir, model_sha) if require_temperature else {"temperature": 1.0}
    model, architecture = build_model(run_dir / "best_model.pt")
    model.to(device).eval()
    info = {"model_sha256": model_sha, "checkpoint_path": str(run_dir / "best_model.pt"),
            "config": config, "image_size": architecture["image_size"], "device": device,
            "class_to_index": CLASSES, "temperature": float(temperature["temperature"]),
            "temperature_artifact_sha256": sha256_file(run_dir / "temperature.json") if require_temperature else None,
            "detector_threshold": 0.5, "raw_logit_definition": "z_fake - z_real",
            "probability_arithmetic": "sigmoid(raw_logit / T) in float64; model FP32"}
    return model, eval_transform(architecture["image_size"]), info


@torch.inference_mode()
def predict_logits(image, model, transform, info):
    pixels = transform(image.convert("RGB")).unsqueeze(0).to(info["device"])
    logits = model(pixels)
    if logits.shape != (1, 2) or not torch.isfinite(logits).all():
        raise ValueError("Neplatný logit detektora.")
    raw = float((logits[0, 1] - logits[0, 0]).item())
    calibrated = raw / float(info["temperature"])
    probability = 1.0 / (1.0 + math.exp(-calibrated)) if calibrated >= 0 else math.exp(calibrated) / (1.0 + math.exp(calibrated))
    return {"raw_logit": raw, "logit": calibrated, "probability_fake": probability}


def smoke(model, rows, args, architecture):
    size = args.smoke_image_size or architecture["image_size"]
    smoke_rows = []
    for label in (0, 1):
        candidates = [row for row in rows if int(row["label"]) == label]
        if not candidates:
            raise ValueError("Smoke vyžaduje obe triedy.")
        smoke_rows.append(sorted(candidates, key=lambda row: row["sample_id"])[0])
    # Independently load every source/class cell, including the large real TIFF,
    # even when the GPU microbatch contains only two examples.
    probe = ManifestDataset(smoke_rows, eval_transform(size))
    for index in range(len(probe)):
        pixels, label, _ = probe[index]
        if pixels.shape != (3, size, size) or not torch.isfinite(pixels).all():
            raise RuntimeError(f"Neplatné predspracovanie smoke vzorky {smoke_rows[index]['sample_id']}.")
    smoke_rows = (smoke_rows * math.ceil(args.batch_size / len(smoke_rows)))[:args.batch_size]
    dataset = ManifestDataset(smoke_rows, MildTrainingTransform(size, args.augmentation_policy))
    images, labels, _ = next(iter(loader(dataset, args, seed=args.seed)))
    model.to(args.device)
    set_training_stage(model, False, args.train_batchnorm)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=args.device == "cuda", init_scale=1024.0)
    backbone_parameter = next(model.features.parameters())
    head_parameter = model.classifier[-1].weight
    before_backbone, before_head = backbone_parameter.detach().clone(), head_parameter.detach().clone()
    with torch.amp.autocast(device_type=args.device, enabled=args.device == "cuda"):
        logits = model(images.to(args.device))
        loss = nn.functional.cross_entropy(logits, labels.to(args.device), reduction="sum")
    if not torch.isfinite(loss):
        raise FloatingPointError("Smoke test má nefinitnú stratu.")
    scaler.scale(loss).backward()
    step_accumulated(model, optimizer, scaler, len(smoke_rows), args.max_grad_norm)
    if torch.equal(before_backbone, backbone_parameter) or torch.equal(before_head, head_parameter):
        raise RuntimeError("Smoke test nepotvrdil aktualizáciu váh celej siete; AMP mohol preskočiť krok.")
    if not torch.isfinite(backbone_parameter).all() or not torch.isfinite(head_parameter).all():
        raise FloatingPointError("Smoke krok vytvoril nefinitné váhy.")
    model.eval()
    with torch.inference_mode():
        output = model(images[:1].to(args.device)).softmax(1)
    if not torch.isfinite(output).all() or output.shape != (1, 2):
        raise RuntimeError("Neplatný výstup smoke testu.")
    print(json.dumps({"smoke": "OK", "production_training": False, "weights_saved": False,
                      "full_backbone_backward_and_adamw_step": True, "image_size": size,
                      "input_cells_verified": sorted({f"{row['source']}/{row['label']}" for row in smoke_rows}),
                      "batch_size": len(smoke_rows), "loss": float(loss.detach()) / len(smoke_rows),
                      "peak_cuda_memory_gib": torch.cuda.max_memory_allocated() / 1024 ** 3 if args.device == "cuda" else None}, ensure_ascii=False, indent=2))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="V4: jemné doladenie zmrazeného detektora na pripravených manifestoch.")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume", type=Path, metavar="RUN_DIR")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--full-hashes", action="store_true", help="Aj pri --check/--smoke čítať a overiť hash každého obrázka.")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--accumulation-steps", type=int, default=4)
    parser.add_argument("--samples-per-epoch", type=int, default=24000)
    parser.add_argument("--max-group-repeats", type=int, default=3)
    parser.add_argument("--augmentation-policy", choices=("v3", "social"), default="v3")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-historical-val-per-class", type=int, default=1000, help="0 znamená celú historickú validáciu.")
    parser.add_argument("--head-lr", type=float, default=1e-4)
    parser.add_argument("--backbone-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--head-warmup-epochs", type=int, default=1)
    parser.add_argument("--train-batchnorm", action="store_true", help="Aktualizovať BatchNorm štatistiky po zahrievaní; predvolene sú pre malé dávky zmrazené.")
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20261015)
    parser.add_argument("--smoke-image-size", type=int, help="Iba technický smoke test; predvolene plná veľkosť checkpointu.")
    args = parser.parse_args(argv)
    for key in ("epochs", "batch_size", "accumulation_steps", "samples_per_epoch", "patience", "max_group_repeats"):
        if getattr(args, key) <= 0:
            parser.error(f"--{key.replace('_', '-')} musí byť kladné.")
    if args.num_workers < 0 or args.max_historical_val_per_class < 0 or not 0 <= args.head_warmup_epochs <= args.epochs:
        parser.error("Neplatný počet pracovníkov, limit validácie alebo dĺžka zahrievania.")
    finite_parameters = [args.label_smoothing, args.head_lr, args.backbone_lr, args.max_grad_norm, args.weight_decay, args.min_delta]
    if not all(math.isfinite(value) for value in finite_parameters) or not 0 <= args.label_smoothing < 1 or args.head_lr <= 0 or args.backbone_lr <= 0 or args.max_grad_norm <= 0 or args.weight_decay < 0 or args.min_delta < 0:
        parser.error("Neplatné hyperparametre.")
    if args.check and args.smoke:
        parser.error("--check a --smoke sa nesmú kombinovať.")
    if args.smoke_image_size is not None and (not args.smoke or args.smoke_image_size < 32):
        parser.error("--smoke-image-size patrí iba k --smoke a musí byť aspoň 32.")
    if args.resume and args.output_dir and args.resume.resolve() != args.output_dir.resolve():
        parser.error("--output-dir musí pri obnovení zodpovedať --resume.")
    return args


def main(argv=None):
    args = parse_args(argv)
    data_dir = args.data_dir.resolve()
    print("Kontrolujem protokol, pôvodný checkpoint a pripravené dáta...", flush=True)
    protocol = verify_dataset(data_dir, full_hashes=args.full_hashes or not (args.check or args.smoke))
    if protocol.get("version") != 4:
        raise ValueError("Tréning V4 vyžaduje dátový protokol V4.")
    checkpoint_info = protocol["initial_checkpoint"]
    expected_initial = "c4b54730991c9615d6e69f9505dd87fda4cb737db0f2f883a17811405b86ad0a"
    if checkpoint_info["sha256"] != expected_initial:
        raise ValueError("V4 sa musí začať zmrazeným detektorom V3 z 8. epochy.")
    if sha256_file(checkpoint_info["path"]) != checkpoint_info["sha256"]:
        raise ValueError("Pôvodný checkpoint nezodpovedá protokolu.")
    rows = load_manifest(data_dir / "train.csv")
    val_rows = select_validation(load_manifest(data_dir / "val.csv"), args.max_historical_val_per_class, args.seed)
    planned_sampler = CappedBalancedSampler(rows, args.samples_per_epoch, args.max_group_repeats, args.seed)
    train_groups = {row["dependency_key"] for row in rows}
    if train_groups & {row["dependency_key"] for row in val_rows}:
        raise ValueError("Tréning a validácia zdieľajú zdrojové skupiny.")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA nie je dostupná. Použi správne prostredie; CPU tréning vyžaduje explicitné --device cpu.")
    print(f"Tréning: {len(rows)} unikátnych obrázkov, {len(planned_sampler)} výberov/epochu (požiadavka {args.samples_per_epoch}, najviac {args.max_group_repeats} na skupinu). Validácia: {len(val_rows)} × 3 podmienky.", flush=True)
    if args.check:
        print(json.dumps({"check": "OK", "train_images": len(rows), "validation_images": len(val_rows),
                          "cuda_available": torch.cuda.is_available(), "device": args.device,
                          "checkpoint_sha256": checkpoint_info["sha256"], "full_image_hashes": args.full_hashes, "sampler": planned_sampler.report}, indent=2))
        return 0
    seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    model, architecture = build_model(checkpoint_info["path"])
    if args.smoke:
        smoke(model, rows, args, architecture)
        return 0
    config = configuration(args, architecture, data_dir)
    run_dir = (args.resume or args.output_dir or data_dir.parent / "training_main").resolve()
    if (run_dir / "complete.json").is_file():
        saved_config, _ = verify_completed_training(run_dir)
        if saved_config != config:
            raise ValueError("Dokončený beh má inú konfiguráciu. Nový pokus potrebuje nový --output-dir.")
        print(f"Tréning bol už dokončený: {run_dir / 'comparison.json'}", flush=True)
        return 0
    if args.resume:
        if not (run_dir / "last_training_state.pt").is_file():
            raise FileNotFoundError("Obnovenie vyžaduje last_training_state.pt z dokončenej epochy.")
    elif run_dir.exists():
        # Interrupting the initial validation leaves no completed epoch. Only a
        # recognized identical run may repeat that initialization; no weights
        # from a partial epoch are ever treated as a committed checkpoint.
        if not (run_dir / "config.json").is_file():
            raise ValueError("Existujúci priečinok nie je rozpoznaný beh V4.")
        saved = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        if saved != config:
            raise ValueError("Rozpracovaný beh má odlišnú konfiguráciu.")
        if (run_dir / "last_training_state.pt").is_file():
            raise ValueError("Existuje dokončená epocha; použi automatické obnovenie cez train_detector_v4.py.")
        for name, expected in config["source_sha256"].items():
            if sha256_file(run_dir / "snapshot" / name) != expected:
                raise ValueError(f"Zmenená snímka: {name}")
    else:
        # Even an empty existing run is not silently overwritten.
        run_dir.mkdir(parents=True, exist_ok=False)
        json_write(run_dir / "config.json", config)
        snapshot = run_dir / "snapshot"
        snapshot.mkdir()
        for name, location in source_paths().items():
            (snapshot / name).write_bytes(location.read_bytes())
        json_write(run_dir / "validation_selection.json", {"seed": args.seed,
                    "max_historical_per_class": args.max_historical_val_per_class,
                    "sample_ids": [row["sample_id"] for row in val_rows], "conditions": CONDITIONS,
                    "scope": config["validation_scope"]})
    print(f"Výstupy nového tréningu: {run_dir}", flush=True)
    lock_path = run_dir / ".training.lock"
    lock_context = process_lock(lock_path)
    lock_context.__enter__()
    try:
        model.to(args.device)
        optimizer = torch.optim.AdamW([
            {"params": [p for name, p in model.named_parameters() if not name.startswith("classifier.")], "lr": args.backbone_lr, "name": "backbone"},
            {"params": model.classifier.parameters(), "lr": args.head_lr, "name": "classifier"},
        ], weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0.0)
        scaler = torch.amp.GradScaler("cuda", enabled=args.device == "cuda")
        if args.resume:
            state = torch.load(run_dir / "last_training_state.pt", map_location="cpu", weights_only=False)
            validate_resume(state, config, run_dir)
            model.load_state_dict(state["model_state_dict"])
            optimizer.load_state_dict(state["optimizer_state_dict"])
            scheduler.load_state_dict(state["scheduler_state_dict"])
            scaler.load_state_dict(state["scaler_state_dict"])
            restore_rng(state["rng_state"])
            start_epoch = int(state["epoch"]) + 1
            best_score, best_epoch, bad_epochs, history = state["selection_score"], state["best_epoch"], state["bad_epochs"], state["history"]
            # A crash after saving a candidate but before committing the epoch
            # may leave best_model.pt ahead of last_training_state.pt.
            shutil.copyfile(best_checkpoint_path(run_dir, best_epoch), run_dir / "best_model.pt")
            del state
            baseline = json.loads((run_dir / "baseline_metrics.json").read_text(encoding="utf-8"))
        else:
            print("Meriam pôvodný model na rovnakej vývojovej validácii pred učením.", flush=True)
            baseline = evaluate(model, val_rows, args, architecture["image_size"], run_dir / "baseline_predictions.csv")
            json_write(run_dir / "baseline_metrics.json", baseline)
            best_score, best_epoch, bad_epochs, history, start_epoch = baseline["selection_score"], 0, 0, [], 1
            save_best(run_dir, model, config, 0, best_score)
            save_resume(run_dir, model, config, 0, best_score, 0, 0, optimizer, scheduler, scaler, history)
        dataset = ManifestDataset(rows, MildTrainingTransform(architecture["image_size"], args.augmentation_policy))
        for epoch in range(start_epoch, args.epochs + 1):
            if bad_epochs >= args.patience:
                print("V predchádzajúcej epoche bola splnená podmienka skorého zastavenia.", flush=True)
                break
            seed_all(args.seed + epoch)
            sampler = CappedBalancedSampler(rows, args.samples_per_epoch, args.max_group_repeats, args.seed + epoch)
            json_write(run_dir / f"sampling_epoch_{epoch:03d}.json", sampler.report)
            data_loader = loader(dataset, args, sampler=sampler, seed=args.seed + epoch)
            current_lr = [group["lr"] for group in optimizer.param_groups]
            train = train_epoch(model, data_loader, optimizer, scaler, args, epoch)
            validation = evaluate(model, val_rows, args, architecture["image_size"])
            improved = validation["selection_score"] > best_score + args.min_delta
            if improved:
                best_score, best_epoch, bad_epochs = validation["selection_score"], epoch, 0
                save_best(run_dir, model, config, epoch, best_score)
            else:
                bad_epochs += 1
            history.append({"epoch": epoch, "learning_rates": current_lr, "training": train,
                            "validation": validation, "improved": improved, "sampling": sampler.report})
            scheduler.step()
            save_resume(run_dir, model, config, epoch, best_score, best_epoch, bad_epochs, optimizer, scheduler, scaler, history)
            json_write(run_dir / "history.json", history)
            print(f"Epocha {epoch}: robustné skóre {validation['selection_score']:.6f}; najlepšie {best_score:.6f} (epocha {best_epoch}).", flush=True)
        best = torch.load(run_dir / "best_model.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(best["model_state_dict"])
        del best
        after = evaluate(model, val_rows, args, architecture["image_size"], run_dir / "best_predictions.csv")
        comparison = {"completed_utc": utc_now(), "scope": config["validation_scope"], "baseline": baseline,
                      "best": after, "best_epoch": best_epoch, "trained_epochs": len(history),
                      "selection_score_delta": after["selection_score"] - baseline["selection_score"],
                      "improved_over_original": best_epoch > 0,
                      "per_condition_delta": {condition: {
                          "hierarchical_balanced_accuracy": after["conditions"][condition]["hierarchical_balanced_accuracy"] - baseline["conditions"][condition]["hierarchical_balanced_accuracy"],
                          "worst_cell_accuracy": after["conditions"][condition]["worst_cell_accuracy"] - baseline["conditions"][condition]["worst_cell_accuracy"]} for condition in CONDITIONS},
                      "selector_status": "A new detector requires new degradation profiles, selector fitting and separate calibration. No selector has been trained or recalibrated by this script.",
                      "test_status": "No external test was run. A new untouched test is required for generalization claims."}
        json_write(run_dir / "comparison.json", comparison)
        files = ["config.json", "validation_selection.json", "baseline_metrics.json", "baseline_predictions.csv", "best_predictions.csv", "comparison.json", "best_model.pt", "last_training_state.pt"]
        if (run_dir / "history.json").is_file():
            files.append("history.json")
        files.extend(f"sampling_epoch_{item['epoch']:03d}.json" for item in history)
        json_write(run_dir / "complete.json", {"completed_utc": utc_now(), "best_epoch": best_epoch,
                  "files": {name: sha256_file(run_dir / name) for name in files}})
        print(f"Tréning skončil. Porovnanie: {run_dir / 'comparison.json'}", flush=True)
        if best_epoch == 0:
            print("Nové váhy nezlepšili výberové skóre. best_model.pt obsahuje pôvodné váhy; nie je deklarované zlepšenie.", flush=True)
        return 0
    finally:
        lock_context.__exit__(None, None, None)


# Release entry point: require an explicit research-job request.
if __name__ == '__main__':
    import sys as _release_sys
    if not any(arg in _release_sys.argv for arg in ('--help', '-h', '--execute-research')):
        print('Research source. Read src/RESEARCH_CODE.txt before running a job. '
              'Use --help to inspect options or --execute-research to run explicitly.')
        raise SystemExit(0)
    if '--execute-research' in _release_sys.argv:
        _release_sys.argv.remove('--execute-research')

if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Prerušené. Dokončenú epochu možno obnoviť cez --resume RUN_DIR.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as error:
        print(f"Chyba: {error}", file=sys.stderr)
        raise SystemExit(2)
