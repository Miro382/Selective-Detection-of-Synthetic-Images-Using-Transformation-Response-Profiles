"""Jednorazová kalibrácia teploty dokončeného detektora na samostatnom splite.

Nemení váhy ani rozhodnutia pri prahu 0,5. Opakované spustenie iba overí
už zmrazený výsledok. --no-temperature explicitne zmrazí T=1 bez učenia.
"""
from __future__ import annotations
import argparse
import json
import math
import os
from pathlib import Path
import sys
import time

# PyCharm môže používať iné prostredie než overený PyTorch runtime.
# Interpreter selection is controlled by the calling environment.

import numpy as np
from PIL import Image
from scipy.optimize import minimize_scalar
from scipy.special import expit
from detector_training_v4 import (json_write, sha256_file, utc_now, verify_completed_training,
    verify_temperature, load_frozen_detector, predict_logits, seed_all, process_lock)
from prepare_data_v4 import verify_dataset, load_manifest


def fit_temperature(raw_logits, labels):
    logits, labels = np.asarray(raw_logits, dtype=np.float64), np.asarray(labels, dtype=np.int64)
    if logits.ndim != 1 or labels.shape != logits.shape or logits.size < 2 or not np.isfinite(logits).all() or set(labels.tolist()) != {0, 1}:
        raise ValueError("Kalibrácia potrebuje konečné logity a obe triedy.")
    def nll(log_temperature):
        z = logits / math.exp(log_temperature)
        return float(np.mean(np.logaddexp(0.0, z) - labels * z))
    lower, upper = math.log(0.01), math.log(100.0)
    result = minimize_scalar(nll, bounds=(lower, upper), method="bounded", options={"xatol": 1e-10})
    if not result.success or not math.isfinite(result.fun):
        raise RuntimeError("Odhad teploty nekonvergoval.")
    # Include T=1 and both boundaries; never report a worse NLL than no scaling.
    candidates = [0.0, lower, upper, float(result.x)]
    best = min(candidates, key=nll)
    temperature = math.exp(best)
    return {"temperature": temperature, "bounds": [0.01, 100.0],
            "at_boundary": abs(best - lower) < 1e-6 or abs(best - upper) < 1e-6,
            "nll_before": nll(0.0), "nll_after": nll(best),
            "brier_before": float(np.mean((expit(logits) - labels) ** 2)),
            "brier_after": float(np.mean((expit(logits / temperature) - labels) ** 2)),
            "decision_changes_at_half": int(np.count_nonzero((logits >= 0) != (logits / temperature >= 0)))}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parent
    parser.add_argument("--run-dir", type=Path, default=root / "detector_experiments" / "detector_v4" / "training_main")
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--no-temperature", action="store_true")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    run = args.run_dir.resolve()
    config, complete = verify_completed_training(run)
    data = (args.data_dir or Path(config["training"]["data_dir"])).resolve()
    if sha256_file(data / "data_protocol.json") != config["data_protocol_sha256"]:
        raise ValueError("Dátový protokol sa po tréningu zmenil.")
    model_sha = complete["files"]["best_model.pt"]
    if (run / "temperature.complete.json").is_file():
        artifact = verify_temperature(run, model_sha)
        if args.no_temperature and artifact["method"] != "fixed_identity":
            raise ValueError("Teplota už bola naučená a zmrazená; nedá sa zmeniť na T=1.")
        print(f"Kalibrácia už dokončená a overená: T={artifact['temperature']:.6g}")
        return 0
    if args.check:
        raise ValueError("Kalibrácia ešte nie je dokončená.")
    verify_dataset(data, full_hashes=False)
    rows = load_manifest(data / "detector_calibration.csv")
    if not rows or any(row["split"] != "detector_calibration" for row in rows):
        raise ValueError("Nesprávny kalibračný split.")
    if len({row["dependency_key"] for row in rows}) != len(rows):
        raise ValueError("Kalibrácia musí obsahovať jedného reprezentanta každej nezávislej skupiny.")
    if any(row.get("platform") not in ("PreSocial", "Original", "original", "clean", "") for row in rows):
        raise ValueError("Kalibrácia vyžaduje originálne reprezentanty.")
    script_sha = sha256_file(Path(__file__))
    if script_sha != config["source_sha256"]["calibrate_detector_v4.py"]:
        raise ValueError("Kalibračný kód sa od tréningu zmenil; použi snapshot tréningu.")
    protocol = {"schema_version": "v4-temperature-1", "model_sha256": model_sha,
                "training_complete_sha256": sha256_file(run / "complete.json"),
                "data_protocol_sha256": sha256_file(data / "data_protocol.json"),
                "manifest_sha256": sha256_file(data / "detector_calibration.csv"),
                "method": "fixed_identity" if args.no_temperature else "scalar_nll_temperature",
                "bounds": [0.01, 100.0], "device": args.device, "seed": 20261015,
                "code_sha256": script_sha, "detector_threshold": 0.5,
                "n_groups": len(rows), "condition": "original_only",
                "scope": "Development score calibration; no risk guarantee and no external test."}
    protocol_path = run / "temperature_protocol.json"
    lock = run / ".temperature.lock"
    lock_context = process_lock(lock)
    lock_context.__enter__()
    try:
        if protocol_path.is_file():
            if json.loads(protocol_path.read_text(encoding="utf-8")) != protocol:
                raise ValueError("Rozpracovaná kalibrácia má iný protokol.")
        else:
            json_write(protocol_path, protocol)
        cache = run / "temperature_logits.jsonl"
        records = []
        if cache.is_file():
            raw = cache.read_bytes()
            # Only an interrupted final line can be discarded.
            if raw and not raw.endswith(b"\n"):
                raw = raw[:raw.rfind(b"\n") + 1]
                cache.write_bytes(raw)
            for line in raw.splitlines():
                records.append(json.loads(line))
        if len(records) > len(rows) or (args.no_temperature and records):
            raise ValueError("Kalibračná cache nezodpovedá protokolu.")
        for record, row in zip(records, rows):
            if record["sample_id"] != row["sample_id"] or record["sha256"] != row["sha256"] or record["label"] != int(row["label"]) or not math.isfinite(record["raw_logit"]):
                raise ValueError("Kalibračná cache má zmenený obsah alebo poradie.")
        if not args.no_temperature and len(records) < len(rows):
            import torch
            seed_all(20261015)
            torch.set_num_threads(4)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.use_deterministic_algorithms(True, warn_only=True)
            model, transform, info = load_frozen_detector(run, args.device, require_temperature=False)
            last_log = time.monotonic()
            with cache.open("a", encoding="utf-8", newline="\n") as stream:
                for row in rows[len(records):]:
                    if sha256_file(row["path"]) != row["sha256"]:
                        raise ValueError(f"Kalibračný obrázok sa zmenil: {row['sample_id']}")
                    with Image.open(row["path"]) as source:
                        source.load()
                        value = predict_logits(source.convert("RGB"), model, transform, info)
                    record = {"sample_id": row["sample_id"], "sha256": row["sha256"],
                              "label": int(row["label"]), "raw_logit": value["raw_logit"]}
                    stream.write(json.dumps(record, allow_nan=False) + "\n")
                    stream.flush()
                    records.append(record)
                    if len(records) % 100 == 0 or time.monotonic() - last_log >= 30:
                        print(f"Kalibračné logity: {len(records)}/{len(rows)}", flush=True)
                        last_log = time.monotonic()
        if args.no_temperature:
            cache.touch(exist_ok=True)
            fit = {"temperature": 1.0, "decision_changes_at_half": 0}
        else:
            if len(records) != len(rows):
                raise ValueError("Kalibračná cache nie je úplná.")
            fit = fit_temperature([r["raw_logit"] for r in records], [r["label"] for r in records])
        artifact = {**protocol, **fit, "completed_utc": utc_now(), "inference_records": len(records)}
        json_write(run / "temperature.json", artifact)
        json_write(run / "temperature.complete.json", {"completed_utc": utc_now(),
            "files": {name: sha256_file(run / name) for name in ("temperature.json", "temperature_protocol.json", "temperature_logits.jsonl")}})
        print(f"Teplota zmrazená: T={artifact['temperature']:.6g}. Rozhodnutia pri prahu 0,5 sa nezmenili.")
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
        print("Prerušené. Rovnaký príkaz pokračuje vo výpočte kalibračných logitov.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(f"Chyba: {exc}", file=sys.stderr)
        raise SystemExit(2)
