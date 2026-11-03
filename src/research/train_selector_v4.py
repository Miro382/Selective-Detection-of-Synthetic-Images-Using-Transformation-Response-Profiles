"""Učenie selektora V4 bez opätovného spúšťania obrazového detektora.

Run: učenie na selector_train, výber na selector_selection, zmrazenie a jedno
overenie na selector_calibration. --fit-only kalibráciu neotvára. V3 sa nemení.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import importlib.util
from importlib import metadata
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parent
SPLITS = ("selector_train", "selector_selection", "selector_calibration")
CODE_FILES = ("train_selector_v4.py", "selector_v4_features.py", "selector_v4_math.py", "selector_v4_report.py")
TRAINING_VARIANTS = {"clean", "jpeg_mild", "jpeg_strong", "resize_mild", "resize_strong", "brightness", "contrast", "combined"}
DEFAULTS = {"target_risk": .02, "selection_risk": .01, "delta": .05, "min_coverage": .1,
            "min_accepted": 30, "seed": 20261015, "selection_criterion": "minimum_selection_aurc_then_candidate_id"}


def now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=lambda x: (_ for _ in ()).throw(ValueError("Nekonečné číslo JSON.")))


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("." + path.name + "." + uuid.uuid4().hex + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def safe_path(folder, relative):
    path = (Path(folder) / relative).resolve()
    if not path.is_relative_to(Path(folder).resolve()):
        raise ValueError("Cesta opúšťa uzamknutý priečinok: " + str(relative))
    return path


def verify_files(folder, files):
    for relative, expected in files.items():
        if sha256(safe_path(folder, relative)) != expected:
            raise ValueError("Zmenený uzamknutý súbor: " + str(relative))


@contextmanager
def lock(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # The OS releases this lock even when Python is killed or power is lost.
    # A remaining zero-byte/one-byte lock file is not a stale reservation.
    with path.open("a+b") as stream:
        stream.seek(0, 2)
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise ValueError("V tomto priečinku už prebieha výpočet: " + str(path)) from exc
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def validate_partitions(partitions):
    if set(partitions) != set(SPLITS):
        raise ValueError("Nesprávne oddelenia profilov V4.")
    seen = {"group": set(), "original_id": set(), "sha256": set()}
    for split in SPLITS:
        rows = partitions[split]
        if not rows:
            raise ValueError("Prázdne oddelenie: " + split)
        local = {key: set() for key in seen}
        for row in rows:
            for key in seen:
                if not isinstance(row[key], str) or not row[key]:
                    raise ValueError("Chýbajúca identita predlohy.")
                local[key].add(row[key])
            if len(row["sha256"]) != 64 or any(c not in "0123456789abcdef" for c in row["sha256"]):
                raise ValueError("Neplatný hash predlohy.")
        if len(local["original_id"]) != len(rows) or len(local["sha256"]) != len(rows):
            raise ValueError("Duplicitné predlohy v oddelení.")
        if split != "selector_train" and len(local["group"]) != len(rows):
            raise ValueError("Výber a overenie musia mať iba jedného reprezentanta skupiny.")
        for key in seen:
            if seen[key] & local[key]:
                raise ValueError("Prekryv oddelení podľa " + key)
            seen[key].update(local[key])
    return digest(sorted(row["sha256"] for row in partitions["selector_calibration"]))


def source_context(source):
    """Kontrolné súčty kalibrácie sa čítajú; jej riadky a štítky ešte nie."""
    source = Path(source).resolve()
    protocol = read_json(source / "protocol.json")
    psha = sha256(source / "protocol.json")
    if psha != (source / "protocol.sha256").read_text().strip() or protocol.get("version") != "selector-v4-profiles-1":
        raise ValueError("Neplatný protokol profilov V4.")
    verify_files(source, protocol["locked_files"])
    partition_file = protocol.get("partition_file", "partitions.json")
    if partition_file not in protocol["locked_files"]:
        raise ValueError("Identita oddelení nie je uzamknutá.")
    partitions = read_json(safe_path(source, partition_file))
    cal_digest = validate_partitions(partitions)
    if "calibration_partition_digest" in protocol and protocol["calibration_partition_digest"] != cal_digest:
        raise ValueError("Nesúlad identity kalibračných predlôh.")
    partition_digest = digest({s: sorted([[r["original_id"], r["group"], r["sha256"]] for r in rows]) for s, rows in sorted(partitions.items())})
    if "partition_digest" in protocol and protocol["partition_digest"] != partition_digest:
        raise ValueError("Nesúlad identity oddelení.")
    if not math.isfinite(protocol["temperature"]) or protocol["temperature"] <= 0 or protocol["detector_threshold"] != .5:
        raise ValueError("Neplatná teplota alebo prah detektora.")
    receipts = {}
    for split in SPLITS:
        receipt = read_json(source / "profiles" / (split + ".receipt.json"))
        wanted = {"protocol_sha256": psha, "model_sha256": protocol["model_sha256"],
                  "model_fingerprint": protocol["model_fingerprint"], "count": protocol["counts"][split]}
        if any(receipt.get(k) != v for k, v in wanted.items()):
            raise ValueError("Nesúlad potvrdenia profilov: " + split)
        if receipt["sha256"] != sha256(source / "profiles" / (split + ".jsonl")):
            raise ValueError("Zmenené alebo neúplné profily: " + split)
        expected_count = len(partitions[split]) * (8 if split == "selector_train" else 1)
        if wanted["count"] != expected_count:
            raise ValueError("Nesprávny počet reprezentantov alebo tréningových variantov.")
        receipts[split] = receipt
    return {"path": str(source), "protocol": protocol, "protocol_sha256": psha,
            "partitions": partitions, "calibration_partition_digest": cal_digest, "receipts": receipts}


def load_records(context, split):
    rows = []
    path = Path(context["path"]) / "profiles" / (split + ".jsonl")
    expected_sha = context["receipts"][split]["sha256"]
    if sha256(path) != expected_sha:
        raise ValueError("Profily sa zmenili po kontrole vstupov: " + split)
    identities = {r["original_id"]: r for r in context["partitions"][split]}
    seen, counts = set(), Counter()
    variants, labels = {}, {}
    p = context["protocol"]
    with path.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line, parse_constant=lambda s: (_ for _ in ()).throw(ValueError("Nekonečné číslo.")))
            if row["sample_id"] in seen or row["original_id"] not in identities or row["group"] != identities[row["original_id"]]["group"]:
                raise ValueError("Duplicitný alebo cudzí profil.")
            if row["sha256"] != identities[row["original_id"]]["sha256"]:
                raise ValueError("Profil má odlišný obsah pôvodného obrázka.")
            variant = row["input_spec"]["id"]
            if row["sample_id"] != row["original_id"] + "::" + variant:
                raise ValueError("Identita profilu nezodpovedá vonkajšej degradácii.")
            variants.setdefault(row["original_id"], set()).add(variant)
            if row["split"] != split or row["model_fingerprint"] != p["model_fingerprint"] or row["temperature"] != p["temperature"]:
                raise ValueError("Profil patrí inému oddeleniu alebo detektoru.")
            if type(row["label"]) is not int or row["label"] not in (0, 1) or type(row["detector_error"]) is not int:
                raise ValueError("Neplatný štítok alebo chyba detektora.")
            if row["detector_error"] != int((row["raw_margins"][0] >= 0) != row["label"]):
                raise ValueError("Štítok chyby nezodpovedá rozhodnutiu detektora.")
            if labels.setdefault(row["original_id"], row["label"]) != row["label"]:
                raise ValueError("Varianty jednej predlohy majú odlišnú skutočnú triedu.")
            seen.add(row["sample_id"])
            counts[row["original_id"]] += 1
            rows.append(row)
    expected = 8 if split == "selector_train" else 1
    if set(counts) != set(identities) or any(n != expected for n in counts.values()) or len(rows) != p["counts"][split]:
        raise ValueError("Neúplné varianty alebo reprezentanti.")
    if split == "selector_train" and any(v != TRAINING_VARIANTS for v in variants.values()):
        raise ValueError("Tréning neobsahuje osem rôznych predpísaných variantov každej predlohy.")
    if sha256(path) != expected_sha:
        raise ValueError("Profily sa zmenili počas čítania: " + split)
    return rows


def environment():
    return {"python": platform.python_version(), "packages": {n: metadata.version(n) for n in ("numpy", "scipy", "scikit-learn")}}


def load_frozen_modules(output):
    code = str(Path(output) / "code")
    sys.path.insert(0, code)
    result = []
    for name in ("selector_v4_math", "selector_v4_features", "selector_v4_report"):
        spec = importlib.util.spec_from_file_location(name, Path(code) / (name + ".py"))
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        result.append(module)
    return result


def prepare_output(output, context, app_root):
    output = Path(output)
    if (output / "protocol.json").exists():
        p = read_json(output / "protocol.json")
        if sha256(output / "protocol.json") != (output / "protocol.sha256").read_text().strip():
            raise ValueError("Zmenený protokol učenia.")
        if p["source_protocol_sha256"] != context["protocol_sha256"] or p["source_receipts"] != context["receipts"] or p["source"] != context["path"]:
            raise ValueError("Profily sa zmenili od prípravy učenia.")
        if p["environment"] != environment():
            raise ValueError("Zmenené numerické prostredie učenia; obnovte pôvodné verzie.")
        verify_files(output, p["locked_files"])
        if sha256(ROOT / "train_selector_v4.py") != p["locked_files"]["code/train_selector_v4.py"]:
            raise ValueError("Zmenený riadiaci skript učenia. Obnovte jeho uzamknutú verziu zo zložky code.")
        return p
    if output.exists() and any(output.iterdir()):
        raise ValueError("Výstupný priečinok existuje bez platného protokolu.")
    output.mkdir(parents=True, exist_ok=True)
    (output / "code").mkdir(exist_ok=True)
    for name in CODE_FILES:
        shutil.copy2(ROOT / name, output / "code" / name)
    p = {"version": "selector-v4-training-1", "created_utc": now(), "source": context["path"],
         "source_protocol_sha256": context["protocol_sha256"], "source_receipts": context["receipts"],
         "defaults": DEFAULTS, "environment": environment(), "app_root": str(Path(app_root).resolve()),
         "calibration_partition_digest": context["calibration_partition_digest"],
         "locked_files": {"code/" + n: sha256(output / "code" / n) for n in CODE_FILES},
         "scope": "Vývoj a jednorazové overenie pri vopred určenej zmesi zdrojov a degradácií. Žiadna záruka pri zmene distribúcie."}
    save_json(output / "protocol.json", p)
    (output / "protocol.sha256").write_text(sha256(output / "protocol.json") + "\n", encoding="utf-8")
    return p


def fit(output, context, math_module, features):
    import numpy as np
    output = Path(output)
    receipt_path = output / "fit.receipt.json"
    if receipt_path.exists():
        verify_files(output, read_json(receipt_path)["files"])
        return read_json(output / "frozen_selection.json")
    if (output / "frozen_selection.json").exists():
        raise ValueError("Zmrazenie nemá platné potvrdenie; nemožno automaticky meniť model.")
    training = load_records(context, "selector_train")
    selection = load_records(context, "selector_selection")
    arrays, validation = features.profile_arrays(training), features.profile_arrays(selection)
    y = np.asarray([r["detector_error"] for r in training], dtype=np.int64)
    val_y = np.asarray([r["detector_error"] for r in selection], dtype=np.int64)
    weights = features.group_weights(training)
    if len(np.unique(y)) != 2:
        raise ValueError("Selektor potrebuje správne aj nesprávne rozhodnutia detektora v tréningu.")
    if len({r["label"] for r in training}) != 2:
        raise ValueError("Tréning selektora musí obsahovať reálne aj syntetické predlohy.")
    error_groups = len({r["group"] for r in training if r["detector_error"]})
    warnings = []
    if error_groups < 100:
        warnings.append(f"Málo skupín s chybou detektora ({error_groups}); výber selektora môže byť nestabilný.")
    candidates = []
    for i, candidate in enumerate(features.candidate_grid(), 1):
        print(f"[{i}/9] Učím/porovnávam {candidate['id']}...", flush=True)
        model = features.fit_candidate(candidate, arrays, y, weights)
        scores = features.predict_candidate(model, validation)
        curve = math_module._risk_curve(val_y, scores, np.ones(len(val_y)))
        candidates.append({**candidate, "model": model,
                           "selection_aurc": math_module.weighted_aurc(val_y, scores, np.ones(len(val_y))),
                           "curve": [{"coverage": float(c), "risk": float(r)} for c, r in zip(curve["coverage"], curve["risk"])],
                           "threshold": math_module.choose_threshold(val_y, scores, np.ones(len(val_y)), DEFAULTS["selection_risk"], DEFAULTS["min_coverage"])})
    # The selected rule may be a simple baseline if no learned model wins.
    chosen = min(candidates, key=lambda x: (x["selection_aurc"], x["id"]))
    summary = {"candidates": candidates, "chosen_id": chosen["id"], "training_rows": len(training),
               "training_groups": len({r["group"] for r in training}), "training_error_groups": error_groups,
               "selection_rows": len(selection), "selection_detector_errors": int(val_y.sum()), "warnings": warnings,
               "group_weight_sum": float(weights.sum()), "selection_is_development": True}
    save_json(output / "selection_results.json", summary)
    frozen = {"version": "selector-v4-frozen-1", "created_utc": now(), "chosen_id": chosen["id"],
              "model": chosen["model"], "research_threshold": chosen["threshold"]["tau"],
              "threshold_selection": chosen["threshold"], "source_protocol_sha256": context["protocol_sha256"],
              "training_protocol_sha256": sha256(output / "protocol.json"),
              "calibration_partition_digest": context["calibration_partition_digest"],
              "selection_results_sha256": sha256(output / "selection_results.json")}
    save_json(output / "frozen_selection.json", frozen)
    save_json(receipt_path, {"created_utc": now(), "files": {n: sha256(output / n) for n in ("selection_results.json", "frozen_selection.json")},
                            "calibration_opened": False})
    print("Model a prah boli zmrazené pred otvorením kalibrácie: " + chosen["id"], flush=True)
    return frozen


def calibration_key(context):
    # A changed detector must not unlock labels already revealed by a previous
    # calibration. Identity depends on raw images, not on model or output path.
    return context["calibration_partition_digest"]


def certify(output, context, training_protocol, frozen, math_module, features):
    import numpy as np
    output = Path(output)
    registry = Path(training_protocol["app_root"]) / "selective_experiments" / ".selector_v4_calibration_registry"
    registry.mkdir(parents=True, exist_ok=True)
    key = calibration_key(context)
    entry = registry / (key + ".json")
    binding = {"output": str(output.resolve()), "frozen_sha256": sha256(output / "frozen_selection.json"),
               "calibration_sha256": context["receipts"]["selector_calibration"]["sha256"],
               "model_sha256": context["protocol"]["model_sha256"],
               "calibration_partition_digest": context["calibration_partition_digest"]}
    calibration_hashes = sorted(r["sha256"] for r in context["partitions"]["selector_calibration"])
    with lock(registry / ".registry.lock"):
        for previous in registry.glob("*.json"):
            if previous == entry:
                continue
            consumed = read_json(previous).get("calibration_hashes")
            if consumed is None:
                raise ValueError("Starší záznam kalibrácie nemá audit predlôh; nemožno potvrdiť nezávislosť nového overenia.")
            if set(consumed) & set(calibration_hashes):
                raise ValueError("Nová kalibračná množina sa prekrýva s už rezervovanými/odhalenými predlohami.")
        if entry.exists():
            existing = read_json(entry)
            if existing["binding"] != binding:
                raise ValueError("Tieto kalibračné predlohy už boli rezervované pre iné zmrazené pravidlo. Opakované ladenie/overenie je zakázané.")
            if existing["status"] == "complete":
                result = existing["result"]
                save_json(output / "calibration_result.json", result)
                return result
            if existing["status"] != "reserved":
                raise ValueError("Neznámy stav kalibračného registra.")
        else:
            save_json(entry, {"status": "reserved", "created_utc": now(), "binding": binding, "calibration_hashes": calibration_hashes})
        # First access to calibration rows occurs only AFTER immutable fit receipt,
        # frozen selection and global one-use reservation have been verified.
        verify_files(output, read_json(output / "fit.receipt.json")["files"])
        rows = load_records(context, "selector_calibration")
        scores = features.predict_candidate(frozen["model"], features.profile_arrays(rows))
        y = np.asarray([r["detector_error"] for r in rows], dtype=np.int64)
        accepted = scores <= frozen["research_threshold"]
        n, k = int(accepted.sum()), int(y[accepted].sum())
        upper = math_module.clopper_pearson_upper(k, n, DEFAULTS["delta"])
        passed = n >= DEFAULTS["min_accepted"] and upper <= DEFAULTS["target_risk"]
        result = {"status": "passed" if passed else "failed_certification", "total": len(rows),
                  "accepted": n, "review": len(rows) - n, "coverage": n / len(rows), "errors": k,
                  "risk": k / n if n else None, "risk_upper_bound": upper,
                  "confidence": 1 - DEFAULTS["delta"], "target_risk": DEFAULTS["target_risk"],
                  "minimum_accepted": DEFAULTS["min_accepted"], "research_threshold": frozen["research_threshold"],
                  "accept_threshold": frozen["research_threshold"] if passed else -1.,
                  "calibration_partition_digest": context["calibration_partition_digest"], "evaluated_utc": now(),
                  "scope": "Jedno vopred zmrazené pravidlo, nezávislí reprezentanti pri rovnakej cieľovej distribúcii. Bez záruky pri posune distribúcie."}
        save_json(entry, {"status": "complete", "binding": binding, "calibration_hashes": calibration_hashes, "result": result})
        save_json(output / "calibration_result.json", result)
        return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=ROOT / "selective_experiments" / "experiment_v4")
    parser.add_argument("--profiles-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--check", action="store_true")
    modes.add_argument("--fit-only", action="store_true")
    modes.add_argument("--validate-only", action="store_true")
    modes.add_argument("--report-only", action="store_true")
    parser.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    source = (args.profiles_dir or args.run_dir / "selector_profiles_v4").resolve()
    output = (args.output_dir or args.run_dir / "selector_training_v4").resolve()
    if not args._worker:
        python = Path(sys.executable)
        if python.exists() and Path(sys.executable).resolve() != python.resolve():
            command = [str(python), "-B", "-u", str(Path(__file__).resolve()), *(argv if argv is not None else sys.argv[1:]), "--_worker"]
            return subprocess.call(command, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONIOENCODING": "utf-8"})
    print("Overujem dokončené profily V4 a oddelenie predlôh...", flush=True)
    context = source_context(source)
    if args.check:
        print("Kontrolné súčty a oddelenie dát: OK. Kalibračné riadky neboli otvorené.")
        return 0
    if args.validate_only and not (output / "fit.receipt.json").exists():
        raise ValueError("Najprv vykonajte --fit-only; chýba zmrazený model.")
    if args.report_only and not (output / "complete.json").exists():
        raise ValueError("Report-only vyžaduje dokončené jednorazové overenie.")
    output.parent.mkdir(parents=True, exist_ok=True)
    with lock(output.parent / ("." + output.name + ".lock")):
        protocol = prepare_output(output, context, ROOT)
        maths, features, report = load_frozen_modules(output)
        if context["protocol"]["profile_config"] != features.PROFILE_CONFIG:
            raise ValueError("Nepodporovaná konfigurácia profilov.")
        if (output / "complete.json").exists():
            verify_files(output, read_json(output / "complete.json")["files"])
            print("Učenie a overenie už boli dokončené. Report: " + str(output / "report.html"))
            return 0
        entry = Path(protocol["app_root"]) / "selective_experiments" / ".selector_v4_calibration_registry" / (calibration_key(context) + ".json")
        if entry.exists():
            reservation = read_json(entry)
            if reservation["binding"]["output"] != str(output.resolve()) or not (output / "fit.receipt.json").exists():
                raise ValueError("Kalibrácia bola už odhalená/rezervovaná. Nové učenie potrebuje nové kalibračné predlohy.")
        frozen = fit(output, context, maths, features)
        if args.fit_only:
            print("Výber je zmrazený. Kalibrácia sa neotvorila. Pokračujte rovnakým skriptom bez --fit-only.")
            return 0
        result = certify(output, context, protocol, frozen, maths, features)
        p = context["protocol"]
        artifact = {"version": "selective-selector-v4-artifact-1", "model": frozen["model"],
                    "model_sha256": p["model_sha256"], "model_fingerprint": p["model_fingerprint"],
                    "profile_config": p["profile_config"], "detector_threshold": p["detector_threshold"],
                    "temperature": p["temperature"], "research_threshold": frozen["research_threshold"],
                    "accept_threshold": result["accept_threshold"], "calibration_metadata": result,
                    "training_protocol_sha256": sha256(output / "protocol.json"),
                    "score_semantics": "Skóre odhadu chyby; samo osebe nie kalibrovaná pravdepodobnosť.",
                    "scope": protocol["scope"]}
        save_json(output / "selector.json", artifact)
        report.write_report(output, read_json(output / "selection_results.json"), result, artifact)
        files = ("selector.json", "calibration_result.json", "selection_results.json", "frozen_selection.json", "fit.receipt.json", "report.html", "risk_coverage.svg")
        save_json(output / "complete.json", {"created_utc": now(), "files": {n: sha256(output / n) for n in files}})
        print("Učenie V4 dokončené. Jednorazové overenie: " + result["status"])
        print("Report: " + str(output / "report.html"))
        if result["status"] != "passed":
            print("Cieľové riziko sa nepotvrdilo. Nasadzovaný prah odmieta všetko; výskumný prah zostáva uložený.")
        print("Externý test ani nasadenie do aplikácie sa nespustili.")
    return 0


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
    except (ValueError, OSError, KeyError) as exc:
        print("Chyba: " + str(exc), file=sys.stderr)
        raise SystemExit(2)
