from collections import Counter
from pathlib import Path
import os
import shutil
import sys
import uuid
import numpy as np
from .common import read, write, sha, lines, environment, load_module, verify
from .features import profile_features, logistic_score, methods


def original_modules(selector):
    code = Path(selector) / "code"
    load_module(code / "selector_v4_math.py", "selector_v4_math")
    features = load_module(code / "selector_v4_features.py", "selector_v4_features")
    trainer = load_module(code / "train_selector_v4.py", "comparison_original_trainer")
    return trainer, features


def validate_rows(rows, plans, p, features):
    if len(rows) != len(plans) or len({r["sample_id"] for r in rows}) != len(rows):
        raise ValueError("Počet alebo jedinečnosť profilov nesedí.")
    keys = ("sample_id", "original_id", "group", "sha256", "label", "input_spec", "split")
    for row, plan in zip(rows, plans):
        if any(row[k] != plan[k] for k in keys):
            raise ValueError("Profil nezodpovedá plánu: " + plan["sample_id"])
        if row["temperature"] != p["temperature"] or row["model_fingerprint"] != p["model_fingerprint"]:
            raise ValueError("Zmiešané modely alebo teploty v profiloch.")
        if row["detector_error"] != int((row["raw_margins"][0] >= 0) != row["label"]):
            raise ValueError("Nesprávny cieľ učenia selektora.")
    arrays = features.profile_arrays(rows)
    raw = np.asarray([r["raw_margins"] for r in rows], dtype=np.float64)
    m = raw / p["temperature"]
    if not np.allclose(profile_features(m), features.matrix(arrays, "symmetric_profile"), atol=1e-12, rtol=1e-12):
        raise ValueError("Nová implementácia príznakov sa líši od zmrazeného selektora.")
    return {"raw_margins": raw, "margins": m,
            "y": np.asarray([r["detector_error"] for r in rows], dtype=np.int64),
            "labels": np.asarray([r["label"] for r in rows], dtype=np.int64)}


def assert_disjoint(metadata):
    seen = {"sha256": set(), "original_id": set(), "origin": set()}
    for split, rows in metadata.items():
        local = {"sha256": {r["sha256"] for r in rows},
                 "original_id": {r["original_id"] for r in rows},
                 "origin": {r.get("origin_group", r["group"]).removeprefix("deepshield/") for r in rows}}
        for key in seen:
            if seen[key] & local[key]:
                raise ValueError(f"Únik medzi oddeleniami ({split}, {key}).")
            seen[key].update(local[key])


def prepare(project, run):
    project, run = Path(project).resolve(), Path(run).resolve()
    if (run / "protocol.json").exists():
        print("Existujúci pripravený protokol: kontrolujem integritu.", flush=True)
        return verify(run)
    # Rozpracovaná príprava sa zapisuje do vyhradeného dočasného priečinka.
    # Nikdy sa nevymazáva alebo nemení pôvodný experiment.
    if run.exists():
        raise ValueError("Výstup existuje bez protokolu; zvoľte nový --run.")
    destination = run
    source = project / "selective_experiments/experiment_v4/selector_profiles_v4"
    selector = project / "selective_experiments/experiment_v4/selector_training_v4"
    external = project / "selective_experiments/external_v4_confirmation_reduced_20261017"
    trainer, features = original_modules(selector)
    print("Kontrolujem pôvodný model, uzamknuté profily a oddelenia dát...", flush=True)
    # Kalibračné profily sa iba hashujú na overenie pôvodného experimentu;
    # nečítajú sa ich záznamy a nikdy sa nepoužijú pri novom učení ani hodnotení.
    context = trainer.source_context(source)
    sp = context["protocol"]
    artifact = read(selector / "selector.json")
    ep = read(external / "protocol.json"); complete = read(external / "complete.json")
    for key in ("temperature", "model_sha256", "model_fingerprint", "profile_config"):
        if ep[key] != sp[key] or artifact[key] != sp[key]:
            raise ValueError("Externý test/pôvodný selektor používal iný detektor: " + key)
    if ep["selector_sha256"] != sha(selector / "selector.json"):
        raise ValueError("Zmenený pôvodný selektor.")
    if complete["protocol_sha256"] != sha(external / "protocol.json"):
        raise ValueError("Zmenený protokol externého testu.")
    if complete["profiles_sha256"] != sha(external / "profiles.jsonl"):
        raise ValueError("Externý test nie je kompletný alebo sa zmenil.")
    if ep["files"]["plan.jsonl"] != sha(external / "plan.jsonl"):
        raise ValueError("Zmenený plán externého testu.")
    for key, expected in sp["environment"].items():
        got = environment()[key]
        if isinstance(expected, dict):
            if any(got[n] != v for n, v in expected.items()):
                raise ValueError("Nezodpovedajú verzie knižníc profilov.")
        elif got != expected:
            raise ValueError("Nezodpovedá verzia Pythonu.")
    run = destination.with_name("." + destination.name + ".preparing-" + uuid.uuid4().hex)
    run.mkdir(parents=True, exist_ok=False)
    data = run / "data"; data.mkdir(exist_ok=True)
    all_meta, counts = {}, {}
    anchors = {str(source / "protocol.json"): sha(source / "protocol.json"),
               str(selector / "selector.json"): sha(selector / "selector.json"),
               str(external / "complete.json"): sha(external / "complete.json"),
               str(external / "profiles.jsonl"): complete["profiles_sha256"]}
    for split in ("train", "selection", "external"):
        print("Importujem existujúce profily: " + split, flush=True)
        if split == "external":
            plans = list(lines(external / "plan.jsonl"))
            rows = [r["profile"] for r in lines(external / "profiles.jsonl")]
            if len(rows) != complete["count"]:
                raise ValueError("Neúplný externý test.")
        else:
            name = "selector_" + split
            rows = trainer.load_records(context, name)
            plans = list(lines(source / "plans" / (name + ".jsonl")))
            anchors[str(source / "profiles" / (name + ".jsonl"))] = context["receipts"][name]["sha256"]
        arrays = validate_rows(rows, plans, sp, features)
        arrays["frozen_scores"] = logistic_score(artifact["model"]["parameters"], profile_features(arrays["margins"]))
        np.savez_compressed(data / (split + ".npz"), **arrays)
        write(data / (split + ".json"), plans)
        all_meta[split] = plans
        counts[split] = {"profiles": len(plans), "originals": len({r["sha256"] for r in plans}),
                         "groups": len({r["group"] for r in plans}),
                         "sources": dict(Counter(r["source"] for r in plans)),
                         "cohorts": dict(Counter(r.get("cohort", "development") for r in plans))}
        del rows, arrays
    assert_disjoint(all_meta)
    # Toto nie je nová kontrola pHash celého pôvodného tréningu: odkazujeme na
    # už vykonanú kontrolu pôvodu a duplicít v uzamknutom externom protokole.
    write(run / "frozen_selector.json", artifact)
    frozen_code = run / "detector_runtime"; frozen_code.mkdir(exist_ok=True)
    for f in (source / "snapshot/code").glob("*.py"):
        shutil.copy2(f, frozen_code / f.name)
    shutil.copy2(external / "protocol.json", run / "external_protocol.json")
    shutil.copy2(source / "protocol.json", run / "source_protocol.json")
    code_files = {f.name: sha(f) for f in sorted(Path(__file__).parent.glob("*.py"))}
    code_copy = run / "code"; code_copy.mkdir(exist_ok=True)
    for name in code_files:
        shutil.copy2(Path(__file__).parent / name, code_copy / name)
    protocol = {"version": "selector-comparison-1", "project": str(project), "source": str(source),
                "selector": str(selector), "external": str(external), "anchors": anchors,
                "environment": environment(), "model_sha256": sp["model_sha256"],
                "model_fingerprint": sp["model_fingerprint"], "temperature": sp["temperature"],
                "frozen_threshold": artifact["accept_threshold"], "counts": counts,
                "methods": {s: methods(s) for s in ("cached", "full")},
                "training": {"C": .1, "seeds": [17,29,43], "hidden": [70,70], "dropout": .5,
                             "epochs": 80, "patience": 10, "batch_size": 512, "lr": .001,
                             "epoch_selection": "selection_AURC; earliest minimum",
                             "MLP_loss": "weighted BCE, inverse group size and balanced error/correct classes",
                             "scaler": "train-only weighted StandardScaler",
                             "device": "cpu"},
                "statistics": {"bootstrap": 1000, "seed": 31019, "coverage": [.8,.9,.95],
                               "primary": "AURC by cohort, paired connected-group bootstrap",
                               "ties": "AURC whole score blocks; fixed coverage SHA256(sample_id) without labels",
                               "multiple_comparisons": "exploratory, pointwise unadjusted intervals; no confirmatory p-values"},
                "scope": "Retrospektívne porovnanie na už vyhodnotených dátach; nie nová nezávislá certifikácia.",
                "calibration": "Pôvodná kalibračná množina je vylúčená z nového učenia aj testu.",
                "code": code_files}
    protocol["files"] = {str(f.relative_to(run)).replace("\\", "/"): sha(f)
                         for f in run.rglob("*") if f.is_file()}
    write(run / "protocol.json", protocol)
    (run / "protocol.sha256").write_text(sha(run / "protocol.json") + "\n", encoding="ascii")
    os.replace(run, destination)
    print("Príprava dokončená. Cache: 40 000 tréningových + 2 000 výberových + 14 000 externých profilov.", flush=True)
    return protocol
