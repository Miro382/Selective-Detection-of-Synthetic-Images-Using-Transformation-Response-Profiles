"""Descriptive leave-one-group-out analysis of archived ablation scores.

Run `python analyze.py --check` to verify packaged inputs and results, or
`python analyze.py` to regenerate the analysis/table. No training, inference,
bootstrap resampling, threshold fitting, or image access is performed.
`--extract-from RUN` prepares the portable inputs from a completed source run.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PAPER = HERE.parent
TIE_NAMESPACE = "article-ablation-confirmation-tie-v1/"
FRACTIONS = {"DeepShield": (9807, 10000), "OpenFake": (3639, 4000)}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def text_json(value):
    return json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"


def extract(run):
    run = Path(run).resolve()
    results = read(run / "results.json")
    protocol = read(run / "run_protocol.json")
    sources = {}
    for name in ("results.json", "run_protocol.json", "plan.jsonl", "profiles.jsonl",
                 "profiles.complete.json", "code/statistics.py"):
        sources[name] = {"sha256": sha(run / name), "bytes": (run / name).stat().st_size}
    expected = {"run_protocol.json": results["inputs"]["run_protocol_sha256"],
                "plan.jsonl": results["inputs"]["plan_sha256"],
                "profiles.jsonl": results["inputs"]["profiles_sha256"],
                "profiles.complete.json": results["inputs"]["profiles_complete_sha256"],
                "code/statistics.py": results["inputs"]["statistics_sha256"]}
    for name, digest in expected.items():
        assert sources[name]["sha256"] == digest, name
    assert sha(run / "run_protocol.json") == (run / "run_protocol.sha256").read_text().strip()
    assert protocol["groups"] == {"DeepShield": 1500, "OpenFake": 43}
    plans = [json.loads(line) for line in (run / "plan.jsonl").read_text(encoding="utf-8").splitlines()]
    profiles = {p["sample_id"]: p for p in
                (json.loads(line) for line in (run / "profiles.jsonl").read_text(encoding="utf-8").splitlines())}
    assert len(plans) == len(profiles) == 7500
    assert len({p["sample_id"] for p in plans}) == 7500
    rows = []
    for plan in plans:
        p = profiles[plan["sample_id"]]
        assert int(p["prediction"] != plan["label"]) == p["error"]
        rows.append({"cohort": plan["dataset"], "error": p["error"],
                     "full_score": repr(p["full_score"]), "resize_score": repr(p["resize_score"]),
                     "tie_key": hashlib.sha256((TIE_NAMESPACE + p["sample_id"]).encode()).hexdigest(),
                     "group_id": hashlib.sha256(("cluster-sensitivity-group-v1/" + plan["group"]).encode()).hexdigest()})
    path = HERE / "scores.csv"
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
    baseline = {name: {"images": c["profiles"], "groups": c["groups"], "detector_errors": c["detector_errors"],
                       "full": {"aurc": c["models"]["full"]["aurc"], **c["models"]["full"]["risks"]["matched"]},
                       "resize": {"aurc": c["models"]["resize"]["aurc"], **c["models"]["resize"]["risks"]["matched"]}}
                for name, c in results["cohorts"].items()}
    (HERE / "baseline.json").write_text(text_json(baseline), encoding="utf-8")
    provenance = {"source_run_name": run.name, "source_files": sources,
                  "scores_sha256": sha(path), "baseline_sha256": sha(HERE / "baseline.json"),
                  "stored_decision": results["decision"], "stored_all_four_conditions_passed": results["all_four_conditions_passed"],
                  "extractor": "analyze.py --extract-from RUN", "source_files_mutated": False,
                  "identifiers": "Original paths/identities removed; original tie hashes retained, groups namespace-hashed.",
                  "scope": "Existing predictions only. No retraining, image access, inference or new bootstrap draws."}
    (HERE / "provenance.json").write_text(text_json(provenance), encoding="utf-8")
    assert all(sha(run / name) == val["sha256"] for name, val in sources.items())


def load():
    provenance = read(HERE / "provenance.json")
    assert sha(HERE / "scores.csv") == provenance["scores_sha256"]
    assert sha(HERE / "baseline.json") == provenance["baseline_sha256"]
    with (HERE / "scores.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 7500 and len({r["tie_key"] for r in rows}) == 7500
    for r in rows:
        r["error"] = int(r["error"])
        r["full_score"] = float(r["full_score"])
        r["resize_score"] = float(r["resize_score"])
        assert r["error"] in (0, 1)
        assert math.isfinite(r["full_score"]) and math.isfinite(r["resize_score"])
        assert r["cohort"] in FRACTIONS
    return rows, provenance


def metric(rows, score, fraction):
    """Same whole-tie-block AURC and exact floor(q*N) as the locked protocol."""
    scores = np.array([r[score] for r in rows], dtype=np.float64)
    errors = np.array([r["error"] for r in rows], dtype=np.int64)
    ties = np.array([r["tie_key"] for r in rows])
    order = np.lexsort((ties, scores))
    ordered_scores = scores[order]
    ends = np.r_[np.flatnonzero(ordered_scores[1:] != ordered_scores[:-1]), len(rows) - 1]
    counts = np.arange(1, len(rows) + 1, dtype=np.int64)
    mistakes = np.cumsum(errors[order], dtype=np.int64)
    coverage = counts[ends] / len(rows)
    aurc = float(np.diff(np.r_[0.0, coverage]) @ (mistakes[ends] / counts[ends]))
    k = fraction[0] * len(rows) // fraction[1]
    accepted_errors = int(mistakes[k - 1]) if k else 0
    return {"aurc": aurc, "accepted": k, "errors": accepted_errors,
            "risk": accepted_errors / k if k else None, "coverage": k / len(rows)}


def summarize(rows, cohort):
    full = metric(rows, "full_score", FRACTIONS[cohort])
    resize = metric(rows, "resize_score", FRACTIONS[cohort])
    return {"images": len(rows), "groups": len({r["group_id"] for r in rows}),
            "detector_errors": sum(r["error"] for r in rows), "full": full, "resize": resize,
            "delta_aurc": resize["aurc"] - full["aurc"],
            "delta_risk_pp": 100 * (resize["risk"] - full["risk"]),
            "delta_errors": resize["errors"] - full["errors"]}


def analyze():
    rows, provenance = load()
    baseline = read(HERE / "baseline.json")
    original = {}
    comparisons = 0
    for cohort in FRACTIONS:
        current = summarize([r for r in rows if r["cohort"] == cohort], cohort)
        for key in ("images", "groups", "detector_errors"):
            assert current[key] == baseline[cohort][key]
            comparisons += 1
        for model in ("full", "resize"):
            for key in ("aurc", "accepted", "errors", "risk", "coverage"):
                old, new = baseline[cohort][model][key], current[model][key]
                if isinstance(old, int):
                    assert new == old
                else:
                    assert math.isclose(new, old, abs_tol=1e-15, rel_tol=0), (cohort, model, key, old, new)
                comparisons += 1
        original[cohort] = current
    of = [r for r in rows if r["cohort"] == "OpenFake"]
    groups = sorted({r["group_id"] for r in of})
    deleted = []
    for group in groups:
        retained = [r for r in of if r["group_id"] != group]
        c = summarize(retained, "OpenFake")
        c.update({"omitted_group": group, "omitted_images": len(of) - len(retained)})
        deleted.append(c)
    largest = max(deleted, key=lambda r: r["omitted_images"])
    assert largest["omitted_images"] == 3412
    return {"version": "posthoc-cluster-sensitivity-v1", "source_scores_sha256": provenance["scores_sha256"],
            "scope": "Retrospective descriptive deletion sensitivity, not a new confirmatory test or replacement decision.",
            "metric_definition": "Whole-tie-block AURC; retained sample re-ranked, k=floor(3639*N/4000); original hash tie ordering.",
            "source_point_checks": comparisons, "source_point_tolerance": 1e-15,
            "original": original, "leave_one_group_out": deleted, "largest_group_deleted": largest,
            "summary": {"deletions": len(deleted),
                        "aurc_negative": sum(d["delta_aurc"] < 0 for d in deleted),
                        "risk_negative": sum(d["delta_risk_pp"] < 0 for d in deleted),
                        "risk_zero": sum(d["delta_risk_pp"] == 0 for d in deleted),
                        "risk_positive": sum(d["delta_risk_pp"] > 0 for d in deleted),
                        "delta_aurc_range": [min(d["delta_aurc"] for d in deleted), max(d["delta_aurc"] for d in deleted)],
                        "delta_risk_pp_range": [min(d["delta_risk_pp"] for d in deleted), max(d["delta_risk_pp"] for d in deleted)]},
            "uncertainty_intervals_computed": False,
            "locked_decision_unchanged": provenance["stored_decision"],
            "training_inference_or_new_data": False}


def render_table(result):
    rows = []
    for name, r in (("Complete OF sample", result["original"]["OpenFake"]),
                    ("Largest group omitted", result["largest_group_deleted"])):
        rows.append(f"{name} & {r['images']:,} & {r['groups']} & {r['full']['accepted']:,} & "
                    f"{r['full']['aurc']:.6f} / {r['resize']['aurc']:.6f} & "
                    f"{r['full']['errors']} / {r['resize']['errors']} & {r['delta_risk_pp']:+.4f} " + r"\\")
    return r"""\begin{table*}[t]
\caption{Retrospective deletion sensitivity for the independent OpenFake ablation. AURC and accepted-error pairs are full / resize-only; risk differences are resize-only minus full, in percentage points. Each retained sample is reranked and both methods accept $\lfloor(3639/4000)N\rfloor$ cases. These are descriptive results, without new intervals or a revised replacement decision.}
\label{tab:cluster_sensitivity}
\centering\small
\setlength{\tabcolsep}{4pt}
\begin{tabular}{lrrrcrr}
\toprule
Sample & Images & Groups & Accepted & AURC: full / resize & Errors: full / resize & $\Delta$ risk \\
\midrule
""" + "\n".join(rows) + "\n" + r"""\bottomrule
\end{tabular}
\end{table*}
"""


def render_csv(result):
    stream = io.StringIO(newline="")
    columns = ["omitted_group", "omitted_images", "images", "groups", "accepted", "full_aurc", "resize_aurc",
               "full_errors", "resize_errors", "delta_aurc", "delta_risk_pp"]
    writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    for d in result["leave_one_group_out"]:
        writer.writerow({"omitted_group": d["omitted_group"], "omitted_images": d["omitted_images"],
                         "images": d["images"], "groups": d["groups"], "accepted": d["full"]["accepted"],
                         "full_aurc": d["full"]["aurc"], "resize_aurc": d["resize"]["aurc"],
                         "full_errors": d["full"]["errors"], "resize_errors": d["resize"]["errors"],
                         "delta_aurc": d["delta_aurc"], "delta_risk_pp": d["delta_risk_pp"]})
    return stream.getvalue()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extract-from", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.extract_from:
        if args.check:
            parser.error("--check must not extract or mutate inputs")
        extract(args.extract_from)
    result = analyze()
    outputs = {HERE / "results.json": text_json(result), HERE / "leave_one_group_out.csv": render_csv(result),
               PAPER / "tables/cluster_sensitivity.tex": render_table(result)}
    for path, text in outputs.items():
        if args.check:
            assert path.read_text(encoding="utf-8") == text, str(path)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8", newline="\n")
    print(text_json({"status": "passed", "point_checks": result["source_point_checks"],
                     "summary": result["summary"], "largest_group_deleted": result["largest_group_deleted"]}))


if __name__ == "__main__":
    main()
