"""Rebuild the adapted-detector tables and plot from packaged aggregate data.

Default: portable rendering from figure_data/adaptation_results.json only.
--check: verify numerical consistency and exact generated table contents.
--extract-from RUN: read completed results once and archive aggregate/provenance.
No training, image reads, detector inference, or new bootstrap draws are performed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "figure_data/adaptation_results.json"
PROVENANCE = HERE / "evidence/adaptation_sources.json"
DETECTORS = ("clip", "resnet50")
COHORTS = ("OpenFake", "DeepShield")
METHODS = ("confidence", "tta_mean", "lr_full", "lr_confidence", "lr_jpeg_only", "lr_resize_only")
NAMES = {"clip": "CLIP", "resnet50": "ResNet-50"}
SHORT = {"OpenFake": "OF", "DeepShield": "DS"}
LABELS = {"confidence": "Confidence", "tta_mean": "TTA mean", "lr_full": "Full profile",
          "lr_confidence": "Learned confidence", "lr_jpeg_only": "JPEG-only", "lr_resize_only": "Resize-only"}
COLORS = {"confidence": "#777777", "tta_mean": "#E69F00", "lr_full": "#0072B2",
          "lr_jpeg_only": "#009E73", "lr_resize_only": "#AA4499"}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def seal_ok(value):
    if "integrity_sha256" not in value:
        return
    body = {key: val for key, val in value.items() if key != "integrity_sha256"}
    encoded = json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    if hashlib.sha256(encoded).hexdigest() != value["integrity_sha256"]:
        raise ValueError("Invalid result integrity seal")


def extract(run):
    run = Path(run).resolve()
    files = {}

    def source(relative, expected=None):
        path = (run / relative).resolve()
        if not path.is_relative_to(run):
            raise ValueError("Source escaped experiment root")
        actual = sha(path)
        if expected is not None and actual != expected:
            raise ValueError("Source checksum mismatch: " + relative)
        files[relative] = {"sha256": actual, "bytes": path.stat().st_size}
        return path

    protocol = read(source("protocol.json"))
    if source("protocol.sha256").read_text(encoding="ascii").strip() != sha(run / "protocol.json"):
        raise ValueError("Protocol checksum mismatch")
    freeze = read(source("evaluation_freeze.json"))
    audit = read(source("input_audit.json"))
    summary = read(source("report/summary.json"))
    if not summary["all_detectors_complete"]:
        raise ValueError("Run is not complete")
    data = {"version": "adapted-detector-article-aggregates-v1", "protocol_sha256": sha(run / "protocol.json"),
            "config": protocol["config"], "counts": protocol["counts"], "input_audit": audit,
            "scope": "Retrospective reuse of the original EfficientNet cohorts; separately refitted selectors.",
            "detectors": {}}
    for det in DETECTORS:
        train = read(source(f"{det}/detector/complete.json", summary["checkpoint_selection"][det]["receipt_sha256"]))
        best = read(source(f"{det}/detector/best_validation.json", train["files"]["best_validation.json"]))
        initial = read(source(f"{det}/detector/initial_validation.json", train["files"]["initial_validation.json"]))
        source(f"{det}/detector/history.json", train["files"]["history.json"])
        selector = read(source(f"{det}/selector/complete.json", freeze["development"][det]))
        seal_ok(selector)
        assessment = read(source(f"{det}/selector/assessment.json", selector["assessment_sha256"]))
        seal_ok(assessment)
        source(f"{det}/selector/frozen.json", selector["frozen_sha256"])
        result = read(source(f"{det}/evaluation/results.json", summary["results_sha256"][det]))
        seal_ok(result)
        for relative, expected in result["files"].items():
            source(f"{det}/evaluation/{relative}", expected)
        entry = {"completed_utc": result["completed_utc"], "training": {
            "best_epoch": train["best_epoch"], "epochs_completed": train["epochs_completed"],
            "metadata": train["metadata"], "initial_validation": initial["validation"],
            "best_validation": best["validation"]}, "assessment": assessment, "cohorts": {}}
        for cohort in COHORTS:
            original = result["cohorts"][cohort]
            c = {key: original[key] for key in ("role", "images", "groups", "primary_cohort", "sample_status",
                 "detector", "contrasts", "coverage_fractions", "frozen_working_point", "bootstrap")}
            c["methods"] = {}
            for method in METHODS:
                m = original["methods"][method]
                c["methods"][method] = {"aurc": m["aurc"], "risk_at_coverage": m["risk_at_coverage"],
                    "curve": {key: m["curve"][key] for key in ("coverage", "risk")}}
            entry["cohorts"][cohort] = c
        data["detectors"][det] = entry
    DATA.parent.mkdir(parents=True, exist_ok=True)
    DATA.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n", encoding="utf-8")
    PROVENANCE.parent.mkdir(parents=True, exist_ok=True)
    PROVENANCE.write_text(json.dumps({"version": "adapted-detector-article-provenance-v1",
        "source_run": str(run), "source_files": files, "aggregate_sha256": sha(DATA),
        "source_scope": "Read-only extraction of completed frozen experimental results; no new measurements.",
        "portable_build": "python build_adaptation_assets.py; python build_adaptation_assets.py --check"},
        ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return data


def ci(value, scale=1, digits=6):
    return "$[" + f"{scale*value['lower_95']:.{digits}f},{scale*value['upper_95']:.{digits}f}" + "]$"


def accuracy_percent(metrics, count):
    exact = Decimal(100) * Decimal(count-metrics["errors"]) / Decimal(count)
    return str(exact.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def pairs(data):
    for det in DETECTORS:
        for cohort in COHORTS:
            yield det, cohort, data["detectors"][det]["cohorts"][cohort]


def validate(data):
    checks = 0
    for det, cohort, c in pairs(data):
        assert c["images"] == (4000 if cohort == "OpenFake" else 10000)
        assert c["groups"] == (213 if cohort == "OpenFake" else 10000)
        assert c["methods"]["confidence"]["curve"] == c["methods"]["lr_confidence"]["curve"]
        checks += 3
        for checkpoint in ("initial", "adapted"):
            dm = c["detector"][checkpoint]
            assert math.isclose(dm["accuracy"], (c["images"]-dm["errors"])/c["images"], abs_tol=1e-12)
            checks += 1
        for method in METHODS:
            m = c["methods"][method]
            x, y = m["curve"]["coverage"], m["curve"]["risk"]
            assert len(x) == len(y) and x[-1] == 1
            assert all(b > a for a, b in zip(x, x[1:]))
            integrated = math.fsum((right-left)*risk for left, right, risk in zip([0]+x[:-1], x, y))
            assert math.isclose(integrated, m["aurc"], abs_tol=1e-12)
            for key, fixed in m["risk_at_coverage"].items():
                numerator, denominator = c["coverage_fractions"][key]
                assert fixed["accepted"] == numerator*c["images"]//denominator
                assert math.isclose(fixed["risk"], fixed["errors"]/fixed["accepted"], abs_tol=1e-12)
                checks += 2
            checks += 3
        contrast = c["contrasts"]["lr_full_minus_confidence"]
        full, ref = c["methods"]["lr_full"], c["methods"]["confidence"]
        assert math.isclose(contrast["aurc"]["estimate"], full["aurc"]-ref["aurc"], abs_tol=1e-12)
        assert contrast["aurc"]["upper_95"] < 0
        assert math.isclose(contrast["risk_original"]["estimate"],
            full["risk_at_coverage"]["original"]["risk"]-ref["risk_at_coverage"]["original"]["risk"], abs_tol=1e-12)
        checks += 3
    return checks


def table_detectors(data):
    lines = [r"\begin{table*}[t]",
        r"\caption{Detector adaptation on the V4 training manifest: unconditional test accuracy and area under the receiver operating characteristic curve (AUROC) before and after adaptation. Each checkpoint pair uses identical test images and the same detector-specific preprocessing. OF: 4,000 cross-dataset images; DS: 10,000 in-domain held-out images. The selected checkpoint is epoch 10 for both detectors.}",
        r"\label{tab:adapted_detectors}", r"\centering\small", r"\begin{tabular}{llrrrr}", r"\toprule",
        r" & & \multicolumn{2}{c}{Accuracy (\%)} & \multicolumn{2}{c}{AUROC} \\",
        r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}",
        r"Detector & Cohort & Initial & Adapted & Initial & Adapted \\", r"\midrule"]
    for det, cohort, c in pairs(data):
        a, b = c["detector"]["initial"], c["detector"]["adapted"]
        lines.append(f"{NAMES[det]} & {SHORT[cohort]} & {accuracy_percent(a,c['images'])} & {accuracy_percent(b,c['images'])} & {a['auroc']:.6f} & {b['auroc']:.6f} " + r"\\")
    return "\n".join(lines+[r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""])


def table_selectors(data):
    lines = [r"\begin{table*}[t]",
        r"\caption{Full profile versus confidence after detector adaptation. Top: AURC and its relative reduction. Bottom: accepted errors at the original matched coverage, $q=3,639/4,000$ on OF and $q=9,807/10,000$ on DS. All differences are full minus confidence; risk differences are percentage points. Intervals are pointwise paired 95\% intervals from 5,000 whole-group bootstrap replicates. Each replicate accepts $\lfloor qN^{(b)}\rfloor$ predictions per method. Learned confidence has the same ranking as confidence.}",
        r"\label{tab:adapted_selectors}", r"\centering\small", r"\setlength{\tabcolsep}{5pt}",
        r"\begin{tabular}{llrrrl}", r"\toprule",
        r"Detector & Cohort & AURC: confidence & AURC: full & Reduction (\%) & 95\% interval for $\Delta$AURC \\", r"\midrule"]
    for det, cohort, c in pairs(data):
        a, b = c["methods"]["confidence"]["aurc"], c["methods"]["lr_full"]["aurc"]
        z = c["contrasts"]["lr_full_minus_confidence"]["aurc"]
        lines.append(f"{NAMES[det]} & {SHORT[cohort]} & {a:.6f} & {b:.6f} & {100*(a-b)/a:.2f} & {ci(z)} " + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\par\vspace{5pt}", r"\begin{tabular}{llrrrrl}", r"\toprule",
        r"Detector & Cohort & Accepted & Errors: confidence & Errors: full & $\Delta$ risk & 95\% interval \\", r"\midrule"]
    for det, cohort, c in pairs(data):
        a = c["methods"]["confidence"]["risk_at_coverage"]["original"]
        b = c["methods"]["lr_full"]["risk_at_coverage"]["original"]
        z = c["contrasts"]["lr_full_minus_confidence"]["risk_original"]
        lines.append(f"{NAMES[det]} & {SHORT[cohort]} & {a['accepted']:,} & {a['errors']} & {b['errors']} & ${100*z['estimate']:.4f}$ & {ci(z,100,4)} " + r"\\")
    return "\n".join(lines+[r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""])


def table_all_methods(data):
    lines = [r"\begin{table*}[t]",
        r"\caption{All six selection methods after detector adaptation. Each cell gives unscaled AURC / accepted errors at matched $k$: 3,639 for OpenFake (OF), 9,807 for DeepShield (DS). The JPEG-only and resize-only representations are ablations. Point estimates against these ablations and TTA are descriptive; the paired intervals in the main article compare the full profile with confidence and learned confidence.}",
        r"\label{tab:adapted_all_methods}", r"\centering\small", r"\begin{tabular}{lrrrr}", r"\toprule",
        r" & \multicolumn{2}{c}{CLIP} & \multicolumn{2}{c}{ResNet-50} \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}", r"Selector & OF & DS & OF & DS \\", r"\midrule"]
    for method in METHODS:
        values = [LABELS[method]]
        for _, _, c in pairs(data):
            m = c["methods"][method]
            values.append(f"{m['aurc']:.6f} / {m['risk_at_coverage']['original']['errors']}")
        lines.append(" & ".join(values)+r" \\")
    return "\n".join(lines+[r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""])


def figure(data):
    os.environ.setdefault("MPLCONFIGDIR", str(HERE / "figures/.mplconfig"))
    os.environ.setdefault("SOURCE_DATE_EPOCH", "1793491200")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 9,
        "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
        "legend.fontsize": 8, "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(2, 2, figsize=(7.05, 4.55), sharex=True)
    shown = ("confidence", "tta_mean", "lr_jpeg_only", "lr_resize_only", "lr_full")
    for i, det in enumerate(DETECTORS):
        for j, cohort in enumerate(COHORTS):
            ax = axes[i, j]
            c = data["detectors"][det]["cohorts"][cohort]
            for method in shown:
                curve = c["methods"][method]["curve"]
                ax.plot([100*x for x in curve["coverage"]], [100*x for x in curve["risk"]],
                    color=COLORS[method], linewidth=1.65 if method == "lr_full" else 0.85,
                    label=LABELS[method], alpha=1 if method == "lr_full" else 0.9,
                    solid_capstyle="round")
            ax.set_title(f"({chr(97+2*i+j)}) {NAMES[det]} / {cohort}", loc="left", pad=6)
            limit = 20 if cohort == "OpenFake" and det == "clip" else 26 if cohort == "OpenFake" else 10
            ax.set_ylim(0, limit)
            ax.set_xlim(0, 100)
            ax.xaxis.set_major_locator(MultipleLocator(20))
            ax.yaxis.set_major_locator(MultipleLocator(5 if cohort == "OpenFake" else 2))
            ax.grid(color="#DCE3E7", linewidth=0.45, alpha=0.8)
            ax.set_axisbelow(True)
            ax.spines[["top", "right"]].set_visible(False)
            ax.spines[["bottom", "left"]].set_color("#8D979E")
            ax.tick_params(length=3, color="#8D979E")
            ax.set_ylabel("Accepted risk (%)")
            if i == 1:
                ax.set_xlabel("Coverage (%)")
            conf, full = c["methods"]["confidence"]["aurc"], c["methods"]["lr_full"]["aurc"]
            ax.text(.035, .94, f"AURC reduction: {100*(conf-full)/conf:.2f}%", transform=ax.transAxes,
                ha="left", va="top", fontsize=7.8, color=COLORS["lr_full"],
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.3, "alpha": .9})
    handles, labels = axes[0, 0].get_legend_handles_labels()
    order = [4, 0, 1, 2, 3]
    fig.legend([handles[k] for k in order], [labels[k] for k in order], loc="lower center",
        ncol=5, frameon=False, bbox_to_anchor=(.51, .005), handlelength=2.1, columnspacing=1.5)
    fig.subplots_adjust(left=.075, right=.97, top=.945, bottom=.155, wspace=.25, hspace=.32)
    destination = HERE / "figures/adapted_risk_coverage.pdf"
    fig.savefig(destination, metadata={"Title": "Risk-coverage curves after matched-data detector adaptation",
        "Author": "Murin et al.", "Subject": "Archived empirical curves; all stored score thresholds shown"})
    fig.savefig(destination.with_suffix(".png"), dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extract-from", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check and args.extract_from:
        parser.error("--check is read-only and cannot be combined with --extract-from")
    data = extract(args.extract_from) if args.extract_from else read(DATA)
    provenance = read(PROVENANCE)
    if sha(DATA) != provenance["aggregate_sha256"]:
        raise ValueError("Packaged aggregate checksum mismatch")
    count = validate(data)
    outputs = {"tables/adapted_detectors.tex": table_detectors(data),
               "tables/adapted_selectors.tex": table_selectors(data),
               "tables/adapted_all_methods.tex": table_all_methods(data)}
    for relative, text in outputs.items():
        path = HERE / relative
        if args.check:
            if path.read_text(encoding="utf-8") != text:
                raise ValueError("Generated table differs from aggregate: " + relative)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8", newline="\n")
    if not args.check:
        figure(data)
    print(f"Validated {count} numerical consistency checks and {len(outputs)} tables; " +
          ("read-only verification complete." if args.check else "assets rebuilt from packaged aggregate."))


if __name__ == "__main__":
    main()
