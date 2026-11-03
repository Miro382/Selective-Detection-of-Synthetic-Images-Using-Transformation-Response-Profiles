"""Render archived selector comparisons; no inference, training, or resampling.

Default input is the portable figure_data/selector_coverage.json aggregate.
--refresh-data rebuilds it from packaged evidence. --verify-primary additionally
checks the original JSON reports and their hashes; it never opens images/models.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator
import numpy as np

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "figure_data/selector_coverage.json"
COHORTS = ("DeepShield", "OpenFake")
COLORS = {"main": "#0072B2", "baseline": "#777777", "lr": "#009E73",
          "margins": "#AA4499", "ours_logits": "#E69F00", "bahat_logits": "#D55E00"}
LABELS = {
    "frozen_v4": "Full profile (frozen)", "confidence": "Confidence",
    "tta_mean": "TTA mean", "lr_confidence": "LR: confidence",
    "lr_full": "LR: full profile refit", "lr_no_summaries": "LR: no summaries",
    "lr_no_orientation": "LR: no class orientation", "lr_no_confidence": "LR: no original confidence",
    "lr_summaries_only": "LR: confidence + summaries", "lr_jpeg_only": "LR: JPEG views only",
    "lr_resize_only": "LR: resize views only", "lr_without_jpeg_q90": "LR: omit JPEG 90",
    "lr_without_jpeg_q75": "LR: omit JPEG 75", "lr_without_jpeg_q60": "LR: omit JPEG 60",
    "lr_without_resize_0p75": "LR: omit resize 0.75", "lr_without_resize_0p5": "LR: omit resize 0.50",
    "lr_balanced_no_summaries": "LR: balanced, no summaries",
}
for _seed in (17, 29, 43):
    LABELS[f"mlp_margins_s{_seed}"] = f"MLP: oriented margins, seed {_seed}"
    LABELS[f"mlp_ours_logits_s{_seed}"] = f"MLP: our-view logits, seed {_seed}"
    LABELS[f"mlp_bahat_logits_s{_seed}"] = f"MLP: Bahat-view logits, seed {_seed}"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def refresh(verify_primary=False):
    history_path = ROOT / "evidence/history_selector.json"
    extended_path = ROOT / "evidence/extended_ablation_table.json"
    history, extended = read(history_path), read(extended_path)
    primary_checks = []
    historical = []
    for row in history["V1_V2_paired_table"]:
        if verify_primary:
            path = Path(row["source_report"])
            assert sha(path) == history["sources"][str(path)]["sha256"], path
            report = read(path)
            for short, long in (("V1", "comparison_3"), ("V2", "selective"), ("confidence", "original_confidence")):
                recorded = report["methods"][long]["at_primary_coverage"]
                assert recorded["aurc"] == row["aurc"][short]
                assert recorded["errors"] == row["errors"][short]
                assert recorded["accepted"] == row["accepted"]
                assert recorded["total"] == row["n"] == 2000
            primary_checks.append({"path": str(path), "sha256": sha(path), "values_match": True})
        historical.append({k: row[k] for k in ("condition", "n", "groups", "accepted", "coverage", "errors", "aurc")})
    if verify_primary:
        rec = extended["source_inventory"]["reports/full/results.json"]
        path = Path(rec["path"])
        assert sha(path) == rec["sha256"]
        report = read(path)
        for row in extended["all_methods"]:
            for cohort in COHORTS:
                assert report["cohorts"][cohort]["methods"][row["method"]]["aurc"] == row["cohorts"][cohort]["aurc"]
                assert report["cohorts"][cohort]["methods"][row["method"]]["matched_frozen_coverage"] == row["cohorts"][cohort]["matched_frozen_coverage"]
        primary_checks.append({"path": str(path), "sha256": sha(path), "values_match": True})
    methods = []
    for row in extended["all_methods"]:
        name = row["method"]
        family = "main" if name == "frozen_v4" else row["spec"]["kind"]
        if family == "mlp":
            family = "margins" if row["spec"]["features"] == "no_summaries" else row["spec"]["features"]
        methods.append({"method": name, "label": LABELS[name], "family": family,
                        "dimension": row["dimension"], "spec": row["spec"],
                        "aurc": {c: row["cohorts"][c]["aurc"] for c in COHORTS},
                        "matched_frozen_coverage": {c: row["cohorts"][c]["matched_frozen_coverage"] for c in COHORTS}})
    assert len(methods) == 26 and len(historical) == 6
    data = {"schema": "selector-coverage-figure-data-v1", "new_training": False,
            "new_inference": False, "new_bootstrap": False,
            "inputs": [{"path": str(p.relative_to(ROOT)), "sha256": sha(p)} for p in (history_path, extended_path)],
            "primary_report_checks": primary_checks,
            "historical_v1_v2": historical, "exploratory_methods": methods,
            "cohorts": extended["cohorts"],
            "scope": "Historical V1/V2 matched conditions and a separate retrospective 26-entry comparison. No cross-study pooling or new intervals."}
    write(DATA, data)
    return data


def finish(fig, name):
    output = ROOT / "figures"
    for ext in ("pdf", "png"):
        fig.savefig(output / f"{name}.{ext}", dpi=220, facecolor="white",
                    metadata={"CreationDate": None, "ModDate": None} if ext == "pdf" else None)
    plt.close(fig)


def historical_figure(data):
    by_condition = {r["condition"]: r for r in data["historical_v1_v2"]}
    order = ("clean", "jpeg_q40", "resize_0p5", "brightness_0p85", "contrast_1p15", "mixture")
    names = ("Clean", "JPEG quality 40", "Resize 0.50", "Brightness 0.85", "Contrast 1.15", "Mixture")
    rows = [by_condition[c] for c in order]
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 3.8), sharey=True)
    fig.subplots_adjust(left=0.235, right=0.985, bottom=0.18, top=0.91, wspace=0.19)
    y = np.arange(6)
    colors = (COLORS["baseline"], "#E69F00", COLORS["main"])
    for ax, metric, title, xlabel in zip(axes, ("aurc", "errors"),
            ("(a) Curve ranking", "(b) Matched acceptance"), ("AURC (lower is better)", "Accepted errors (lower is better)")):
        for j, method in enumerate(("confidence", "V1", "V2")):
            values = [r[metric][method] for r in rows]
            bars = ax.barh(y + (j-1)*0.22, values, height=0.205, color=colors[j], edgecolor="white", linewidth=0.3,
                          label={"confidence":"Confidence", "V1":"V1: clean-trained", "V2":"V2: augmented"}[method], zorder=3)
            if metric == "errors":
                ax.bar_label(bars, padding=2, fontsize=7)
        ax.set_title(title, loc="left", fontsize=9, fontweight="bold", pad=9)
        ax.set_xlabel(xlabel, fontsize=8)
        ax.grid(axis="x", color="#dddddd", linewidth=0.5, zorder=0)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0, labelsize=8)
        ax.tick_params(axis="x", labelsize=8)
        ax.set_ylim(5.6, -0.6)
    axes[0].set_yticks(y, [f"{name}   (k={r['accepted']:,})" for name, r in zip(names, rows)])
    axes[0].set_xlim(0, 0.15)
    axes[0].set_xticks((0, .05, .10, .15))
    axes[1].set_xlim(0, 118)
    axes[1].set_xticks((0, 25, 50, 75, 100))
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.58, 0.025), ncol=3, frameon=False, fontsize=8)
    finish(fig, "selector_v1_v2_development")


def landscape_figure(data):
    rows = data["exploratory_methods"]
    assert len(rows) == 26
    baseline = next(r for r in rows if r["method"] == "frozen_v4")
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 5.95), sharey=True)
    fig.subplots_adjust(left=0.37, right=0.94, bottom=0.17, top=0.90, wspace=0.40)
    y = np.arange(len(rows))
    for i, (ax, cohort) in enumerate(zip(axes, COHORTS)):
        ratios = [r["aurc"][cohort] / baseline["aurc"][cohort] for r in rows]
        ax.set_xscale("log", base=2)
        ax.axvline(1, color="#555555", linewidth=1, zorder=2)
        for k, (row, ratio) in enumerate(zip(rows, ratios)):
            ax.plot([1, ratio], [k, k], color=COLORS[row["family"]], linewidth=1.0, zorder=3)
            ax.scatter(ratio, k, s=30 if row["family"] == "main" else 20,
                       color=COLORS[row["family"]], edgecolors="white", linewidths=0.45, zorder=4)
            count = row["matched_frozen_coverage"][cohort]["errors"]
            ax.text(1.06, k, str(count), transform=ax.get_yaxis_transform(), fontsize=7.3,
                    ha="center", va="center", clip_on=False,
                    fontweight="bold" if row["family"] == "main" else "normal")
        accepted = baseline["matched_frozen_coverage"][cohort]["accepted"]
        ax.text(1.06, 1.012, "Errors", transform=ax.transAxes, fontsize=7.1, ha="center", va="bottom")
        ax.set_title(f"({chr(97+i)}) {cohort} (k={accepted:,})\nFull AURC = {baseline['aurc'][cohort]:.6f}", loc="left", fontsize=7.8, pad=9)
        ax.set_yticks(y, [r["label"] for r in rows])
        ax.set_ylim(len(rows)-0.4, -0.75)
        ax.tick_params(axis="y", length=0, labelsize=7.3, pad=5)
        ax.tick_params(axis="x", labelsize=8)
        ax.grid(axis="x", color="#e3e3e3", linewidth=0.5, zorder=0)
        ax.spines[["top", "right", "left"]].set_visible(False)
        for yy in (0.5, 2.5, 16.5, 19.5, 22.5):
            ax.axhline(yy, color="#dddddd", linewidth=0.5, zorder=0)
        ticks = [1, 2, 4, 8] if i == 0 else [0.625, .75, 1, 1.25]
        ax.set_xlim((0.84, 8.2) if i == 0 else (0.61, 1.30))
        ax.xaxis.set_major_locator(FixedLocator(ticks))
        ax.xaxis.set_major_formatter(FixedFormatter([f"{t:g}" for t in ticks]))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_xlabel("AURC / frozen full-profile AURC\n(log scale; below 1 is better)", fontsize=7.6)
    legend = [("main", "Frozen main"), ("baseline", "Unfitted references"), ("lr", "Logistic variants"),
              ("margins", "MLP margins"), ("ours_logits", "MLP our-view logits"), ("bahat_logits", "MLP Bahat-view logits")]
    handles = [Line2D([0], [0], marker="o", color=COLORS[k], linestyle="-", markersize=4, linewidth=1, label=v) for k,v in legend]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.55, .012), frameon=False, ncol=3, fontsize=7.4,
               columnspacing=1.1, handletextpad=.5)
    finish(fig, "selector_family_landscape")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--refresh-data", action="store_true")
    ap.add_argument("--verify-primary", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    if args.verify_primary and not args.refresh_data:
        ap.error("--verify-primary requires --refresh-data")
    names = ("selector_v1_v2_development", "selector_family_landscape")
    outputs = [ROOT / "figures" / f"{name}.{ext}" for name in names for ext in ("pdf", "png")]
    if not args.overwrite and any(p.exists() for p in outputs):
        raise FileExistsError("A figure exists; pass --overwrite to regenerate these two figures only.")
    data = refresh(args.verify_primary) if args.refresh_data else read(DATA)
    assert data["schema"] == "selector-coverage-figure-data-v1"
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "pdf.fonttype": 42,
                         "ps.fonttype": 42, "axes.unicode_minus": True, "savefig.facecolor": "white"})
    historical_figure(data)
    landscape_figure(data)
    receipt = {"schema": "selector-coverage-figures-receipt-v1", "aggregate_sha256": sha(DATA),
               "builder_sha256": sha(Path(__file__)), "historical_conditions": 6, "exploratory_entries": 26,
               "new_training": False, "new_inference": False, "new_bootstrap": False,
               "primary_reports_checked": data["primary_report_checks"],
               "outputs": {str(p.relative_to(ROOT)): sha(p) for p in outputs}}
    write(ROOT / "evidence/selector_coverage_figures_receipt.json", receipt)
    print(json.dumps({"figures": 2, "historical_conditions": 6, "exploratory_entries": 26,
                      "source_report_checks": len(data["primary_report_checks"]), "new_computations": "ratios only"}))


if __name__ == "__main__":
    main()
