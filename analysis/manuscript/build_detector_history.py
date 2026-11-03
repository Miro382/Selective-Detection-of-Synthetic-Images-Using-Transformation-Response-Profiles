"""Zostaví prenosný obrázok histórie detektora zo zmrazených lokálnych čísel.

Pôvodné dáta ani modely sa nenačítavajú. Voliteľné --verify-sources iba overí
SHA pôvodných zdrojových súborov, pokiaľ sú dostupné na tomto počítači.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

DATA_SHA256 = "8364af2a081c05daf8521eb09aee77c0f48f93961016ad41153e5f2d9562daf0"
ROOT = Path(__file__).resolve().parent


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def checked_data(path: Path, verify_sources: bool = False) -> dict:
    if sha256(path) != DATA_SHA256:
        raise ValueError("Zmenili sa overené lokálne údaje grafu.")
    data = json.loads(path.read_text(encoding="utf-8"))
    if data["schema"] != "detector-development-figure-v1":
        raise ValueError("Neznáma schéma údajov.")
    for version, expected_n in (("V3", 2880), ("V4", 4880)):
        run = data["runs"][version]
        assert run["n"] == expected_n
        assert [e["epoch"] for e in run["epochs"]] == list(range(1, 11))
        selected = run["epochs"][run["selected_epoch"] - 1]
        assert selected["selection_score"] == run["selected_selection_score"]
        for condition in data["condition_order"]:
            record = run["conditions"][condition]
            assert record["n"] == expected_n
            for prefix in ("before", "after"):
                assert math.isclose(
                    record[prefix + "_accuracy"],
                    1 - record[prefix + "_errors"] / expected_n,
                    rel_tol=0, abs_tol=1e-15,
                )
            observed = selected["validation_conditions"][condition]
            assert observed["errors"] == record["after_errors"]
            assert observed["accuracy"] == record["after_accuracy"]
        assert all(e["training_n"] == 24000 for e in run["epochs"])
    if verify_sources:
        for source in data["source_anchors"]:
            path = Path(source["path"])
            if sha256(path) != source["sha256"]:
                raise ValueError("Nesúhlasí pôvodný zdroj: " + str(path))
    return data


def build(data: dict, output: Path) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8.5,
        "axes.labelsize": 8.5, "axes.titlesize": 9,
        "xtick.labelsize": 8, "ytick.labelsize": 8,
        "legend.fontsize": 8, "axes.linewidth": 0.6,
        "lines.linewidth": 1.4, "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    blue, orange = "#0072B2", "#D55E00"
    fig, axes = plt.subplots(2, 2, figsize=(7.1, 5.25))
    fig.subplots_adjust(left=0.085, right=0.985, top=0.93,
                        bottom=0.19, hspace=0.53, wspace=0.27)
    for ax, version, panel in zip(axes[0], ("V3", "V4"), ("a", "b")):
        run = data["runs"][version]
        positions = np.arange(3)
        width = 0.35
        for offset, prefix, label, color in (
            (-width / 2, "before", run["initial_label"], blue),
            (width / 2, "after", run["selected_label"], orange),
        ):
            values = [100 * run["conditions"][c][prefix + "_accuracy"]
                      for c in data["condition_order"]]
            bars = ax.bar(positions + offset, values, width, label=label,
                          color=color, edgecolor="white", linewidth=0.4, zorder=3)
            for bar, value in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width() / 2, value + 1.2,
                        f"{value:.1f}", ha="center", va="bottom", fontsize=8)
        ax.set_title(f"({panel}) {run['initial_label']} → {version}: matched validation\n"
                     f"{run['n']:,} images per condition", loc="left", pad=6)
        ax.set_ylabel("Accuracy (%)")
        ax.set_xticks(positions, data["condition_labels"])
        ax.set_ylim(0, 108)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.legend(loc="lower right", frameon=True, framealpha=0.95,
                  edgecolor="none", ncol=2, handlelength=1.0,
                  columnspacing=0.8, borderpad=0.4)
    loss_ax, score_ax = axes[1]
    for version, color, marker in (("V3", blue, "o"), ("V4", orange, "s")):
        run = data["runs"][version]
        epochs = [r["epoch"] for r in run["epochs"]]
        loss_ax.plot(epochs, [r["train_loss"] for r in run["epochs"]],
                     marker=marker, markersize=3, color=color, label=version)
        scores = [100 * r["selection_score"] for r in run["epochs"]]
        score_ax.plot(epochs, scores, marker=marker, markersize=3, color=color,
                      label=f"{version} hierarchy")
        best_epoch = run["selected_epoch"]
        score_ax.scatter([best_epoch], [scores[best_epoch - 1]], marker="*",
                         s=90, color=color, edgecolor="black", linewidth=0.45,
                         zorder=5)
    loss_ax.set_title("(c) Fine-tuning training loss\nDifferent sampled training distributions",
                      loc="left", pad=6)
    loss_ax.set_ylabel("Cross-entropy loss")
    loss_ax.set_ylim(0, 0.43)
    loss_ax.legend(loc="upper right", frameon=False, ncol=2)
    score_ax.set_title("(d) Each run's own selection criterion\nDifferent validation sets and hierarchies",
                       loc="left", pad=6)
    score_ax.set_ylabel("Selection criterion (%)")
    score_ax.set_ylim(80, 100)
    score_ax.legend(loc="lower right", frameon=False)
    score_ax.text(0.03, 0.97, "Stars: selected epochs 8 / 10",
                  transform=score_ax.transAxes, fontsize=8, va="top")
    for ax in axes[1]:
        ax.set_xlabel("Epoch (24,000 sampled draws)")
        ax.set_xticks([1, 2, 4, 6, 8, 10])
        ax.set_xlim(0.7, 10.35)
    for ax in axes.flat:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#D4D8DC", linewidth=0.45, alpha=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(length=3, width=0.6)
    fig.text(0.085, 0.035,
             "Adaptive development validation, not external confirmation; compare checkpoints within each top panel.\n"
             "Different training data and selection hierarchies preclude causal attribution across runs.",
             ha="left", va="bottom", fontsize=8, color="#303840")
    output.mkdir(parents=True, exist_ok=True)
    pdf = output / "detector_development.pdf"
    png = output / "detector_development.png"
    fixed_time = datetime(2026, 10, 29, tzinfo=timezone.utc)
    fig.savefig(pdf, metadata={"Title": "Detector development on adaptive validation",
                              "Author": "", "Creator": "build_detector_history.py",
                              "Subject": "Source data SHA256: " + DATA_SHA256,
                              "CreationDate": fixed_time, "ModDate": fixed_time})
    fig.savefig(png, dpi=300, metadata={"Software": "build_detector_history.py",
                                       "SourceDataSHA256": DATA_SHA256})
    plt.close(fig)
    return [pdf, png]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "figure_data/detector_development.json")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "figures")
    parser.add_argument("--verify-sources", action="store_true")
    args = parser.parse_args()
    data = checked_data(args.data, args.verify_sources)
    print("Lokálne údaje overené: " + DATA_SHA256)
    for path in build(data, args.out_dir):
        print(str(path) + " SHA256=" + sha256(path))


if __name__ == "__main__":
    main()
