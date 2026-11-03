"""Plot archived historical transfer aggregates without fitting or inference."""
from pathlib import Path
import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
CONDITIONS = ("clean", "jpeg_q40", "resize_0p5")
COLORS = {"confidence": "#777777", "selector_v4": "#0072B2",
          "tta_mean": "#E69F00", "max_shift": "#AA4499",
          "profile_candidate": "#009E73"}
LABELS = {"confidence": "Confidence", "selector_v4": "Full profile V4",
          "tta_mean": "TTA mean", "max_shift": "Maximum shift",
          "profile_candidate": "V5 profile candidate"}


def main():
    source = ROOT / "figure_data" / "historical_transfer.json"
    data = json.loads(source.read_text(encoding="utf-8"))
    assert data["schema"] == "historical-selector-transfer-v1"
    assert all(data[k] is False for k in ("new_training", "new_inference", "new_bootstrap"))
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.35), sharey=True)
    settings = [
        ("CF", "(a) Historical Community Forensics", ("confidence", "tta_mean", "max_shift", "selector_v4")),
        ("OF_A", "(b) Historical OpenFake A", ("confidence", "profile_candidate", "selector_v4")),
    ]
    for ax, (cohort, title, methods) in zip(axes, settings):
        rows = data["cohorts"][cohort]
        assert rows["n"] == 4000 and rows["profiles"] == 12000
        x = np.arange(len(CONDITIONS))
        width = .18
        for j, method in enumerate(methods):
            y = [rows["conditions"][c]["aurc"][method] for c in CONDITIONS]
            assert all(0 <= v <= 1 for v in y)
            ax.bar(x + (j - (len(methods)-1)/2)*width, y, width=width,
                   color=COLORS[method], edgecolor="white", linewidth=.5,
                   label=LABELS[method], zorder=3)
        ax.set_title(title, loc="left", fontsize=8.5, pad=22)
        ax.text(0, 1.045, f"N = 4,000; G = {rows['groups']}", transform=ax.transAxes,
                fontsize=8, color="#555555")
        ax.set_xticks(x, ["Clean", "JPEG 40", "Resize 0.5"])
        ax.set_ylim(0, .23)
        ax.set_yticks(np.arange(0, .201, .05))
        ax.grid(axis="y", color="#d8d8d8", linewidth=.5, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(loc="upper center", bbox_to_anchor=(.5, -.18),
                  ncol=2 if cohort == "CF" else 1, frameon=False, fontsize=7.6,
                  handlelength=1.5, columnspacing=1.0, labelspacing=.5)
    axes[0].set_ylabel("AURC (lower is better)")
    fig.subplots_adjust(left=.095, right=.985, top=.79, bottom=.31, wspace=.17)
    fig.text(.5, .015, "V5 selected confidence exactly; its retained profile candidate is exploratory.",
             ha="center", fontsize=7.7)
    output = ROOT / "figures"
    paths = [output / "historical_transfer.pdf", output / "historical_transfer.png"]
    fig.savefig(paths[0], metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(paths[1], dpi=220)
    plt.close(fig)
    receipt = {"source": str(source.relative_to(ROOT)),
               "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
               "outputs": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
               "new_training": False, "new_inference": False, "new_bootstrap": False}
    (ROOT / "evidence" / "historical_transfer_render_receipt.json").write_text(
        json.dumps(receipt, indent=2)+"\n", encoding="utf-8")
    print("Historical transfer figure rendered from archived aggregates.")


if __name__ == "__main__":
    main()
