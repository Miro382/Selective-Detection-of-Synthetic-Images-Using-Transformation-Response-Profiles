"""Render stored Bahat contrasts; no inference, fitting, or bootstrap is performed."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


ROOT = Path(__file__).resolve().parent
SEEDS = (17, 29, 43)
COHORTS = ("DeepShield", "OpenFake")


def load_contrasts(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema") != "article-contribution-audit-v1":
        raise ValueError("Unexpected contribution evidence schema.")
    for flag in ("new_inference", "new_training", "new_bootstrap"):
        if data.get(flag) is not False:
            raise ValueError(f"Evidence does not declare {flag}=False.")
    panels = {}
    for cohort in COHORTS:
        risk = {r["seed"]: r for r in data["bahat_at95_coverage"] if r["cohort"] == cohort}
        if set(risk) != set(SEEDS):
            raise ValueError(f"Missing or extra risk seeds for {cohort}.")
        risk_rows = []
        for seed in SEEDS:
            row = risk[seed]
            if row["coverage"] != 0.95 or row["bootstrap_repetitions"] != 5000:
                raise ValueError("Unexpected coverage or risk-bootstrap repetitions.")
            point = float(row["delta_risk_percentage_points"])
            lo, hi = map(float, row["ci95_percentage_points"])
            if not lo <= point <= hi < 0:
                raise ValueError("Risk interval or contrast direction differs from audited evidence.")
            risk_rows.append((point, lo, hi))
        family = data["transformation_family"][cohort]["paired_seed_contrasts"]
        family_by_seed = {}
        for row in family:
            seed = int(row["left_method"].rsplit("_s", 1)[1])
            if row["left_method"] != f"mlp_ours_logits_s{seed}" or row["right_method"] != f"mlp_bahat_logits_s{seed}":
                raise ValueError("Unexpected transformation-family contrast.")
            family_by_seed[seed] = row
        if set(family_by_seed) != set(SEEDS) or len(family) != len(SEEDS):
            raise ValueError(f"Missing, duplicate, or extra transformation seeds for {cohort}.")
        family_rows = []
        for seed in SEEDS:
            row = family_by_seed[seed]
            interval = row["ci95"]
            if interval["repetitions"] != 1000:
                raise ValueError("Unexpected transformation-bootstrap repetitions.")
            point = float(row["estimate_left_minus_right"])
            lo, hi = float(interval["lower_95"]), float(interval["upper_95"])
            if not lo <= point <= hi < 0:
                raise ValueError("AURC interval or contrast direction differs from audited evidence.")
            family_rows.append(tuple(1000 * x for x in (point, lo, hi)))
        panels[cohort] = {"risk": risk_rows, "family": family_rows}
    return panels


def draw_panel(ax, rows, title: str, xlabel: str, color: str) -> None:
    points = [r[0] for r in rows]
    errors = [[r[0] - r[1] for r in rows], [r[2] - r[0] for r in rows]]
    ax.axvline(0, color="#666666", linewidth=0.8, linestyle="-", zorder=1)
    ax.errorbar(points, (2, 1, 0), xerr=errors, fmt="o", color=color,
                ecolor=color, markersize=4.4, elinewidth=1.4, capsize=3.2,
                markeredgecolor="white", markeredgewidth=0.6, zorder=3)
    xmin = min(r[1] for r in rows)
    ax.set_xlim(xmin * 1.10, -xmin * 0.085)
    ax.set_ylim(-0.55, 2.55)
    ax.set_yticks((2, 1, 0), [f"Seed {s}" for s in SEEDS])
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=3))
    ax.set_title(title, loc="left", fontsize=8.8, fontweight="bold", pad=9)
    ax.set_xlabel(xlabel, fontsize=8.2, labelpad=4)
    ax.grid(axis="x", color="#e2e2e2", linewidth=0.5, zorder=0)
    ax.tick_params(axis="both", length=3, labelsize=8)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color("#aaaaaa")
    ax.tick_params(axis="y", length=0, pad=4)


def build(evidence: Path, output: Path, overwrite: bool = False) -> tuple[Path, Path]:
    panels = load_contrasts(evidence)
    output.mkdir(parents=True, exist_ok=True)
    targets = (output / "bahat_comparison.pdf", output / "bahat_comparison.png")
    if not overwrite and any(p.exists() for p in targets):
        raise FileExistsError("Figure exists; pass --overwrite to regenerate this figure only.")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.4,
                         "pdf.fonttype": 42, "ps.fonttype": 42,
                         "axes.unicode_minus": True, "savefig.facecolor": "white"})
    fig, axes = plt.subplots(2, 2, figsize=(7.1, 4.95))
    draw_panel(axes[0, 0], panels["DeepShield"]["risk"],
               "(a) DeepShield: full selector − Bahat", "Δ risk at 95% coverage (percentage points)", "#17668a")
    draw_panel(axes[0, 1], panels["OpenFake"]["risk"],
               "(b) OpenFake: full selector − Bahat", "Δ risk at 95% coverage (percentage points)", "#17668a")
    draw_panel(axes[1, 0], panels["DeepShield"]["family"],
               "(c) DeepShield: matched MLP views", r"Δ AURC: proposed − reference views ($\times 10^{-3}$)", "#267b5d")
    draw_panel(axes[1, 1], panels["OpenFake"]["family"],
               "(d) OpenFake: matched MLP views", r"Δ AURC: proposed − reference views ($\times 10^{-3}$)", "#267b5d")
    fig.subplots_adjust(left=0.105, right=0.985, top=0.885, bottom=0.145, hspace=0.70, wspace=0.30)
    fig.text(0.5, 0.985, "Negative differences favor the proposed selector or views", ha="center", va="top", fontsize=8.8)
    fig.text(0.5, 0.035, "Pointwise 95% paired group intervals; top: 5,000 replicates; bottom: 1,000. Panel scales differ.",
             ha="center", va="bottom", fontsize=8)
    digest = hashlib.sha256(evidence.read_bytes()).hexdigest()
    fig.savefig(targets[0], metadata={"Title": "Bahat comparison: high-coverage risk and matched transformation controls",
                                     "Subject": f"Stored evidence SHA256 {digest}; no bootstrap recomputation"})
    fig.savefig(targets[1], dpi=240, metadata={"Description": f"Stored evidence SHA256 {digest}; no bootstrap recomputation"})
    plt.close(fig)
    return targets


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, default=ROOT / "evidence" / "contribution_audit.json")
    parser.add_argument("--output", type=Path, default=ROOT / "figures")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    for target in build(args.evidence.resolve(), args.output.resolve(), args.overwrite):
        print(target)


if __name__ == "__main__":
    main()
