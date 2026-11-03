"""Build two portable history tables from the saved, full-precision evidence.

No experiment directory, model, image, or numerical package is needed.
Usage: python -B build_selector_history_tables.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def fmt(value: float, digits: int = 6) -> str:
    return f"{value:.{digits}f}"


def build(evidence: Path, output: Path) -> None:
    raw = evidence.read_bytes()
    data = json.loads(raw)
    if data["schema"] != "article-selector-history-v1":
        raise ValueError("Unsupported selector history evidence schema")
    v = data["versions"]
    assert [v[x]["feature_count"] for x in ["V1", "V2", "V3", "V4"]] == [13, 13, 9, 9]
    provenance = "% Generated from evidence/history_selector.json; SHA256=" + hashlib.sha256(raw).hexdigest() + "\n"
    # Widths plus eight inter-column gaps at 3 pt total 6.93 in.
    evolution = provenance + r"""\begin{table*}[t]
\caption{Selector development stages. Counts refer to selector fitting, not detector training. An error target is an incorrectly classified input variant. The stages change data and assessment protocols and do not constitute a matched cross-version performance ranking.}
\label{tab:selector_evolution}
\centering
\begingroup
\fontsize{8}{9.6}\selectfont
\setlength{\tabcolsep}{3pt}
\renewcommand{\arraystretch}{1.12}
\begin{tabular}{@{}p{0.30in}p{1.72in}p{1.27in}p{1.40in}p{1.90in}@{}}
\toprule
Stage & Representation and fit & Fitting sample & Model selection & Risk assessment \\
\midrule
"""
    # All numeric content is obtained from the evidence; prose identifies procedures.
    a, b = v["V1"], v["V2"]
    rows = []
    for name, x, change, group_text, selection in [
        ("V1", a, "Raw probability responses and three image-quality features; clean inputs", "4,000 origins / 4,000 SHA groups", "Fixed full family; no CV"),
        ("V2", b, "Same representation; eight outer training variants per origin", "4,000 origins / 4,000 SHA groups", "Same fixed family; no CV"),
    ]:
        fit = x["effective_training"]
        cal = x["calibration"]
        rows.append(
            f"{name} & {x['feature_count']} features; $C_{{\\rm reg}}={x['C']:g}$. {change}. Unweighted fit. & "
            f"{group_text}; {fit['sample_count']:,} profiles; {fit['error_count']:,} error targets. & {selection}. & "
            f"{cal['sample_count']:,} calibration inputs; {cal['grid_size']} Bonferroni thresholds; "
            f"{100*cal['target_risk']:g}\\% target. "
            f"{cal['error_count']}/{cal['accepted_count']:,} accepted errors; $U={100*cal['risk_upper_bound']:.3f}\\%$. \\")
    c = v["V3"]; fit = c["counts"]; cal = c["calibration"]
    rows.append(
        f"V3 & {c['feature_count']} class-oriented features; $C_{{\\rm reg}}={c['C']:g}$. "
        "Clipped-probability logits; class/generator-balanced weights. & "
        f"{fit['train_originals']:,} origins / {fit['train_groups']:,} source groups; {fit['train_profiles']:,} profiles; "
        f"{c['selected']['detector_errors']:,} error targets. & "
        f"Five group folds; {len(c['candidates'])} candidates; mean-fold AURC. Threshold from OOF predictions. & "
        f"One frozen rule; {cal['sample_count']:,} representatives; {100*cal['target_risk']:g}\\% target. "
        f"{cal['error_count']}/{cal['accepted_count']:,} accepted errors; $U={100*cal['risk_upper_bound']:.3f}\\%$. "
        "Failed; reject all. \\")
    d = v["V4"]; fit = d["selector_training"]; counts = fit["profile_counts_and_errors"]; cal = d["calibration_and_threshold"]["result"]
    rows.append(
        f"V4 & {d['feature_count']} class-oriented features; $C_{{\\rm reg}}={d['C']:g}$. "
        "Native calibrated margins; inverse-profile group weights. & "
        f"{counts['selector_train']['groups']:,} origins / groups; {counts['selector_train']['profiles']:,} profiles; "
        f"{counts['selector_train']['detector_error_profiles']:,} error targets. & "
        f"Separate {counts['selector_selection']['profiles']:,}-input selection set; "
        f"{len(fit['selection']['candidates'])} candidates; selection AURC; no CV. & "
        f"One frozen rule; {cal['total']:,} representatives; {100*cal['target_risk']:g}\\% target. "
        f"{cal['errors']}/{cal['accepted']:,} accepted errors; $U={100*cal['risk_upper_bound']:.3f}\\%$. Passed. \\")
    # Raw strings above terminate rows with one backslash; make the TeX terminator explicit.
    evolution += "\n\\addlinespace[2pt]\n".join(row.rstrip("\\") + r"\\" for row in rows)
    evolution += r"""
\bottomrule
\end{tabular}
\endgroup
\end{table*}
"""

    paired = provenance + r"""\begin{table*}[t]
\caption{Matched historical V1--V2 comparison on the same 2,000 test originals in every condition. All error counts use the V2 accepted count $k$. AURC follows the archived sample-prefix convention. The last column contains stored pointwise 95\% intervals for V2 minus confidence, not V2 minus V1 (1,000 paired group bootstrap replicates). CI denotes confidence interval.}
\label{tab:selector_v1_v2}
\centering
\begingroup
\fontsize{8}{9.6}\selectfont
\setlength{\tabcolsep}{3pt}
\renewcommand{\arraystretch}{1.12}
\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lrrrrrrl@{}}
\toprule
 & & \multicolumn{3}{c}{Accepted errors} & \multicolumn{2}{c}{AURC} & AURC difference \\
\cmidrule(lr){3-5}\cmidrule(lr){6-7}
Condition & $k$ & Conf. & V1 & V2 & V1 & V2 & V2 $-$ conf., 95\% CI \\
\midrule
"""
    labels = {"mixture": "Mixture", "clean": "Clean", "jpeg_q40": "JPEG 40", "resize_0p5": r"Resize $0.5$", "brightness_0p85": r"Brightness $0.85$", "contrast_1p15": r"Contrast $1.15$"}
    rr = data["V1_V2_paired_table"]
    assert len(rr) == 6 and all(x["n"] == 2000 and x["groups"] == 2000 for x in rr)
    for x in rr:
        e, q = x["errors"], x["aurc"]
        ci = x["V2_minus_confidence_aurc_bootstrap"]["percentile_interval_95"]
        paired += f"{labels[x['condition']]} & {x['accepted']:,} & {e['confidence']} & {e['V1']} & {e['V2']} & {fmt(q['V1'])} & {fmt(q['V2'])} & $[{fmt(ci[0])},{fmt(ci[1])}]$ " + r"\\" + "\n"
    paired += r"""\bottomrule
\end{tabular*}
\endgroup
\end{table*}
"""
    output.mkdir(parents=True, exist_ok=True)
    for name, content in [("selector_evolution.tex", evolution), ("selector_v1_v2.tex", paired)]:
        target = output / name
        target.write_text(content, encoding="utf-8")
        print(f"{target.name}: {hashlib.sha256(target.read_bytes()).hexdigest()}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, default=HERE / "evidence" / "history_selector.json")
    parser.add_argument("--output-dir", type=Path, default=HERE / "tables")
    args = parser.parse_args()
    build(args.evidence, args.output_dir)


if __name__ == "__main__":
    main()
