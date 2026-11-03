ARTICLE ANALYSES

Run from the repository root:

  python analysis/verify_results.py --check

The verification reconstructs 100 method/cohort AURCs and 432 accepted-count
outcomes from ten score bundles. It recomputes energy and MC-dropout scores
from stored native logits, checks adapted-detector accuracy and AUROC, and
recomputes 51 percentile intervals from the archived bootstrap draws.

No image files, detector weights, network access, inference, training, or new
bootstrap sampling are required. NumPy is the only non-standard dependency.
These checks verify consistency with archived measurements. They do not turn
retrospective comparisons into independent replications.

Additional read-only checks:

  python analysis/manuscript/class_analysis/build_class_analysis.py --check
  python analysis/manuscript/cluster_sensitivity/analyze.py --check
  python analysis/manuscript/build_adaptation_assets.py --check
  python analysis/manuscript/build_retrospective_table.py --check

The class analysis checks 48 published accepted/error outcomes and generates
the class-specific tables. The group analysis reproduces 26 original points
and all 43 OpenFake group-deletion diagnostics. The adaptation check verifies
296 consistency relationships and the three adapted-detector tables.

FIGURES AND TABLES

With Matplotlib installed:

  python analysis/rebuild_manuscript_assets.py

This creates work/manuscript, then rebuilds the numerical tables and figures
there. The supplied sources and evidence remain unchanged. Choose another new
directory with --output if the default already exists. Generated PNGs and PDFs
are plots of numerical results, not dataset images. The generators also retain
supplementary developmental results and the earlier fixed-checkpoint stress
tests; those are distinct from the adapted-detector experiment.

The package contains the numerical assets, not the IEEE article template or a
complete manuscript build. Diagrams embedded directly in the manuscript's
LaTeX and the article's prose are not part of this analysis release.

Figure/table source files are retained from the research archive. Portability
changes are confined to local-path redaction, compact JSON serialization, and
the corresponding release checksum updates. Original source hashes remain in
evidence/source_files.json. Plot files may differ in PDF metadata or font
rendering across Matplotlib versions; the numerical checks are independent of
that presentation layer.

INTERVAL SCOPE

The default verification checks stored draws for the adapted detectors and
the later matched-coverage Bahat/initial-CLIP comparisons. Original principal,
MC/energy, and independent-ablation intervals are retained as archived results;
the default command does not recompute those bootstrap samples. Their original
statistical implementations and protocol settings are included separately in
the release. Group labels in the independent-ablation portable CSV have been
hashed; their sorted order is not the original bootstrap RNG index order.
