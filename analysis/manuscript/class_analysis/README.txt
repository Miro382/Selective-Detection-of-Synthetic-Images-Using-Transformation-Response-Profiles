CLASS-CONDITIONAL RETROSPECTIVE ANALYSIS

Scope
-----
This sidecar reads the stored predictions of the principal EfficientNetV2-M
and adapted CLIP/ResNet-50 detectors on the original 10,000 DeepShield and
4,000 OpenFake images. It reports no new independent test or inferential
interval. No training, fitting, image decoding, detector inference, download,
or mutation of original results is performed.

Inputs
------
inputs/ contains six compressed CSV files, one per detector/cohort.
Each row supplies reference class (0 real, 1 synthetic), original predicted
class, full-profile score, confidence score, and opaque image/group/tie keys.
Image and group keys are namespace-prefixed SHA-256 digests. Tie keys retain
the precise original namespace-prefixed SHA-256 digest, preserving the exact
original label-independent ordering for tied scores. No original image,
path, sample name or checkpoint is included. metadata.json fixes file hashes,
archived thresholds, expected aggregate outcomes, and evaluation conventions.
source_provenance.json identifies the original input files by relative paths
and hashes; it is informational for users without that source archive.

Reproduce from these inputs (Python >=3.11, NumPy)
-----------------------------------------------
  python build_class_analysis.py --check

This read-only command checks input hashes, reconstructs all metrics, checks
48 archived accepted-count/error-count outcomes, and verifies exact agreement
with results.json, results.csv and the two companion manuscript table files
../tables/class_conditional.tex and ../tables/class_conditional_details.tex.
The default command without --check regenerates those four outputs only.
The optional --extract-from ARCHIVE_ROOT command is intended for a fresh
sidecar before supplied inputs exist; it exports only stored numerical data.
It refuses to overwrite an existing metadata.json.

Definitions
-----------
FP: accepted real image predicted synthetic.
FN: accepted synthetic image predicted real.
Per-class coverage: accepted true-class count divided by all observations
of that true class. Per-class accepted risk: errors divided by the accepted
true-class count. Risk is undefined if that accepted count is zero.

original_matched: both methods take the first k scores under the original
tie rule, k=9,807 on DeepShield and k=3,639 on OpenFake.
frozen_threshold: full profile applies its archived threshold; confidence
accepts exactly the same number of images. Confidence's own threshold is not
used. Class labels are used only in reporting, never to select accepted cases.
unconditional: every original detector decision is included.

Interpretation
--------------
At original matched coverage, all six full-profile comparisons reduce FN and
increase FP relative to confidence. Improvements in the total error count
therefore do not imply improvements for both error types. Frozen-threshold
results may have different class coverages and need not favor the full profile
at every operating point. No unreported class-level significance is claimed.
