RETROSPECTIVE CONNECTED-GROUP SENSITIVITY

This standalone analysis evaluates the influence of each of the 43 connected
groups in the independent OpenFake resize-only/full-profile comparison.
It does not alter the original frozen models, protocol, bootstrap intervals,
or four-condition replacement decision. It uses no images, network access,
detector inference, model training, or fresh bootstrap sampling.

Requirements: Python 3.11+ and NumPy. Tested with Python 3.12.14 and NumPy 2.3.5.

From this directory:
    python analyze.py --check
checks the portable-input hashes, recomputes the 26 source-point comparisons,
all 43 deletion contrasts, and checks the packaged JSON, CSV and article table.
The ordinary command `python analyze.py` regenerates those three derived files.
For source extraction only, `python analyze.py --extract-from RUN` reads the
archived run and verifies its source checksums; it never writes into that run.

Files:
  scores.csv: all 7,500 archived error/score rows; no image data or paths.
    cohort: DeepShield or OpenFake
    error: 1 iff the fixed detector's original prediction is incorrect
    full_score, resize_score: stored scores; smaller means accept earlier
    tie_key: original SHA256(article-ablation-confirmation-tie-v1/sample_id)
    group_id: namespace-hashed original connected-group identifier
  baseline.json: original archived points for both cohorts, preserved as anchors
  provenance.json: source hashes, portable-input hashes, unchanged locked decision
  results.json: original points, all deletion results, ranges and sign counts
  leave_one_group_out.csv: one row per OpenFake group deletion
  analyze.py: deterministic analysis and exact manuscript-table generator

The matched acceptance count is the exact integer floor(3639*N/4000).
The retained sample is reranked separately after each deletion. AURC uses
whole equal-score blocks, while fixed-count acceptance uses original tie hashes.
All differences are resize-only minus full. Risk differences are percentage
points. AURC differences are unscaled fractions. No inferential intervals are
constructed for this retrospective diagnostic. Deletion scenarios overlap
and cannot be interpreted as independent confirmatory replications.

Summary: AURC differences remain negative for all 43 omissions. Matched-risk
differences are negative for 42 omissions and positive only when the largest
3,412-image group is removed. The locked decision remains retain_full.
