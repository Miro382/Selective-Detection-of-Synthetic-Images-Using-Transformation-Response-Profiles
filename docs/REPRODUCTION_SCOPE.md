# Reproduction scope

The files distinguish recorded measurements, derived numerical analyses, and
code for running experiments. These are different levels of reproduction.

## Numerical verification

The default `python verify.py` command checks the published distribution and
recomputes the supported statistics from stored numerical outputs. The
packaged metadata identifies the original experiment and source-file hashes.
Exported copies may replace local paths and sample identifiers with portable
or opaque identifiers. Their release hashes therefore need not equal the
hashes of the original archive files.

Opaque sample and group keys retain the within-analysis relationships needed
for paired comparisons. The ordering keys preserve the original deterministic
tie handling. They are not evidence that all possible duplicates or historical
training overlaps have been ruled out. The provenance statements and limitations
of the associated manuscript continue to apply.

An aggregate-only historical result can be checked against its recorded values
and table calculations; it cannot support a new per-image statistical analysis.
The analysis documentation identifies which comparisons have individual score
rows and which are retained only as aggregate history.

## Inference and training

The code and small selector parameter files describe the implemented methods.
Detector inference additionally needs the correct detector checkpoint,
preprocessing, and source images. Training additionally needs the intended
training and validation selections, external pretrained assets, and the
experiment-specific software environment. Checkpoint and manifest hashes
identify those assets without redistributing them.

The package verification command never starts a research run. Training and
image-based evaluation must be invoked explicitly after the required assets
have been supplied in a separate working directory. Original research outputs
should be treated as immutable when making a new run.

## What is not included

- Source dataset images or image archives.
- Third-party dataset listings, captions, or annotation collections.
- Pretrained or adapted detector checkpoint binaries.
- Third-party model repositories, IEEE template files, and proprietary fonts.
- Unfinished prospective experiments or results that have not been measured.
- Editorial conversations, review drafts, temporary files, or credentials.

The independent ablation decision, the main detector comparisons, and the
retrospective diagnostics retain their original interpretation. Packaging
does not create a new independent test or change the experimental outcomes.
