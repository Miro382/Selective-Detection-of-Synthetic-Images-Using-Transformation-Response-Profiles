# Selective Detection of Synthetic Images Using Transformation Response Profiles

Research materials for the manuscript by Miroslav Murin, Miroslav Michalko,
Ondrej Kainz, and Ivana Semanová, Technical University of Košice.
Corresponding author: Ondrej Kainz, ondrej.kainz@tuke.sk.

The method ranks the decisions of a frozen image detector using its responses
to JPEG compression and resizing. Six detector views form a nine-feature
profile. A separate logistic selector estimates an error score and leaves
the detector's original class decision unchanged.

This distribution includes research code, small fitted selector models,
experiment configurations, numerical evaluation outputs, and scripts that
recompute reported results. Images and detector checkpoints are obtained
separately. The manuscript and the IEEE typesetting assets are not included.

## Verify the reported results

Use Python 3.11 or newer in a separate environment. From this directory:

```sh
python -m pip install -r requirements.txt
python verify.py
```

Verification checks file hashes, then runs the numerical checks listed in
`verification.json`. These checks use the packaged scores and labels; they
do not need a GPU, dataset downloads, detector inference, or model fitting.
An integrity-only check has no external Python dependencies:

```sh
python verify.py --integrity-only
```

Optional figure rendering requires:

```sh
python -m pip install -r requirements-figures.txt
```

See the instructions under `analysis/` for individual analyses and figure
commands. Numerical verification and figure rendering are separate tasks.

## Contents

| Directory | Contents |
| --- | --- |
| `analysis/` | Portable checks, retrospective analyses, and figure/table code |
| `evidence/` | Numerical outputs, aggregates, protocols, and source fingerprints |
| `src/` | Detector, profile, selector, and reference-method research code |
| `models/` | Fitted selector parameters and detector checkpoint fingerprints |
| `configs/` | Recorded experiment settings and portable configuration examples |
| `docs/` | Dataset sources, third-party notices, and reproduction scope |

`release_manifest.json` records the SHA-256 digest and size of each distributed
file. It detects changes to the assembled package; it is not a digital
signature or an independent certification of an experiment.
The `.gitattributes` file preserves file bytes across Git checkouts so that
platform-specific line-ending conversion does not invalidate these hashes.

## Experimental scope

The principal detector is EfficientNetV2-M. DeepShield is an in-domain held-out
test: other DeepShield images contributed to detector and selector development.
OpenFake is the cross-dataset test relative to the documented fitting sources.
Its source was known during earlier exploration. Comparisons with MC dropout,
energy, the learned references, and the adapted detectors reuse these evaluation
images and should not be counted as independent data collections.

The additional-detector experiment adapts a CLIP linear head and ResNet-50 using
the principal detector's training manifest and fits a separate selector for
each detector. The independent 7,500-image experiment compares the full selector
with the three-view resizing alternative under four locked replacement
conditions. Only one condition passes; the full selector remains the principal
method. The ablation is not an independent repeat of every baseline comparison.

Scores, class labels, and opaque grouping/tie keys allow numerical reanalysis
without redistributing images. Reported intervals are conditional on the fitted
models and the recorded evaluation design. Stored MC-dropout scores also
condition on the recorded dropout draws.

## Models and training

The fitted selector parameters are small and are supplied separately from
detector weights. The source tree includes the training and evaluation logic
used by the completed experiments, with its dependencies and configuration
requirements documented alongside the code. Archived workflows require the
corresponding data manifests and externally obtained model assets. These are
not replaced with invented example data or silently downloaded by verification.

Recomputing results from stored outputs is supported by the supplied checks.
Recreating the complete detector-training history requires additional external
assets and the original data selections. No claim of bitwise reproducible GPU
retraining is made. See `docs/REPRODUCTION_SCOPE.md` and the source documentation
before running a new training or evaluation job.

## License

The authors' contributions to this repository, including research code,
documentation, fitted selector parameters, and numerical evaluation outputs,
are distributed under the [MIT License](LICENSE).

This license does not cover third-party dataset images, annotations, detector
checkpoints, or software obtained from external providers. These assets are not
bundled and remain subject to their respective licenses and access terms. The
small fitted selector parameters included in `models/` are distinct from the
externally obtained detector weights. Dataset metadata, file fingerprints, and
derived numerical outputs do not grant rights to the source datasets or images.
See [data sources](docs/DATA_SOURCES.txt) and
[third-party notices](docs/THIRD_PARTY_NOTICES.txt) for provider information.

## Citation and acknowledgments

The accompanying repository is available on
[GitHub](https://github.com/Miro382/Selective-Detection-of-Synthetic-Images-Using-Transformation-Response-Profiles).
Citation metadata is supplied in [CITATION.cff](CITATION.cff); manuscript
publication details will be added after publication.
Assistance with this distribution is recorded in
[ACKNOWLEDGMENTS.md](ACKNOWLEDGMENTS.md).
