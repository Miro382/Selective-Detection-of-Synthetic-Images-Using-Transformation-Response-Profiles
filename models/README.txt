MODEL ARTIFACTS

All included fitted models are small JSON artifacts. No pickle or executable
model serialization is needed to inspect them.

efficientnet/full_selector.json
    The frozen nine-feature logistic selector, temperature, original threshold,
    calibration assessment, and checkpoint fingerprints.
efficientnet/resize_only_ablation.json
    The later resize-only comparator used in the independent ablation. It does
    not replace the main selector.
efficientnet_controls/
    Stored logistic and MLP comparator parameters, including all three MLP
    seeds. JSON state arrays are coefficients of the small selector networks,
    not image-detector checkpoints.
clip/selectors.json and resnet50/selectors.json
    Separately fitted full, confidence-only, JPEG-only, and resize-only selectors
    for the adapted detectors, with temperature and threshold-selection records.
detector_checkpoints.json
    Architecture/preprocessing metadata and SHA-256 fingerprints identifying the
    omitted principal and adapted detector checkpoints.

Detector weights are intentionally omitted for repository size and upstream
licensing reasons. A fingerprint identifies the archived file; it is not a
download link or proof that an arbitrary replacement model is equivalent.

The portable full-selector implementation and recorded-score checks are in
src/profile_selector.py. Historical control-model scoring and model construction
are in src/research/selective_experiments/selector_comparison_bahat/code/.
