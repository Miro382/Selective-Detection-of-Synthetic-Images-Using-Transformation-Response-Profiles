"""Score stored detector margins with the article's frozen logistic selectors.

The input columns are the original view, JPEG 90/75/60, and resize 0.75/0.5.
Margins are native fake-minus-real logits before temperature scaling. A scalar
fake logit has the same interpretation. This module does not load images or
detector weights and does not fit parameters.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODEL_FILES = {
    'efficientnet': 'models/efficientnet/full_selector.json',
    'clip': 'models/clip/selectors.json',
    'resnet50': 'models/resnet50/selectors.json',
}
VIEW_ORDER = ('original', 'jpeg_q90', 'jpeg_q75', 'jpeg_q60',
              'resize_0p75', 'resize_0p5')


def sigmoid(values):
    values = np.asarray(values, dtype=np.float64)
    magnitude = np.exp(-np.abs(values))
    return np.where(values >= 0, 1.0 / (1.0 + magnitude),
                    magnitude / (1.0 + magnitude))


def profile_features(raw_margins, temperature):
    raw = np.asarray(raw_margins, dtype=np.float64)
    if raw.ndim != 2 or raw.shape[1] != 6 or not np.isfinite(raw).all():
        raise ValueError('Expected an N-by-6 array of finite native margins.')
    if isinstance(temperature, bool) or not np.isfinite(temperature) or temperature <= 0:
        raise ValueError('Temperature must be finite and positive.')
    margins = raw / temperature
    if not np.isfinite(margins).all():
        raise ValueError('Temperature-scaled margins are not finite.')
    orientation = np.where(margins[:, :1] >= 0, 1.0, -1.0)
    probabilities = sigmoid(margins)
    return np.concatenate((
        -np.abs(margins[:, :1]),
        orientation * (margins[:, 1:] - margins[:, :1]),
        probabilities.var(axis=1)[:, None],
        np.max(np.abs(probabilities[:, 1:] - probabilities[:, :1]), axis=1)[:, None],
        np.mean((margins[:, 1:] >= 0) != (margins[:, :1] >= 0), axis=1)[:, None],
    ), axis=1)


def model_parameters(detector, root=ROOT):
    artifact = json.loads((Path(root) / MODEL_FILES[detector]).read_text('utf-8'))
    if detector == 'efficientnet':
        if artifact['model']['family'] != 'symmetric_profile':
            raise ValueError('The principal artifact is not the nine-feature model.')
        parameters = artifact['model']['parameters']
        temperature = artifact['temperature']
        mean, scale = parameters['scaler']['mean'], parameters['scaler']['scale']
    else:
        parameters = artifact['selector']['models']['lr_full']
        if parameters['columns'] != list(range(6)):
            raise ValueError('The adapted artifact does not use all six views.')
        temperature = artifact['selector']['temperature']
        mean, scale = parameters['mean'], parameters['scale']
    mean, scale, coefficient = (np.asarray(x, dtype=np.float64) for x in
                                (mean, scale, parameters['coef']))
    if any(x.shape != (9,) or not np.isfinite(x).all() for x in (mean, scale, coefficient)):
        raise ValueError('A full selector must have nine finite parameters and scales.')
    intercept = float(parameters['intercept'])
    if not np.isfinite(intercept) or np.any(scale <= 0):
        raise ValueError('Invalid intercept or feature scale.')
    return temperature, mean, scale, coefficient, intercept


def score(raw_margins, detector, root=ROOT):
    temperature, mean, scale, coefficient, intercept = model_parameters(detector, root)
    features = profile_features(raw_margins, temperature)
    return sigmoid(((features - mean) / scale) @ coefficient + intercept)


def check(root=ROOT):
    """Compare release scoring with every stored test score, without inference."""
    root = Path(root)
    report = []
    for detector in MODEL_FILES:
        for cohort in ('DeepShield', 'OpenFake'):
            path = root / 'evidence' / 'scores' / f'{detector}_{cohort}.npz'
            with np.load(path, allow_pickle=False) as data:
                margin_key = 'raw_margins' if detector == 'efficientnet' else 'margins'
                expected_key = 'score__frozen_v4' if detector == 'efficientnet' else 'score__lr_full'
                raw = data[margin_key]
                expected = data[expected_key]
            actual = score(raw, detector, root)
            if actual.shape != expected.shape or not np.isfinite(expected).all():
                raise AssertionError(f'Invalid recorded score array: {path.name}')
            difference = float(np.max(np.abs(actual - expected)))
            if not np.allclose(actual, expected, rtol=0, atol=1e-12):
                raise AssertionError(f'Score mismatch: {path.name}, max absolute difference {difference}')
            report.append({'detector': detector, 'cohort': cohort, 'rows': len(actual),
                           'max_absolute_difference': difference})
    return {'status': 'passed', 'comparison': 'frozen parameters against stored detector-output scores',
            'absolute_tolerance': 1e-12, 'checks': report}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='Compare all released full-profile scores.')
    parser.add_argument('--detector', choices=tuple(MODEL_FILES))
    parser.add_argument('--input', type=Path, help='NPZ file containing native six-view margins.')
    parser.add_argument('--array', default='raw_margins', help='NPZ array name; use margins for adapted-detector evidence.')
    parser.add_argument('--output', type=Path, help='New output CSV path; existing files are never overwritten.')
    args = parser.parse_args(argv)
    if args.check:
        if args.detector or args.input or args.output:
            parser.error('--check cannot be combined with a scoring job.')
        print(json.dumps(check(), indent=2))
        return 0
    if not all((args.detector, args.input, args.output)):
        parser.print_help()
        return 0
    with np.load(args.input, allow_pickle=False) as data:
        values = score(data[args.array], args.detector)
    with args.output.open('x', encoding='utf-8', newline='') as stream:
        np.savetxt(stream, values, delimiter=',', header='error_score', comments='', fmt='%.17g')
    print(f'Wrote {len(values)} scores to {args.output}')
    return 0


if __name__ == '__main__':
    sys.dont_write_bytecode = True
    raise SystemExit(main())
