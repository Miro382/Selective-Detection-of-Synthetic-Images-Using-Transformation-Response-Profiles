"""Build the retrospective matched-coverage table from the distributed JSON.

Uses only the Python standard library and retrospective_ci_20261030/results.json.
No original research directories, checkpoints, images, NumPy, fitting, or
bootstrap reruns are required. Run with --check to verify the existing table
without writing it. Paths resolve relative to this script, not the caller's cwd.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_RESULTS = ROOT / 'retrospective_ci_20261030/results.json'
DEFAULT_OUTPUT = ROOT / 'tables/retrospective_matched_intervals.tex'
CAPTION = (
    r'Retrospective paired intervals at the original matched coverage. '
    r'Differences are full profile minus Bahat-style adaptation. '
    r'The original sample uses the displayed $k$; each bootstrap replicate uses '
    r'$\lfloor (k/N)N^{(b)}\rfloor$ for both methods. '
    r'Intervals are pointwise 95\% from 5,000 whole-group replicates (seed 20261030). '
    r'Risk differences are percentage points; negative values favor the full selector.'
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def render(data: dict) -> bytes:
    """Return the published table, including its original CRLF line endings."""
    require(data['retrospective'] is True, 'Expected retrospective results.')
    require(data['seed'] == 20261030 and data['repetitions'] == 5000,
            'The table caption requires seed 20261030 and 5,000 replicates.')
    require(data['direction'] == 'profile minus baseline', 'Unexpected contrast direction.')
    lines = [
        r'\begin{table*}[t]',
        r'\caption{' + CAPTION + '}',
        r'\label{tab:matchedci}',
        r'\centering\small',
        r'\begin{tabular}{@{}lrrrrr@{}}',
        r'\toprule',
        r'Cohort & $k$ & Seed & Errors: full / reference & $\Delta$ risk & 95\% interval \\',
        r'\midrule',
    ]
    for name, label, expected_n, expected_k in (
        ('DeepShield', 'DS (in-domain held-out)', 10000, 9807),
        ('OpenFake', 'OF (cross-dataset)', 4000, 3639),
    ):
        cohort = data['comparisons']['efficientnet_' + name]
        full = cohort['methods']['frozen_v4']
        k, errors = full['accepted'], full['errors']
        require(cohort['n'] == expected_n and k == expected_k,
                'Unexpected original sample/acceptance count for ' + name)
        for row, seed in enumerate((17, 29, 43)):
            method = 'mlp_bahat_logits_s' + str(seed)
            reference = cohort['methods'][method]
            difference = cohort['differences'][method]
            ci = difference['interval_pp']
            delta, lo, hi = difference['estimate_pp'], ci['lower_95'], ci['upper_95']
            require(reference['accepted'] == k, 'Methods must accept the same count.')
            require(difference['metric'] == 'risk', 'Expected a selective-risk contrast.')
            require(ci['valid_repetitions'] == 5000, 'Incomplete bootstrap interval.')
            require(all(math.isfinite(x) for x in (delta, lo, hi)) and lo <= hi,
                    'Invalid interval values.')
            require(math.isclose(delta, 100 * (errors - reference['errors']) / k,
                                 rel_tol=0., abs_tol=1e-12),
                    'Risk difference does not match the displayed accepted errors.')
            first = label if row == 0 else ''
            lines.append(
                f"{first} & {k:,} & {seed} & {errors} / {reference['errors']} & "
                f"${delta:.5f}$ & $[{lo:.5f},{hi:.5f}]$ " + r'\\'
            )
    lines.extend((r'\bottomrule', r'\end{tabular}', r'\end{table*}'))
    return ('\r\n'.join(lines) + '\r\n').encode('utf-8')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=DEFAULT_RESULTS)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--check', action='store_true', help='Compare bytes; do not write the table.')
    args = parser.parse_args()
    generated = render(json.loads(args.results.read_text(encoding='utf-8')))
    if args.check:
        if not args.output.is_file() or args.output.read_bytes() != generated:
            raise SystemExit('Table differs from the supplied aggregate results: ' + str(args.output))
        status = 'Verified byte-identical'
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(generated)
        status = 'Wrote'
    print(f"{status}: {args.output}")
    print('SHA256: ' + hashlib.sha256(generated).hexdigest())


if __name__ == '__main__':
    main()
