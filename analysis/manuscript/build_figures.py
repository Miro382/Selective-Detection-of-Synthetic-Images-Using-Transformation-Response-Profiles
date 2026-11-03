"""Render manuscript figures from completed, frozen local measurements only.

No image reads, model construction, inference, training, or new bootstrap draws.
Run with the AudioDetection/.venv-audio Python, optionally --output PATH.
Portable package: python build_figures.py --from-aggregate
The portable mode uses figures/figure_data.json and its archived source receipt;
it never opens the historical experiment paths. It is also selected automatically
when any historical run is absent. Only files below this manuscript directory
may be written. Existing source_receipt.json remains the historical provenance;
aggregate_render_receipt.json records a subsequent portable rendering.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
os.environ['MPLCONFIGDIR'] = str(HERE / 'figures' / '.mplconfig')
os.environ.setdefault('SOURCE_DATE_EPOCH', '1793232000')
sys.dont_write_bytecode = True

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

REFERENCE = ROOT / 'article_reference_experiment/runs/retrospective_20261028'
PUBLIC = ROOT / 'article_detector_experiment/external_testing/runs/independent_20261028'
ABLATION = ROOT / 'article_ablation_confirmation/runs/independent_20261029'
RC_METHODS = ('confidence', 'tta_mean', 'mc_entropy_30', 'energy_t1',
              'frozen_v4', 'mlp_bahat_logits_s43')
PUBLIC_METHODS = ('confidence', 'tta_mean', 'lr_confidence', 'lr_full',
                  'lr_jpeg_only', 'lr_resize_only')
RC_LABELS = {'confidence': 'Confidence', 'tta_mean': 'TTA mean',
             'mc_entropy_30': 'MC entropy (30)', 'energy_t1': 'Energy (T = 1)',
             'frozen_v4': 'Full profile', 'mlp_bahat_logits_s43': 'Bahat MLP (seed 43)'}
PUBLIC_LABELS = {'confidence': 'Confidence', 'tta_mean': 'TTA mean',
                 'lr_confidence': 'LR confidence*', 'lr_full': 'Full profile',
                 'lr_jpeg_only': 'JPEG profile', 'lr_resize_only': 'Resize profile'}
COLORS = {'confidence': '#777777', 'tta_mean': '#E69F00',
          'mc_entropy_30': '#AA4499', 'energy_t1': '#009E73',
          'frozen_v4': '#0072B2', 'mlp_bahat_logits_s43': '#D55E00',
          'lr_confidence': '#D55E00', 'lr_full': '#0072B2',
          'lr_jpeg_only': '#009E73', 'lr_resize_only': '#AA4499'}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


class Sources:
    def __init__(self):
        self.items = {}

    def record(self, path, expected=None):
        path = Path(path).resolve()
        actual = sha(path)
        if expected is not None and actual != expected:
            raise ValueError(f'SHA mismatch: {path}')
        self.items[path.as_posix()] = {'sha256': actual, 'bytes': path.stat().st_size,
                                      'expected_sha256': expected}
        return path

    def read(self, path, expected=None):
        return json.loads(self.record(path, expected).read_text(encoding='utf-8-sig'))

    def complete(self, directory):
        receipt = self.read(directory / 'complete.json')
        for name, expected in receipt['files'].items():
            target = (directory / name).resolve()
            if not target.is_relative_to(directory.resolve()):
                raise ValueError('Completion target escapes run.')
            self.record(target, expected)
        return receipt

    def protocol(self, directory, name='protocol'):
        sidecar = self.record(directory / (name + '.sha256'))
        expected = sidecar.read_text(encoding='ascii').strip()
        return self.read(directory / (name + '.json'), expected)

    def recheck(self):
        for path, value in self.items.items():
            if sha(path) != value['sha256']:
                raise ValueError(f'Source changed while plotting: {path}')


def strict_close(actual, expected, description, atol=1e-13):
    if not np.isclose(actual, expected, rtol=0, atol=atol):
        raise ValueError(f'{description}: {actual!r} != {expected!r}')


def load_reference(sources):
    complete = sources.complete(REFERENCE)
    protocol = sources.protocol(REFERENCE)
    for name, expected in protocol['files'].items():
        sources.record(REFERENCE / name, expected)
    result = sources.read(REFERENCE / 'results.json')
    for name, filename in [('protocol_sha256', 'protocol.json'),
                           ('profiles_receipt_sha256', 'profiles.complete.json'),
                           ('profiles_sha256', 'mc_profiles.jsonl'),
                           ('runtime_sha256', 'runtime.json')]:
        if result['inputs'][name] != complete['inputs'][name]:
            raise ValueError('Reference result/completion mismatch.')
        sources.record(REFERENCE / filename, result['inputs'][name])
    receipt = sources.read(REFERENCE / 'profiles.complete.json')
    sources.record(REFERENCE / 'mc_profiles.jsonl', receipt['sha256'])
    plans = [json.loads(line) for line in (REFERENCE / 'plan.jsonl').read_text(encoding='utf-8').splitlines()]
    rows = [json.loads(line) for line in (REFERENCE / 'mc_profiles.jsonl').read_text(encoding='utf-8').splitlines()]
    with np.load(REFERENCE / 'inputs.npz', allow_pickle=False) as bundle:
        arrays = {name: bundle[name] for name in bundle.files}
    if len(plans) != len(rows) or len(rows) != receipt['count'] or len(rows) != len(arrays['errors']):
        raise ValueError('Reference row count mismatch.')
    if [p['sample_id'] for p in plans] != [row['sample_id'] for row in rows]:
        raise ValueError('MC rows no longer align with frozen plans.')
    labels = np.asarray([p['label'] for p in plans])
    original_prediction = (arrays['logits'][:, 1] >= arrays['logits'][:, 0]).astype(np.int64)
    if not np.array_equal(arrays['errors'], original_prediction != labels):
        raise ValueError('Errors are not those of the original deterministic predictions.')
    deterministic = np.asarray([row['deterministic_logits'] for row in rows])
    mc = np.asarray([row['mc_logits'] for row in rows])
    if not np.allclose(deterministic, arrays['logits'], rtol=1e-6, atol=2e-5):
        raise ValueError('Cached deterministic logits mismatch.')
    code = REFERENCE / 'snapshot/code/scores.py'
    spec = importlib.util.spec_from_file_location('_manuscript_frozen_scores', code)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # Pure NumPy scores; no detector runtime is imported.
    additions = module.new_scores(deterministic, mc)['scores']
    score_map = {name: arrays['score__' + name] for name in RC_METHODS if name not in additions}
    score_map.update({name: np.asarray(additions[name]) for name in RC_METHODS if name in additions})
    cohorts = {}
    for name in ('DeepShield', 'OpenFake'):
        indices = np.asarray([i for i, plan in enumerate(plans) if plan['cohort'] == name])
        errors = arrays['errors'][indices]
        ids = [plans[i]['sample_id'] for i in indices]
        metrics = result['cohorts'][name]
        if len(indices) != metrics['profiles'] or int(errors.sum()) != metrics['detector']['errors']:
            raise ValueError('Reference cohort mismatch.')
        methods = {}
        for method in RC_METHODS:
            prepared = module._prepared(errors, score_map[method][indices], ids)
            point = module._statistics(prepared, np.ones(len(indices), dtype=np.int64))
            target = metrics['methods'][method]
            strict_close(point['aurc'], target['aurc'], name + '/' + method + '/AURC')
            if point['risk_at_coverage'] != target['risk_at_coverage']:
                raise ValueError('Frozen fixed-coverage scores do not reproduce stored metrics.')
            _, ordered_errors, ends = prepared
            accepted = ends + 1
            cumulative_errors = np.cumsum(ordered_errors, dtype=np.int64)[ends]
            coverage = accepted / len(indices)
            risk = cumulative_errors / accepted
            strict_close(float(np.diff(np.r_[0., coverage]) @ risk), target['aurc'], 'Block integral')
            methods[method] = {'aurc': target['aurc'], 'accepted': accepted.tolist(),
                               'coverage': coverage.tolist(), 'errors': cumulative_errors.tolist(),
                               'risk': risk.tolist(), 'exact_score_blocks': len(ends),
                               'risk_at_coverage': target['risk_at_coverage']}
        cohorts[name] = {'n': len(indices), 'groups': metrics['groups'],
                         'detector_errors': int(errors.sum()), 'methods': methods}
    return {'scope': 'Retrospective descriptive reuse; not a new independent holdout.',
            'curves': 'Whole exact-score blocks; steps-pre matches the stored right-endpoint block AURC.',
            'target': 'Errors of the original fixed deterministic two-logit prediction.',
            'selection_of_displayed_mlp': 'Seed 43 is displayed as requested; all three seeds remain in the results table. No claim of preselected best seed.',
            'cohorts': cohorts}


def load_public(sources):
    complete = sources.complete(PUBLIC)
    sources.protocol(PUBLIC)
    sources.record(PUBLIC / 'protocol.json', complete['inputs']['run_sha256'])
    result = sources.read(PUBLIC / 'results.json')
    evidence = sources.read(HERE / 'evidence/new_experiments.json')
    source_expected = evidence['sources'][(PUBLIC / 'results.json').as_posix()]['sha256']
    sources.record(PUBLIC / 'results.json', source_expected)
    cohorts = {}
    for detector in ('clip', 'resnet50'):
        cohorts[detector] = {}
        for dataset in ('DeepShield', 'OpenFake', 'CommunityForensics'):
            records = [row for row in evidence['tables']['public_detector_all_methods']
                       if row['scope'] == 'external' and row['detector'] == detector and row['dataset'] == dataset]
            if len(records) != 6:
                raise ValueError('Expected all six public methods.')
            authoritative = result['results'][detector][dataset]
            values = {}
            for method in PUBLIC_METHODS:
                row = next(row for row in records if row['method'] == method)
                strict_close(row['aurc'], authoritative['methods'][method]['aurc'], 'Public AURC')
                values[method] = row['aurc']
            cohorts[detector][dataset] = {'n': records[0]['n'], 'groups': records[0]['groups'],
                                         'aurc': values}
    return {'cohorts': cohorts, 'primary': result['primary'],
            'caveat': 'CLIP/OpenFake: LR confidence AURC is lower than full profile; no stored paired full-minus-LR-confidence CI. CLIP JPEG also has lower point AURC than full on OpenFake and CommunityForensics. ResNet full has the lowest AURC among the six displayed methods on all three cohorts.'}


def load_ablation(sources):
    complete = sources.complete(ABLATION)
    sources.protocol(ABLATION, 'run_protocol')
    result = sources.read(ABLATION / 'results.json')
    if result['inputs'] != complete['inputs']:
        raise ValueError('Ablation result/completion mismatch.')
    sources.record(ABLATION / 'run_protocol.json', result['inputs']['run_protocol_sha256'])
    contrasts = []
    for cohort in ('DeepShield', 'OpenFake'):
        data = result['cohorts'][cohort]
        for metric in ('aurc', 'matched'):
            ci = data['intervals'][metric]
            contrasts.append({'cohort': cohort, 'metric': metric,
                              'estimate': data['delta_resize_minus_full'][metric],
                              'lower_95': ci['lower_95'], 'upper_95': ci['upper_95'],
                              'passed': result['conditions'][cohort + ':' + metric]['passed'],
                              'n': data['profiles'], 'groups': data['groups'],
                              'matched_error_discordant_groups': data['paired_error_discordance']['discordant_groups'],
                              'matched_q': data['matched_fraction'],
                              'matched_k': data['models']['full']['risks']['matched']['accepted']})
    return {'contrast': 'resize-minus-full', 'decision': result['decision'],
            'intervals': 'Original pointwise paired group-bootstrap percentile 95% intervals; no recomputation.',
            'conditions': contrasts, 'all_four_conditions_passed': result['all_four_conditions_passed'],
            'guard': 'At least 10 matched-error-discordant groups required; DeepShield has 1 and OpenFake has 20.'}


def load_aggregate(sources, path):
    """Validate the packaged data, without accessing any historical source path."""
    path = path.resolve()
    receipt_path = path.parent / 'source_receipt.json'
    archived_receipt = sources.read(receipt_path)
    expected = archived_receipt['outputs']['figure_data.json']['sha256']
    data = sources.read(path, expected)
    if data.get('version') != 'article-manuscript-figures-v1':
        raise ValueError('Unsupported packaged figure-data version.')
    cohorts = data['efficientnet_retrospective']['cohorts']
    if set(cohorts) != {'DeepShield', 'OpenFake'}:
        raise ValueError('Packaged reference cohorts changed.')
    for name, value in cohorts.items():
        n = value['n']
        if n != {'DeepShield': 10000, 'OpenFake': 4000}[name] or set(value['methods']) != set(RC_METHODS):
            raise ValueError('Packaged reference count or methods changed.')
        for method, curve in value['methods'].items():
            accepted = np.asarray(curve['accepted'], dtype=np.int64)
            errors = np.asarray(curve['errors'], dtype=np.int64)
            coverage = np.asarray(curve['coverage'], dtype=np.float64)
            risk = np.asarray(curve['risk'], dtype=np.float64)
            if (len(accepted) != curve['exact_score_blocks'] or accepted[-1] != n
                    or accepted[0] < 1 or not np.all(np.diff(accepted) > 0)
                    or not np.all(np.diff(errors) >= 0) or np.any(errors > accepted)
                    or errors[-1] != value['detector_errors']
                    or not np.array_equal(coverage, accepted / n)
                    or not np.array_equal(risk, errors / accepted)):
                raise ValueError('Packaged whole-score-block curve is inconsistent.')
            strict_close(float(np.diff(np.r_[0., coverage]) @ risk), curve['aurc'],
                         f'Packaged AURC {name}/{method}')
    public = data['public_detectors']['cohorts']
    if set(public) != {'clip', 'resnet50'}:
        raise ValueError('Packaged public detector set changed.')
    for detector in public.values():
        if set(detector) != {'DeepShield', 'OpenFake', 'CommunityForensics'}:
            raise ValueError('Packaged public cohort set changed.')
        for cohort in detector.values():
            if set(cohort['aurc']) != set(PUBLIC_METHODS):
                raise ValueError('Packaged public method set changed.')
            if not all(np.isfinite(x) and 0 <= x <= 1 for x in cohort['aurc'].values()):
                raise ValueError('Invalid packaged public AURC.')
    ablation = data['independent_ablation']
    conditions = {(v['cohort'], v['metric']) for v in ablation['conditions']}
    if conditions != {(d, m) for d in ('DeepShield', 'OpenFake') for m in ('aurc', 'matched')}:
        raise ValueError('Packaged ablation contrasts changed.')
    for value in ablation['conditions']:
        if (not all(np.isfinite(value[k]) for k in ('estimate', 'lower_95', 'upper_95'))
                or value['lower_95'] > value['upper_95']
                or bool(value['passed']) != bool(value['upper_95'] < 0)):
            raise ValueError('Packaged ablation interval or condition is inconsistent.')
    return data


def style():
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8.2,
                         'axes.labelsize': 8.2, 'axes.titlesize': 8.5,
                         'xtick.labelsize': 8.0, 'ytick.labelsize': 8.0,
                         'legend.fontsize': 8.0, 'axes.linewidth': .65,
                         'lines.linewidth': 1.25, 'pdf.fonttype': 42,
                         'ps.fonttype': 42, 'path.simplify': False,
                         'savefig.facecolor': 'white'})


def clean_axis(ax):
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', color='#D8D8D8', linewidth=.45, zorder=0)
    ax.tick_params(width=.6, length=3)


def save(fig, output, stem):
    # Fixed 7.1-inch page width; do not let bbox_inches='tight' change physical size.
    fig.savefig(output / (stem + '.pdf'), metadata={'Creator': 'Frozen local manuscript evidence',
                                                  'CreationDate': None, 'ModDate': None})
    fig.savefig(output / (stem + '.png'), dpi=300)
    plt.close(fig)


def risk_coverage(data, output):
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.95))
    for panel, (ax, cohort) in enumerate(zip(axes, ('DeepShield', 'OpenFake'))):
        cohort_data = data['cohorts'][cohort]
        for method in RC_METHODS:
            value = cohort_data['methods'][method]
            x, y = np.asarray(value['coverage']) * 100, np.asarray(value['risk']) * 100
            # Add the first block value at coverage zero only for a complete step rectangle.
            ax.step(np.r_[0., x], np.r_[y[0], y], where='pre',
                    color=COLORS[method], linestyle='-',
                    linewidth=1.8 if method == 'frozen_v4' else 1.15,
                    label=RC_LABELS[method], zorder=5 if method == 'frozen_v4' else 3)
        ax.set_title(f'({chr(97 + panel)}) {cohort} (N = {cohort_data["n"]:,})', loc='left', pad=7)
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 2.0 if cohort == 'DeepShield' else 40.0)
        if cohort == 'OpenFake':
            ax.set_yticks([0, 10, 20, 30, 40])
        ax.set_xlabel('Coverage (%)')
        ax.set_ylabel('Selective risk (%)')
        ax.set_xticks([0, 20, 40, 60, 80, 100])
        clean_axis(ax)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, frameon=False,
               bbox_to_anchor=(.5, .005), columnspacing=1.9, handlelength=3.0)
    fig.subplots_adjust(left=.075, right=.985, bottom=.30, top=.89, wspace=.28)
    save(fig, output, 'efficientnet_risk_coverage')


def public_aurc(data, output):
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.10), sharey=True)
    datasets = ('DeepShield', 'OpenFake', 'CommunityForensics')
    width = .119
    for panel, (ax, detector) in enumerate(zip(axes, ('clip', 'resnet50'))):
        for index, method in enumerate(PUBLIC_METHODS):
            values = [data['cohorts'][detector][name]['aurc'][method] for name in datasets]
            ax.bar(np.arange(3) + (index - 2.5) * width, values, width=width * .94,
                   label=PUBLIC_LABELS[method], color=COLORS[method],
                   edgecolor='white', linewidth=.35, zorder=3)
        ax.set_xticks(range(3), ['DeepShield', 'OpenFake', 'Community\nForensics'])
        ax.set_title(f'({chr(97 + panel)}) ' + ('CLIP ViT-L/14 + linear head' if detector == 'clip' else 'ResNet-50'),
                     loc='left', pad=7)
        ax.set_ylim(0, .75)
        ax.set_yticks(np.arange(0, .71, .1))
        ax.set_ylabel('AURC (lower is better)')
        clean_axis(ax)
        if detector == 'clip':
            # The visible star is explained by the manuscript caption and figure_data.
            x = 1 + (PUBLIC_METHODS.index('lr_confidence') - 2.5) * width
            y = data['cohorts'][detector]['OpenFake']['aurc']['lr_confidence']
            ax.text(x, y + .012, '*', ha='center', va='bottom', color=COLORS['lr_confidence'],
                    fontsize=12, fontweight='bold')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, frameon=False,
               bbox_to_anchor=(.5, .005), columnspacing=2.2, handlelength=1.7)
    fig.subplots_adjust(left=.075, right=.985, bottom=.32, top=.89, wspace=.25)
    save(fig, output, 'public_detectors_aurc')


def ablation_forest(data, output):
    fig, axes = plt.subplots(2, 2, figsize=(7.1, 3.65))
    for col, cohort in enumerate(('DeepShield', 'OpenFake')):
        for row, metric in enumerate(('aurc', 'matched')):
            ax = axes[row, col]
            value = next(x for x in data['conditions'] if x['cohort'] == cohort and x['metric'] == metric)
            factor = 1000. if metric == 'aurc' else 100.
            estimate, low, high = [value[key] * factor for key in ('estimate', 'lower_95', 'upper_95')]
            width = max(high, 0.) - min(low, 0.)
            ax.set_xlim(min(low, 0.) - width * .14, max(high, 0.) + width * .15)
            ax.axvline(0, ymin=0, ymax=.76, color='#777777', linestyle='-', linewidth=.8, zorder=1)
            ax.errorbar(estimate, .59, xerr=[[estimate - low], [high - estimate]],
                        fmt='o', color='#0072B2', markersize=5, elinewidth=1.35,
                        capsize=3.5, markeredgecolor='white', markeredgewidth=.55, zorder=3)
            ax.set_ylim(0, 1)
            ax.set_yticks([])
            ax.spines[['top', 'right', 'left']].set_visible(False)
            ax.xaxis.set_major_locator(MaxNLocator(5))
            ax.tick_params(width=.6, length=3)
            label = 'AURC' if metric == 'aurc' else f'Risk at k = {value["matched_k"]:,}'
            ax.set_title(f'({chr(97 + row * 2 + col)}) {cohort}: {label}', loc='left', pad=7)
            ax.set_xlabel(r'$\Delta$ AURC ($\times 10^{-3}$)' if metric == 'aurc'
                          else r'$\Delta$ risk (percentage points)', labelpad=3)
            ax.text(.02, .13, r'$\leftarrow$ Resize better', transform=ax.transAxes, fontsize=8.0,
                    ha='left', va='center')
            if metric == 'matched':
                ax.text(.98, .89, f'{value["matched_error_discordant_groups"]} discordant group' +
                        ('' if value['matched_error_discordant_groups'] == 1 else 's'),
                        transform=ax.transAxes, fontsize=8.0, ha='right', va='center', color='#444444')
            else:
                ax.text(.98, .89, f'N = {value["n"]:,}; G = {value["groups"]:,}', transform=ax.transAxes,
                        fontsize=8.0, ha='right', va='center', color='#444444')
    fig.subplots_adjust(left=.06, right=.985, top=.91, bottom=.14, wspace=.25, hspace=.95)
    save(fig, output, 'independent_ablation_contrasts')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=HERE / 'figures')
    parser.add_argument('--from-aggregate', action='store_true',
                        help='Render only the SHA-verified packaged figure_data.json; no historical paths are opened.')
    parser.add_argument('--aggregate', type=Path, default=HERE / 'figures/figure_data.json',
                        help='Packaged figure data, next to its original source_receipt.json.')
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(HERE) or output == HERE:
        raise ValueError('Output must be a subdirectory of the new manuscript directory.')
    output.mkdir(parents=True, exist_ok=True)
    sources = Sources()
    sources.record(Path(__file__))
    aggregate_mode = args.from_aggregate or not all((p / 'complete.json').is_file()
                                                    for p in (REFERENCE, PUBLIC, ABLATION))
    if aggregate_mode:
        data = load_aggregate(sources, args.aggregate)
    else:
        data = {'version': 'article-manuscript-figures-v1',
                'efficientnet_retrospective': load_reference(sources),
                'public_detectors': load_public(sources),
                'independent_ablation': load_ablation(sources),
                'figure_specification': {'width_inches': 7.1, 'font_points': [8.0, 8.2, 8.5],
                                         'pdf_vector': True, 'png_dpi': 300,
                                         'confidence_intervals': 'Original stored 95% intervals only.',
                                         'risk_curve_resampling': 'None; every exact-score block is included.'}}
    style()
    risk_coverage(data['efficientnet_retrospective'], output)
    public_aurc(data['public_detectors'], output)
    ablation_forest(data['independent_ablation'], output)
    sources.recheck()
    if aggregate_mode:
        # Preserve archived bytes, including full-precision values and metadata.
        if (output / 'figure_data.json').resolve() != args.aggregate.resolve():
            (output / 'figure_data.json').write_bytes(args.aggregate.read_bytes())
    else:
        (output / 'figure_data.json').write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    owned_names = {stem + suffix for stem in ('efficientnet_risk_coverage',
                   'public_detectors_aurc', 'independent_ablation_contrasts')
                   for suffix in ('.pdf', '.png')} | {'figure_data.json'}
    outputs = {p.name: {'sha256': sha(p), 'bytes': p.stat().st_size}
               for p in output.iterdir() if p.name in owned_names}
    receipt = {'version': 'article-manuscript-figure-receipt-v1',
               'mode': 'packaged_aggregate' if aggregate_mode else 'historical_frozen_scores',
               'script_sha256': sha(Path(__file__)), 'sources': sources.items, 'outputs': outputs,
               'environment': {'python': platform.python_version(), 'numpy': np.__version__,
                               'matplotlib': matplotlib.__version__},
               'verification': {'completed_results_hashes': not aggregate_mode,
                                'frozen_inputs_hashes': not aggregate_mode,
                                'packaged_aggregate_sha256_matches_archived_receipt': aggregate_mode,
                                'reference_all_displayed_block_aurcs_reproduced': True,
                                'reference_fixed_coverage_metrics_reproduced': not aggregate_mode,
                                'public_evidence_matches_authoritative_results': not aggregate_mode,
                                'new_bootstrap': False, 'image_decoding': False, 'inference': False}}
    receipt_name = 'aggregate_render_receipt.json' if aggregate_mode else 'source_receipt.json'
    (output / receipt_name).write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'output': output.as_posix(), 'figures': 3, 'source_files': len(sources.items),
                      'mode': receipt['mode'], 'all_metric_checks_passed': True}, ensure_ascii=False))


if __name__ == '__main__':
    main()
