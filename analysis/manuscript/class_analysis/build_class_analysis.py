"""Reproduce descriptive class-conditional results from stored detector scores.

Default writes only this sidecar's aggregate files and its two manuscript tables.
--check performs an exact read-only regeneration check. --extract-from is an
optional one-time exporter from the original ImageDetection archive; it never
loads images, checkpoints, experiment modules, or fitting/inference software.
Requires Python >=3.11 and NumPy. All supplied per-image inputs are anonymized.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import platform
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
MANUSCRIPT = HERE.parent
COHORTS = ('DeepShield', 'OpenFake')
DETECTORS = ('efficientnet', 'clip', 'resnet50')
DISPLAY = {'efficientnet': 'EfficientNet', 'clip': 'CLIP', 'resnet50': 'ResNet-50'}
SHORT = {'DeepShield': 'DS', 'OpenFake': 'OF'}
MATCHED_K = {'DeepShield': 9807, 'OpenFake': 3639}
FIELDS = ('image_key', 'group_key', 'tie_key', 'label', 'prediction', 'full_score', 'confidence_score')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def encoded(value):
    return json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n'


def opaque(namespace, value):
    return hashlib.sha256((namespace + str(value)).encode('utf-8')).hexdigest()


def export(archive):
    """Copy only evaluation labels, predictions, scores and opaque ordering keys."""
    archive = Path(archive).resolve()
    if (HERE / 'inputs/metadata.json').exists():
        raise FileExistsError('Inputs already exported; use default or --check.')
    source_files = {}
    known = {}
    prior = read(MANUSCRIPT / 'retrospective_ci_20261030/protocol.json')['source_sha256']
    for path, digest in prior.items():
        try:
            relative = Path(path).relative_to(archive).as_posix()
        except ValueError:
            continue
        known[relative] = digest
    ap = read(MANUSCRIPT / 'evidence/adaptation_sources.json')
    adaptation = Path('article_detector_adaptation/runs/matched_v4_20261030')
    for relative, info in ap['source_files'].items():
        known[(adaptation / relative).as_posix()] = info['sha256']

    def source(relative):
        relative = Path(relative).as_posix()
        path = archive / relative
        digest = sha(path)
        if relative in known:
            assert digest == known[relative], ('Changed archived input', relative)
        source_files[relative] = {'sha256': digest, 'bytes': path.stat().st_size,
                                  'matched_previous_manuscript_digest': relative in known}
        return path

    base = Path('selective_experiments/selector_comparison_bahat')
    rows = read(source(base / 'data/external.json'))
    with np.load(source(base / 'data/external.npz'), allow_pickle=False) as f:
        original = {k: f[k].copy() for k in f.files}
    with np.load(source(base / 'reports/full/scores.npz'), allow_pickle=False) as f:
        scored = {k: f[k].copy() for k in f.files}
    original_result = read(source(base / 'reports/full/results.json'))
    source(base / 'code/metrics.py')
    source(adaptation / 'snapshot/helpers/evaluation.py')
    source(adaptation / 'snapshot/code/selector.py')
    assert len(rows) == len(original['labels']) == 14000
    assert len({r['sample_id'] for r in rows}) == len(rows)
    assert np.array_equal(original['frozen_scores'], scored['frozen_v4'])
    assert np.array_equal(np.array([r['label'] for r in rows]), original['labels'])
    reference_labels = {r['sample_id']: r['label'] for r in rows}
    reference_groups = {r['sample_id']: r['group'] for r in rows}
    entries = []
    input_dir = HERE / 'inputs'
    input_dir.mkdir(parents=True, exist_ok=True)
    for detector in DETECTORS:
        for cohort in COHORTS:
            if detector == 'efficientnet':
                mask = np.array([r['cohort'] == cohort for r in rows])
                ids = np.array([r['sample_id'] for r in rows])[mask]
                groups = np.array([r['group'] for r in rows])[mask]
                labels, errors = original['labels'][mask], original['y'][mask]
                predictions = (original['margins'][mask, 0] >= 0).astype(int)
                full, confidence = scored['frozen_v4'][mask], scored['confidence'][mask]
                result = original_result['cohorts'][cohort]
                methods = result['methods']
                tau = methods['frozen_v4']['fixed_threshold']['threshold']
                namespace = 'comparison-tie-v1/'
                matched = {name: methods[key]['matched_frozen_coverage']
                           for name, key in [('full', 'frozen_v4'), ('confidence', 'confidence')]}
                frozen = {'full': methods['frozen_v4']['fixed_threshold'], 'confidence': matched['confidence']}
                expected_errors = result['detector_errors']
                assert np.array_equal(original['y'][mask], predictions != labels)
            else:
                folder = adaptation / detector / 'evaluation' / cohort
                with np.load(source(folder / 'scores.npz'), allow_pickle=False) as f:
                    stored = {k: f[k].copy() for k in f.files}
                result = read(source(folder / 'results.json'))
                assessment = read(source(adaptation / detector / 'selector/assessment.json'))
                ids, groups = stored['sample_ids'], stored['groups']
                labels, errors = stored['labels'], stored['errors']
                predictions = (stored['margins'][:, 0] >= 0).astype(int)
                names = list(stored['method_names'])
                full = stored['scores'][:, names.index('lr_full')]
                confidence = stored['scores'][:, names.index('confidence')]
                point = result['frozen_working_point']
                tau = point['threshold']
                assert tau == assessment['research_threshold'] == assessment['deployment_threshold']
                assert assessment['passed']
                namespace = result['bootstrap']['tie_namespace']
                risk_names = list(result['coverage_fractions'])
                original_name = next(name for name in risk_names if result['coverage_fractions'][name]
                                     == [MATCHED_K[cohort], len(ids)])
                matched = {name: result['methods'][key]['risk_at_coverage'][original_name]
                           for name, key in [('full', 'lr_full'), ('confidence', 'confidence')]}
                frozen = {'full': point, 'confidence': point['matched_references']['confidence']}
                expected_errors = result['detector']['adapted']['errors']
            assert len(ids) == (10000 if cohort == 'DeepShield' else 4000)
            assert np.array_equal(errors, predictions != labels)
            assert int(errors.sum()) == expected_errors
            assert set(ids) == {r['sample_id'] for r in rows if r['cohort'] == cohort}
            assert all(reference_labels[str(i)] == int(y) for i, y in zip(ids, labels))
            assert all(reference_groups[str(i)] == str(g) for i, g in zip(ids, groups))
            assert np.isfinite(full).all() and np.isfinite(confidence).all()
            filename = detector + '_' + cohort + '.csv.gz'
            stream = io.StringIO(newline='')
            writer = csv.writer(stream, lineterminator='\n')
            writer.writerow(FIELDS)
            for i, g, y, h, sf, sc in zip(ids, groups, labels, predictions, full, confidence):
                writer.writerow((opaque('article-class-image-v1/', i),
                                 opaque('article-class-group-v1/', g), opaque(namespace, i),
                                 int(y), int(h), repr(float(sf)), repr(float(sc))))
            (input_dir / filename).write_bytes(gzip.compress(stream.getvalue().encode('utf-8'), mtime=0))
            entries.append({'detector': detector, 'cohort': cohort, 'file': filename,
                'sha256': sha(input_dir / filename), 'images': len(ids),
                'groups': len(set(groups)), 'frozen_threshold': tau,
                'tie_namespace': namespace, 'original_k': MATCHED_K[cohort],
                'expected': {'unconditional_errors': int(expected_errors),
                    'original_matched': {m: {k: int(v[k]) for k in ('accepted', 'errors')}
                                         for m, v in matched.items()},
                    'frozen_threshold': {m: {k: int(v[k]) for k in ('accepted', 'errors')}
                                         for m, v in frozen.items()}}})
    provenance = {'version': 'class-conditional-source-provenance-v1',
        'scope': 'Read-only export of existing predictions; no images, fitting, inference or bootstrap.',
        'source_root_role': 'Original ImageDetection experiment archive (not needed to reproduce supplied analysis).',
        'source_files': source_files, 'python': platform.python_version(), 'numpy': np.__version__}
    metadata = {'version': 'class-conditional-inputs-v1',
        'retrospective': True, 'new_independent_test': False,
        'label_convention': '0 real, 1 synthetic; predictions retained from original frozen detector.',
        'anonymization': 'Image and group identifiers are namespace-prefixed SHA-256 digests; '
            'tie_key is the exact original namespace-prefixed digest, preserving original tie order. '
            'No image pixels, original paths, original IDs, or model weights are distributed.',
        'entries': entries}
    (input_dir / 'metadata.json').write_text(encoded(metadata), encoding='utf-8')
    (HERE / 'source_provenance.json').write_text(encoded(provenance), encoding='utf-8')
    for relative, info in source_files.items():
        assert sha(archive / relative) == info['sha256'], ('Source changed during read-only export', relative)
    print('Exported six anonymized stored-score inputs; verified', len(source_files), 'source hashes.')


def load_input(entry):
    path = HERE / 'inputs' / entry['file']
    assert sha(path) == entry['sha256'], ('Changed input', path.name)
    with gzip.open(path, 'rt', encoding='utf-8', newline='') as stream:
        reader = csv.DictReader(stream)
        assert tuple(reader.fieldnames) == FIELDS
        rows = list(reader)
    values = {key: np.array([r[key] for r in rows]) for key in FIELDS[:3]}
    for key in ('label', 'prediction'):
        values[key] = np.array([int(r[key]) for r in rows], dtype=np.int64)
        assert np.isin(values[key], (0, 1)).all()
    for key in ('full_score', 'confidence_score'):
        values[key] = np.array([float(r[key]) for r in rows], dtype=np.float64)
        assert np.isfinite(values[key]).all()
    assert len(rows) == entry['images']
    assert len(set(values['image_key'])) == len(rows)
    assert len(set(values['group_key'])) == entry['groups']
    assert len(set(values['tie_key'])) == len(rows)
    return values


def summarize(v, accepted):
    y, h = v['label'], v['prediction']
    e = y != h
    classes = {}
    for c, name in ((0, 'real'), (1, 'synthetic')):
        mask = y == c
        n, n_a, k = int(mask.sum()), int((mask & accepted).sum()), int((mask & accepted & e).sum())
        classes[name] = {'total': n, 'accepted': n_a, 'rejected': n - n_a,
                         'coverage': n_a / n, 'errors': k,
                         'accepted_risk': k / n_a if n_a else None}
    n_a, k = int(accepted.sum()), int((accepted & e).sum())
    assert sum(c['accepted'] for c in classes.values()) == n_a
    assert sum(c['errors'] for c in classes.values()) == k
    return {'total': len(y), 'accepted': n_a, 'rejected': len(y) - n_a,
            'coverage': n_a / len(y), 'errors': k, 'risk': k / n_a if n_a else None,
            'accepted_fp': classes['real']['errors'], 'accepted_fn': classes['synthetic']['errors'],
            'accepted_tn': classes['real']['accepted'] - classes['real']['errors'],
            'accepted_tp': classes['synthetic']['accepted'] - classes['synthetic']['errors'],
            'classes': classes}


def analyze():
    meta = read(HERE / 'inputs/metadata.json')
    cohorts, checks = [], []
    for entry in meta['entries']:
        v = load_input(entry)
        n = entry['images']
        base = summarize(v, np.ones(n, dtype=bool))
        assert base['errors'] == entry['expected']['unconditional_errors']
        assert base['classes']['real']['total'] == base['classes']['synthetic']['total'] == n // 2
        frozen = v['full_score'] <= entry['frozen_threshold']
        orders = {method: np.lexsort((v['tie_key'], v[method + '_score']))
                  for method in ('full', 'confidence')}
        policies = {}
        for policy, k in (('original_matched', entry['original_k']), ('frozen_threshold', int(frozen.sum()))):
            methods = {}
            for method in ('full', 'confidence'):
                chosen = np.zeros(n, dtype=bool)
                chosen[orders[method][:k]] = True
                if policy == 'frozen_threshold' and method == 'full':
                    assert np.array_equal(chosen, frozen)
                    chosen = frozen
                result = summarize(v, chosen)
                expected = entry['expected'][policy][method]
                for key in ('accepted', 'errors'):
                    assert result[key] == expected[key], (entry['detector'], entry['cohort'], policy, method, key)
                    checks.append([entry['detector'], entry['cohort'], policy, method, key, result[key]])
                methods[method] = result
            assert methods['full']['accepted'] == methods['confidence']['accepted']
            methods['full_minus_confidence'] = {
                'accepted_fp': methods['full']['accepted_fp'] - methods['confidence']['accepted_fp'],
                'accepted_fn': methods['full']['accepted_fn'] - methods['confidence']['accepted_fn'],
                'accepted_errors': methods['full']['errors'] - methods['confidence']['errors'],
                'real_coverage_pp': 100 * (methods['full']['classes']['real']['coverage'] - methods['confidence']['classes']['real']['coverage']),
                'synthetic_coverage_pp': 100 * (methods['full']['classes']['synthetic']['coverage'] - methods['confidence']['classes']['synthetic']['coverage'])}
            policies[policy] = methods
        cohorts.append({'detector': entry['detector'], 'cohort': entry['cohort'], 'groups': entry['groups'],
                        'frozen_threshold': entry['frozen_threshold'], 'unconditional': base, 'policies': policies})
    return {'version': 'class-conditional-results-v1', 'scope': 'Descriptive retrospective class-conditional analysis on reused cohorts; no new intervals or model selection.',
        'definitions': {'coverage_by_true_class': 'accepted count of true class c / all inputs of true class c',
            'accepted_class_risk': 'errors among accepted inputs of true class c / accepted count of true class c',
            'FP': 'accepted real image classified as synthetic', 'FN': 'accepted synthetic image classified as real',
            'original_matched': 'fixed global count 9807/10000 DS or 3639/4000 OF for both methods',
            'frozen_threshold': 'full profile uses its archived threshold; confidence is ranked to exactly the same count',
            'class_thresholds': 'None: no per-class threshold, label-dependent acceptance, or refitting is introduced.'},
        'cohorts': cohorts, 'aggregate_reproduction_checks': checks}


def as_csv(results):
    output = io.StringIO(newline='')
    names = ['detector','cohort','policy','method','total','accepted','rejected','coverage','errors','risk',
             'accepted_fp','accepted_fn','accepted_tn','accepted_tp','real_total','real_accepted','real_coverage',
             'real_accepted_risk','synthetic_total','synthetic_accepted','synthetic_coverage','synthetic_accepted_risk']
    writer = csv.DictWriter(output, fieldnames=names, lineterminator='\n')
    writer.writeheader()
    for c in results['cohorts']:
        items = [('unconditional','detector',c['unconditional'])]
        items += [(p,m,c['policies'][p][m]) for p in ('original_matched','frozen_threshold') for m in ('confidence','full')]
        for policy,method,v in items:
            row = {k: v[k] for k in names if k in v}
            row.update(detector=c['detector'],cohort=c['cohort'],policy=policy,method=method)
            for name in ('real','synthetic'):
                row.update({name+'_'+key: v['classes'][name][key] for key in ('total','accepted','coverage','accepted_risk')})
            writer.writerow(row)
    return output.getvalue()


def number(x):
    return f'{x:,}'


def main_table(results):
    lines = [r'\begin{table*}[t]',
      r'\caption{Class-conditional outcomes at the original matched global acceptance counts: $k=9{,}807$ of $10{,}000$ on DS and $k=3{,}639$ of $4{,}000$ on OF. $C_0,C_1$ are percentages of all real/synthetic inputs accepted, respectively (5,000 per class on DS; 2,000 on OF). FP counts accepted real images classified as synthetic; FN counts accepted synthetic images classified as real. Rows use the same detector decisions and global $k$; class membership is used only for retrospective reporting.}',
      r'\label{tab:class_conditional}',r'\centering\small',r'\begin{tabular}{lllrrrr}',r'\toprule',
      r'Detector & Cohort & Score & $C_0$ (\%) & $C_1$ (\%) & FP & FN \\',r'\midrule']
    for c in results['cohorts']:
        for method in ('confidence','full'):
            v = c['policies']['original_matched'][method]
            lines.append(f"{DISPLAY[c['detector']]} & {SHORT[c['cohort']]} & {'Confidence' if method=='confidence' else 'Full profile'} & "
                f"{100*v['classes']['real']['coverage']:.2f} & {100*v['classes']['synthetic']['coverage']:.2f} & "
                f"{number(v['accepted_fp'])} & {number(v['accepted_fn'])} "+r'\\')
        if c != results['cohorts'][-1]: lines.append(r'\addlinespace[2pt]')
    return '\n'.join(lines + [r'\bottomrule',r'\end{tabular}',r'\end{table*}',''])


def detail_tables(results):
    lines = [r'\begin{table*}[t]',
        r'\caption{Unconditional detector outcomes on the same evaluation inputs. Both classes contain 5,000 images on DS and 2,000 on OF. Here FP and FN include all predictions, before abstention; Acc.$_0$ and Acc.$_1$ are true-class accuracies.}',
        r'\label{tab:class_unconditional}',r'\centering\small',r'\begin{tabular}{llrrrr}',r'\toprule',
        r'Detector & Cohort & FP & FN & Acc.$_0$ (\%) & Acc.$_1$ (\%) \\',r'\midrule']
    for c in results['cohorts']:
        v = c['unconditional']
        lines.append(f"{DISPLAY[c['detector']]} & {SHORT[c['cohort']]} & {number(v['accepted_fp'])} & {number(v['accepted_fn'])} & "
            f"{100*(1-v['classes']['real']['accepted_risk']):.2f} & {100*(1-v['classes']['synthetic']['accepted_risk']):.2f} "+r'\\')
    lines += [r'\bottomrule',r'\end{tabular}',r'\end{table*}','',r'\begin{table*}[t]',
        r"\caption{Class-conditional outcomes at each full selector's archived frozen threshold. Confidence is ranked to the same total accepted count $k$, rather than assigned its own threshold. $k_0,k_1$ are accepted true-class counts, $C_0,C_1$ their class coverages, and $R$ the overall accepted risk. FP and FN have the definitions in main-article Table~\ref{main-tab:class_conditional}.}",
        r'\label{tab:class_frozen}',r'\centering\small',r'\setlength{\tabcolsep}{4pt}',
        r'\begin{tabular}{lllrrrrrrr}',r'\toprule',
        r'Detector & Cohort & Score & $k_0$ & $k_1$ & $C_0$ (\%) & $C_1$ (\%) & FP & FN & $R$ (\%) \\',r'\midrule']
    for c in results['cohorts']:
        for method in ('confidence','full'):
            v = c['policies']['frozen_threshold'][method]
            lines.append(f"{DISPLAY[c['detector']]} & {SHORT[c['cohort']]} & {'Confidence' if method=='confidence' else 'Full profile'} & "
                f"{number(v['classes']['real']['accepted'])} & {number(v['classes']['synthetic']['accepted'])} & "
                f"{100*v['classes']['real']['coverage']:.2f} & {100*v['classes']['synthetic']['coverage']:.2f} & "
                f"{number(v['accepted_fp'])} & {number(v['accepted_fn'])} & {100*v['risk']:.3f} "+r'\\')
        if c != results['cohorts'][-1]: lines.append(r'\addlinespace[2pt]')
    lines += [r'\bottomrule',r'\end{tabular}',r'\end{table*}','']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--extract-from', type=Path, help='Optional original ImageDetection root; reads stored scores only.')
    parser.add_argument('--check', action='store_true', help='Verify inputs, totals and exact regenerated outputs without writes.')
    args = parser.parse_args()
    if args.extract_from:
        if args.check: parser.error('--extract-from and --check are mutually exclusive')
        export(args.extract_from)
    results = analyze()
    products = {HERE/'results.json':encoded(results), HERE/'results.csv':as_csv(results),
                MANUSCRIPT/'tables/class_conditional.tex':main_table(results),
                MANUSCRIPT/'tables/class_conditional_details.tex':detail_tables(results)}
    for path, text in products.items():
        if args.check:
            assert path.read_text(encoding='utf-8') == text, ('Generated output differs', path)
        else:
            path.parent.mkdir(parents=True,exist_ok=True)
            path.write_text(text, encoding='utf-8', newline='\n')
    print(('Checked' if args.check else 'Generated'),len(products),'outputs;',
          len(results['aggregate_reproduction_checks']),'exact published accepted/error checks; six cohort/detector inputs.')


if __name__ == '__main__':
    main()
