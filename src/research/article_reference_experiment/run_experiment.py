"""Príprava a explicitné spustenie oddeleného porovnania MC dropout + energy."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
import uuid

sys.dont_write_bytecode = True
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from article_reference_experiment.common import ROOT, PROJECT, canonical, digest, sha, read, write, now, environment, output_path, lock

DEFAULT_SOURCE = PROJECT / 'selective_experiments/selector_comparison_bahat'
VERSION = 'article-reference-comparison-v1'
SAMPLES = 30
SEED = 20261028
BASELINES = ('confidence', 'tta_mean', 'frozen_v4', 'lr_confidence',
             'mlp_bahat_logits_s17', 'mlp_bahat_logits_s29', 'mlp_bahat_logits_s43')


def sample_seed(sample_id):
    return int(hashlib.sha256(f'article-reference-mc-v1/{SEED}/{sample_id}'.encode()).hexdigest()[:8], 16)


def plans_at(run):
    with (Path(run) / 'plan.jsonl').open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream]


def verify(run):
    run = output_path(run)
    protocol = read(run / 'protocol.json')
    if sha(run / 'protocol.json') != (run / 'protocol.sha256').read_text(encoding='ascii').strip():
        raise ValueError('Zmenený protokol nového experimentu.')
    if protocol['version'] != VERSION or protocol['environment'] != environment():
        raise ValueError('Zmenená verzia alebo numerické prostredie; použite deklarovaný Python.')
    for relative, expected in protocol['files'].items():
        path = (run / relative).resolve()
        if not path.is_relative_to(run) or sha(path) != expected:
            raise ValueError('Zmenený alebo neplatný vstup: ' + relative)
    for name, expected in protocol['code'].items():
        if Path(name).name != name or sha(ROOT / name) != expected:
            raise ValueError('Implementácia sa zmenila po príprave: ' + name)
    for path, expected in protocol['source_anchors'].items():
        if sha(path) != expected:
            raise ValueError('Zmenený pôvodný zdroj: ' + path)
    return protocol


def prepare(run, mode, source=DEFAULT_SOURCE):
    import numpy as np
    from article_reference_experiment.source import inspect_source, read_split
    run, source = output_path(run), Path(source).resolve()
    if run.exists():
        protocol = verify(run)
        if protocol['mode'] != mode or protocol['comparison_run'] != str(source):
            raise ValueError('Existujúci beh má iný režim alebo zdroj.')
        return protocol
    source_protocol, anchors = inspect_source(source)
    split = 'external' if mode == 'retrospective' else 'selection'
    inputs = read_split(source, split)
    all_plans = inputs['plans']
    indices = list(range(len(all_plans)))
    if mode == 'pilot':
        indices = sorted(sorted(indices, key=lambda i: digest(['reference-pilot-v1', SEED, all_plans[i]['sample_id']]))[:64])
    plans = [{**all_plans[i], 'source_index': i} for i in indices]
    if not plans or len({p['sample_id'] for p in plans}) != len(plans):
        raise ValueError('Prázdny alebo nejedinečný plán.')
    if set(inputs['baseline_scores']) != set(BASELINES):
        raise ValueError('Chýba niektorá zmrazená referencia.')
    temporary = run.with_name('.' + run.name + '.preparing-' + uuid.uuid4().hex)
    (temporary / 'snapshot/code').mkdir(parents=True)
    with (temporary / 'plan.jsonl').open('xb') as stream:
        for plan in plans:
            stream.write(canonical(plan) + b'\n')
    arrays = {name: np.asarray(inputs[name])[indices] for name in ('logits', 'raw_margins', 'errors')}
    arrays.update({'score__' + name: np.asarray(inputs['baseline_scores'][name])[indices] for name in BASELINES})
    np.savez_compressed(temporary / 'inputs.npz', **arrays)
    write(temporary / 'source_protocol.json', source_protocol)
    code = {path.name: sha(path) for path in sorted(ROOT.glob('*.py'))}
    for name in code:
        shutil.copyfile(ROOT / name, temporary / 'snapshot/code' / name)
    scope = {'pilot': 'Technický pilot na vývojových vstupoch; bez záverov o externom prínose.',
             'development': 'Vývojové porovnanie na pôvodnej výberovej množine.',
             'retrospective': 'Doplnkové retrospektívne porovnanie na už vyhodnotených 14000 obrázkoch; nie nový nezávislý potvrdzujúci test.'}[mode]
    protocol = {'version': VERSION, 'created_utc': now(), 'mode': mode, 'comparison_run': str(source),
                'source_split': split, 'scope': scope, 'environment': environment(), 'source_anchors': anchors,
                'counts': {'profiles': len(plans), 'groups': len({p['group'] for p in plans}),
                           'cohorts': dict(Counter(p.get('cohort', 'development') for p in plans)),
                           'input_specs': dict(Counter(p['input_spec']['id'] for p in plans))},
                'model_sha256': source_protocol['model_sha256'], 'old_temperature': source_protocol['temperature'],
                'methods': {'existing': list(BASELINES), 'new': ['mc_entropy_30', 'mc_entropy_6', 'energy_t1'],
                            'mc_samples': SAMPLES, 'mc_temperature': 1., 'energy_temperature': 1.,
                            'mc_primary': 'Predictive entropy of mean of 30 softmax probabilities; native trained head dropout only.',
                            'mc_secondary': 'Same calculation using the first 6 draws; not an independent repetition.',
                            'energy': 'Negative logsumexp of the original two native deterministic logits, T=1; no centering.',
                            'bahat': 'All three previously frozen seeds are reported separately; no winner chosen on test.'},
                'statistics': {'bootstrap': 200 if mode == 'pilot' else 5000, 'seed': SEED,
                               'primary_descriptive': ['mc_entropy_30', 'energy_t1'], 'coverages': [.8, .9, .95],
                               'unit': 'Whole original/sampling groups; paired resampling across methods.',
                               'intervals': 'Pointwise 95%, descriptive; fixed models and fixed realization of MC masks.'},
                'policy': {'classifier': 'Original deterministic prediction remains fixed for every method.',
                           'training': 'No training, temperature fitting or test-label threshold tuning.',
                           'preprocessing': 'Preserve each original input_spec, then native EfficientNet preprocessing.',
                           'rng': 'Per-sample SHA256 seed; restored caller RNG state; same first six draws.',
                           'cost': 'One deterministic backbone feature computation, deterministic head and 30 stochastic head calls; report actual times. Not 30 full backbones.',
                           'unsupported': {'clip': 'No native trained dropout; one-logit energy is not uniquely defined.',
                                           'resnet50': 'No native trained dropout; one-logit energy is not uniquely defined.'},
                           'doctor': 'Unperturbed binary DOCTOR is rank-equivalent to confidence; not counted as another independent baseline.',
                           'conformal': 'Related-work context only; no conformal experiment or risk guarantee.'},
                'code': code}
    protocol['files'] = {p.relative_to(temporary).as_posix(): sha(p) for p in temporary.rglob('*') if p.is_file()}
    for path, expected in anchors.items():
        if sha(path) != expected:
            raise ValueError('Zdroj sa zmenil počas prípravy.')
    write(temporary / 'protocol.json', protocol)
    (temporary / 'protocol.sha256').write_text(sha(temporary / 'protocol.json') + '\n', encoding='ascii')
    os.rename(temporary, run)
    print(f'Pripravené: {mode}; {len(plans)} vstupov; {run}', flush=True)
    return protocol


def input_arrays(run):
    import numpy as np
    with np.load(Path(run) / 'inputs.npz', allow_pickle=False) as bundle:
        return {name: bundle[name] for name in bundle.files}


def scan(run, plans, arrays, fingerprint, repair=False):
    import numpy as np
    path = Path(run) / 'mc_profiles.jsonl'
    chain, rows = digest(['reference-mc-journal-v1', fingerprint]), []
    if not path.exists():
        return rows, chain
    with path.open('r+b' if repair else 'rb') as stream:
        while True:
            position = stream.tell()
            line = stream.readline()
            if not line:
                break
            if not line.endswith(b'\n'):
                if repair:
                    stream.truncate(position)
                    break
                raise ValueError('Nedopísaný koniec MC žurnálu.')
            row = json.loads(line)
            if len(rows) >= len(plans):
                raise ValueError('Nadbytočný MC profil.')
            plan = plans[len(rows)]
            body = {k: v for k, v in row.items() if k != 'chain'}
            expected = digest([chain, body])
            if row.get('chain') != expected or row['fingerprint'] != fingerprint or row['plan_sha256'] != digest(plan):
                raise ValueError('Poškodený alebo zmiešaný MC žurnál.')
            det, mc = np.asarray(row['deterministic_logits']), np.asarray(row['mc_logits'])
            cached = arrays['logits'][len(rows)]
            if (det.shape != (2,) or mc.shape != (SAMPLES, 2) or not np.isfinite(det).all() or not np.isfinite(mc).all()
                    or not np.allclose(det, cached, rtol=1e-6, atol=2e-5)
                    or bool(det[1] >= det[0]) != bool(cached[1] >= cached[0])
                    or row['seed'] != sample_seed(plan['sample_id']) or row['samples'] != SAMPLES
                    or row['image_sha256'] != plan['sha256']):
                raise ValueError('Neplatná alebo nereprodukovateľná MC inferencia.')
            chain = expected
            rows.append(row)
    return rows, chain


def load_profiles(run, require_complete=True):
    plans, arrays = plans_at(run), input_arrays(run)
    metadata = read(Path(run) / 'runtime.json')
    if metadata['protocol_sha256'] != sha(Path(run) / 'protocol.json'):
        raise ValueError('Runtime patrí inému protokolu.')
    receipt_path = Path(run) / 'profiles.complete.json'
    receipt = read(receipt_path) if receipt_path.exists() else None
    if require_complete and receipt is None:
        raise ValueError('Inferencia nie je dokončená.')
    rows, chain = scan(run, plans, arrays, digest(metadata), repair=not receipt and not require_complete)
    if receipt and (receipt['count'] != len(rows) or len(rows) != len(plans) or receipt['chain'] != chain
                    or receipt['fingerprint'] != digest(metadata) or receipt['sha256'] != sha(Path(run) / 'mc_profiles.jsonl')):
        raise ValueError('Neplatné potvrdenie dokončených MC profilov.')
    return rows


def extract(run, protocol, device='cuda', max_new=None):
    from article_reference_experiment.runtime import load_runtime
    run = Path(run)
    if (run / 'profiles.complete.json').exists():
        load_profiles(run)
        return True
    plans, arrays = plans_at(run), input_arrays(run)
    runtime = load_runtime(Path(protocol['comparison_run']), read(run / 'source_protocol.json'), device=device)
    metadata = {'protocol_sha256': sha(run / 'protocol.json'), 'adapter': runtime.metadata,
                'device': device, 'samples': SAMPLES, 'seed': SEED}
    if (run / 'runtime.json').exists():
        if read(run / 'runtime.json') != metadata:
            raise ValueError('Zmenený runtime pri obnovení.')
    else:
        write(run / 'runtime.json', metadata)
    fingerprint = digest(metadata)
    rows, chain = scan(run, plans, arrays, fingerprint, repair=True)
    before, new = time.perf_counter(), 0
    with (run / 'mc_profiles.jsonl').open('ab') as stream:
        for index in range(len(rows), len(plans)):
            if max_new is not None and new >= max_new:
                os.fsync(stream.fileno())
                return False
            plan = plans[index]
            value = runtime.analyze(plan, expected_logits=arrays['logits'][index], seed=sample_seed(plan['sample_id']), samples=SAMPLES)
            body = {**value, 'fingerprint': fingerprint, 'plan_sha256': digest(plan), 'sample_id': plan['sample_id']}
            body['chain'] = digest([chain, body])
            chain = body['chain']
            stream.write(canonical(body) + b'\n')
            stream.flush()
            new += 1
            if new % 16 == 0 or index + 1 == len(plans):
                os.fsync(stream.fileno())
                write(run / 'progress.json', {'stage': 'inference', 'profiles': index + 1, 'required': len(plans), 'updated_utc': now()})
                print(f'MC dropout: {index+1}/{len(plans)}; {(time.perf_counter()-before)/new:.3f} s/nový vstup', flush=True)
        os.fsync(stream.fileno())
    verified, verified_chain = scan(run, plans, arrays, fingerprint)
    if len(verified) != len(plans) or verified_chain != chain:
        raise ValueError('Neúplná kontrola MC žurnálu.')
    write(run / 'profiles.complete.json', {'count': len(plans), 'chain': chain, 'fingerprint': fingerprint,
                                         'sha256': sha(run / 'mc_profiles.jsonl')})
    return True


def analyse(run, protocol):
    import numpy as np
    from article_reference_experiment.scores import new_scores, evaluate
    run = Path(run)
    verify(run)
    rows = load_profiles(run)
    inputs = {'protocol_sha256': sha(run / 'protocol.json'), 'profiles_sha256': sha(run / 'mc_profiles.jsonl'),
              'runtime_sha256': sha(run / 'runtime.json'), 'profiles_receipt_sha256': sha(run / 'profiles.complete.json')}
    if (run / 'complete.json').exists():
        receipt = read(run / 'complete.json')
        if receipt['inputs'] != inputs:
            raise ValueError('Výsledok patrí iným vstupom.')
        for name, expected in receipt['files'].items():
            if sha(run / name) != expected:
                raise ValueError('Zmenený dokončený výsledok.')
        return read(run / 'results.json')
    plans, arrays = plans_at(run), input_arrays(run)
    additions = new_scores(np.asarray([r['deterministic_logits'] for r in rows]), np.asarray([r['mc_logits'] for r in rows]))
    scores = {name: arrays['score__' + name] for name in BASELINES}
    scores.update({name: np.asarray(values) for name, values in additions['scores'].items()})
    cohorts = {}
    for cohort in sorted({p.get('cohort', 'development') for p in plans}):
        ix = [i for i, p in enumerate(plans) if p.get('cohort', 'development') == cohort]
        cohorts[cohort] = evaluate(arrays['errors'][ix], {name: value[ix] for name, value in scores.items()},
                                  [plans[i]['sample_id'] for i in ix], [plans[i]['group'] for i in ix],
                                  repetitions=protocol['statistics']['bootstrap'], seed=SEED, profile='frozen_v4')
    report = {'version': VERSION, 'created_utc': now(), 'inputs': inputs, 'scope': protocol['scope'],
              'cohorts': cohorts, 'diagnostics': additions['diagnostics'], 'statistics': protocol['statistics'],
              'trained_on_external_labels': False, 'new_independent_confirmation': False,
              'mc_samples': SAMPLES, 'timing_rows': [r.get('timing', {}) for r in rows]}
    write(run / 'results.json', report)
    summary = [protocol['scope'], f"Dokončených {len(rows)} vstupov; 30 natívnych MC head forwardov na vstup.",
               'Energia používa natívne dva logity s T=1. Predikovaná trieda sa pre selektory nemení.',
               'Nejde o nové nezávislé potvrdenie; pôvodné výsledky ani dizertačka sa nemenili.']
    for cohort, result in cohorts.items():
        summary.extend(['', f"Množina: {cohort}; N={result['profiles']}; skupiny={result['groups']}",
                        f"Presnosť pôvodného detektora: {100 * result['detector']['accuracy']:.3f} %",
                        'Metóda | AURC | riziko pri 80 % | pri 90 % | pri 95 %'])
        for name, value in result['methods'].items():
            risks = [value['risk_at_coverage'][str(q)]['risk'] for q in (.8, .9, .95)]
            summary.append(name + ' | ' + f"{value['aurc']:.6f}" + ' | ' +
                           ' | '.join('N/A' if x is None else f'{x:.6f}' for x in risks))
        summary.append('Rozdiel AURC profil − referencia; záporná hodnota zvýhodňuje profil. Bodové opisné 95 % intervaly:')
        for name, value in result['paired_differences'].items():
            ci = value['intervals'].get('aurc')
            interval = 'N/A' if ci is None else f"[{ci['lower_95']:.6f}, {ci['upper_95']:.6f}]"
            summary.append(f"{name}: {value['aurc']:+.6f}; {interval}")
    summary.extend(['', 'Zmeny triedy pri MC priemerovaní (iba diagnostika; nenahrádzajú pôvodné predikcie):'])
    for samples, value in additions['diagnostics']['class_flips'].items():
        summary.append(f"K={samples}: {value['count']}/{len(rows)} ({100 * value['rate']:.3f} %)")
    if rows and all('wall_seconds' in r.get('timing', {}) for r in rows):
        summary.append(f"Súčet nameraných časov vstupov: {sum(r['timing']['wall_seconds'] for r in rows):.3f} s; bez načítania modelu a bootstrapu.")
    (run / 'summary.txt').write_text('\n'.join(summary) + '\n', encoding='utf-8')
    # Konečné potvrdenie až po opätovnom overení vstupov a žurnálu.
    verify(run)
    load_profiles(run)
    write(run / 'complete.json', {'inputs': inputs, 'complete': True, 'new_independent_confirmation': False,
                                 'files': {name: sha(run / name) for name in ('results.json', 'summary.txt')}})
    write(run / 'progress.json', {'stage': 'complete', 'profiles': len(rows), 'required': len(rows), 'updated_utc': now()})
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--mode', choices=('pilot', 'development', 'retrospective'), required=True)
    parser.add_argument('--stage', choices=('prepare', 'verify', 'execute'), required=True)
    parser.add_argument('--source', type=Path, default=DEFAULT_SOURCE)
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    parser.add_argument('--max-new', type=int)
    args = parser.parse_args(argv)
    if args.max_new is not None and (args.max_new < 1 or args.stage != 'execute'):
        parser.error('--max-new musí byť kladný a patrí k execute.')
    run = output_path(args.run)
    with lock(run.parent / ('.' + run.name + '.lock')):
        protocol = prepare(run, args.mode, args.source) if args.stage == 'prepare' else verify(run)
        if protocol['mode'] != args.mode:
            raise ValueError('Režim nezodpovedá uzamknutému protokolu.')
        if args.stage == 'execute':
            try:
                if extract(run, protocol, args.device, args.max_new):
                    analyse(run, protocol)
                    print('Dokončené: ' + str(run / 'results.json'), flush=True)
                else:
                    print('Čiastočný beh uložený. Rovnaký príkaz bezpečne pokračuje.', flush=True)
            except Exception as error:
                write(run / 'progress.json', {'stage': 'failed', 'error': str(error), 'updated_utc': now()})
                raise
        else:
            print(json.dumps({'verified': True, 'mode': protocol['mode'], 'counts': protocol['counts'], 'scope': protocol['scope']}, ensure_ascii=False), flush=True)


# Release entry point: require an explicit research-job request.
if __name__ == '__main__':
    import sys as _release_sys
    if not any(arg in _release_sys.argv for arg in ('--help', '-h', '--execute-research')):
        print('Research source. Read src/RESEARCH_CODE.txt before running a job. '
              'Use --help to inspect options or --execute-research to run explicitly.')
        raise SystemExit(0)
    if '--execute-research' in _release_sys.argv:
        _release_sys.argv.remove('--execute-research')

if __name__ == '__main__':
    main()
