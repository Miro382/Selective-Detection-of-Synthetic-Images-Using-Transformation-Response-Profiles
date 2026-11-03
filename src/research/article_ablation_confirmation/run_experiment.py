"""Vykoná iba nový auditovaný test podľa nemenného predtestového návrhu."""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import time
import uuid

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))
VERSION = 'article-ablation-execution-v1'
DEFAULT_DESIGN = ROOT / 'protocols/strict_20261028'
DEFAULT_DATA = ROOT / 'data/independent_20261029'
DEFAULT_RUN = ROOT / 'runs/independent_20261029'
DEFAULT_SMOKE = ROOT / 'technical_checks/runtime_20261029.json'
SMOKE_INDICES = [0, 100, 1000, 4999, 5000, 9999, 10000, 13999]


def now():
    return datetime.now(timezone.utc).isoformat()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            value.update(block)
    return value.hexdigest()


def read(path):
    def invalid(value):
        raise ValueError('Neplatná JSON hodnota: ' + value)
    return json.loads(Path(path).read_text(encoding='utf-8'), parse_constant=invalid)


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name('.' + path.name + '.' + uuid.uuid4().hex)
    with temporary.open('xb') as stream:
        stream.write(canonical(value) + b'\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def plans_at(run):
    with (Path(run) / 'plan.jsonl').open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream]


def output_path(path):
    path = Path(path).resolve()
    if path.parent != (ROOT / 'runs').resolve() or path.name.startswith('.'):
        raise ValueError('Beh musí byť nový article_ablation_confirmation/runs/<názov>.')
    return path


@contextmanager
def lock(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+b') as stream:
        stream.seek(0, 2)
        if not stream.tell():
            stream.write(b'0')
            stream.flush()
        stream.seek(0)
        if os.name == 'nt':
            import msvcrt
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == 'nt':
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def scorer(models):
    """Iba pôvodná zmrazená definícia príznakov a nemenné koeficienty."""
    models = Path(models)
    manifest = read(models / 'manifest.json')
    for name in ('comparison/code/features.py', 'models/frozen_selector.json', 'models/lr_resize_only.json'):
        if sha(models / name) != manifest['copies'][name]['sha256']:
            raise ValueError('Zmenená zmrazená definícia selektora: ' + name)
    path = models / 'comparison/code/features.py'
    spec = importlib.util.spec_from_file_location('_ablation_verified_features_' + sha(path), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    full = read(models / 'models/frozen_selector.json')['model']['parameters']
    resize = read(models / 'models/lr_resize_only.json')['parameters']
    temperature = manifest['detector']['temperature']
    return module, full, resize, temperature


def score_margins(bundle, margins):
    import numpy as np
    features, full, resize, _ = bundle
    values = np.asarray(margins, dtype=np.float64).reshape(1, 6)
    return (float(features.logistic_score(full, features.profile_features(values))[0]),
            float(features.logistic_score(resize, features.profile_features(values, (0, 4, 5)))[0]))


def smoke(design=DEFAULT_DESIGN, destination=DEFAULT_SMOKE, device='cuda'):
    import numpy as np
    from article_ablation_confirmation import freeze_protocol, inference_runtime
    from article_reference_experiment.source import read_split
    design, destination = Path(design).resolve(), Path(destination).resolve()
    if destination.parent != (ROOT / 'technical_checks').resolve():
        raise ValueError('Technická kontrola musí byť v novom technical_checks.')
    protocol = freeze_protocol.verify(design)
    models = Path(protocol['frozen_models'])
    bindings = {'design_sha256': sha(design / 'protocol.json'), 'model_manifest_sha256': sha(models / 'manifest.json'),
                'runtime_sha256': sha(ROOT / 'inference_runtime.py'), 'device': device}
    if destination.exists():
        result = read(destination)
        if result['bindings'] != bindings or result['passed'] is not True or result['indices'] != SMOKE_INDICES:
            raise ValueError('Existujúca technická kontrola patrí inému kódu; neprepíše sa.')
        return result
    comparison = Path(read(models / 'manifest.json')['comparison_run'])
    data = read_split(comparison, 'external', indices=SMOKE_INDICES)
    with np.load(comparison / 'logits/external.npz', allow_pickle=False) as archive:
        expected = archive['ours_logits'][SMOKE_INDICES].copy()
    runtime = inference_runtime.load_runtime(models, device=device)
    bundle = scorer(models)
    checks = []
    for i, plan in enumerate(data['plans']):
        value = runtime.analyze(plan, expected_logits=expected[i])
        scores = score_margins(bundle, data['raw_margins'][i] / bundle[3])
        if (not np.allclose([value['full_score'], value['resize_score']], scores, rtol=1e-10, atol=1e-12)
                or not np.isclose(value['full_score'], data['baseline_scores']['frozen_v4'][i], rtol=1e-10, atol=1e-12)
                or value['error'] != int(data['errors'][i])):
            raise ValueError('Historická kontrola selektorov alebo chýb neprešla.')
        checks.append({'sample_id': plan['sample_id'], 'max_logit_difference': float(np.max(np.abs(np.asarray(value['native_logits']) - expected[i]))),
                       'max_score_difference': float(np.max(np.abs(np.asarray([value['full_score'], value['resize_score']]) - scores)))})
    result = {'version': VERSION, 'created_utc': now(), 'bindings': bindings, 'indices': SMOKE_INDICES,
              'passed': True, 'historical_only': True, 'new_test_metrics_computed': False,
              'metadata': runtime.metadata, 'checks': checks}
    write(destination, result)
    print('Technická GPU/CPU kontrola prešla na 8 historických obrázkoch.', flush=True)
    return result


def prepare(run=DEFAULT_RUN, data=DEFAULT_DATA, design=DEFAULT_DESIGN, smoke_path=DEFAULT_SMOKE, device='cuda'):
    from article_ablation_confirmation import freeze_protocol, download_data, inference_runtime
    run, data, design, smoke_path = output_path(run), Path(data).resolve(), Path(design).resolve(), Path(smoke_path).resolve()
    if run.exists():
        result = verify(run)
        if (result['data'] != str(data) or result['design'] != str(design) or result['device'] != device
                or result['smoke'] != str(smoke_path)):
            raise ValueError('Existujúci beh patrí inému nastaveniu.')
        return result
    plan_design = freeze_protocol.verify(design)
    verified = download_data.verify_data(data)
    if verified['protocol']['design_sha256'] != sha(design / 'protocol.json'):
        raise ValueError('Obsahový audit patrí inému predtestovému návrhu.')
    plans = verified['manifest']
    counts = dict(Counter(row['dataset'] for row in plans))
    if counts != plan_design['counts'] or len({p['sample_id'] for p in plans}) != len(plans):
        raise ValueError('Neúplný alebo duplicitný obrazový manifest.')
    if any(type(p['label']) is not int or p['label'] not in (0, 1)
           or p['input_spec'] != {'version': 1, 'id': 'clean', 'operations': []} for p in plans):
        raise ValueError('Nesprávne triedy alebo zmenená primárna čistá množina.')
    models = Path(plan_design['frozen_models'])
    check = read(smoke_path)
    if (check['passed'] is not True or check['indices'] != SMOKE_INDICES or check['bindings'] !=
            {'design_sha256': sha(design / 'protocol.json'), 'model_manifest_sha256': sha(models / 'manifest.json'),
             'runtime_sha256': sha(ROOT / 'inference_runtime.py'), 'device': device}):
        raise ValueError('Chýba platná historická kontrola presného runtime.')
    runtime = inference_runtime.load_runtime(models, device=device)
    if runtime.metadata != check['metadata']:
        raise ValueError('Runtime sa od technickej kontroly zmenil.')
    source_paths = [design / 'protocol.json', models / 'manifest.json', smoke_path]
    source_paths += [data / name for name in ('protocol.json', 'protocol.sha256', 'audit.json', 'manifest.jsonl', 'complete.json')]
    anchors = {str(path): sha(path) for path in source_paths}
    code = {path.name: sha(path) for path in sorted(ROOT.glob('*.py'))}
    if not {'download_data.py', 'content_audit.py', 'inference_runtime.py', 'statistics.py', 'report_results.py', 'run_experiment.py'} <= set(code):
        raise ValueError('Chýba dokončený vykonávací kód.')
    run.parent.mkdir(parents=True, exist_ok=True)
    stage = run.with_name('.' + run.name + '.preparing-' + uuid.uuid4().hex)
    (stage / 'code').mkdir(parents=True)
    with (stage / 'plan.jsonl').open('xb') as stream:
        for row in plans:
            stream.write(canonical(row) + b'\n')
    for name in code:
        shutil.copyfile(ROOT / name, stage / 'code' / name)
    protocol = {'version': VERSION, 'created_utc': now(), 'ready_for_inference': True,
                'design': str(design), 'design_sha256': sha(design / 'protocol.json'),
                'frozen_models': str(models), 'model_manifest_sha256': sha(models / 'manifest.json'),
                'data': str(data), 'data_protocol_sha256': sha(data / 'protocol.json'), 'smoke': str(smoke_path),
                'counts': counts, 'groups': {d: len({p['group'] for p in plans if p['dataset'] == d}) for d in counts},
                'environment': plan_design['environment'], 'device': device, 'runtime_metadata': runtime.metadata,
                'source_anchors': anchors, 'code': code, 'new_training': False,
                'adoption_rule': 'Unchanged strict_20261028; no automatic model/thesis modification.'}
    protocol['files'] = {p.relative_to(stage).as_posix(): sha(p) for p in stage.rglob('*') if p.is_file()}
    for name, checksum in anchors.items():
        if sha(name) != checksum:
            raise ValueError('Vstup sa zmenil počas prípravy.')
    write(stage / 'run_protocol.json', protocol)
    (stage / 'run_protocol.sha256').write_text(sha(stage / 'run_protocol.json') + '\n', encoding='ascii')
    write(stage / 'preparation.complete.json', {'run_protocol_sha256': sha(stage / 'run_protocol.json'),
                                              'ready_for_inference': True, 'external_inference_performed': False})
    os.rename(stage, run)
    print('Pripravený uzamknutý vykonávací beh: ' + str(run), flush=True)
    return protocol


def verify(run, deep=True):
    from article_ablation_confirmation import freeze_protocol, freeze_models, download_data
    run = output_path(run)
    protocol = read(run / 'run_protocol.json')
    checksum = sha(run / 'run_protocol.json')
    if ((run / 'run_protocol.sha256').read_text(encoding='ascii').strip() != checksum
            or read(run / 'preparation.complete.json')['run_protocol_sha256'] != checksum
            or protocol['version'] != VERSION or protocol['ready_for_inference'] is not True):
        raise ValueError('Neplatný vykonávací protokol.')
    for name, expected in protocol['files'].items():
        path = (run / name).resolve()
        if not path.is_relative_to(run) or sha(path) != expected:
            raise ValueError('Zmenený uzamknutý vstup behu: ' + name)
    for name, expected in protocol['code'].items():
        if Path(name).name != name or sha(ROOT / name) != expected:
            raise ValueError('Kód sa po uzamknutí zmenil: ' + name)
    for path, expected in protocol['source_anchors'].items():
        if sha(path) != expected:
            raise ValueError('Zmenený zdroj vykonávacieho behu: ' + path)
    if deep:
        design = freeze_protocol.verify(Path(protocol['design']))
        freeze_models.verify(Path(protocol['frozen_models']))
        data = download_data.verify_data(Path(protocol['data']))
        if (protocol['counts'] != design['counts'] or data['manifest'] != plans_at(run)
                or data['protocol']['design_sha256'] != protocol['design_sha256']
                or sha(Path(protocol['design']) / 'protocol.json') != protocol['design_sha256']):
            raise ValueError('Dáta alebo kohorty sa zmenili.')
    return protocol


def scan(run, protocol, metadata, repair=False):
    import numpy as np
    run = Path(run)
    plans = plans_at(run)
    fingerprint = digest(metadata)
    chain, rows = digest(['ablation-profiles-v1', fingerprint]), []
    journal = run / 'profiles.jsonl'
    if not journal.exists():
        return rows, chain
    bundle = scorer(protocol['frozen_models'])
    with journal.open('r+b' if repair else 'rb') as stream:
        while True:
            position, line = stream.tell(), stream.readline()
            if not line:
                break
            if not line.endswith(b'\n'):
                if repair:
                    stream.truncate(position)
                    break
                raise ValueError('Nedopísaný profilový žurnál.')
            value = json.loads(line)
            if len(rows) >= len(plans):
                raise ValueError('Nadbytočný profil.')
            plan = plans[len(rows)]
            body = {key: item for key, item in value.items() if key != 'chain'}
            expected_chain = digest([chain, body])
            if (value.get('chain') != expected_chain or value.get('fingerprint') != fingerprint
                    or value.get('plan_sha256') != digest(plan) or value.get('sample_id') != plan['sample_id']
                    or value.get('image_sha256') != plan['sha256']
                    or value.get('input_spec') != plan['input_spec']
                    or any(value.get(key) != plan[key] for key in ('rgb_sha256', 'phash'))):
                raise ValueError('Poškodený alebo zmiešaný profilový žurnál.')
            logits = np.asarray(value['native_logits'], dtype=np.float64)
            if (logits.shape != (6, 2) or not np.isfinite(logits).all()
                    or not np.array_equal(logits, logits.astype(np.float32).astype(np.float64))):
                raise ValueError('Neplatné natívne logity.')
            raw = np.subtract(logits[:, 1].astype(np.float32), logits[:, 0].astype(np.float32)).astype(np.float64)
            calibrated = raw / bundle[3]
            prediction = int(raw[0] >= 0)
            expected_scores = score_margins(bundle, calibrated)
            if (np.asarray(value['raw_margins']).shape != (6,) or not np.array_equal(value['raw_margins'], raw)
                    or np.asarray(value['calibrated_margins']).shape != (6,)
                    or not np.allclose(value['calibrated_margins'], calibrated, atol=1e-14, rtol=1e-14)
                    or type(value['prediction']) is not int or type(value['error']) is not int
                    or value['prediction'] != prediction or value['error'] != int(prediction != plan['label'])
                    or not np.allclose([value['full_score'], value['resize_score']], expected_scores, atol=1e-12, rtol=1e-10)):
                raise ValueError('Nekonzistentné skóre, chyba alebo kalibrované marginy.')
            rows.append(value)
            chain = expected_chain
    return rows, chain


def load_profiles(run, protocol=None):
    run = Path(run)
    protocol = protocol or verify(run, deep=False)
    metadata = read(run / 'runtime.json')
    if (metadata['run_protocol_sha256'] != sha(run / 'run_protocol.json')
            or metadata.get('adapter') != protocol['runtime_metadata']
            or metadata.get('device') != protocol['device']):
        raise ValueError('Runtime patrí inému behu.')
    receipt = read(run / 'profiles.complete.json')
    rows, chain = scan(run, protocol, metadata)
    if (receipt['count'] != len(rows) or len(rows) != sum(protocol['counts'].values())
            or receipt['sha256'] != sha(run / 'profiles.jsonl') or receipt['chain'] != chain
            or receipt['fingerprint'] != digest(metadata) or receipt['runtime_sha256'] != sha(run / 'runtime.json')):
        raise ValueError('Nesúhlasí potvrdenie úplnej inferencie.')
    return rows


def extract(run, protocol, max_new=None):
    from article_ablation_confirmation import inference_runtime
    run = Path(run)
    if (run / 'profiles.complete.json').exists():
        load_profiles(run, protocol)
        return True
    runtime = inference_runtime.load_runtime(Path(protocol['frozen_models']), device=protocol['device'])
    if runtime.metadata != protocol['runtime_metadata']:
        raise ValueError('Zmenené zariadenie alebo runtime počas pokračovania.')
    metadata = {'run_protocol_sha256': sha(run / 'run_protocol.json'), 'device': protocol['device'], 'adapter': runtime.metadata}
    if (run / 'runtime.json').exists():
        if read(run / 'runtime.json') != metadata:
            raise ValueError('Žurnál vznikol v inom runtime.')
    else:
        write(run / 'runtime.json', metadata)
    plans = plans_at(run)
    existing, chain = scan(run, protocol, metadata, repair=True)
    before, new = time.perf_counter(), 0
    write(run / 'progress.json', {'stage': 'inference', 'completed': len(existing), 'required': len(plans), 'updated_utc': now()})
    with (run / 'profiles.jsonl').open('ab') as stream:
        for index in range(len(existing), len(plans)):
            if max_new is not None and new >= max_new:
                os.fsync(stream.fileno())
                return False
            plan = plans[index]
            result = runtime.analyze(plan)
            body = {**result, 'sample_id': plan['sample_id'], 'plan_sha256': digest(plan), 'fingerprint': digest(metadata)}
            body['chain'] = digest([chain, body])
            chain = body['chain']
            stream.write(canonical(body) + b'\n')
            stream.flush()
            new += 1
            if new % 16 == 0 or index + 1 == len(plans):
                os.fsync(stream.fileno())
                write(run / 'progress.json', {'stage': 'inference', 'completed': index + 1, 'required': len(plans), 'updated_utc': now()})
            if new % 100 == 0 or index + 1 == len(plans):
                print(f'Profily: {index + 1}/{len(plans)}; {(time.perf_counter()-before)/new:.3f} s/nový vstup', flush=True)
        os.fsync(stream.fileno())
    rows, verified_chain = scan(run, protocol, metadata)
    if len(rows) != len(plans) or verified_chain != chain:
        raise ValueError('Nedokončený profilový žurnál.')
    write(run / 'profiles.complete.json', {'count': len(rows), 'chain': chain, 'sha256': sha(run / 'profiles.jsonl'),
                                          'fingerprint': digest(metadata), 'runtime_sha256': sha(run / 'runtime.json')})
    return True


def results_inputs(run):
    run = Path(run)
    return {'run_protocol_sha256': sha(run / 'run_protocol.json'), 'plan_sha256': sha(run / 'plan.jsonl'),
            'profiles_sha256': sha(run / 'profiles.jsonl'), 'profiles_complete_sha256': sha(run / 'profiles.complete.json'),
            'statistics_sha256': sha(ROOT / 'statistics.py')}


def analyse(run, protocol):
    from article_ablation_confirmation import statistics
    run = Path(run)
    # Pred prvou štatistikou znovu auditovať vstupy; hotový výsledok má
    # nemenné kotvy a receipty, jeho čítanie neopakuje obsahový audit.
    verify(run, deep=not (run / 'complete.json').exists())
    rows = load_profiles(run, protocol)
    inputs = results_inputs(run)
    if (run / 'complete.json').exists():
        receipt = read(run / 'complete.json')
        required_files = {'results.json', 'plan.jsonl', 'profiles.jsonl', 'profiles.complete.json'}
        if (receipt['complete'] is not True or receipt['inputs'] != inputs
                or set(receipt['files']) != required_files):
            raise ValueError('Výsledok patrí iným vstupom.')
        for name, expected in receipt['files'].items():
            if sha(run / name) != expected:
                raise ValueError('Zmenený dokončený výsledok.')
        result = read(run / 'results.json')
        if result['inputs'] != inputs:
            raise ValueError('Výsledok odkazuje na iné vstupy.')
        return result
    plans = plans_at(run)
    write(run / 'progress.json', {'stage': 'statistics', 'completed': len(rows), 'required': len(rows), 'updated_utc': now()})
    result = statistics.evaluate([r['error'] for r in rows], [r['full_score'] for r in rows], [r['resize_score'] for r in rows],
                                 [p['sample_id'] for p in plans], [p['group'] for p in plans], [p['dataset'] for p in plans],
                                 repetitions=5000, seed=20261028)
    result['inputs'] = inputs
    write(run / 'results.json', result)
    verify(run, deep=False)
    load_profiles(run, protocol)
    if results_inputs(run) != inputs:
        raise ValueError('Vstupy sa počas vyhodnotenia zmenili.')
    write(run / 'complete.json', {'complete': True, 'inputs': inputs,
                                  'files': {name: sha(run / name) for name in ('results.json', 'plan.jsonl', 'profiles.jsonl', 'profiles.complete.json')}})
    write(run / 'progress.json', {'stage': 'complete', 'completed': len(rows), 'required': len(rows),
                                 'decision': result['decision'], 'updated_utc': now()})
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('smoke', 'prepare', 'verify', 'execute', 'report'), required=True)
    parser.add_argument('--run', type=Path, default=DEFAULT_RUN)
    parser.add_argument('--data', type=Path, default=DEFAULT_DATA)
    parser.add_argument('--design', type=Path, default=DEFAULT_DESIGN)
    parser.add_argument('--smoke', type=Path, default=DEFAULT_SMOKE)
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    parser.add_argument('--max-new', type=int)
    args = parser.parse_args(argv)
    if args.max_new is not None and (args.stage != 'execute' or args.max_new < 1):
        parser.error('--max-new musí byť kladný a patrí iba k execute.')
    if args.stage == 'smoke':
        smoke(args.design, args.smoke, args.device)
        return
    run = output_path(args.run)
    with lock(run.parent / ('.' + run.name + '.lock')):
        if args.stage == 'prepare':
            prepare(run, args.data, args.design, args.smoke, args.device)
            return
        protocol = verify(run, deep=args.stage != 'report')
        if args.stage == 'execute':
            try:
                if extract(run, protocol, args.max_new):
                    result = analyse(run, protocol)
                    print('Vyhodnotenie dokončené: ' + result['decision'], flush=True)
                else:
                    print('Čiastočný beh uložený; rovnaký príkaz pokračuje.', flush=True)
            except Exception as error:
                write(run / 'progress.json', {'stage': 'failed', 'error': str(error), 'updated_utc': now()})
                raise
        elif args.stage == 'report':
            analyse(run, protocol)
            from article_ablation_confirmation.report_results import render_report
            print(render_report(run), flush=True)
        else:
            print(json.dumps({'verified': True, 'counts': protocol['counts']}, ensure_ascii=False))


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
