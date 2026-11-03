"""Budúci externý test iba z dokončeného zmrazeného protokolu.

Bez argumentov sa nič nespúšťa. --verify-only iba číta a overuje vstupy.
--execute explicitne spustí inferenciu; --resume obnoví existujúci nový beh.
Nikdy sa neučí selektor, teplota, škálovanie ani prah na externých triedach.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
import importlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import time
import types
import uuid

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import freeze

DETECTORS = ('clip', 'resnet50')
EXPECTED_COUNTS = {'OpenFake': 4000, 'DeepShield': 2000, 'CommunityForensics': 800}
RUN_VERSION = 'article-independent-external-run-v1'
SPLIT = 'external_test'


def now():
    return datetime.now(timezone.utc).isoformat()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name('.' + path.name + '.' + uuid.uuid4().hex + '.tmp')
    with temporary.open('xb') as stream:
        stream.write(freeze.canonical(value) + b'\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def allowed_output(path):
    path = Path(path).resolve()
    base = (ROOT / 'runs').resolve()
    if path == base or not path.is_relative_to(base):
        raise ValueError('Externý výsledok musí byť nový podpriečinok external_testing/runs.')
    return path


@contextmanager
def lock(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+b') as stream:
        if stream.tell() == 0:
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


def load_frozen_modules(folder):
    """Relatívne importy idú výhradne do overenej kópie vývojového kódu."""
    name = '_article_external_frozen_' + freeze.sha(folder / 'protocol.json')[:20]
    package = types.ModuleType(name)
    package.__path__ = [str(folder / 'code')]
    sys.modules[name] = package
    common = importlib.import_module(name + '.common')
    # Overenie prostredia neimportuje torch a nenačítava obrazové modely.
    return name, common


def verify_inputs(frozen_path):
    folder = Path(frozen_path).resolve()
    protocol = freeze.verify_frozen(folder)
    if protocol.get('version') != 'article-independent-external-freeze-v1':
        raise ValueError('Nepodporovaný zmrazený externý protokol.')
    if freeze.sha(Path(freeze.__file__)) != freeze.sha(folder / 'preparation/freeze.py'):
        raise ValueError('Zmenená implementácia validátora po zmrazení.')
    if set(protocol['detectors']) != set(DETECTORS):
        raise ValueError('Test vyžaduje obidva zmrazené detektory.')
    primary = protocol['primary']
    if (primary['dataset'] != 'OpenFake' or primary['pointwise_interval'] != .95
            or primary['contrasts'] != ['clip:lr_full-confidence', 'resnet50:lr_full-confidence']
            or protocol['bootstrap']['repetitions'] != 5000 or protocol['bootstrap']['seed'] != 20261027
            or protocol['secondary']['fixed_coverages'] != [.8, .9, .95]):
        raise ValueError('Runner nepodporuje zmenený vopred určený štatistický plán.')
    # Znovu overiť aj pôvodné plány, históriu a checkpointy. Žiadny obrazový model.
    for path, checksum in protocol['inputs']['input_anchors'].items():
        if freeze.sha(path) != checksum:
            raise ValueError('Zmenená pôvodná kotva zmrazenia: ' + path)
    name, common = load_frozen_modules(folder)
    if common.environment() != protocol['environment']:
        raise ValueError('Zmenilo sa numerické prostredie externého testu; obnovte zmrazené prostredie.')
    rows = list(freeze.lines(folder / 'plan/manifest.jsonl'))
    if Counter(r['dataset'] for r in rows) != Counter(EXPECTED_COUNTS):
        raise ValueError('Manifest nezodpovedá pevnému rozsahu 4000/2000/800.')
    seen = set()
    dependencies = {}
    for row in rows:
        if row['sample_id'] in seen or type(row['label']) is not int or row['label'] not in (0, 1):
            raise ValueError('Neplatná alebo opakovaná identita externého testu.')
        seen.add(row['sample_id'])
        if row['input_spec'] != {'version': 1, 'id': 'clean', 'operations': []}:
            raise ValueError('Externý test musí mať čisté originály.')
        if not row.get('dependency_keys') or not row.get('group'):
            raise ValueError('Chýba skupina externého testu.')
        for dependency in row['dependency_keys']:
            if dependencies.setdefault(dependency, row['group']) != row['group']:
                raise ValueError('Rovnaká pôvodová alebo sampling závislosť je rozdelená medzi skupiny.')
        if freeze.sha(row['path']) != row['sha256']:
            raise ValueError('Zmenený externý obrázok: ' + row['sample_id'])
    for dataset in EXPECTED_COUNTS:
        subset = [r for r in rows if r['dataset'] == dataset]
        actual = {'real': sum(r['label'] == 0 for r in subset), 'fake': sum(r['label'] == 1 for r in subset),
                  'groups': len({r['group'] for r in subset})}
        if actual != protocol['counts'][dataset] or not actual['real'] or not actual['fake']:
            raise ValueError('Nesúhlasia zmrazené počty alebo triedy: ' + dataset)
    assets = ROOT.parent / 'assets'
    metadata = {}
    for detector in DETECTORS:
        info = freeze.read(folder / 'models' / detector / 'detector.json')
        model = freeze.read(folder / 'models' / detector / 'selector.json')
        entry = protocol['detectors'][detector]
        if (freeze.digest(info) != entry['fingerprint']
                or freeze.sha(folder / 'models' / detector / 'selector.json') != entry['selector_sha256']
                or model['temperature'] != entry['temperature']
                or info['detector'] != detector or info['batch_size'] != 6
                or info['tf32'] is not False or info['adapter']['tf32'] is not False
                or info['adapter']['batch_size'] != 6 or info['device'] != info['adapter']['device']):
            raise ValueError('Nezhodný zmrazený detektor, teplota alebo režim inferencie.')
        manifest = assets / detector / 'manifest.json'
        if freeze.sha(manifest) != info['adapter']['asset_manifest_sha256']:
            raise ValueError('Zmenený manifest obrazového modelu: ' + detector)
        for relative, item in info['adapter']['assets'].items():
            if freeze.sha(freeze.inside(assets / detector, relative)) != item['sha256']:
                raise ValueError('Zmenený checkpoint obrazového modelu: ' + detector)
        metadata[detector] = info
    return {'folder': folder, 'protocol': protocol, 'rows': [{**r, 'split': SPLIT} for r in rows],
            'package': name, 'common': common, 'metadata': metadata, 'assets': assets,
            'frozen_sha256': freeze.sha(folder / 'protocol.json')}


def create_or_verify_run(context, output, resume=False):
    output = allowed_output(output)
    inputs = {'frozen_path': str(context['folder']), 'frozen_sha256': context['frozen_sha256'],
              'runner_sha256': freeze.sha(Path(__file__)), 'validator_sha256': freeze.sha(Path(freeze.__file__))}
    if output.exists():
        if not resume:
            raise ValueError('Výstup už existuje. Obnovenie vyžaduje explicitné --resume.')
        protocol = freeze.checked_protocol(output, {})
        if protocol['version'] != RUN_VERSION or protocol['inputs'] != inputs:
            raise ValueError('Beh patrí inému protokolu alebo zmenenej implementácii runnera.')
        for relative, checksum in protocol['files'].items():
            if freeze.sha(freeze.inside(output, relative)) != checksum:
                raise ValueError('Zmenený snapshot externého runnera.')
    else:
        temporary = output.with_name('.' + output.name + '.preparing-' + uuid.uuid4().hex)
        (temporary / 'snapshot').mkdir(parents=True)
        for source, name in ((Path(__file__), 'run_test.py'), (Path(freeze.__file__), 'freeze.py')):
            shutil.copyfile(source, temporary / 'snapshot' / name)
        protocol = {'version': RUN_VERSION, 'created_utc': now(), 'inputs': inputs,
                    'environment': context['protocol']['environment'],
                    'inference': {'detectors': list(DETECTORS), 'originals_per_detector': len(context['rows']),
                                  'views_per_original': 6, 'learn_on_external_labels': False},
                    'files': {'snapshot/' + name: freeze.sha(temporary / 'snapshot' / name)
                              for name in ('run_test.py', 'freeze.py')}}
        save(temporary / 'protocol.json', protocol)
        (temporary / 'protocol.sha256').write_text(freeze.sha(temporary / 'protocol.json') + '\n', encoding='ascii')
        os.rename(temporary, output)
    context['output'] = output
    context['run_sha256'] = freeze.sha(output / 'protocol.json')
    return protocol


def inference_metadata(context, detector):
    return {'version': 'article-external-inference-v1', 'run_sha256': context['run_sha256'],
            'frozen_sha256': context['frozen_sha256'], 'detector': detector,
            'development_fingerprint': context['protocol']['detectors'][detector]['fingerprint'],
            'configuration': context['metadata'][detector]}


def load_adapter(context, detector):
    adapters = importlib.import_module(context['package'] + '.adapters')
    info = context['metadata'][detector]
    import torch
    torch.manual_seed(20261027)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    adapter = adapters.load_detector(detector, context['assets'], device=info['device'], batch_size=info['batch_size'])
    if adapter.metadata != info['adapter']:
        raise ValueError('Načítaný model, preprocessing alebo nastavenie sa líši od vývoja.')
    return adapter


def validated_journal(context, detector, repair=False, require_complete=False):
    profiles = importlib.import_module(context['package'] + '.profiles')
    folder = context['output'] / detector
    metadata = inference_metadata(context, detector)
    fingerprint = freeze.digest(metadata)
    path, receipt_path = folder / 'profiles.jsonl', folder / 'profiles.complete.json'
    if (folder / 'detector.json').exists() and freeze.read(folder / 'detector.json') != metadata:
        raise ValueError('Žurnál patrí inému detektoru alebo nastaveniu.')
    if path.exists() and not (folder / 'detector.json').exists():
        raise ValueError('Žurnálu chýba identita detektora.')
    receipt = freeze.read(receipt_path) if receipt_path.exists() else None
    if receipt and (receipt['sha256'] != freeze.sha(path) or receipt['fingerprint'] != fingerprint):
        raise ValueError('Zmenený dokončený žurnál externých profilov.')
    rows, chain = profiles.scan(path, context['rows'], fingerprint, repair=repair and not receipt)
    # Pôvodný scanner kontroluje kritické inferenčné polia; externý test overí celý manifest.
    for row, plan in zip(rows, context['rows']):
        if any(row.get(key) != value for key, value in plan.items()):
            raise ValueError('Profil má zmenené pole externého manifestu.')
    complete = bool(receipt and receipt['count'] == len(context['rows']) == len(rows) and receipt['chain'] == chain)
    if receipt and not complete:
        raise ValueError('Neúplné alebo neplatné potvrdenie externých profilov.')
    if require_complete and not complete:
        raise ValueError('Analýza vyžaduje kompletnú inferenciu oboch detektorov.')
    return rows, chain, complete


def extract_detector(context, detector, max_new=None):
    profiles = importlib.import_module(context['package'] + '.profiles')
    import numpy as np
    rows, chain, complete = validated_journal(context, detector, repair=True)
    if complete:
        return {'complete': True, 'new_profiles': 0}
    folder = context['output'] / detector
    folder.mkdir(exist_ok=True)
    metadata = inference_metadata(context, detector)
    if not (folder / 'detector.json').exists():
        save(folder / 'detector.json', metadata)
    fingerprint = freeze.digest(metadata)
    adapter = load_adapter(context, detector)
    if adapter.metadata != context['metadata'][detector]['adapter']:
        raise ValueError('Načítaný detektor nezodpovedá zmrazenému nastaveniu.')
    augmentation_path = context['folder'] / 'code/augmentation.py'
    spec = importlib.util.spec_from_file_location(context['package'] + '.augmentation', augmentation_path)
    augmentation = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(augmentation)
    device = context['metadata'][detector]['device']
    images = list(profiles.variants(profiles.load_input(context['rows'][0], augmentation.apply_spec)))
    profiles.synchronization(device)
    adapter.predict_logits(images)  # zahrievacia dávka, bez výberu podľa štítkov
    batched = np.asarray(adapter.predict_logits(images), dtype=float)
    singles = np.asarray([adapter.predict_logits([im])[0] for im in images], dtype=float)
    if (batched.shape != (6,) or not np.isfinite(batched).all()
            or not np.allclose(batched, singles, rtol=1e-4, atol=1e-4)
            or not np.array_equal(batched >= 0, singles >= 0)):
        raise ValueError('Nezhodná jednotlivá a dávková externá inferencia.')
    new, start = 0, time.perf_counter()
    with (folder / 'profiles.jsonl').open('ab') as stream:
        for plan in context['rows'][len(rows):]:
            if max_new is not None and new >= max_new:
                os.fsync(stream.fileno())
                return {'complete': False, 'new_profiles': new}
            inputs = list(profiles.variants(profiles.load_input(plan, augmentation.apply_spec)))
            profiles.synchronization(device)
            before = time.perf_counter()
            margins = np.asarray(adapter.predict_logits(inputs), dtype=float)
            profiles.synchronization(device)
            if margins.shape != (6,) or not np.isfinite(margins).all():
                raise ValueError('Externý detektor nevrátil šesť konečných logitov.')
            row = {**plan, 'margins': margins.tolist(), 'fingerprint': fingerprint,
                   'error': int(int(margins[0] >= 0) != plan['label']),
                   'inference_seconds': time.perf_counter() - before}
            row['chain'] = freeze.digest([chain, row])
            chain = row['chain']
            stream.write(freeze.canonical(row) + b'\n')
            stream.flush()
            rows.append(row)
            new += 1
            if new % 32 == 0:
                os.fsync(stream.fileno())
                save(context['output'] / 'progress.json', {'updated_utc': now(), 'stage': 'inference',
                     'detector': detector, 'profiles': len(rows), 'required': len(context['rows'])})
                print(f'{detector}: {len(rows)}/{len(context["rows"])} externých profilov', flush=True)
        os.fsync(stream.fileno())
    save(folder / 'profiles.complete.json', {'count': len(rows), 'sha256': freeze.sha(folder / 'profiles.jsonl'),
         'chain': chain, 'fingerprint': fingerprint})
    save(folder / 'timing' / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') + '.json'),
         {'new_profiles': new, 'image_inferences': new * 6, 'wall_seconds': time.perf_counter() - start,
          'sanity_check': {'same_decisions': True, 'batch_single_max_abs': float(np.max(np.abs(batched - singles)))}})
    return {'complete': True, 'new_profiles': new}


def analysis_inputs(context):
    inputs = {'run_sha256': context['run_sha256'], 'frozen_sha256': context['frozen_sha256'], 'detectors': {}}
    all_rows = {}
    # Obe kompletné inferencie sú povinné pred prvým výpočtom testovej metriky.
    for detector in DETECTORS:
        rows, _, _ = validated_journal(context, detector, require_complete=True)
        folder = context['output'] / detector
        all_rows[detector] = rows
        inputs['detectors'][detector] = {name: freeze.sha(folder / name)
                                       for name in ('profiles.jsonl', 'profiles.complete.json', 'detector.json')}
    return inputs, all_rows


def verify_complete(context):
    receipt = freeze.read(context['output'] / 'complete.json')
    inputs, _ = analysis_inputs(context)
    if receipt['inputs'] != inputs or receipt.get('external_metrics_computed') is not True:
        raise ValueError('Dokončená analýza nepatrí aktuálnym profilom.')
    for relative, checksum in receipt['files'].items():
        if freeze.sha(freeze.inside(context['output'], relative)) != checksum:
            raise ValueError('Zmenený dokončený externý výsledok.')
    return freeze.read(context['output'] / 'results.json')


def analyse(context):
    # Pred importom skórovacieho kódu znovu overiť snapshot; inferencia môže trvať hodiny.
    current = verify_inputs(context['folder'])
    if current['frozen_sha256'] != context['frozen_sha256'] or current['rows'] != context['rows']:
        raise ValueError('Zmrazené vstupy sa zmenili počas externej inferencie.')
    inputs, all_rows = analysis_inputs(context)
    if (context['output'] / 'complete.json').exists():
        return verify_complete(context)
    evaluation = importlib.import_module(context['package'] + '.evaluation')
    results = {}
    for detector in DETECTORS:
        model = freeze.read(context['folder'] / 'models' / detector / 'selector.json')
        results[detector] = {}
        for dataset in EXPECTED_COUNTS:
            rows = [r for r in all_rows[detector] if r['dataset'] == dataset]
            scores = evaluation.score_records(rows, model)
            if set(scores) != set(context['protocol']['methods']):
                raise ValueError('Skórovanie nevrátilo presne zmrazené metódy.')
            results[detector][dataset] = evaluation.evaluate_records(
                rows, scores, bootstrap=context['protocol']['bootstrap']['repetitions'],
                seed=context['protocol']['bootstrap']['seed'])
            print(f'Vyhodnotené {detector}/{dataset}; párový skupinový bootstrap 5000.', flush=True)
    primary = {}
    for detector in DETECTORS:
        result = results[detector]['OpenFake']
        interval = result['paired_group_bootstrap']['intervals'].get('lr_full', {}).get('aurc')
        primary[detector] = {'delta_aurc': result['differences_vs_confidence']['lr_full']['aurc'],
                             'interval_95': interval,
                             'supports_improvement': bool(interval and interval['valid_repetitions'] == 5000
                                                          and interval['upper_95'] < 0)}
    report = {'version': RUN_VERSION, 'created_utc': now(), 'inputs': inputs, 'results': results,
              'primary': {'dataset': 'OpenFake', 'contrast': 'lr_full-confidence', 'detectors': primary,
                          'intersection_union_supports_both': all(r['supports_improvement'] for r in primary.values()),
                          'policy': context['protocol']['primary']},
              'independence': context['protocol']['independence'],
              'scope': 'Dva verejné detektory CLIP a ResNet-50. EfficientNet nie je súčasťou tohto externého behu.',
              'external_metrics_computed': True, 'trained_on_external_labels': False,
              'risk_guarantee_transferred': False}
    summary = [report['scope'], 'Primárny kontrast: OpenFake, ΔAURC = lr_full − confidence; záporný rozdiel zvýhodňuje profil.']
    for detector, value in primary.items():
        interval = value['interval_95']
        bounds = (f'[{interval["lower_95"]:.6f}, {interval["upper_95"]:.6f}]' if interval else 'nedostupný')
        summary.append(f'{detector}: ΔAURC={value["delta_aurc"]:.6f}; bodový 95 % interval {bounds}.')
    summary.extend(['Spoločné tvrdenie o prínose pre oba detektory: ' + ('podporené.' if report['primary']['intersection_union_supports_both'] else 'nepodporené.'),
                    'DeepShield, CommunityForensics, ablačné metódy a pevné pokrytia sú sekundárne.',
                    'Výsledok neposkytuje záruku 2 % rizika ani dôkaz čistej kauzality architektúry.',
                    context['protocol']['independence']['limitation']])
    # Potvrdenie až po opätovnej kontrole všetkých profilov a nemenných vstupov.
    if analysis_inputs(context)[0] != inputs:
        raise ValueError('Profily sa zmenili počas analýzy.')
    freeze.verify_frozen(context['folder'])
    save(context['output'] / 'results.json', report)
    text_path = context['output'] / 'summary.txt'
    temporary = text_path.with_suffix('.writing-' + uuid.uuid4().hex)
    temporary.write_text('\n'.join(summary) + '\n', encoding='utf-8')
    os.replace(temporary, text_path)
    save(context['output'] / 'complete.json', {'inputs': inputs, 'external_inference_performed': True,
         'external_metrics_computed': True, 'trained_on_external_labels': False,
         'files': {name: freeze.sha(context['output'] / name) for name in ('results.json', 'summary.txt')}})
    return report


def execute(frozen_path, output, resume=False, max_new=None):
    if max_new is not None and (type(max_new) is not int or max_new <= 0):
        raise ValueError('max-new musí byť kladné celé číslo.')
    output = allowed_output(output)
    with lock(output.parent / ('.' + output.name + '.lock')):
        context = verify_inputs(frozen_path)
        create_or_verify_run(context, output, resume)
        if (output / 'complete.json').exists():
            return verify_complete(context)
        remaining = max_new
        try:
            for detector in DETECTORS:
                status = extract_detector(context, detector, remaining)
                if remaining is not None:
                    remaining -= status['new_profiles']
                if not status['complete'] or (remaining == 0 and detector != DETECTORS[-1]):
                    return {'complete': False, 'stage': 'inference', 'external_metrics_computed': False}
            return analyse(context)
        except Exception as error:
            save(output / 'progress.json', {'updated_utc': now(), 'stage': 'failed', 'error': str(error)})
            raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frozen', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--verify-only', action='store_true')
    mode.add_argument('--execute', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--max-new', type=int)
    args = parser.parse_args(argv)
    if args.verify_only and (args.resume or args.max_new is not None):
        parser.error('--resume a --max-new patria iba k --execute.')
    if args.verify_only:
        output = allowed_output(args.output)
        context = verify_inputs(args.frozen)
        result = {'verified': True, 'external_inference_performed': False,
                  'profiles_per_detector': len(context['rows']), 'output_exists': output.exists()}
    else:
        result = execute(args.frozen, args.output, args.resume, args.max_new)
        result = {'complete': result.get('external_metrics_computed', False),
                  'output': str(args.output.resolve()), 'external_metrics_computed': result.get('external_metrics_computed', False)}
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


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
