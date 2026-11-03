"""Nemenná kópia dvoch EXISTUJÚCICH selektorov; bez učenia a inferencie.

Tento krok nezamyká testovacie dáta, endpointy ani pravidlo adopcie. Jeho
úspech preto neznamená pripravenosť na potvrdzujúci test. Checkpoint detektora
sa nekopíruje: zachovávajú sa a overujú obe pôvodné cesty a kontrolné súčty.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import importlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import sys
import uuid

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
FROZEN_ROOT = ROOT / 'frozen_models'
COMPARISON_RUN = PROJECT / 'selective_experiments/selector_comparison_bahat'
SCHEMA = 'article-ablation-frozen-models-v1'
# Identity artefaktov nezávisle overené pred prípravou tohto modulu.
EXPECTED_HASHES = {
    'protocol.json': 'c42d9a3457aff3b7ba70ad11ef9f6d67e92602e292e339e87eccfeaa46a40eab',
    'source_protocol.json': '7b2ec05a9e5a1dc3674cf9a2f0acf7a6d4cedacf0908b3c8191a10d698f4041f',
    'frozen_selector.json': 'c563c4ccbf6c668d7f6914ee6325518d6f9164725f4bcdd7eaacdc9fb4a50a4a',
    'models/lr_resize_only.json': 'da57322fd1fcf4c5604a37d9215266f873c4e95985a2ac7626fbaa9d0ba87182',
    'models/lr_resize_only.receipt.json': 'dc5d41ba80a67281519dee1abe87a9bfdca16184119479f70be0a68a644bd73d',
    'models/frozen_v4.json': 'ceed98e59ce872ef98efdd54749d85bfaf04e0c56dfd5884d40d13a3f5f96098',
    'models/frozen_v4.receipt.json': '1940e7e951f31f176adc9fb2c7aa28d77ee584570a41e58f0828918b921ff409',
}
PROFILE_CONFIG = {'version': 'degradation-profile-v1', 'jpeg_qualities': [90, 75, 60],
                  'resize_scales': [0.75, 0.5]}
TRANSFORMS = ['original', 'jpeg_q90', 'jpeg_q75', 'jpeg_q60', 'resize_0p75', 'resize_0p5']
FEATURE_NAMES = ['negative_absolute_calibrated_margin', 'oriented_margin_delta_jpeg_q90',
                 'oriented_margin_delta_jpeg_q75', 'oriented_margin_delta_jpeg_q60',
                 'oriented_margin_delta_resize_0p75', 'oriented_margin_delta_resize_0p5',
                 'prediction_variance', 'max_abs_shift', 'flip_fraction']
RUNTIME_NAMES = {'calibrate_detector_v4.py', 'detector_training_v4.py',
                 'extract_selector_profiles_v4.py', 'prepare_data_v4.py', 'sampler_v4.py',
                 'selector_augmentation_v4.py', 'selector_profile_runtime_v4.py',
                 'v4_data_sources.py', 'v4_deepshield_audit.py', 'v4_deepshield_remote_zip.py'}
STATUS = {'models_frozen': True, 'ready_for_testing': False, 'primary_decision': 'deferred',
          'data_protocol_frozen': False, 'endpoints_frozen': False, 'training_performed': False,
          'detector_inference_performed': False}


class FreezeError(ValueError):
    """Neplatný zdroj, prostredie alebo neúplná zmrazená kópia."""


def _sha(path):
    checksum = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            checksum.update(block)
    return checksum.hexdigest()


def _read(path):
    def invalid(value):
        raise FreezeError('Nekonečná hodnota JSON: ' + value)
    return json.loads(Path(path).read_text(encoding='utf-8'), parse_constant=invalid)


def _json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'),
                       allow_nan=False) + '\n').encode('utf-8')


def _write_new(path, body):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        stream.write(body)
        stream.flush()
        os.fsync(stream.fileno())


def _environment():
    return {'python': platform.python_version(), 'packages': {name: metadata.version(name) for name in
            ('numpy', 'scipy', 'scikit-learn', 'torch', 'torchvision', 'Pillow')}}


@contextmanager
def _no_bytecode():
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        yield
    finally:
        sys.dont_write_bytecode = previous


def _inspect_source(run):
    # Len overený read-only reader. Žiadny fit(), runtime() ani read_split().
    with _no_bytecode():
        source = importlib.import_module('article_reference_experiment.source')
        return source.inspect_source(run)


def _inside(folder, relative):
    folder = Path(folder).resolve()
    path = (folder / relative).resolve()
    if Path(relative).is_absolute() or path == folder or not path.is_relative_to(folder):
        raise FreezeError('Cesta opúšťa povolený adresár: ' + str(relative))
    return path


def _output(path):
    path = Path(path).resolve()
    base = FROZEN_ROOT.resolve()
    if path.parent != base or path.name.startswith('.'):
        raise FreezeError('Výstup musí byť article_ablation_confirmation/frozen_models/<názov_behu>.')
    return path


def _checked(path, expected, anchors):
    path = Path(path).resolve()
    actual = _sha(path)
    if actual != expected or (str(path) in anchors and anchors[str(path)] != actual):
        raise FreezeError('Zmenený SHA256 zdroja: ' + str(path))
    anchors[str(path)] = actual
    return path


def _anchored_json(path, anchors):
    path = Path(path).resolve()
    if str(path) not in anchors:
        raise FreezeError('Súbor nemá overenú kotvu: ' + str(path))
    value = _read(path)
    _checked(path, anchors[str(path)], anchors)
    return value


def _parameters(value, dimensions):
    try:
        arrays = [value['coef'], value['scaler']['mean'], value['scaler']['scale']]
        if any(not isinstance(a, list) or len(a) != dimensions for a in arrays):
            raise FreezeError('Nesprávny rozmer selektora alebo scalera.')
        values = [x for a in arrays for x in a] + [value['intercept']]
        if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in values):
            raise FreezeError('Neplatné numerické parametre selektora.')
        if any(x <= 0 for x in value['scaler']['scale']):
            raise FreezeError('Neplatné škálovanie príznakov.')
    except (KeyError, TypeError) as error:
        raise FreezeError('Chýbajú parametre zmrazeného selektora.') from error


def _model_metadata(protocol, frozen, full, resize, protocol_sha):
    if full.get('spec') != {'id': 'frozen_v4', 'kind': 'frozen', 'features': 'full'}:
        raise FreezeError('Nesprávny pôvodný referenčný model.')
    if resize.get('spec') != {'id': 'lr_resize_only', 'kind': 'lr', 'features': 'resize_only', 'C': .1}:
        raise FreezeError('Nesprávna existujúca resize ablácia.')
    for model in (full, resize):
        if model.get('protocol_sha256') != protocol_sha:
            raise FreezeError('Model patrí inému porovnávaciemu protokolu.')
    if (full.get('threshold_status') != 'original_frozen_rule' or
            resize.get('threshold_status') != 'development_only_not_certified'):
        raise FreezeError('Zmenený význam uloženého prahu.')
    if (frozen.get('profile_config') != PROFILE_CONFIG or frozen.get('detector_threshold') != .5 or
            frozen.get('model', {}).get('family') != 'symmetric_profile' or
            frozen['model'].get('feature_names') != FEATURE_NAMES or frozen['model'].get('C') != .1):
        raise FreezeError('Pôvodný selektor má iné príznaky alebo transformácie.')
    for key in ('model_sha256', 'model_fingerprint', 'temperature'):
        if frozen.get(key) != protocol.get(key):
            raise FreezeError('Zmiešaný detektor alebo teplota: ' + key)
    if frozen.get('accept_threshold') != protocol['frozen_threshold'] or full.get('threshold') != protocol['frozen_threshold']:
        raise FreezeError('Zmenený pôvodný akceptačný prah.')
    temperature = protocol['temperature']
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)) or not math.isfinite(temperature) or temperature <= 0:
        raise FreezeError('Neplatná zmrazená teplota.')
    answer = {}
    for name, parameters, dimensions, columns, names, threshold, status in (
        ('frozen_v4', frozen['model']['parameters'], 9, list(range(6)), FEATURE_NAMES,
         full['threshold'], full['threshold_status']),
        ('lr_resize_only', resize['parameters'], 6, [0, 4, 5],
         [FEATURE_NAMES[i] for i in (0, 4, 5, 6, 7, 8)], resize['threshold'], resize['threshold_status']),
    ):
        _parameters(parameters, dimensions)
        if threshold is not None and (isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or
                                      not math.isfinite(threshold) or not 0 <= threshold <= 1):
            raise FreezeError('Neplatný uložený prah.')
        answer[name] = {'dimension': dimensions, 'view_columns': columns, 'feature_names': names,
                        'C': .1, 'scaler': parameters['scaler'], 'parameters': parameters,
                        'threshold': threshold, 'threshold_status': status,
                        'parameters_changed': False, 'original_prediction_changed': False,
                        'summaries_recomputed_from_retained_views': True}
    return answer


def _collect():
    run = COMPARISON_RUN.resolve()
    anchors = {}
    # Odmietne aj koherentne prepísaný model spolu s jeho novou receipt.
    for relative, expected in EXPECTED_HASHES.items():
        _checked(_inside(run, relative), expected, anchors)
    protocol, inspected = _inspect_source(run)
    for path, expected in anchors.items():
        if inspected.get(path) != expected:
            # Individuálne resize receipt pôvodný reader ešte nezahŕňa.
            if path != str((run / 'models/lr_resize_only.receipt.json').resolve()):
                raise FreezeError('Pôvodný reader neoveril požadovanú kotvu: ' + path)
    anchors.update(inspected)
    protocol_sha = anchors[str((run / 'protocol.json').resolve())]
    if protocol != _anchored_json(run / 'protocol.json', anchors) or protocol['environment'] != _environment():
        raise FreezeError('Zmenené numerické prostredie alebo porovnávací protokol.')
    for name in ('frozen_v4', 'lr_resize_only'):
        path = run / 'models' / (name + '.receipt.json')
        _checked(path, EXPECTED_HASHES['models/' + name + '.receipt.json'], anchors)
        receipt = _anchored_json(path, anchors)
        expected_files = {name + '.json': anchors[str((run / 'models' / (name + '.json')).resolve())]}
        if receipt.get('protocol_sha256') != protocol_sha or receipt.get('files') != expected_files:
            raise FreezeError('Neplatné individuálne potvrdenie modelu: ' + name)
    frozen = _anchored_json(run / 'frozen_selector.json', anchors)
    full = _anchored_json(run / 'models/frozen_v4.json', anchors)
    resize = _anchored_json(run / 'models/lr_resize_only.json', anchors)
    models = _model_metadata(protocol, frozen, full, resize, protocol_sha)
    source_protocol = _anchored_json(run / 'source_protocol.json', anchors)
    source_root = Path(protocol['source']).resolve()
    detector_root = Path(source_protocol['detector_run']).resolve()
    original_selector = Path(protocol['selector']).resolve() / 'selector.json'
    if _anchored_json(original_selector, anchors) != frozen:
        raise FreezeError('Kópia plného selektora sa líši od pôvodného artefaktu.')
    if source_protocol.get('profile_config') != PROFILE_CONFIG:
        raise FreezeError('Pôvodný profilový protokol má iné transformácie.')
    for key in ('model_sha256', 'model_fingerprint', 'temperature'):
        if source_protocol.get(key) != protocol.get(key):
            raise FreezeError('Profilový protokol používa iný detektor: ' + key)
    copies = {}

    def include(path, target):
        path = Path(path).resolve()
        if str(path) not in anchors:
            raise FreezeError('Kópia nemá nezávisle overenú kotvu: ' + str(path))
        copies[target] = {'source': str(path), 'sha256': anchors[str(path)]}

    for name in ('lr_resize_only.json', 'lr_resize_only.receipt.json', 'frozen_v4.json', 'frozen_v4.receipt.json'):
        include(run / 'models' / name, 'models/' + name)
    include(run / 'frozen_selector.json', 'models/frozen_selector.json')
    for name in ('protocol.json', 'protocol.sha256', 'source_protocol.json', 'external_protocol.json'):
        include(run / name, 'comparison/' + name)
    for name in ('full.complete.json', 'cached.complete.json'):
        include(run / 'models' / name, 'comparison/receipts/models_' + name)
    for name in protocol['code']:
        include(_inside(run / 'code', name), 'comparison/code/' + name)
    runtime = {Path(p).name for p in protocol['files'] if p.startswith('detector_runtime/')}
    if runtime != RUNTIME_NAMES:
        raise FreezeError('Neúplná pôvodná snímka runtime.')
    for name in sorted(runtime):
        relative = 'snapshot/code/' + name
        expected = source_protocol['locked_files'].get(relative)
        if expected != protocol['files']['detector_runtime/' + name]:
            raise FreezeError('Nesúhlasí pôvodný a porovnávací runtime: ' + name)
        _checked(source_root / relative, expected, anchors)
        include(run / 'detector_runtime' / name, 'comparison/detector_runtime/' + name)
    for name in ('config.json', 'temperature.json', 'complete.json', 'temperature.complete.json', 'best_model.pt'):
        expected = source_protocol['locked_files']['snapshot/detector/' + name]
        _checked(source_root / 'snapshot/detector' / name, expected, anchors)
        _checked(detector_root / name, expected, anchors)
        if name != 'best_model.pt':
            include(source_root / 'snapshot/detector' / name, 'detector/' + name)
    config = _anchored_json(source_root / 'snapshot/detector/config.json', anchors)
    if config.get('model') != 'efficientnet_v2_m' or config.get('class_to_index') != {'real': 0, 'fake': 1}:
        raise FreezeError('Zmenená architektúra alebo poradie tried detektora.')
    for path, target in ((Path(__file__), 'verification/freeze_models.py'),
                         (PROJECT / 'article_reference_experiment/source.py', 'verification/reference_source.py')):
        _checked(path, _sha(path), anchors)
        include(path, target)
    return {'schema': SCHEMA, 'scope': 'existing_models_only_no_test_or_adoption_decision',
            **STATUS, 'comparison_run': str(run), 'comparison_protocol_sha256': protocol_sha,
            'environment': protocol['environment'], 'source_anchors': anchors, 'copies': copies,
            'expected_original_hashes': EXPECTED_HASHES, 'models': models,
            'feature_definition': 'comparison/code/features.py',
            'profile_config': PROFILE_CONFIG, 'view_order': TRANSFORMS,
            'margin_definition': '(logit_fake - logit_real) / frozen_temperature',
            'resize_definition': 'round(original_size * scale), min 1; LANCZOS down and back to original size',
            'prediction_definition': 'unchanged original logit_fake >= logit_real',
            'detector': {'model': config['model'], 'class_to_index': config['class_to_index'],
                         'temperature': protocol['temperature'], 'model_sha256': protocol['model_sha256'],
                         'model_fingerprint': protocol['model_fingerprint'],
                         'checkpoint_paths': [str(detector_root / 'best_model.pt'),
                                              str(source_root / 'snapshot/detector/best_model.pt')]}}


@contextmanager
def _lock(path):
    with path.open('a+b') as stream:
        stream.seek(0, 2)
        if stream.tell() == 0:
            stream.write(b'0')
            stream.flush()
        stream.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise FreezeError('Rovnaké zmrazenie už pripravuje iný proces.') from error
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == 'nt':
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _verify(output, *, check_sources=True):
    for name in ('manifest.json', 'manifest.sha256', 'complete.json'):
        if not (output / name).is_file():
            raise FreezeError('Existujúci adresár je neúplný; nebude prepísaný: ' + str(output))
    manifest_sha = _sha(output / 'manifest.json')
    if (output / 'manifest.sha256').read_text(encoding='ascii').strip() != manifest_sha:
        raise FreezeError('Zmenený manifest zmrazených modelov.')
    manifest = _read(output / 'manifest.json')
    complete = _read(output / 'complete.json')
    if (manifest.get('schema') != SCHEMA or complete.get('schema') != SCHEMA or
            complete.get('manifest_sha256') != manifest_sha or
            any(manifest.get(k) != v or complete.get(k) != v for k, v in STATUS.items())):
        raise FreezeError('Neplatné dokončenie alebo rozsah zmrazenia.')
    if (manifest.get('environment') != _environment() or
            manifest.get('comparison_run') != str(COMPARISON_RUN.resolve()) or
            manifest.get('expected_original_hashes') != EXPECTED_HASHES):
        raise FreezeError('Zmenené numerické prostredie alebo identita zdroja.')
    anchors, copies = manifest['source_anchors'], manifest['copies']
    if any(not Path(path).is_absolute() for path in anchors):
        raise FreezeError('Zdrojová kotva musí mať absolútnu cestu.')
    expected_files = {relative: entry['sha256'] for relative, entry in copies.items()}
    expected_files['manifest.json'] = manifest_sha
    expected_files['manifest.sha256'] = _sha(output / 'manifest.sha256')
    if complete.get('files') != expected_files:
        raise FreezeError('Potvrdenie nezahŕňa všetky kópie.')
    actual_files = {p.relative_to(output).as_posix() for p in output.rglob('*') if p.is_file()}
    if actual_files != set(expected_files) | {'complete.json'}:
        raise FreezeError('Neúplný alebo dodatočne rozšírený zmrazený adresár.')
    for relative, entry in copies.items():
        target = _inside(output, relative)
        if target.is_symlink() or anchors.get(entry['source']) != entry['sha256'] or _sha(target) != entry['sha256']:
            raise FreezeError('Zmenená kópia alebo chýbajúca pôvodná kotva: ' + relative)
    for relative, expected in EXPECTED_HASHES.items():
        if anchors.get(str(_inside(COMPARISON_RUN, relative))) != expected:
            raise FreezeError('Zmrazenie neobsahuje pôvodnú identitu modelov: ' + relative)
    protocol = _read(output / 'comparison/protocol.json')
    expected_models = _model_metadata(protocol, _read(output / 'models/frozen_selector.json'),
                                      _read(output / 'models/frozen_v4.json'),
                                      _read(output / 'models/lr_resize_only.json'),
                                      _sha(output / 'comparison/protocol.json'))
    if (manifest['models'] != expected_models or manifest['profile_config'] != PROFILE_CONFIG or
            manifest['view_order'] != TRANSFORMS or manifest['environment'] != protocol['environment'] or
            manifest['comparison_protocol_sha256'] != _sha(output / 'comparison/protocol.json')):
        raise FreezeError('Popis modelov nesúhlasí s nemennými kópiami.')
    if check_sources:
        for path, expected in anchors.items():
            _checked(path, expected, {})
    return manifest


def verify(output: Path) -> dict:
    """Bez zápisu overí kópie, pôvodné kotvy a numerické prostredie."""
    return _verify(_output(output))


def freeze(output: Path) -> dict:
    """Atómovo skopíruje existujúce modely; existujúci výstup iba overí."""
    output = _output(output)
    if output.exists():
        return verify(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with _lock(output.parent / ('.' + output.name + '.lock')):
        if output.exists():
            return verify(output)
        manifest = _collect()
        manifest['created_utc'] = datetime.now(timezone.utc).isoformat()
        staging = output.parent / ('.' + output.name + '.staging-' + uuid.uuid4().hex)
        staging.mkdir()
        try:
            for relative, entry in manifest['copies'].items():
                origin = Path(entry['source'])
                _checked(origin, entry['sha256'], {})
                target = _inside(staging, relative)
                target.parent.mkdir(parents=True, exist_ok=True)
                with origin.open('rb') as incoming, target.open('xb') as outgoing:
                    shutil.copyfileobj(incoming, outgoing)
                    outgoing.flush()
                    os.fsync(outgoing.fileno())
                if _sha(target) != entry['sha256']:
                    raise FreezeError('Zdroj sa zmenil počas kopírovania: ' + str(origin))
            _write_new(staging / 'manifest.json', _json_bytes(manifest))
            manifest_sha = _sha(staging / 'manifest.json')
            _write_new(staging / 'manifest.sha256', (manifest_sha + '\n').encode('ascii'))
            files = {p.relative_to(staging).as_posix(): _sha(p) for p in staging.rglob('*') if p.is_file()}
            _write_new(staging / 'complete.json', _json_bytes({
                'schema': SCHEMA, 'manifest_sha256': manifest_sha, **STATUS, 'files': files}))
            # Finálna kontrola zahŕňa aj zmenu originálov počas kopírovania.
            _verify(staging)
            if output.exists():
                raise FreezeError('Výstup vznikol počas prípravy; nebude prepísaný.')
            os.rename(staging, output)
            return manifest
        finally:
            if staging.exists():
                # Mažeme výlučne vlastnú, overenú staging cestu pod frozen_models.
                if staging.resolve().parent != output.parent or not staging.name.startswith('.' + output.name + '.staging-'):
                    raise FreezeError('Neplatná cesta dočasného adresára.')
                shutil.rmtree(staging)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--verify', action='store_true', help='Iba overiť existujúce zmrazenie.')
    args = parser.parse_args()
    result = verify(args.output) if args.verify else freeze(args.output)
    print('Existujúce modely boli overené: ' + str(_output(args.output)), flush=True)
    print('Rozhodnutie o hlavnom modeli je odložené. Dáta a testovací protokol ešte nie sú zmrazené.', flush=True)
    return result


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
    # Pri spustení súboru priamo sprístupni súrodenecký read-only reader.
    if str(PROJECT) not in sys.path:
        sys.path.insert(0, str(PROJECT))
    main()
