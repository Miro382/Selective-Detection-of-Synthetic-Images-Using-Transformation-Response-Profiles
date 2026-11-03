"""Read-only import pôvodného dvojlogitového EfficientNet porovnania.

Neučí modely, neotvára obrázky ani nespúšťa detektor. Referenčné selektory
skóruje overenou uloženou implementáciou na existujúcich logitoch. Externé
dáta sú už použitý retrospektívny benchmark, nie nový potvrdzujúci holdout.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import math
from pathlib import Path
import platform
import sys

import numpy as np

BASELINES = ('confidence', 'tta_mean', 'frozen_v4', 'lr_confidence',
             'mlp_bahat_logits_s17', 'mlp_bahat_logits_s29', 'mlp_bahat_logits_s43')
SPLITS = ('selection', 'external')
EXTERNAL_SCOPE = 'retrospective_existing_external_14000_not_new_confirmatory_holdout'
_REQUIRED_CODE = {'__init__.py', 'common.py', 'features.py', 'models.py', 'metrics.py', 'extract.py'}


class SourceError(ValueError):
    """Zdroj je neúplný, zmenený alebo vnútorne nekonzistentný."""


def _sha(path):
    checksum = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b''):
            checksum.update(chunk)
    return checksum.hexdigest()


def _read(path):
    def invalid(value):
        raise SourceError('Neplatná konečná hodnota JSON: ' + value)
    return json.loads(Path(path).read_text(encoding='utf-8'), parse_constant=invalid)


def _inside(folder, relative):
    folder = Path(folder).resolve()
    path = (folder / relative).resolve()
    if Path(relative).is_absolute() or path == folder or not path.is_relative_to(folder):
        raise SourceError('Cesta opúšťa zmrazený zdroj: ' + str(relative))
    return path


def _checked(path, expected, anchors):
    path = Path(path).resolve()
    actual = _sha(path)
    if actual != expected:
        raise SourceError('Nesúhlasí SHA256 zdroja: ' + str(path))
    previous = anchors.get(str(path))
    if previous is not None and previous != actual:
        raise SourceError('Zdroj sa zmenil počas overovania: ' + str(path))
    anchors[str(path)] = actual
    return path


def _anchored_read(path, anchors):
    path = Path(path).resolve()
    before = _sha(path)
    value = _read(path)
    _checked(path, before, anchors)
    return value


def _environment():
    names = ('numpy', 'scipy', 'scikit-learn', 'torch', 'torchvision', 'Pillow')
    return {'python': platform.python_version(),
            'packages': {name: importlib.metadata.version(name) for name in names}}


@contextmanager
def _no_bytecode():
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        yield
    finally:
        sys.dont_write_bytecode = previous


def _package(run, protocol_sha):
    """Relatívne importy zostanú vo verifikovanej snímke, nie v živom kóde."""
    identity = hashlib.sha256((str(run) + ':' + protocol_sha).encode()).hexdigest()[:24]
    name = '_article_reference_frozen_comparison_' + identity
    with _no_bytecode():
        if name not in sys.modules:
            spec = importlib.util.spec_from_file_location(
                name, run / 'code/__init__.py', submodule_search_locations=[str(run / 'code')])
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            try:
                spec.loader.exec_module(module)
            except Exception:
                sys.modules.pop(name, None)
                raise
        return {part: importlib.import_module(name + '.' + part)
                for part in ('common', 'features', 'models', 'extract')}


def _receipt(folder, name, protocol_sha, anchors, required=()):
    receipt = _anchored_read(folder / name, anchors)
    if receipt.get('protocol_sha256') != protocol_sha or not isinstance(receipt.get('files'), dict):
        raise SourceError('Potvrdenie patrí inému protokolu: ' + str(folder / name))
    if not set(required) <= set(receipt['files']):
        raise SourceError('Potvrdenie neuzamklo všetky potrebné súbory: ' + name)
    for relative, expected in receipt['files'].items():
        _checked(_inside(folder, relative), expected, anchors)
    return receipt


def _check_detector(run, protocol, anchors):
    source = Path(protocol['source']).resolve()
    source_protocol = _read(run / 'source_protocol.json')
    external_protocol = _read(run / 'external_protocol.json')
    frozen_selector = _read(run / 'frozen_selector.json')
    for other in (source_protocol, external_protocol, frozen_selector):
        for key in ('model_sha256', 'model_fingerprint', 'temperature'):
            if other.get(key) != protocol.get(key):
                raise SourceError('Zmiešaný detektor alebo teplota zdrojov: ' + key)
    temperature = protocol['temperature']
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)) or not math.isfinite(temperature) or temperature <= 0:
        raise SourceError('Neplatná zmrazená teplota.')
    original = source / 'protocol.json'
    _checked(original, _sha(run / 'source_protocol.json'), anchors)
    sidecar = source / 'protocol.sha256'
    _checked(sidecar, _sha(sidecar), anchors)
    if sidecar.read_text(encoding='ascii').strip() != anchors[str(original.resolve())]:
        raise SourceError('Zmenený pôvodný profilový protokol.')
    detector_run = Path(source_protocol['detector_run']).resolve()
    names = ('config.json', 'temperature.json', 'best_model.pt', 'complete.json', 'temperature.complete.json')
    for name in names:
        relative = 'snapshot/detector/' + name
        expected = source_protocol['locked_files'].get(relative)
        if not expected:
            raise SourceError('Zdroj neuzamkol detektorový artefakt: ' + relative)
        _checked(source / relative, expected, anchors)
        _checked(detector_run / name, expected, anchors)
    config = _read(source / 'snapshot/detector/config.json')
    calibrated = _read(source / 'snapshot/detector/temperature.json')
    complete = _read(source / 'snapshot/detector/complete.json')
    temperature_complete = _read(source / 'snapshot/detector/temperature.complete.json')
    if config.get('model') != 'efficientnet_v2_m' or config.get('class_to_index') != {'real': 0, 'fake': 1}:
        raise SourceError('Očakáva sa pôvodný EfficientNet V2-M s triedami real=0, fake=1.')
    if (source_protocol['locked_files']['snapshot/detector/best_model.pt'] != protocol['model_sha256'] or
            calibrated.get('temperature') != temperature or calibrated.get('model_sha256') != protocol['model_sha256'] or
            source_protocol.get('temperature_artifact_sha256') != anchors[str((detector_run / 'temperature.json').resolve())] or
            complete.get('files', {}).get('best_model.pt') != protocol['model_sha256'] or
            complete.get('files', {}).get('config.json') != anchors[str((detector_run / 'config.json').resolve())] or
            temperature_complete.get('files', {}).get('temperature.json') != source_protocol['temperature_artifact_sha256']):
        raise SourceError('Detektor, teplota a potvrdenia dokončenia nesúhlasia.')


def _load_npz(path):
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def _validated_split(run, split, protocol, package, receipt):
    data = _load_npz(run / 'data' / (split + '.npz'))
    plans = _read(run / 'data' / (split + '.json'))
    extra = _load_npz(run / 'logits' / (split + '.npz'))
    count = protocol['counts'][split]['profiles']
    if not isinstance(plans, list) or len(plans) != count or len({r['sample_id'] for r in plans}) != count:
        raise SourceError('Neúplné alebo duplicitné poradie vzoriek: ' + split)
    required = {'raw_margins': (count, 6), 'margins': (count, 6),
                'y': (count,), 'labels': (count,), 'frozen_scores': (count,)}
    for key, shape in required.items():
        if key not in data or data[key].shape != shape or not np.isfinite(data[key]).all():
            raise SourceError('Neplatná dátová cache: ' + split + '/' + key)
    for key in ('ours_logits', 'bahat_logits'):
        if key not in extra or extra[key].shape != (count, 6, 2) or not np.isfinite(extra[key]).all():
            raise SourceError('Neplatná dvojlogitová cache: ' + split + '/' + key)
    labels = np.asarray([r['label'] for r in plans])
    if not np.isin(labels, [0, 1]).all() or not np.array_equal(data['labels'], labels):
        raise SourceError('Triedy cache nesúhlasia s plánom.')
    if not np.array_equal(data['y'], ((data['raw_margins'][:, 0] >= 0) != labels).astype(np.int64)):
        raise SourceError('Nesprávne označené chyby pôvodného detektora.')
    if not np.array_equal(data['margins'], data['raw_margins'] / protocol['temperature']):
        raise SourceError('Kalibrované marginy nesúhlasia so zmrazenou teplotou.')
    frozen = _read(run / 'frozen_selector.json')
    expected_scores = package['features'].logistic_score(
        frozen['model']['parameters'], package['features'].profile_features(data['margins']))
    if not np.allclose(data['frozen_scores'], expected_scores, atol=1e-12, rtol=1e-12):
        raise SourceError('Cache pôvodného selektora nezodpovedá zmrazeným parametrom.')
    # repair=False je zásadné: reader nesmie opravovať ani skracovať starý žurnál.
    records, chain = package['extract'].scan(run / 'logits' / (split + '.jsonl'),
                                            plans, data['raw_margins'], _sha(run / 'protocol.json'), split,
                                            repair=False)
    if len(records) != count or receipt.get('count') != count or receipt.get('chain') != chain:
        raise SourceError('Neúplný alebo neplatný kontrolný reťazec logitov: ' + split)
    for i, record in enumerate(records):
        for key in ('ours_logits', 'bahat_logits'):
            if not np.array_equal(extra[key][i], np.asarray(record[key])):
                raise SourceError('NPZ a JSONL majú iné logity alebo poradie: ' + split)
    return data, plans, extra


def _inspect(comparison_run):
    run = Path(comparison_run).resolve()
    anchors = {}
    sidecar = run / 'protocol.sha256'
    _checked(sidecar, _sha(sidecar), anchors)
    protocol_path = _checked(run / 'protocol.json', sidecar.read_text(encoding='ascii').strip(), anchors)
    protocol_sha = anchors[str(protocol_path)]
    protocol = _read(protocol_path)
    if protocol.get('version') != 'selector-comparison-1':
        raise SourceError('Nepodporovaný pôvodný porovnávací experiment.')
    if protocol.get('environment') != _environment():
        raise SourceError('Zmenené numerické prostredie; použite pôvodný interpreter the recorded package versions.')
    if not _REQUIRED_CODE <= set(protocol.get('code', {})):
        raise SourceError('Chýba zmrazená implementácia porovnania.')
    required_files = {'source_protocol.json', 'external_protocol.json', 'frozen_selector.json',
                      *(f'data/{split}.{suffix}' for split in SPLITS for suffix in ('json', 'npz'))}
    if not required_files <= set(protocol.get('files', {})):
        raise SourceError('Protokol neuzamkol všetky používané dáta.')
    for relative, expected in protocol['files'].items():
        _checked(_inside(run, relative), expected, anchors)
    for name, expected in protocol['code'].items():
        if Path(name).name != name or protocol['files'].get('code/' + name) != expected:
            raise SourceError('Nesúhlasí zoznam zmrazeného kódu: ' + name)
        _checked(_inside(run / 'code', name), expected, anchors)
    for filename, expected in protocol.get('anchors', {}).items():
        if not Path(filename).is_absolute():
            raise SourceError('Pôvodná kotva musí mať absolútnu cestu.')
        _checked(filename, expected, anchors)
    _check_detector(run, protocol, anchors)
    package = _package(run, protocol_sha)
    # Verifikácia vlastným copied common.py kontroluje aj jeho skutočné __file__.
    if package['common'].verify(run) != protocol:
        raise SourceError('Pôvodný overovač vrátil iný protokol.')
    models = {}
    for suite in ('cached', 'full'):
        specs = protocol['methods'][suite]
        expected_names = {spec['id'] + '.json' for spec in specs}
        receipt = _receipt(run / 'models', suite + '.complete.json', protocol_sha, anchors, expected_names)
        if (set(receipt['files']) != expected_names or receipt.get('suite') != suite or
                receipt.get('external_used_for_fitting') is not False or receipt.get('calibration_used') is not False):
            raise SourceError('Učenie porovnávacích modelov nemá platné dokončenie: ' + suite)
    specs = {spec['id']: spec for spec in protocol['methods']['full']}
    for name in BASELINES:
        if name not in specs:
            raise SourceError('Chýba dohodnutá referenčná metóda: ' + name)
        _receipt(run / 'models', name + '.receipt.json', protocol_sha, anchors, (name + '.json',))
        model = _read(run / 'models' / (name + '.json'))
        if model.get('spec') != specs[name] or model.get('protocol_sha256') != protocol_sha:
            raise SourceError('Model nepatrí zmrazenému porovnaniu: ' + name)
        models[name] = model
    split_data = {}
    for split in SPLITS:
        receipt = _receipt(run / 'logits', split + '.complete.json', protocol_sha, anchors,
                           (split + '.jsonl', split + '.npz'))
        split_data[split] = _validated_split(run, split, protocol, package, receipt)
    return protocol, anchors, package, models, split_data


def inspect_source(comparison_run: Path) -> tuple[dict, dict]:
    """Overí pôvodné artefakty; vráti nezmenený protokol a absolútne SHA kotvy."""
    protocol, anchors, _, _, _ = _inspect(comparison_run)
    return protocol, anchors


def read_split(comparison_run: Path, split: str, indices=None) -> dict:
    """Vráti overenú selection/external podmnožinu v zadanom poradí.

    indices je jednorozmerná sekvencia jedinečných nezáporných celých indexov.
    Logity sú pôvodné real/fake páry bez triedenia alebo delenia teplotou.
    Skóre baseline sú pravdepodobnosti chyby (nižšie je lepšie) presne podľa
    pôvodného score(); nejde o nové učenie ani detektorovú inferenciu.
    """
    if split not in SPLITS:
        raise SourceError('Povolené sú iba selection a retrospective external.')
    _, _, package, models, split_data = _inspect(comparison_run)
    data, plans, extra = split_data[split]
    if indices is None:
        selected = np.arange(len(plans), dtype=np.int64)
    else:
        selected = np.asarray(indices)
        if selected.ndim == 1 and selected.size == 0:
            selected = np.empty(0, dtype=np.int64)
        if selected.ndim != 1 or selected.dtype.kind not in 'iu' or selected.dtype.kind == 'b':
            raise SourceError('indices musia byť jednorozmerné celočíselné indexy.')
        if np.any(selected < 0) or np.any(selected >= len(plans)) or len(np.unique(selected)) != len(selected):
            raise SourceError('Indexy sú mimo rozsahu alebo sa opakujú.')
    subset = {key: values[selected] for key, values in data.items()}
    subextra = {key: values[selected] for key, values in extra.items()}
    scores = {}
    if len(selected):
        import torch
        previous_threads = torch.get_num_threads()
        # Pôvodné mlp_score vytvorí sieť a nastaví počet CPU threads. Čítanie
        # nesmie posunúť RNG následného MC experimentu ani ponechať túto zmenu.
        try:
            with torch.random.fork_rng(devices=[]), _no_bytecode():
                for name in BASELINES:
                    score = np.asarray(package['models'].score(models[name], subset, subextra), dtype=np.float64)
                    if score.shape != (len(selected),) or not np.isfinite(score).all():
                        raise SourceError('Neplatné uložené referenčné skóre: ' + name)
                    scores[name] = score
        finally:
            torch.set_num_threads(previous_threads)
    else:
        scores = {name: np.empty(0, dtype=np.float64) for name in BASELINES}
    return {'plans': [plans[int(i)] for i in selected],
            'logits': subextra['ours_logits'][:, 0, :].copy(),
            'raw_margins': subset['raw_margins'].copy(),
            'errors': subset['y'].astype(np.int64, copy=True), 'baseline_scores': scores}
