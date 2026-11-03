"""Pôvodný FP32 EfficientNet a dva zmrazené selektory, bez učenia a MC.

Pohľady, predspracovanie, vonkajší input_spec aj príznaky sa vykonávajú
priamo overeným kódom z frozen_models. Runtime nevytvára žiadne súbory.
"""
from __future__ import annotations

from contextlib import contextmanager
import copy
import hashlib
import importlib.util
import io
import json
import math
from pathlib import Path
import platform
import sys
import threading
import time

import numpy as np
from PIL import Image
import torch

VERSION = 'article-ablation-native-six-view-runtime-v1'
VIEW_ORDER = ('original', 'jpeg_q90', 'jpeg_q75', 'jpeg_q60', 'resize_0p75', 'resize_0p5')
RUNTIME_MODULES = ('v4_data_sources', 'prepare_data_v4', 'sampler_v4',
                   'detector_training_v4', 'selector_augmentation_v4', 'v4_deepshield_audit')
NUMERICAL_POLICY = {'dtype': 'float32', 'batch_size': 1, 'autocast': False,
                    'num_threads': 4, 'cudnn_benchmark': False, 'cudnn_deterministic': True,
                    'cuda_matmul_allow_tf32': False, 'cudnn_allow_tf32': False,
                    'deterministic_algorithms': True, 'deterministic_warn_only': True}
_PROCESS_LOCK = threading.RLock()


class RuntimeErrorV4(ValueError):
    """Neplatný zmrazený zdroj, obrazový plán alebo výstup detektora."""


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            result.update(block)
    return result.hexdigest()


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


@contextmanager
def _no_bytecode():
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        yield
    finally:
        sys.dont_write_bytecode = previous


def _snapshot_path(frozen, manifest, relative):
    frozen = Path(frozen).resolve()
    path = (frozen / relative).resolve()
    if not path.is_relative_to(frozen) or relative not in manifest['copies']:
        raise RuntimeErrorV4('Súbor nepatrí overenej snímke: ' + relative)
    return path, manifest['copies'][relative]['sha256']


def _load_checked(frozen, manifest, relative, name, *, package=False):
    """Kompiluje presne overené bajty; nezapisuje pyc a nečíta druhú verziu."""
    path, expected = _snapshot_path(frozen, manifest, relative)
    body = path.read_bytes()
    if hashlib.sha256(body).hexdigest() != expected:
        raise RuntimeErrorV4('Zmenený kód pred importom: ' + relative)
    spec = importlib.util.spec_from_file_location(name, path,
                submodule_search_locations=[str(path.parent)] if package else None)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    exec(compile(body, str(path), 'exec'), module.__dict__)
    return module


@contextmanager
def _components(frozen, manifest):
    """Dočasné absolútne importy V4 sa vždy obnovia, aj pri chybe."""
    identity = hashlib.sha256(str(Path(frozen).resolve()).encode()).hexdigest()[:20]
    package = '_ablation_frozen_components_' + identity
    names = list(RUNTIME_MODULES) + [package] + [package + '.' + p for p in ('common', 'features', 'extract')]
    missing = object()
    with _PROCESS_LOCK, _no_bytecode():
        previous = {name: sys.modules.get(name, missing) for name in names}
        old_path = list(sys.path)
        try:
            modules = {}
            for name in RUNTIME_MODULES:
                modules[name] = _load_checked(frozen, manifest,
                    'comparison/detector_runtime/' + name + '.py', name)
            _load_checked(frozen, manifest, 'comparison/code/__init__.py', package, package=True)
            for part in ('common', 'features', 'extract'):
                modules[part] = _load_checked(frozen, manifest, 'comparison/code/' + part + '.py', package + '.' + part)
            yield modules
        finally:
            sys.path[:] = old_path
            for name, value in previous.items():
                if value is missing:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = value


@contextmanager
def _numerical_context():
    """Presná politika pôvodného loadera; globálny stav volajúceho sa obnoví."""
    with _PROCESS_LOCK:
        previous = (torch.get_num_threads(), torch.backends.cudnn.benchmark,
                    torch.backends.cudnn.deterministic, torch.backends.cuda.matmul.allow_tf32,
                    torch.backends.cudnn.allow_tf32, torch.are_deterministic_algorithms_enabled(),
                    torch.is_deterministic_algorithms_warn_only_enabled())
        try:
            if previous[0] != 4:
                torch.set_num_threads(4)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.use_deterministic_algorithms(True, warn_only=True)
            yield
        finally:
            if torch.get_num_threads() != previous[0]:
                torch.set_num_threads(previous[0])
            torch.backends.cudnn.benchmark = previous[1]
            torch.backends.cudnn.deterministic = previous[2]
            torch.backends.cuda.matmul.allow_tf32 = previous[3]
            torch.backends.cudnn.allow_tf32 = previous[4]
            torch.use_deterministic_algorithms(previous[5], warn_only=previous[6])


def _synchronize(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def _expected_pairs(expected):
    if expected is None:
        return None
    expected = np.asarray(expected, dtype=np.float64)
    if expected.shape != (6, 2) or not np.isfinite(expected).all():
        raise RuntimeErrorV4('Referenčná cache musí mať tvar 6 × 2 a konečné logity.')
    return expected


def _check_expected(actual, raw, expected):
    if expected is None:
        return None
    if not np.allclose(actual, expected, atol=2e-5, rtol=1e-6):
        raise RuntimeErrorV4('Natívne logity sa nezhodujú s historickou cache.')
    reference_raw = (expected[:, 1].astype(np.float32) - expected[:, 0].astype(np.float32)).astype(np.float64)
    if not np.allclose(raw, reference_raw, atol=2e-5, rtol=1e-6):
        raise RuntimeErrorV4('FP32 marginy sa nezhodujú s historickou cache.')
    if not np.array_equal(raw >= 0, reference_raw >= 0):
        raise RuntimeErrorV4('Zmenené rozhodnutie pôvodného alebo transformovaného pohľadu.')
    return {'max_abs_logit_difference': float(np.max(np.abs(actual - expected))),
            'max_abs_raw_margin_difference': float(np.max(np.abs(raw - reference_raw))),
            'all_six_view_decisions_equal': True, 'atol': 2e-5, 'rtol': 1e-6}


def _scores(features, parameters, calibrated):
    values = np.asarray(calibrated, dtype=np.float64).reshape(1, 6)
    full = features.representation(values, 'full')
    resize = features.representation(values, 'resize_only')
    if full.shape != (1, 9) or resize.shape != (1, 6) or not np.isfinite(full).all() or not np.isfinite(resize).all():
        raise RuntimeErrorV4('Nesprávne rozmery alebo hodnoty pôvodných príznakov.')
    scores = [float(features.logistic_score(parameters[name], x)[0])
              for name, x in (('frozen_v4', full), ('lr_resize_only', resize))]
    if any(not math.isfinite(v) or not 0 <= v <= 1 for v in scores):
        raise RuntimeErrorV4('Neplatné skóre zmrazeného selektora.')
    return scores


class AblationRuntime:
    """Samostatne testovateľné jadro; produkcia ho vytvára cez load_runtime."""
    def __init__(self, model, transform, apply_spec, views, fingerprint, features,
                 parameters, device, metadata):
        self.device = torch.device(device)
        if self.device.type not in ('cpu', 'cuda'):
            raise RuntimeErrorV4('Podporované zariadenia sú iba CPU a CUDA.')
        self.model = model.eval().requires_grad_(False)
        for tensor in list(self.model.parameters()) + list(self.model.buffers()):
            if tensor.is_floating_point() and tensor.dtype != torch.float32:
                raise RuntimeErrorV4('Detektor musí používať pôvodné FP32 váhy a buffery.')
        self.transform, self.apply_spec, self.views = transform, apply_spec, views
        self.fingerprint, self.features = fingerprint, features
        self.parameters = copy.deepcopy(parameters)
        self.metadata = copy.deepcopy(metadata)
        self.temperature = float(metadata['temperature'])
        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise RuntimeErrorV4('Neplatná zmrazená teplota.')

    def analyze(self, plan, expected_logits=None):
        """Vráti JSON údaje jedného obrázka; nikdy nemení jeho pôvodnú predikciu."""
        started = time.perf_counter()
        expected = _expected_pairs(expected_logits)
        if type(plan.get('label')) is not int or plan['label'] not in (0, 1):
            raise RuntimeErrorV4('Plán musí obsahovať pôvodnú triedu label=0 alebo 1.')
        if not isinstance(plan.get('input_spec'), dict):
            raise RuntimeErrorV4('Plán musí explicitne zachovať input_spec.')
        path = Path(plan['path']).resolve()
        raw_bytes = path.read_bytes()
        image_sha = hashlib.sha256(raw_bytes).hexdigest()
        if image_sha != plan.get('sha256'):
            raise RuntimeErrorV4('SHA256 obrázka nezodpovedá plánu: ' + str(path))
        try:
            # Pôvodný audit používa RGB bez EXIF, rozmery <QQ a pôvodný pHash.
            # Neexistuje fallback pre animácie/MPO ani chybné dekódovanie.
            fingerprints = self.fingerprint(path)
            if fingerprints['sha256'] != image_sha:
                raise RuntimeErrorV4('Obrázok sa zmenil medzi čítaním a auditom.')
            for key in ('rgb_sha256', 'phash'):
                if plan.get(key) not in (None, '') and plan[key] != fingerprints[key]:
                    raise RuntimeErrorV4('Odtlačok obrázka nezodpovedá plánu: ' + key)
            spec = copy.deepcopy(plan['input_spec'])
            with Image.open(io.BytesIO(raw_bytes)) as opened:
                opened.load()
                base = self.apply_spec(opened.convert('RGB'), copy.deepcopy(spec))
            if base.mode != 'RGB':
                raise RuntimeErrorV4('Vonkajší input_spec nevrátil RGB obrázok.')
            input_seconds = time.perf_counter() - started
            native, raw_margins, timings = [], [], []
            # Odmietneme nechcený prechod do train namiesto tichého zapnutia MC.
            if any(module.training for module in self.model.modules()):
                raise RuntimeErrorV4('Detektor alebo jeho vrstva opustili eval režim.')
            with _numerical_context(), torch.inference_mode(), torch.autocast(self.device.type, enabled=False):
                iterator = iter(self.views(base))
                for name in VIEW_ORDER:
                    view_started = time.perf_counter()
                    try:
                        image = next(iterator)
                    except StopIteration as error:
                        raise RuntimeErrorV4('Chýba pôvodný profilový pohľad.') from error
                    view_seconds = time.perf_counter() - view_started
                    before_input = time.perf_counter()
                    pixels = self.transform(image.convert('RGB')).unsqueeze(0).to(self.device)
                    size = self.metadata['image_size']
                    if pixels.shape != (1, 3, size, size) or pixels.dtype != torch.float32 or not torch.isfinite(pixels).all():
                        raise RuntimeErrorV4('Predspracovanie nevrátilo platný pôvodný FP32 tenzor.')
                    _synchronize(self.device)
                    preprocessing_seconds = time.perf_counter() - before_input
                    before_forward = time.perf_counter()
                    logits = self.model(pixels)
                    if logits.shape != (1, 2) or logits.dtype != torch.float32 or not torch.isfinite(logits).all():
                        raise RuntimeErrorV4('Detektor nevrátil dva konečné natívne FP32 logity.')
                    # Odčítať na zariadení v FP32, až potom preniesť do float64.
                    margin = logits[0, 1] - logits[0, 0]
                    native.append(logits[0].detach().cpu().numpy().astype(np.float64))
                    raw_margins.append(float(margin.item()))
                    _synchronize(self.device)
                    timings.append({'view': name, 'transform_seconds': view_seconds,
                                    'preprocessing_seconds': preprocessing_seconds,
                                    'forward_seconds': time.perf_counter() - before_forward})
                if next(iterator, None) is not None:
                    raise RuntimeErrorV4('Runtime vrátil viac než šesť pôvodných pohľadov.')
            native = np.asarray(native, dtype=np.float64)
            raw_margins = np.asarray(raw_margins, dtype=np.float64)
            if not np.isfinite(raw_margins).all():
                raise RuntimeErrorV4('FP32 rozdiel logitov nie je konečný.')
            check = _check_expected(native, raw_margins, expected)
            calibrated = raw_margins / self.temperature
            scoring_started = time.perf_counter()
            full_score, resize_score = _scores(self.features, self.parameters, calibrated)
            scoring_seconds = time.perf_counter() - scoring_started
            prediction = int(raw_margins[0] >= 0)
            result = {'native_logits': native.tolist(), 'raw_margins': raw_margins.tolist(),
                      'calibrated_margins': calibrated.tolist(), 'prediction': prediction,
                      'error': int(prediction != plan['label']), 'full_score': full_score,
                      'resize_score': resize_score, 'image_sha256': image_sha,
                      'rgb_sha256': fingerprints['rgb_sha256'], 'phash': fingerprints['phash'],
                      'input_spec': spec, 'cache_comparison': check,
                      'execution_counts': {'full_model_forwards': 6, 'mc_forwards': 0},
                      'timing': {'input_preparation_seconds': input_seconds, 'views': timings,
                                 'scoring_seconds': scoring_seconds}}
        finally:
            if sha(path) != image_sha:
                raise RuntimeErrorV4('Obrázok sa zmenil počas inferencie: ' + str(path))
        result['timing']['wall_seconds'] = time.perf_counter() - started
        return result


def _hardware(device):
    result = {'device': device, 'machine': platform.machine(), 'processor': platform.processor(),
              'torch_version': str(torch.__version__), 'torch_cuda_version': torch.version.cuda,
              'cudnn_version': torch.backends.cudnn.version()}
    if device == 'cuda':
        index = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(index)
        result['gpu'] = {'index': index, 'name': props.name, 'total_memory': props.total_memory,
                         'compute_capability': [props.major, props.minor],
                         'multi_processor_count': props.multi_processor_count,
                         'uuid': str(props.uuid) if getattr(props, 'uuid', None) is not None else None}
    return result


def load_runtime(frozen_models: Path, device='cuda') -> AblationRuntime:
    """Overí zdroje a načíta pôvodný checkpoint; nevykoná žiadny forward."""
    if device not in ('cpu', 'cuda') or (device == 'cuda' and not torch.cuda.is_available()):
        raise RuntimeErrorV4('Požadované CPU/CUDA zariadenie nie je dostupné.')
    frozen = Path(frozen_models).resolve()
    with _no_bytecode():
        from article_ablation_confirmation import freeze_models
        manifest = freeze_models.verify(frozen)
    source = _read(frozen / 'comparison/source_protocol.json')
    detector_run = Path(source['detector_run']).resolve()
    # Inicializácia náhodných váh pri build_model nesmie posunúť RNG volajúceho.
    with _numerical_context(), torch.random.fork_rng(devices=[]), _components(frozen, manifest) as modules:
        model, transform, info = modules['detector_training_v4'].load_frozen_detector(detector_run, device=device)
        if (info['model_sha256'] != manifest['detector']['model_sha256'] or
                info['temperature'] != manifest['detector']['temperature'] or
                info['temperature_artifact_sha256'] != manifest['copies']['detector/temperature.json']['sha256'] or
                info['class_to_index'] != {'real': 0, 'fake': 1} or info['image_size'] != 512 or
                info['config']['model'] != 'efficientnet_v2_m' or info['detector_threshold'] != .5 or
                str(Path(info['checkpoint_path']).resolve()) not in manifest['detector']['checkpoint_paths']):
            raise RuntimeErrorV4('Načítaný detektor, teplota alebo predspracovanie nesúhlasia so zmrazením.')
        if sha(info['checkpoint_path']) != info['model_sha256']:
            raise RuntimeErrorV4('Checkpoint sa zmenil počas načítania.')
        metadata = {'version': VERSION, 'frozen_models_manifest_sha256': sha(frozen / 'manifest.json'),
                    'model_sha256': info['model_sha256'], 'checkpoint_path': info['checkpoint_path'],
                    'temperature': info['temperature'], 'class_to_index': info['class_to_index'],
                    'architecture': 'efficientnet_v2_m', 'image_size': 512, 'view_order': list(VIEW_ORDER),
                    'feature_dimensions': {'frozen_v4': 9, 'lr_resize_only': 6},
                    'view_columns': {'frozen_v4': list(range(6)), 'lr_resize_only': [0, 4, 5]},
                    'original_prediction_rule': 'FP32(logit_fake - logit_real) >= 0',
                    'environment': manifest['environment'], 'hardware': _hardware(device),
                    'numerical_policy': NUMERICAL_POLICY,
                    'all_modules_eval': all(not m.training for m in model.modules()),
                    'dropout_enabled': False, 'batchnorm_training': False, 'stochastic_depth_training': False,
                    'fingerprint_version': modules['v4_deepshield_audit'].FINGERPRINT_VERSION,
                    'allow_mpo_primary': False,
                    'code_sha256': {relative: entry['sha256'] for relative, entry in manifest['copies'].items()
                                   if relative.startswith('comparison/') and relative.endswith('.py')},
                    'model_artifact_sha256': {relative: entry['sha256'] for relative, entry in manifest['copies'].items()
                                             if relative.startswith('models/')},
                    'runtime_sha256': sha(Path(__file__))}
        result = AblationRuntime(model, transform, modules['selector_augmentation_v4'].apply_spec,
                                modules['extract'].ours, modules['v4_deepshield_audit'].fingerprint,
                                modules['features'], {name: value['parameters'] for name, value in manifest['models'].items()},
                                device, metadata)
    return result
