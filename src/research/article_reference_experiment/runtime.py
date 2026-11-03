"""Natívne MC dropout zmrazeného EfficientNetV2-M bez učenia a zápisov.

Backbone sa počíta raz. Tridsať vzoriek opakuje iba pôvodnú klasifikačnú
hlavu s jej dvoma trénovanými Dropout vrstvami. BN a StochasticDepth ostávajú
v eval; rozhodnutie detektora ostáva rozhodnutím deterministickej inferencie.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import importlib.util
import io
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image
import torch
from torch import nn
from torchvision.ops import StochasticDepth

sys.dont_write_bytecode = True
NATIVE_DROPOUT = {'classifier.0': .3, 'classifier.3': .15}
MODULES = ('v4_data_sources', 'prepare_data_v4', 'sampler_v4',
           'detector_training_v4', 'selector_augmentation_v4')


def sha(path):
    checksum = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            checksum.update(block)
    return checksum.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def dropout_inventory(model):
    return [{'name': name, 'type': type(module).__name__, 'p': float(module.p),
             'inplace': bool(module.inplace), 'training': bool(module.training)}
            for name, module in model.named_modules()
            if isinstance(module, nn.modules.dropout._DropoutNd)]


def _active_dropout(model):
    found = [(name, module) for name, module in model.named_modules()
             if isinstance(module, nn.modules.dropout._DropoutNd) and module.p > 0]
    if not found:
        raise ValueError('Model nemá natívny dropout s p > 0; nový dropout sa nevkladá.')
    if any(not name.startswith('classifier.') for name, _ in found):
        raise ValueError('Cache backbone podporuje iba dropout v klasifikačnej hlave.')
    if any(not 0 < module.p < 1 for _, module in found):
        raise ValueError('Dropout vyžaduje pravdepodobnosť v otvorenom intervale (0, 1).')
    for name in ('features', 'avgpool', 'classifier'):
        if not isinstance(getattr(model, name, None), nn.Module):
            raise ValueError('Model nepodporuje rozklad features/avgpool/classifier.')
    return found


def _synchronize(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def _logits(value, name):
    if value.shape != (1, 2) or not torch.isfinite(value).all():
        raise ValueError(name + ' musí vrátiť dva konečné prirodzené logity.')
    return value[0].float().detach().cpu().numpy().astype(np.float64)


def _check_expected(actual, expected):
    expected = np.asarray(expected, dtype=np.float64)
    if expected.shape != (2,) or not np.isfinite(expected).all():
        raise ValueError('Referenčná cache musí obsahovať dva konečné logity.')
    if not np.allclose(actual, expected, rtol=1e-6, atol=2e-5):
        raise ValueError('Deterministické logity sa nezhodujú s existujúcou cache.')
    # Pôvodný EfficientNet odčíta dvojicu v FP32, nie až po prenose do float64.
    actual_margin = np.float32(actual[1]) - np.float32(actual[0])
    expected_margin = np.float32(expected[1]) - np.float32(expected[0])
    if bool(actual_margin >= 0) != bool(expected_margin >= 0):
        raise ValueError('Dávka alebo prostredie zmenili pôvodné rozhodnutie detektora.')
    return float(np.max(np.abs(actual - expected)))


def sample_cached_head(model, pixels, expected_logits, seed, samples=30):
    """Samostatne testovateľné jadro; používa zariadenie explicitného tenzora.

Po úspechu aj výnimke obnoví pôvodné training príznaky a CPU/CUDA RNG.
U produkčného loadera sú pôvodné príznaky vždy eval. Nerobí model.forward,
augmentáciu, optimalizáciu ani žiadny zápis na disk.
"""
    if type(samples) is not int or samples < 2:
        raise ValueError('Počet MC vzoriek musí byť celé číslo aspoň 2.')
    if type(seed) is not int or not 0 <= seed < 2**64:
        raise ValueError('Seed musí byť celé číslo v rozsahu uint64.')
    if pixels.ndim != 4 or pixels.shape[0] != 1:
        raise ValueError('Runtime vyžaduje práve jeden obrazový vstup NCHW.')
    if not torch.is_floating_point(pixels) or not torch.isfinite(pixels).all():
        raise ValueError('Obrazový tenzor musí obsahovať konečné reálne hodnoty.')
    dropouts = _active_dropout(model)
    original_flags = [(module, bool(module.training)) for module in model.modules()]
    device = pixels.device
    if device.type not in ('cpu', 'cuda'):
        raise ValueError('Runtime podporuje iba CPU alebo CUDA.')
    cuda_devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == 'cuda' else []
    started = time.perf_counter()
    try:
        model.eval()
        with torch.random.fork_rng(devices=cuda_devices), torch.inference_mode():
            # torch.manual_seed by menil aj iné CUDA generátory mimo fork_rng.
            torch.random.default_generator.manual_seed(seed)
            if cuda_devices:
                torch.cuda.default_generators[cuda_devices[0]].manual_seed(seed)
            _synchronize(device)
            before = time.perf_counter()
            features = torch.flatten(model.avgpool(model.features(pixels)), 1)
            _synchronize(device)
            backbone_seconds = time.perf_counter() - before
            before = time.perf_counter()
            deterministic = _logits(model.classifier(features), 'Deterministická hlava')
            _synchronize(device)
            deterministic_head_seconds = time.perf_counter() - before
            max_difference = _check_expected(deterministic, expected_logits)

            for _, dropout in dropouts:
                dropout.train(True)
            active_ids = {id(module) for _, module in dropouts}
            if any(module.training and id(module) not in active_ids for module in model.modules()):
                raise RuntimeError('Aktivovala sa iná vrstva než natívny dropout.')
            _synchronize(device)
            before = time.perf_counter()
            draws = []
            for _ in range(samples):
                # Jednotlivé forwardy zodpovedajú jednému vzorkovaniu masky na obraz.
                value = model.classifier(features.clone())
                if value.shape != (1, 2) or not torch.isfinite(value).all():
                    raise ValueError('MC hlava nevrátila dva konečné logity.')
                draws.append(value[0].float().clone())
            mc_logits = torch.stack(draws).cpu().numpy().astype(np.float64)
            _synchronize(device)
            mc_head_seconds = time.perf_counter() - before
        return {'deterministic_logits': deterministic.tolist(), 'mc_logits': mc_logits.tolist(),
                'seed': seed, 'samples': samples, 'max_abs_difference_from_cache': max_difference,
                'execution_counts': {'backbone_forwards': 1, 'deterministic_head_forwards': 1,
                                     'mc_head_forwards': samples, 'total_head_forwards': samples + 1,
                                     'full_model_forwards': 0},
                'timing': {'backbone_seconds': backbone_seconds,
                           'deterministic_head_seconds': deterministic_head_seconds,
                           'mc_head_seconds': mc_head_seconds,
                           'model_wall_seconds': time.perf_counter() - started},
                'prediction_rule': 'Pôvodné deterministické rozhodnutie: FP32(z_fake-z_real) >= 0.',
                'mc_training_modules': [name for name, _ in dropouts]}
    finally:
        # Priama obnova zachová aj pôvodné zmiešané flags, bez rekurzívneho train().
        for module, training in original_flags:
            module.training = training


@contextmanager
def _frozen_modules(comparison_run, protocol):
    code = comparison_run / 'detector_runtime'
    previous = {name: sys.modules.get(name) for name in MODULES}
    old_path = list(sys.path)
    loaded = {}
    try:
        sys.path.insert(0, str(code))
        for name in MODULES:
            path = code / (name + '.py')
            relative = 'detector_runtime/' + name + '.py'
            if sha(path) != protocol['files'][relative]:
                raise ValueError('Zmenený zmrazený runtime: ' + relative)
            spec = importlib.util.spec_from_file_location(name, path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
            loaded[name] = module
        yield loaded
    finally:
        sys.path[:] = old_path
        for name, module in previous.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


class DropoutRuntime:
    def __init__(self, model, transform, apply_spec, device, metadata):
        self.model = model.eval().requires_grad_(False)
        self.transform = transform
        self.apply_spec = apply_spec
        self.device = torch.device(device)
        self.metadata = metadata

    def analyze(self, plan, expected_logits, seed, samples=30):
        started = time.perf_counter()
        path = Path(plan['path'])
        raw = path.read_bytes()
        image_sha = hashlib.sha256(raw).hexdigest()
        if image_sha != plan['sha256']:
            raise ValueError('Obrázok nezodpovedá plánu: ' + str(path))
        try:
            with Image.open(io.BytesIO(raw)) as image:
                image.load()
                # Zachovať vonkajší variant V4, nie vždy načítať čistý originál.
                base = self.apply_spec(image.convert('RGB'), plan['input_spec'])
            pixels = self.transform(base).unsqueeze(0).to(self.device)
            _synchronize(self.device)
            input_seconds = time.perf_counter() - started
            result = sample_cached_head(self.model, pixels, expected_logits, seed, samples)
        finally:
            if sha(path) != image_sha:
                raise ValueError('Obrázok sa zmenil počas inferencie: ' + str(path))
        result['image_sha256'] = image_sha
        result['timing']['input_preparation_seconds'] = input_seconds
        result['timing']['wall_seconds'] = time.perf_counter() - started
        return result


def load_runtime(comparison_run: Path, protocol: dict, device='cuda'):
    """Načíta pôvodný V4 model zo snapshotu comparison, nič nedoúča."""
    comparison_run = Path(comparison_run).resolve()
    if device not in ('cpu', 'cuda'):
        raise ValueError('Zariadenie musí byť cpu alebo cuda.')
    if device == 'cuda' and not torch.cuda.is_available():
        raise ValueError('Požadované CUDA nie je dostupné.')
    source_path = comparison_run / 'source_protocol.json'
    if sha(source_path) != protocol['files']['source_protocol.json']:
        raise ValueError('Zmenený zdrojový protokol EfficientNetu.')
    source = read(source_path)
    if (source['model_sha256'] != protocol['model_sha256']
            or source['temperature'] != protocol['temperature']):
        raise ValueError('Comparison používa iný model alebo teplotu než zdrojový protokol.')
    with torch.random.fork_rng(devices=[]), _frozen_modules(comparison_run, protocol) as modules:
        detector = modules['detector_training_v4']
        model, transform, info = detector.load_frozen_detector(Path(source['detector_run']), device=device)
        apply_spec = modules['selector_augmentation_v4'].apply_spec
    model.eval().requires_grad_(False)
    if (info['model_sha256'] != protocol['model_sha256']
            or info['temperature'] != protocol['temperature']
            or info['class_to_index'] != {'real': 0, 'fake': 1}
            or info['image_size'] != 512 or info['config']['model'] != 'efficientnet_v2_m'
            or not math.isfinite(info['temperature']) or info['temperature'] <= 0):
        raise ValueError('Načítaný model, teplota, predspracovanie alebo poradie tried nezodpovedajú protokolu.')
    inventory = dropout_inventory(model)
    _active_dropout(model)
    if {r['name']: r['p'] for r in inventory} != NATIVE_DROPOUT:
        raise ValueError('Zmrazený EfficientNet nemá očakávané dva natívne dropout moduly.')
    metadata = {'version': 'article-reference-native-mcd-runtime-v1',
                'model_sha256': info['model_sha256'], 'checkpoint_path': info['checkpoint_path'],
                'source_protocol_sha256': sha(source_path),
                'loader_sha256': sha(comparison_run / 'detector_runtime/detector_training_v4.py'),
                'augmentation_sha256': sha(comparison_run / 'detector_runtime/selector_augmentation_v4.py'),
                'architecture': 'efficientnet_v2_m', 'image_size': 512, 'device': device,
                'temperature': info['temperature'], 'class_to_index': info['class_to_index'],
                'raw_output': '[z_real, z_fake], FP32, bez aplikovania teploty',
                'dropout_inventory': inventory,
                'batchnorm_count': sum(isinstance(m, nn.modules.batchnorm._BatchNorm) for m in model.modules()),
                'stochastic_depth_count': sum(isinstance(m, StochasticDepth) for m in model.modules()),
                'all_modules_eval_at_rest': all(not m.training for m in model.modules()),
                'parameters_require_grad': any(p.requires_grad for p in model.parameters()),
                'batchnorm_during_mc': 'eval', 'stochastic_depth_during_mc': 'eval',
                'feature_cache': 'features -> avgpool -> flatten, raz na vstup',
                'mc_scope': 'Iba pôvodné dva trénované Dropout moduly hlavy; bez vloženia nových vrstiev.',
                'rng_policy': 'fork_rng CPU a použitého CUDA zariadenia; osobitný seed vzorky; obnova aj po chybe.'}
    return DropoutRuntime(model, transform, apply_spec, device, metadata)
