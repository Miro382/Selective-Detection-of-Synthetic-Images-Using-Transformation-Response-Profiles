"""Samostatná adaptácia detektorov; spúšťa ju výhradne nový experiment runner.

CLIP ponecháva obrazový encoder zmrazený a učí autorskú lineárnu hlavu.
ResNet-50 po jednej zahrievacej epoche učí aj backbone, s fixnými BN štatistikami.
Tento modul nikdy nezapisuje do pôvodných experimentov ani nesťahuje aktíva.
"""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import io
import math
import os
from pathlib import Path
import random
import shutil
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from .common import ROOT, read, write, sha, digest, load_helper


NAMES = ('clip', 'resnet50')
CONDITIONS = ('clean', 'jpeg_q40', 'resize_0p5')
NORMALIZATION = {
    'clip': ((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    'resnet50': ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
}


def _context(run, name):
    run = Path(run).resolve()
    if name not in NAMES:
        raise ValueError('Neznámy detektor: ' + str(name))
    if not run.is_relative_to((ROOT / 'runs').resolve()) or run == (ROOT / 'runs').resolve():
        raise ValueError('Výstup musí byť samostatný podpriečinok article_detector_adaptation/runs.')
    protocol = read(run / 'protocol.json')
    fingerprint = sha(run / 'protocol.json')
    if (run / 'protocol.sha256').read_text(encoding='ascii').strip() != fingerprint:
        raise ValueError('Zmenený protokol adaptácie.')
    return run, protocol['config'], fingerprint


def _torch_write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    try:
        with tmp.open('wb') as stream:
            torch.save(value, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def _atomic_copy(source, target):
    target = Path(target)
    tmp = target.with_name(target.name + '.tmp')
    try:
        with Path(source).open('rb') as src, tmp.open('wb') as dst:
            shutil.copyfileobj(src, dst)
            dst.flush()
            os.fsync(dst.fileno())
        os.replace(tmp, target)
    finally:
        if tmp.exists():
            tmp.unlink()


def _module(path, name):
    # Asset files have already been verified against their locked manifest.
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _ClipDetector(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.fc = nn.Linear(768, 1)
        self.encoder.requires_grad_(False)
        self.encoder.eval()

    def forward(self, images):
        # No extra L2 normalization: this is the author detector's representation.
        with torch.no_grad():
            features = self.encoder.encode_image(images)
        return self.fc(features)


class EvaluationTransform:
    def __init__(self, name, image_size, resize_shorter):
        self.name, self.image_size, self.resize_shorter = name, int(image_size), int(resize_shorter)

    def __call__(self, image):
        image = TF.resize(image.convert('RGB'), self.resize_shorter, InterpolationMode.BICUBIC)
        image = TF.center_crop(image, [self.image_size, self.image_size])
        mean, std = NORMALIZATION[self.name]
        return TF.normalize(TF.to_tensor(image), mean, std)


def _build_initial(config, name):
    directory = Path(config['paths']['assets']).resolve() / name
    manifest = read(directory / 'manifest.json')
    for relative, entry in manifest['files'].items():
        path = (directory / relative).resolve()
        if not path.is_relative_to(directory) or not path.is_file() or sha(path) != entry['sha256']:
            raise ValueError('Chýbajúce alebo zmenené lokálne aktívum: ' + str(path))
    if name == 'clip':
        author = _module(directory / 'clip_model.py', 'adaptation_clip_model')
        archive = torch.jit.load(str(directory / 'ViT-L-14.pt'), map_location='cpu').eval()
        encoder = author.build_model(archive.state_dict()).float()
        del archive
        model = _ClipDetector(encoder)
        model.fc.load_state_dict(torch.load(directory / 'fc_weights.pth', map_location='cpu', weights_only=True), strict=True)
        trainable = 'linear_head_only; CLIP encoder frozen'
    else:
        author = _module(directory / 'resnet.py', 'adaptation_wang_resnet')
        model = author.resnet50(num_classes=1)
        state = torch.load(directory / 'blur_jpg_prob0.5.pth', map_location='cpu', weights_only=True)
        model.load_state_dict(state['model'], strict=True)
        trainable = 'head warm-up, then all parameters; BatchNorm running statistics frozen'
    preprocessing = dict(config['preprocessing'])
    if int(preprocessing['image_size']) != 224 or int(preprocessing['resize_shorter']) != 256:
        raise ValueError('Tento protokol adaptácie používa resize256/crop224.')
    mean, std = NORMALIZATION[name]
    preprocessing.update({'rgb': True, 'resize_interpolation': 'BICUBIC',
                          'mean': list(mean), 'std': list(std)})
    metadata = {'name': name, 'detector': name, 'class_to_index': {'real': 0, 'fake': 1},
                'output': 'raw_logit_fake', 'decision': 'fake iff raw_logit >= 0',
                'preprocessing': preprocessing, 'precision_at_training': 'float32',
                'precision_at_evaluation': 'float32',
                'tf32': False, 'trainable_policy': trainable,
                'asset_manifest_sha256': sha(directory / 'manifest.json'),
                'assets': {k: v['sha256'] for k, v in manifest['files'].items()}}
    transform = EvaluationTransform(name, preprocessing['image_size'], preprocessing['resize_shorter'])
    return model.float(), transform, metadata


def _device(config):
    device = torch.device(config['training']['device'])
    if device.type not in ('cpu', 'cuda'):
        raise ValueError('Podporované zariadenia sú cpu a cuda.')
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA nie je dostupná. Zmena zariadenia vyžaduje nový protokol.')
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    return device


def _model_state(model, name):
    source = model.fc if name == 'clip' else model
    return {key: value.detach().cpu().clone() for key, value in source.state_dict().items()}


def _restore_model(model, name, state):
    (model.fc if name == 'clip' else model).load_state_dict(state, strict=True)


def _verify_complete(run, name, protocol_sha):
    directory = run / name / 'detector'
    receipt = read(directory / 'complete.json')
    if receipt.get('version') != 'article-detector-adaptation-training-v1' or receipt.get('protocol_sha256') != protocol_sha or receipt.get('name') != name:
        raise ValueError('Dokončený detektor patrí inému protokolu.')
    for relative, expected in receipt['files'].items():
        path = (directory / relative).resolve()
        if not path.is_relative_to(directory.resolve()) or sha(path) != expected:
            raise ValueError('Zmenený dokončený artefakt detektora: ' + str(path))
    return receipt


class Adapter:
    def __init__(self, model, transform, metadata, device, batch_size):
        self.model = model.to(device).float().eval()
        self.model.requires_grad_(False)
        self.transform, self.metadata = transform, metadata
        self.device, self.batch_size = device, int(batch_size)
        if self.batch_size < 1:
            raise ValueError('Inferenčná dávka musí byť kladná.')

    def predict_logits(self, images):
        if not images:
            return np.empty(0, dtype=np.float64)
        values = []
        with torch.inference_mode():
            for start in range(0, len(images), self.batch_size):
                tensors = torch.stack([self.transform(image) for image in images[start:start+self.batch_size]]).to(self.device)
                logits = self.model(tensors).reshape(-1)
                if logits.numel() != len(tensors) or not torch.isfinite(logits).all():
                    raise FloatingPointError('Neplatné logity adaptovaného detektora.')
                values.extend(logits.double().cpu().tolist())
        return np.asarray(values, dtype=np.float64)


def load_adapter(run, name, checkpoint='best'):
    run, config, protocol_sha = _context(run, name)
    if checkpoint not in ('initial', 'best'):
        raise ValueError('Checkpoint musí byť initial alebo best.')
    receipt = _verify_complete(run, name, protocol_sha) if checkpoint == 'best' else None
    model, transform, metadata = _build_initial(config, name)
    if checkpoint == 'best':
        path = run / name / 'detector' / 'best.pt'
        state = torch.load(path, map_location='cpu', weights_only=True)
        if state['protocol_sha256'] != protocol_sha or state['metadata'] != metadata:
            raise ValueError('Checkpoint a aktuálne aktíva/predspracovanie sa nezhodujú.')
        _restore_model(model, name, state['model_state'])
        fingerprint = sha(path)
        metadata['selected_epoch'] = receipt['best_epoch']
    else:
        fingerprint = digest({'name': name, 'asset_manifest_sha256': metadata['asset_manifest_sha256'],
                              'preprocessing': metadata['preprocessing']})
    metadata.update({'checkpoint': checkpoint, 'checkpoint_fingerprint': fingerprint,
                     'protocol_sha256': protocol_sha})
    return Adapter(model, transform, metadata, _device(config), config['training']['batch_size'])


def _read_rows(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row['label'] = int(row['label'])
        if row['label'] not in (0, 1) or len(row['sha256']) != 64:
            raise ValueError('Neplatný manifest detektora.')
    if not rows or len({row['sample_id'] for row in rows}) != len(rows):
        raise ValueError('Prázdny alebo duplicitný manifest detektora.')
    return rows


def _image(row):
    # Decode the bytes whose hash was checked, not a second potentially changed read.
    data = Path(row['path']).read_bytes()
    if hashlib.sha256(data).hexdigest() != row['sha256']:
        raise ValueError('Zmenený tréningový/validačný obrázok: ' + row['path'])
    with Image.open(io.BytesIO(data)) as source:
        source.load()
        return source.convert('RGB')


def _jpeg(image, quality):
    buffer = io.BytesIO()
    image.save(buffer, format='JPEG', quality=int(quality), subsampling=0, optimize=False)
    buffer.seek(0)
    with Image.open(buffer) as decoded:
        decoded.load()
        return decoded.convert('RGB')


def _resize_restore(image, scale):
    shape = image.size
    reduced = tuple(max(1, round(side * scale)) for side in shape)
    return image.resize(reduced, Image.Resampling.LANCZOS).resize(shape, Image.Resampling.LANCZOS)


def _degrade(image, condition):
    if condition == 'clean':
        return image
    if condition == 'jpeg_q40':
        return _jpeg(image, 40)
    if condition == 'resize_0p5':
        return _resize_restore(image, 0.5)
    raise ValueError('Neznáma validačná podmienka.')


def _random_crop(image, size, rng):
    width, height = image.size
    area = width * height
    for _ in range(10):
        target = area * rng.uniform(0.85, 1.0)
        ratio = math.exp(rng.uniform(math.log(0.95), math.log(1.05)))
        w, h = round(math.sqrt(target * ratio)), round(math.sqrt(target / ratio))
        if 0 < w <= width and 0 < h <= height:
            top, left = rng.randint(0, height - h), rng.randint(0, width - w)
            break
    else:
        ratio = width / height
        if ratio < 0.95:
            w, h = width, round(width / 0.95)
        elif ratio > 1.05:
            h, w = height, round(height * 1.05)
        else:
            w, h = width, height
        top, left = (height-h)//2, (width-w)//2
    return TF.resized_crop(image, top, left, h, w, [size, size], InterpolationMode.BICUBIC)


def _training_transform(image, transform, rng):
    if rng.random() < 0.25:
        return transform(image)
    if rng.random() < 0.35:
        image = _resize_restore(image, rng.uniform(0.5, 0.9))
    if rng.random() < 0.4:
        image = ImageEnhance.Brightness(image).enhance(rng.uniform(0.8, 1.2))
        image = ImageEnhance.Contrast(image).enhance(rng.uniform(0.8, 1.2))
    if rng.random() < 0.1:
        image = image.filter(ImageFilter.GaussianBlur(rng.uniform(0.2, 0.8)))
    if rng.random() < 0.45:
        image = _jpeg(image, rng.randint(40, 95))
    image = TF.resize(image, transform.resize_shorter, InterpolationMode.BICUBIC)
    image = _random_crop(image, transform.image_size, rng)
    if rng.random() < 0.5:
        image = TF.hflip(image)
    mean, std = NORMALIZATION[transform.name]
    return TF.normalize(TF.to_tensor(image), mean, std)


class _TrainingRows(Dataset):
    def __init__(self, rows, indices, transform, seed, epoch):
        self.rows, self.indices, self.transform = rows, indices, transform
        self.seed, self.epoch = seed, epoch

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        row = self.rows[self.indices[index]]
        seed = int(hashlib.sha256(f'{self.seed}:augmentation:{self.epoch}:{index}'.encode()).hexdigest()[:16], 16)
        tensor = _training_transform(_image(row), self.transform, random.Random(seed))
        return tensor, row['label']


class _ValidationRows(Dataset):
    def __init__(self, rows, transform, condition):
        self.rows, self.transform, self.condition = rows, transform, condition

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        return self.transform(_degrade(_image(row), self.condition)), row['label']


def _loader(dataset, cfg, device, seed):
    return DataLoader(dataset, batch_size=int(cfg['batch_size']), shuffle=False,
                      num_workers=int(cfg['num_workers']), pin_memory=device.type == 'cuda',
                      persistent_workers=False, generator=torch.Generator().manual_seed(int(seed)),
                      **({'prefetch_factor': 2} if cfg['num_workers'] else {}))


def _metrics(rows, margins):
    from sklearn.metrics import roc_auc_score
    y = np.asarray([int(row['label']) for row in rows], dtype=np.int64)
    margins = np.asarray(margins, dtype=np.float64)
    if margins.shape != y.shape or not np.isfinite(margins).all() or set(y.tolist()) != {0, 1}:
        raise ValueError('Neplatné validačné predikcie.')
    pred, correct = margins >= 0, (margins >= 0) == y
    tree = {}
    for i, row in enumerate(rows):
        label = int(row['label'])
        level = row['domain'] if label == 0 else row['family']
        tree.setdefault(label, {}).setdefault(level, {}).setdefault(row['source'], {}).setdefault(row['generator'], []).append(bool(correct[i]))
    class_scores = {}
    for label, levels in tree.items():
        class_scores[str(label)] = float(np.mean([
            np.mean([np.mean([np.mean(values) for values in generators.values()])
                     for generators in sources.values()]) for sources in levels.values()]))
    tn, fp = int(((y == 0) & ~pred).sum()), int(((y == 0) & pred).sum())
    fn, tp = int(((y == 1) & ~pred).sum()), int(((y == 1) & pred).sum())
    return {'n': len(rows), 'errors': int((~correct).sum()), 'accuracy': float(correct.mean()),
            'balanced_accuracy': float(np.mean([correct[y == label].mean() for label in (0, 1)])),
            'confusion_real_fake': [[tn, fp], [fn, tp]],
            'fake_f1': 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
            'auroc': float(roc_auc_score(y, margins)),
            'log_loss': float(np.mean(np.logaddexp(0.0, margins)-y*margins)),
            'class_macro_accuracy': class_scores,
            'hierarchical_balanced_accuracy': float(np.mean(list(class_scores.values())))}


@torch.inference_mode()
def _validate(model, rows, transform, cfg, device):
    model.eval()
    metrics = {}
    for condition in CONDITIONS:
        print(f'  Validácia {condition}: {len(rows)} obrázkov', flush=True)
        values, last_log = [], time.monotonic()
        dataset = _ValidationRows(rows, transform, condition)
        for images, _ in _loader(dataset, cfg, device, cfg['seed']):
            logits = model(images.to(device, non_blocking=True)).reshape(-1)
            values.extend(logits.double().cpu().tolist())
            if time.monotonic() - last_log >= 30:
                print(f'    {len(values)}/{len(rows)}', flush=True)
                last_log = time.monotonic()
        metrics[condition] = _metrics(rows, values)
    return {'conditions': metrics,
            'selection_score': float(np.mean([v['hierarchical_balanced_accuracy'] for v in metrics.values()]))}


def _stage(model, name, epoch, cfg):
    head_only = name == 'clip' or epoch <= int(cfg['head_warmup_epochs'])
    model.train()
    if name == 'clip':
        model.encoder.requires_grad_(False)
        model.encoder.eval()
        model.fc.requires_grad_(True)
    else:
        for key, parameter in model.named_parameters():
            parameter.requires_grad_(not head_only or key.startswith('fc.'))
        if head_only:
            model.eval()
            model.fc.train()
        for module in model.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()
    return head_only


def _rng_state():
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def _restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if state['cuda'] is not None:
        torch.cuda.set_rng_state_all(state['cuda'])


def _train_epoch(model, name, data, cfg, epoch, optimizer, scaler, device):
    head_only = _stage(model, name, epoch, cfg)
    optimizer.zero_grad(set_to_none=True)
    total_loss = total_correct = total_seen = window_samples = windows = 0
    started = last_log = time.monotonic()
    for index, (images, labels) in enumerate(data):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).float()
        with torch.amp.autocast(device_type=device.type, enabled=False):
            logits = model(images).reshape(-1)
            loss = nn.functional.binary_cross_entropy_with_logits(logits, labels, reduction='sum')
        if not torch.isfinite(loss):
            raise FloatingPointError('Nefinitná tréningová strata; žiadne vstupy sa nepreskakujú.')
        scaler.scale(loss).backward()
        count = labels.numel()
        total_loss += float(loss.detach())
        total_correct += int(((logits >= 0) == labels.bool()).sum())
        total_seen += count
        window_samples += count
        if (index + 1) % int(cfg['accumulation_steps']) == 0 or index + 1 == len(data):
            scaler.unscale_(optimizer)
            for parameter in model.parameters():
                if parameter.grad is not None:
                    parameter.grad.div_(window_samples)
            nn.utils.clip_grad_norm_(model.parameters(), float(cfg.get('max_grad_norm', 1.0)),
                                     error_if_nonfinite=not scaler.is_enabled())
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            window_samples, windows = 0, windows + 1
        if time.monotonic() - last_log >= 30:
            print(f'  {name}, epocha {epoch}: {total_seen}/{len(data.dataset)}, strata {total_loss/total_seen:.5f}', flush=True)
            last_log = time.monotonic()
    return {'n': total_seen, 'loss': total_loss/total_seen, 'accuracy': total_correct/total_seen,
            'optimizer_windows': windows, 'seconds': time.monotonic()-started, 'head_only': head_only}


def train_detector(run, name):
    run, config, protocol_sha = _context(run, name)
    directory = run / name / 'detector'
    if (directory / 'complete.json').exists():
        return _verify_complete(run, name, protocol_sha)
    cfg = config['training']
    if cfg.get('precision') != 'float32':
        raise ValueError('Tento pripravený protokol používa tréning v FP32 bez AMP.')
    if cfg.get('label_smoothing', 0) != 0 or cfg.get('train_batchnorm', False):
        raise ValueError('Pripravený protokol vyžaduje smoothing0 a zmrazené BN štatistiky.')
    train_path, val_path = run / 'plans/detector_train.csv', run / 'plans/detector_validation.csv'
    contract = {'name': name, 'protocol_sha256': protocol_sha, 'implementation_sha256': sha(Path(__file__)),
                'training_plan_sha256': sha(train_path), 'validation_plan_sha256': sha(val_path)}
    if (directory / 'contract.json').exists():
        if read(directory / 'contract.json') != contract:
            raise ValueError('Beh detektora sa zmenil; pôvodný výstup sa neprepíše.')
    else:
        if directory.exists() and any(directory.iterdir()):
            raise ValueError('Neznámy existujúci výstup detektora; zvoľte nový beh.')
        directory.mkdir(parents=True, exist_ok=True)
        write(directory / 'contract.json', contract)
    rows, validation_rows = _read_rows(train_path), _read_rows(val_path)
    if len(rows) != 104706 or len(validation_rows) != 4880:
        raise ValueError('Plný zhodný protokol očakáva 104706 tréningových a 4880 validačných vstupov.')
    random.seed(int(cfg['seed']))
    np.random.seed(int(cfg['seed']) % 2**32)
    torch.manual_seed(int(cfg['seed']))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(cfg['seed']))
    device = _device(config)
    model, transform, metadata = _build_initial(config, name)
    model.to(device)
    groups = [{'params': list(model.fc.parameters()), 'lr': float(cfg['head_lr'])}]
    if name == 'resnet50':
        groups.append({'params': [p for key, p in model.named_parameters() if not key.startswith('fc.')],
                       'lr': float(cfg['backbone_lr'])})
    optimizer = torch.optim.AdamW(groups, weight_decay=float(cfg['weight_decay']))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(cfg['epochs']))
    scaler = torch.amp.GradScaler('cuda', enabled=False)
    sampler_class = load_helper(run, 'sampler_v4').CappedBalancedSampler
    history, best_score, best_epoch, bad_epochs, completed_epoch = [], -float('inf'), 0, 0, 0
    best_relative, best_sha = None, None

    def save_best(epoch, validation):
        relative = f'checkpoints/best_epoch_{epoch:03d}.pt'
        value = {'version': 'adapted-detector-checkpoint-v1', 'name': name,
                 'protocol_sha256': protocol_sha, 'metadata': metadata,
                 'epoch': epoch, 'validation': validation, 'model_state': _model_state(model, name)}
        _torch_write(directory / relative, value)
        return relative, sha(directory / relative)

    def save_last(epoch):
        value = {'version': 'adapted-detector-resume-v1', 'contract': contract, 'metadata': metadata,
                 'epoch': epoch, 'model_state': _model_state(model, name),
                 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
                 'scaler': scaler.state_dict(), 'rng': _rng_state(), 'history': history,
                 'best_score': best_score, 'best_epoch': best_epoch, 'bad_epochs': bad_epochs,
                 'best_relative': best_relative, 'best_sha256': best_sha,
                 'initial_validation_sha256': sha(directory / 'initial_validation.json')}
        _torch_write(directory / 'last.pt', value)
        write(directory / 'history.json', history)

    if (directory / 'last.pt').exists():
        # Trusted checkpoint generated only inside this protocol's private run.
        state = torch.load(directory / 'last.pt', map_location='cpu', weights_only=False)
        if state['contract'] != contract or state['metadata'] != metadata:
            raise ValueError('Obnova odmietnutá: iný protokol, kód alebo počiatočné aktíva.')
        if sha(directory / 'initial_validation.json') != state['initial_validation_sha256']:
            raise ValueError('Zmenená počiatočná validácia.')
        best_relative, best_sha = state['best_relative'], state['best_sha256']
        best_path = (directory / best_relative).resolve()
        if not best_path.is_relative_to(directory.resolve()) or sha(best_path) != best_sha:
            raise ValueError('Chýba alebo sa zmenil vybraný generačný checkpoint.')
        _restore_model(model, name, state['model_state'])
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        scaler.load_state_dict(state['scaler'])
        _restore_rng(state['rng'])
        history, best_score, best_epoch, bad_epochs = state['history'], state['best_score'], state['best_epoch'], state['bad_epochs']
        completed_epoch = int(state['epoch'])
        write(directory / 'history.json', history)
        print(f'{name}: obnova po dokončenej epoche {completed_epoch}.', flush=True)
    else:
        print(f'{name}: počiatočný verejný checkpoint, nové zhodné predspracovanie.', flush=True)
        initial = _validate(model, validation_rows, transform, cfg, device)
        write(directory / 'initial_validation.json', {'protocol_sha256': protocol_sha,
              'metadata': metadata, 'validation': initial})
        best_score = initial['selection_score']
        best_relative, best_sha = save_best(0, initial)
        save_last(0)

    if bad_epochs < int(cfg['patience']):
        for epoch in range(completed_epoch+1, int(cfg['epochs'])+1):
            sampler = sampler_class(rows, int(cfg['samples_per_epoch']), int(cfg['max_group_repeats']), int(cfg['seed'])+epoch)
            indices = list(sampler)
            report = dict(sampler.report)
            report['ordered_sample_ids_sha256'] = digest([rows[i]['sample_id'] for i in indices])
            write(directory / f'sampling_epoch_{epoch:03d}.json', report)
            dataset = _TrainingRows(rows, indices, transform, int(cfg['seed']), epoch)
            loader = _loader(dataset, cfg, device, int(cfg['seed'])+epoch)
            train = _train_epoch(model, name, loader, cfg, epoch, optimizer, scaler, device)
            validation = _validate(model, validation_rows, transform, cfg, device)
            scheduler.step()
            improved = validation['selection_score'] > best_score + float(cfg['min_delta'])
            if improved:
                best_score, best_epoch, bad_epochs = validation['selection_score'], epoch, 0
                best_relative, best_sha = save_best(epoch, validation)
            else:
                bad_epochs += 1
            history.append({'epoch': epoch, 'training': train, 'validation': validation,
                            'improved': improved, 'sampling': report})
            save_last(epoch)
            print(f'{name}, epocha {epoch}: skóre {validation["selection_score"]:.6f}; najlepšie {best_score:.6f}, epocha {best_epoch}.', flush=True)
            if bad_epochs >= int(cfg['patience']):
                break
    _atomic_copy(directory / best_relative, directory / 'best.pt')
    best_payload = torch.load(directory / 'best.pt', map_location='cpu', weights_only=True)
    write(directory / 'best_validation.json', {'protocol_sha256': protocol_sha,
          'epoch': best_epoch, 'validation': best_payload['validation']})
    files = ['contract.json', 'best.pt', 'last.pt', 'history.json', 'initial_validation.json', 'best_validation.json', best_relative]
    files.extend(f'sampling_epoch_{entry["epoch"]:03d}.json' for entry in history)
    receipt = {'version': 'article-detector-adaptation-training-v1', 'name': name,
               'protocol_sha256': protocol_sha, 'metadata': metadata, 'best_epoch': best_epoch,
               'best_selection_score': best_score, 'epochs_completed': len(history),
               'initial_selection_score': read(directory / 'initial_validation.json')['validation']['selection_score'],
               'scope': 'Adaptive development validation; no test data used for training or checkpoint selection.',
               'files': {relative: sha(directory / relative) for relative in files}}
    write(directory / 'complete.json', receipt)
    del optimizer, scheduler, scaler, model
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return receipt
