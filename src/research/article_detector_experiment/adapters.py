"""Zmrazené autorské detektory so spoločným výstupom: surový logit fake."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch
from torchvision import transforms

ROOT = Path(__file__).resolve().parent
ALIASES = {'clip': 'clip', 'ojha': 'clip', 'resnet50': 'resnet50', 'wang': 'resnet50'}


def _sha256(path):
    checksum = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            checksum.update(block)
    return checksum.hexdigest()


def _module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _verify_assets(directory):
    manifest = json.loads((directory / 'manifest.json').read_text(encoding='utf-8'))
    for name, item in manifest['files'].items():
        path = directory / name
        if not path.is_file() or _sha256(path) != item['sha256']:
            raise RuntimeError('Nesúhlasí uzamknutý artefakt: ' + str(path))
    return manifest


class _CLIPHead(torch.nn.Module):
    """Presný výpočet autorského CLIPModel.forward, bez L2 normalizácie."""
    def __init__(self, encoder):
        super().__init__()
        self.model = encoder
        self.fc = torch.nn.Linear(768, 1)

    def forward(self, x):
        return self.fc(self.model.encode_image(x))


class DetectorAdapter:
    def __init__(self, model, transform, metadata, device, batch_size):
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.model.requires_grad_(False)
        self.transform = transform
        self.metadata = metadata
        self.batch_size = batch_size

    def predict_logits(self, images):
        """Vráti logity v zadanom poradí, bez sigmoid a bez zmeny obrázkov."""
        if not images:
            return np.empty(0, dtype=np.float32)
        if self.batch_size < 1:
            raise ValueError('Veľkosť dávky musí byť kladná.')
        outputs = []
        with torch.inference_mode():
            for start in range(0, len(images), self.batch_size):
                tensor = torch.stack([
                    self.transform(image.convert('RGB'))
                    for image in images[start:start + self.batch_size]
                ]).to(self.device)
                values = self.model(tensor).reshape(-1)
                if len(values) != tensor.shape[0] or not torch.isfinite(values).all():
                    raise RuntimeError('Neplatný výstup detektora.')
                outputs.append(values.float().cpu().numpy())
        return np.concatenate(outputs)


def load_detector(name, assets_dir=ROOT / 'assets', device='cuda', batch_size=16):
    """Načíta iba overené lokálne aktíva; nikdy nesťahuje do globálnej cache."""
    if name not in ALIASES:
        raise ValueError('Neznámy detektor: ' + name)
    name = ALIASES[name]
    if batch_size < 1:
        raise ValueError('Veľkosť dávky musí byť kladná.')
    if str(device).startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('Požadované CUDA nie je dostupné; CPU treba zvoliť explicitne.')
    directory = Path(assets_dir).resolve() / name
    manifest = _verify_assets(directory)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    if name == 'clip':
        # Ojha CLIPModel volá clip.load(..., device='cpu'): ten build_model
        # prevedie na float32, až následne sa celý detektor presunie na CUDA.
        author = _module(directory / 'clip_model.py', 'article_ojha_clip_model')
        archive = torch.jit.load(str(directory / 'ViT-L-14.pt'), map_location='cpu').eval()
        encoder = author.build_model(archive.state_dict()).float()
        del archive
        model = _CLIPHead(encoder)
        model.fc.load_state_dict(torch.load(directory / 'fc_weights.pth', map_location='cpu', weights_only=True), strict=True)
        mean = (0.48145466, 0.4578275, 0.40821073)
        std = (0.26862954, 0.26130258, 0.27577711)
        architecture = 'CLIP ViT-L/14 + autorská lineárna hlava 768→1'
    else:
        author = _module(directory / 'resnet.py', 'article_wang_resnet')
        model = author.resnet50(num_classes=1)
        checkpoint = torch.load(directory / 'blur_jpg_prob0.5.pth', map_location='cpu', weights_only=True)
        model.load_state_dict(checkpoint['model'], strict=True)
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        architecture = 'ResNet-50 CNNDetection Blur+JPEG(0.5)'
    transform = transforms.Compose([
        transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(mean, std),
    ])
    metadata = {
        'version': 'article-detector-adapter-v1', 'name': name,
        'architecture': architecture, 'output': 'raw_logit_fake',
        'class_order': {'real': 0, 'fake': 1}, 'decision_threshold_logit': 0.0,
        'preprocessing': {'rgb': True, 'resize': None, 'center_crop': [224, 224],
                          'mean': list(mean), 'std': list(std)},
        'precision': 'float32', 'tf32': False, 'device': str(device),
        'torch_version': torch.__version__, 'batch_size': batch_size,
        'asset_manifest_sha256': _sha256(directory / 'manifest.json'),
        'assets': {key: {'sha256': value['sha256'], 'url': value['url'], 'commit': value['commit']}
                   for key, value in manifest['files'].items()},
    }
    return DetectorAdapter(model, transform, metadata, device, batch_size)
