"""Príprava pripnutých autorských zdrojov a váh iba pre nový článkový experiment."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parent
OJHA_COMMIT = '030495aea3300a8b54c0ec37ec7fe1dd7e63c619'
WANG_COMMIT = 'ea0b5622365e3a9cd31d1b54b6b5971131a839ab'
CLIP_SHA256 = 'b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836'


def specs(detector):
    if detector == 'clip':
        base = f'https://raw.githubusercontent.com/WisconsinAIVision/UniversalFakeDetect/{OJHA_COMMIT}/'
        return [
            ('clip_model.py', base + 'models/clip/model.py', OJHA_COMMIT, None),
            ('clip_models.py', base + 'models/clip_models.py', OJHA_COMMIT, None),
            ('clip_loader.py', base + 'models/clip/clip.py', OJHA_COMMIT, None),
            ('validate.py', base + 'validate.py', OJHA_COMMIT, None),
            ('LICENSE', base + 'LICENSE', OJHA_COMMIT, None),
            ('fc_weights.pth', base + 'pretrained_weights/fc_weights.pth', OJHA_COMMIT, None),
            ('ViT-L-14.pt', f'https://openaipublic.azureedge.net/clip/models/{CLIP_SHA256}/ViT-L-14.pt', None, CLIP_SHA256),
        ]
    if detector == 'resnet50':
        base = f'https://raw.githubusercontent.com/PeterWang512/CNNDetection/{WANG_COMMIT}/'
        return [
            ('resnet.py', base + 'networks/resnet.py', WANG_COMMIT, None),
            ('datasets.py', base + 'data/datasets.py', WANG_COMMIT, None),
            ('eval.py', base + 'eval.py', WANG_COMMIT, None),
            ('validate.py', base + 'validate.py', WANG_COMMIT, None),
            ('LICENSE.txt', base + 'LICENSE.txt', WANG_COMMIT, None),
            ('download_weights.sh', base + 'weights/download_weights.sh', WANG_COMMIT, None),
            ('blur_jpg_prob0.5.pth', 'https://www.dropbox.com/s/2g2jagq2jn1fd0i/blur_jpg_prob0.5.pth?dl=1', None, None),
        ]
    raise ValueError('Neznámy detektor: ' + detector)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def prepare(detector, assets_dir=ROOT / 'assets'):
    directory = Path(assets_dir).resolve() / detector
    directory.mkdir(parents=True, exist_ok=True)
    manifest_path = directory / 'manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8')) if manifest_path.exists() else {
        'version': 'article-detector-assets-v1', 'detector': detector, 'files': {},
    }
    for name, url, commit, expected in specs(detector):
        target = directory / name
        prior = manifest['files'].get(name)
        if prior is not None:
            if prior['url'] != url or not target.is_file() or sha256(target) != prior['sha256']:
                raise RuntimeError('Zmenený alebo chýbajúci uzamknutý súbor: ' + str(target))
            if expected and prior['sha256'] != expected:
                raise RuntimeError('Nesúhlasí oficiálny odtlačok CLIP.')
            print('Overené: ' + name, flush=True)
            continue
        if target.exists():
            raise RuntimeError('Neoverený existujúci súbor sa neprepisuje: ' + str(target))
        temporary = target.with_name(target.name + '.part')
        # Čiastočný prenos nepredstavuje platný artefakt; opakovanie ho začne nanovo.
        request = urllib.request.Request(url, headers={'User-Agent': 'article-detector-experiment/1'})
        print('Sťahujem: ' + url, flush=True)
        with urllib.request.urlopen(request, timeout=60) as response, temporary.open('wb') as out:
            actual_url = response.geturl()
            content_type = response.headers.get('Content-Type', '')
            if name.endswith(('.pt', '.pth')) and 'text/html' in content_type:
                raise RuntimeError('Namiesto checkpointu server vrátil HTML: ' + url)
            for block in iter(lambda: response.read(1024 * 1024), b''):
                out.write(block)
        checksum = sha256(temporary)
        if expected and checksum != expected:
            raise RuntimeError('Nesúhlasí SHA256: ' + name)
        temporary.replace(target)
        manifest['files'][name] = {
            'url': url, 'resolved_url': actual_url, 'commit': commit,
            'sha256': checksum, 'bytes': target.stat().st_size,
            'retrieved_utc': datetime.now(timezone.utc).isoformat(),
            'upstream_expected_sha256': expected,
        }
        pending = manifest_path.with_suffix('.json.tmp')
        pending.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        os.replace(pending, manifest_path)
        print(f'Uložené: {name} ({target.stat().st_size:,} bajtov)', flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--detector', choices=['clip', 'resnet50', 'all'], default='clip')
    parser.add_argument('--assets-dir', type=Path, default=ROOT / 'assets')
    args = parser.parse_args()
    for name in (['clip', 'resnet50'] if args.detector == 'all' else [args.detector]):
        prepare(name, args.assets_dir)


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
