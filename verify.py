"""Check release integrity and reproduce the packaged numerical analyses."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / 'release_manifest.json'
EXCLUDED_PARTS = {'.git', '.venv', 'venv', '__pycache__', '.pytest_cache',
                  '.mplconfig', 'outputs', 'work'}
FORBIDDEN_SUFFIXES = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tif', '.tiff',
                      '.heic', '.pt', '.pth', '.ckpt', '.safetensors', '.onnx',
                      '.pkl', '.pickle', '.joblib', '.zip', '.exe', '.dll'}
TEXT_SUFFIXES = {'.py', '.json', '.jsonl', '.csv', '.txt', '.md', '.tex', '.bib',
                 '.yml', '.yaml', '.cff', '.toml', '.ps1', '.sh'}
SECRET_PATTERNS = [re.compile(p) for p in (
    r'\bhf_[A-Za-z0-9]{25,}\b', r'\bgh[pousr]_[A-Za-z0-9]{30,}\b',
    r'\bgithub_pat_[A-Za-z0-9_]{40,}\b', r'\bAKIA[0-9A-Z]{16}\b',
    r'-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----',
    r'\bsk-(?:proj-)?[A-Za-z0-9_-]{32,}\b',
)]


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def inventory():
    return {p.relative_to(ROOT).as_posix(): p for p in ROOT.rglob('*')
            if p.is_file() and not (set(p.relative_to(ROOT).parts) & EXCLUDED_PARTS)
            and p != MANIFEST and p.suffix not in {'.pyc', '.pyo'}}


def integrity():
    if not MANIFEST.is_file():
        raise ValueError('Missing release_manifest.json.')
    manifest = json.loads(MANIFEST.read_text(encoding='utf-8'))
    actual = inventory()
    expected = manifest['files']
    if set(actual) != set(expected):
        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        raise ValueError(f'File inventory differs: missing={missing}, extra={extra}')
    total = 0
    for name, item in expected.items():
        path = actual[name]
        if path.is_symlink() or not path.resolve().is_relative_to(ROOT):
            raise ValueError('File link leaves the release: ' + name)
        size = path.stat().st_size
        total += size
        if size != item['bytes'] or digest(path) != item['sha256']:
            raise ValueError('Checksum or size mismatch: ' + name)
        if size > 50 * 1024 * 1024:
            raise ValueError('File exceeds the release size limit: ' + name)
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            raise ValueError('Excluded asset type: ' + name)
        if path.suffix in TEXT_SUFFIXES or path.name == '.gitignore':
            text = path.read_text(encoding='utf-8-sig')
            if any(pattern.search(text) for pattern in SECRET_PATTERNS):
                raise ValueError('Possible credential in: ' + name)
    if total > 100 * 1024 * 1024:
        raise ValueError('The release exceeds 100 MiB.')
    print(f'Integrity verified: {len(actual)} files, {total / 1024**2:.2f} MiB.', flush=True)


def main():
    if not __debug__:
        raise RuntimeError('Run verification without Python optimization (-O/-OO).')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--integrity-only', action='store_true',
                        help='Check files without importing numerical libraries.')
    args = parser.parse_args()
    integrity()
    if args.integrity_only:
        return
    config = json.loads((ROOT / 'verification.json').read_text(encoding='utf-8'))
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
    environment.pop('PYTHONOPTIMIZE', None)
    for check in config['checks']:
        script = (ROOT / check['script']).resolve()
        if not script.is_relative_to(ROOT) or not script.is_file():
            raise ValueError('Invalid analysis entry point: ' + check['script'])
        print('\n' + check['name'], flush=True)
        subprocess.run([sys.executable, '-B', str(script), *check.get('args', [])],
                       cwd=ROOT, env=environment, check=True)
    print('\nAll packaged verification checks passed.', flush=True)


if __name__ == '__main__':
    main()
