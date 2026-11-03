"""Integrita a zápis výlučne do nového experimentu."""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import uuid

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
SPLITS = ('detector_calibration', 'selector_train', 'selector_selection')


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(',', ':'), allow_nan=False).encode('utf-8')


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def records(path):
    with Path(path).open(encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name('.' + path.name + '.' + uuid.uuid4().hex + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    os.replace(temp, path)


def write_records(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as f:
        for row in rows:
            f.write(canonical(row) + b'\n')


def environment():
    return {'python': platform.python_version(), 'packages': {
        name: metadata.version(name) for name in
        ('torch', 'torchvision', 'Pillow', 'numpy', 'scipy', 'scikit-learn')}}


def output_path(path):
    path = Path(path).resolve()
    # Výsledky môžu vzniknúť len v novom podstrome, nikdy v starých behoch.
    base = (ROOT / 'runs').resolve()
    if path == base or not path.is_relative_to(base):
        raise ValueError('Výstup musí byť samostatný podpriečinok article_detector_experiment/runs.')
    return path


@contextmanager
def lock(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+b') as f:
        if f.tell() == 0:
            f.write(b'0'); f.flush()
        f.seek(0)
        if os.name == 'nt':
            import msvcrt
            try:
                msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise RuntimeError('V tomto behu už pracuje iný proces.') from exc
        else:
            import fcntl
            fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            f.seek(0)
            if os.name == 'nt':
                msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
