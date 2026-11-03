"""Integrita a zápisy výlučne do nového experimentu."""
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import uuid

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent


def now():
    return datetime.now(timezone.utc).isoformat()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def read(path):
    def invalid(value):
        raise ValueError('Nekonečná hodnota JSON: ' + value)
    return json.loads(Path(path).read_text(encoding='utf-8-sig'), parse_constant=invalid)


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name('.' + path.name + '.' + uuid.uuid4().hex)
    with temporary.open('xb') as stream:
        stream.write(canonical(value) + b'\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def environment():
    return {'python': platform.python_version(), 'packages': {name: metadata.version(name) for name in
            ('numpy', 'scipy', 'scikit-learn', 'torch', 'torchvision', 'Pillow')}}


def output_path(path):
    path = Path(path).resolve()
    base = (ROOT / 'runs').resolve()
    if path == base or not path.is_relative_to(base):
        raise ValueError('Výstup musí byť pod article_reference_experiment/runs.')
    return path


@contextmanager
def lock(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+b') as stream:
        stream.seek(0, 2)
        if not stream.tell():
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
