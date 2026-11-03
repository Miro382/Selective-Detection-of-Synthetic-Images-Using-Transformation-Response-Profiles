"""Integrita nového behu; tento modul nespúšťa numerické výpočty pri importe."""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
from importlib import metadata
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import sys
import uuid

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
PACKAGES = ('torch', 'torchvision', 'Pillow', 'numpy', 'scipy', 'scikit-learn')


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(',', ':'), allow_nan=False).encode('utf-8')


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def records(path):
    with Path(path).open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def output_path(path):
    path = Path(path).resolve()
    base = (ROOT / 'runs').resolve()
    if path == base or not path.is_relative_to(base):
        raise ValueError('Výstup musí byť v novom article_detector_adaptation/runs/<názov>.')
    return path


def write(path, value):
    path = output_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name('.' + path.name + '.' + uuid.uuid4().hex + '.tmp')
    with temp.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    os.replace(temp, path)


def write_records(path, rows):
    path = output_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name('.' + path.name + '.' + uuid.uuid4().hex + '.tmp')
    with temp.open('xb') as stream:
        for row in rows:
            stream.write(canonical(row) + b'\n')
        stream.flush(); os.fsync(stream.fileno())
    os.replace(temp, path)


def environment():
    # The executable path is informational, not a numerical identity criterion.
    return {'python': platform.python_version(),
            'packages': {name: metadata.version(name) for name in PACKAGES}}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def load_helper(run, name):
    if Path(name).name != name or not name.replace('_', '').isalnum():
        raise ValueError('Neplatný názov pomocného modulu.')
    run = output_path(run)
    relative = 'snapshot/helpers/' + name + '.py'
    protocol = read(run / 'protocol.json')
    if sha(run / relative) != protocol['files'][relative]:
        raise ValueError('Zmenený pomocný modul: ' + name)
    module_name = '_adaptation_' + digest(str(run))[:12] + '_' + name
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, run / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@contextmanager
def lock(path):
    path = output_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+b') as stream:
        if stream.tell() == 0:
            stream.write(b'0'); stream.flush()
        stream.seek(0)
        if os.name == 'nt':
            import msvcrt
            try:
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise RuntimeError('Tento nový beh už používa iný proces.') from exc
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
