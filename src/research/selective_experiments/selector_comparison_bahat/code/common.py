from __future__ import annotations
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import sys
from contextlib import contextmanager
from importlib import metadata


def canonical(obj):
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def digest(obj):
    return hashlib.sha256(canonical(obj)).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))


def write(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)


def lines(path):
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def environment():
    names = ("numpy", "scipy", "scikit-learn", "torch", "torchvision", "Pillow")
    return {"python": platform.python_version(), "packages": {n: metadata.version(n) for n in names}}


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@contextmanager
def lock(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as f:
        f.seek(0, 2)
        if f.tell() == 0:
            f.write(b"0")
            f.flush()
        f.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError("V tomto výstupe už beží iný proces.") from exc
        try:
            yield
        finally:
            f.seek(0)
            if os.name == "nt":
                msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def verify(run):
    run = Path(run)
    p = read(run / "protocol.json")
    if sha(run / "protocol.json") != (run / "protocol.sha256").read_text().strip():
        raise ValueError("Zmenený protokol porovnania.")
    if p["environment"] != environment():
        raise ValueError("Zmenené prostredie. Použite pôvodný interpreter the recorded package versions.")
    for relative, expected in p["files"].items():
        if sha(run / relative) != expected:
            raise ValueError("Zmenený vstup porovnania: " + relative)
    for name, expected in p["code"].items():
        if sha(Path(__file__).parent / name) != expected:
            raise ValueError("Kód sa zmenil od prípravy. Pre nový kód vytvorte nový --run: " + name)
    return p


def load_data(run, split):
    import numpy as np
    with np.load(Path(run) / "data" / (split + ".npz"), allow_pickle=False) as z:
        a = {k: z[k] for k in z.files}
    return a, read(Path(run) / "data" / (split + ".json"))


def check_receipt(folder, receipt, protocol_sha):
    r = read(Path(folder) / receipt)
    if r["protocol_sha256"] != protocol_sha:
        raise ValueError("Potvrdenie patrí inému protokolu.")
    for name, expected in r["files"].items():
        if sha(Path(folder) / name) != expected:
            raise ValueError("Zmenený výsledok: " + name)
    return r
