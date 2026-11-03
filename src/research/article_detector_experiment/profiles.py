"""Obnoviteľná extrakcia surových odpovedí nových zmrazených detektorov."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import importlib.util
import io
import os
from pathlib import Path
import time

import numpy as np
from PIL import Image

from .common import ROOT, SPLITS, canonical, digest, output_path, read, records, sha, write
from .data import verify


def variants(image):
    """Šesť nezávislých verzií, originál je prvý; vstup sa nemení."""
    image = image.convert('RGB')
    yield image.copy()
    for quality in (90, 75, 60):
        with io.BytesIO() as buffer:
            image.save(buffer, format='JPEG', quality=quality, subsampling=0, optimize=False)
            buffer.seek(0)
            with Image.open(buffer) as decoded:
                yield decoded.convert('RGB').copy()
    for scale in (0.75, 0.5):
        size = tuple(max(1, round(side * scale)) for side in image.size)
        yield image.resize(size, Image.Resampling.LANCZOS).resize(image.size, Image.Resampling.LANCZOS)


def augmentation(run):
    path = run / 'snapshot/augmentation.py'
    spec = importlib.util.spec_from_file_location('article_frozen_augmentation', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.apply_spec


def load_input(plan, apply_spec):
    raw = Path(plan['path']).read_bytes()
    if hashlib.sha256(raw).hexdigest() != plan['sha256']:
        raise ValueError('Obrázok nezodpovedá manifestu: ' + plan['path'])
    with Image.open(io.BytesIO(raw)) as image:
        image.load()
        return apply_spec(image.convert('RGB'), plan['input_spec'])


def scan(path, plans, fingerprint, repair=False):
    """Po prerušení možno odstrániť iba nedopísaný posledný riadok."""
    result = []
    chain = digest(['article-profiles-v1', fingerprint])
    if not path.exists():
        return result, chain
    with path.open('r+b' if repair else 'rb') as f:
        while True:
            position = f.tell()
            line = f.readline()
            if not line:
                break
            if not line.endswith(b'\n'):
                if repair:
                    f.truncate(position)
                    break
                raise ValueError('Nedopísaný koniec žurnálu.')
            import json
            row = json.loads(line)
            body = {k: v for k, v in row.items() if k != 'chain'}
            expected = digest([chain, body])
            if row.get('chain') != expected or len(result) >= len(plans):
                raise ValueError('Poškodený alebo nadbytočný záznam žurnálu.')
            plan = plans[len(result)]
            for key in ('sample_id', 'original_id', 'group', 'label', 'sha256', 'split', 'input_spec'):
                if row[key] != plan[key]:
                    raise ValueError('Žurnál nezodpovedá plánu: ' + key)
            values = np.asarray(row['margins'], dtype=float)
            if values.shape != (6,) or not np.isfinite(values).all() or row['fingerprint'] != fingerprint:
                raise ValueError('Neplatné alebo zmiešané odpovede detektora.')
            if row['error'] != int(int(values[0] >= 0) != plan['label']):
                raise ValueError('Nesprávne označenie chyby detektora.')
            chain = expected
            result.append(row)
    return result, chain


def synchronization(device):
    if str(device).startswith('cuda'):
        import torch
        torch.cuda.synchronize()


def extract(run, detector, assets, device='cuda', batch_size=6, max_new=None):
    from .adapters import load_detector
    import torch
    run = output_path(run)
    protocol = verify(run)
    if detector not in protocol['detectors']:
        raise ValueError('Detektor nie je súčasťou protokolu.')
    if batch_size < 1 or (max_new is not None and max_new < 1):
        raise ValueError('Dávka a max-new musia byť kladné.')
    folder = run / detector
    folder.mkdir(exist_ok=True)
    torch.manual_seed(20261027)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    started = time.perf_counter()
    adapter = load_detector(detector, Path(assets), device=device, batch_size=batch_size)
    metadata = {'protocol_sha256': sha(run / 'protocol.json'), 'adapter': adapter.metadata,
                'device': device, 'batch_size': batch_size, 'tf32': False,
                'detector': detector, 'decision': 'fake iff raw_logit >= 0'}
    fingerprint = digest(metadata)
    metadata_path = folder / 'detector.json'
    if metadata_path.exists():
        if read(metadata_path) != metadata:
            raise ValueError('Zmenený model alebo nastavenie inferencie; použite nový beh.')
    else:
        write(metadata_path, metadata)
    loading_seconds = time.perf_counter() - started
    apply_spec = augmentation(run)
    first_plan = records(run / 'plans/detector_calibration.jsonl')[0]
    warm_images = list(variants(load_input(first_plan, apply_spec)))
    synchronization(device)
    adapter.predict_logits(warm_images)
    synchronization(device)
    # Reálna kontrola zhody jednotlivých a dávkových inferencií.
    batched = np.asarray(adapter.predict_logits(warm_images), dtype=float)
    singles = np.asarray([adapter.predict_logits([im])[0] for im in warm_images], dtype=float)
    if not np.isfinite(batched).all() or not np.allclose(batched, singles, rtol=1e-4, atol=1e-4):
        raise ValueError('Nesúlad jednotlivých a dávkových inferencií.')
    if not np.array_equal(batched >= 0, singles >= 0):
        raise ValueError('Dávkovanie zmenilo klasifikáciu kontrolných obrázkov.')
    check = {'batch_single_max_abs': float(np.max(np.abs(batched - singles))),
             'checked_views': 6, 'same_decisions': True, 'loading_seconds': loading_seconds}
    if str(device).startswith('cuda'):
        torch.cuda.reset_peak_memory_stats()
    new = 0
    inference_seconds = 0.0
    wall_started = time.perf_counter()
    invocation = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    for split in SPLITS:
        plans = records(run / 'plans' / (split + '.jsonl'))
        journal = folder / (split + '.jsonl')
        receipt = folder / (split + '.complete.json')
        if receipt.exists():
            complete = read(receipt)
            if complete['fingerprint'] != fingerprint or complete['sha256'] != sha(journal) or complete['count'] != len(plans):
                raise ValueError('Poškodený dokončený profilový súbor.')
        rows, chain = scan(journal, plans, fingerprint, repair=not receipt.exists())
        print(f'{detector}: {split} {len(rows)}/{len(plans)}', flush=True)
        with journal.open('ab') as f:
            for plan in plans[len(rows):]:
                if max_new is not None and new >= max_new:
                    return {'complete': False, 'new_profiles': new}
                image = load_input(plan, apply_spec)
                inputs = list(variants(image))
                synchronization(device)
                before = time.perf_counter()
                margins = np.asarray(adapter.predict_logits(inputs), dtype=float)
                synchronization(device)
                elapsed = time.perf_counter() - before
                if margins.shape != (6,) or not np.isfinite(margins).all():
                    raise ValueError('Detektor nevrátil šesť konečných logitov.')
                row = {**plan, 'margins': margins.tolist(), 'fingerprint': fingerprint,
                       'error': int(int(margins[0] >= 0) != plan['label']),
                       'inference_seconds': elapsed}
                row['chain'] = digest([chain, row])
                chain = row['chain']
                f.write(canonical(row) + b'\n')
                f.flush()
                rows.append(row)
                new += 1
                inference_seconds += elapsed
                if new % 32 == 0:
                    os.fsync(f.fileno())
                    print(f'  {new} nových profilov; {(time.perf_counter()-wall_started)/new:.3f} s/profil vrátane I/O a úprav', flush=True)
            os.fsync(f.fileno())
        write(receipt, {'count': len(rows), 'sha256': sha(journal), 'chain': chain, 'fingerprint': fingerprint})
    wall_seconds = time.perf_counter() - wall_started
    telemetry = {'complete': True, 'new_profiles': new, 'image_inferences': 6 * new,
                 'wall_seconds': wall_seconds, 'inference_seconds': inference_seconds,
                 'seconds_per_new_profile': wall_seconds / new if new else None,
                 'peak_allocated_gpu_bytes': torch.cuda.max_memory_allocated() if str(device).startswith('cuda') else None,
                 'gpu': torch.cuda.get_device_name() if str(device).startswith('cuda') else None,
                 'sanity_check': check, 'scope': 'Meranie po zahriatí, bez načítania modelu; zahŕňa čítanie a overenie obrazov, úpravy a zápis.'}
    write(folder / 'timing' / (invocation + '.json'), telemetry)
    # Pri obnovení hotového behu sa pôvodné meranie neprepisuje nulovým behom.
    if new:
        write(folder / 'timing.json', telemetry)
    return telemetry


def load_complete(run, detector, split):
    folder = Path(run) / detector
    receipt = read(folder / (split + '.complete.json'))
    path = folder / (split + '.jsonl')
    metadata = read(folder / 'detector.json')
    if metadata['protocol_sha256'] != sha(Path(run) / 'protocol.json'):
        raise ValueError('Profily patria inému protokolu.')
    if receipt['sha256'] != sha(path) or receipt['fingerprint'] != digest(metadata):
        raise ValueError('Neplatný kontrolný súčet profilov.')
    plans = records(Path(run) / 'plans' / (split + '.jsonl'))
    rows, chain = scan(path, plans, digest(metadata))
    if len(rows) != len(plans) or receipt['count'] != len(rows) or receipt['chain'] != chain:
        raise ValueError('Profily nie sú kompletné.')
    return rows
