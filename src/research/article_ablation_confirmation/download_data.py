"""Stiahnutie a obsahový audit zmrazeného holdoutu; bez modelovej inferencie."""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import sys
import uuid
import zlib

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
DEFAULT_DATA = ROOT / 'data/independent_20261029'
DEFAULT_DESIGN = ROOT / 'protocols/strict_20261028'
MAX_BYTES = 40 * 2**30
VERSION = 'article-ablation-content-data-v1'


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b''):
            value.update(block)
    return value.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def lines(path):
    with Path(path).open(encoding='utf-8') as stream:
        for line in stream:
            if not line.strip() or not line.endswith('\n'):
                raise ValueError('Neúplný JSONL: ' + str(path))
            yield json.loads(line)


def now():
    return datetime.now(timezone.utc).isoformat()


def atomic(path, body):
    path = Path(path)
    temporary = path.with_name('.' + path.name + '.writing-' + uuid.uuid4().hex)
    with temporary.open('xb') as stream:
        stream.write(body)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def save(path, value):
    atomic(path, canonical(value) + b'\n')


def write_lines(path, values):
    atomic(path, b''.join(canonical(value) + b'\n' for value in values))


def check_files(folder, files):
    folder = Path(folder).resolve()
    for name, checksum in files.items():
        path = (folder / name).resolve()
        if not path.is_relative_to(folder) or sha(path) != checksum:
            raise ValueError('Zmenený uzamknutý súbor: ' + str(path))


def checked_json(folder, name):
    path = folder / (name + '.json')
    if sha(path) != (folder / (name + '.sha256')).read_text(encoding='ascii').strip():
        raise ValueError('Zmenený protokol: ' + str(path))
    return read(path)


def load_module(name, path, checksum=None):
    if checksum is not None and sha(path) != checksum:
        raise ValueError('Zmenený importovaný modul: ' + str(path))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def allowed(path):
    path = Path(path).resolve()
    if path.parent != (ROOT / 'data').resolve() or path.name.startswith('.'):
        raise ValueError('Výstup musí byť nový article_ablation_confirmation/data/<názov>.')
    return path


@contextmanager
def lock(folder):
    with (folder / '.execution.lock').open('a+b') as stream:
        if stream.tell() == 0:
            stream.write(b'0'); stream.flush()
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


def environment():
    return {'python': sys.version.split()[0],
            'packages': {name: importlib.metadata.version(name)
                         for name in ('numpy', 'scipy', 'Pillow', 'pyarrow', 'requests')}}


def verify_sources(design_path, folder):
    design_path = Path(design_path).resolve()
    design = checked_json(design_path, 'protocol')
    module_path = ROOT / 'freeze_protocol.py'
    verifier = load_module('_ablation_design_verifier', module_path, design['anchors'][str(module_path)])
    design = verifier.verify(design_path)
    if (design['counts'] != {'DeepShield': 1500, 'OpenFake': 6000} or
            not design['statistical_protocol_locked'] or not design['adoption_rule_locked']):
        raise ValueError('Nesúhlasí pevný návrh testu.')
    metadata = Path(design['metadata_plan'])
    meta_protocol = checked_json(metadata, 'protocol')
    planner = verifier.prepare_candidate_plan
    if planner.source_inventory(PROJECT, metadata) != meta_protocol['history_source_inventory']:
        raise ValueError('Po uzamknutí vznikla nová neauditovaná história.')
    for other in (ROOT / 'data').glob('*'):
        if other.is_dir() and other.resolve() != folder.resolve():
            if any((other / 'images').glob('*')) or any((other / 'receipts').glob('*.json')):
                raise ValueError('Iný nový dataset vyžaduje rozšírenie histórie: ' + str(other))
    # Freeze verifier už overil všetkých 6969 receipts aj 6926 raw kotiev.
    anchors = read(metadata / 'history_anchors.json')
    old = PROJECT / 'article_detector_experiment/external_testing'
    if str(old) not in sys.path:
        sys.path.insert(0, str(old))
    legacy = load_module('_ablation_legacy_transport', old / 'prepare_data.py', anchors[str(old / 'prepare_data.py')])
    for module in (legacy.of, legacy.cf, legacy.ds):
        path = str(Path(module.__file__).resolve())
        if path not in anchors or sha(path) != anchors[path]:
            raise ValueError('Importovaný helper nie je z histórie.')
    return design, meta_protocol, legacy


def verify_preparation(folder):
    folder = allowed(folder)
    prep = checked_json(folder, 'preparation')
    if (prep['version'] != VERSION or prep['destination'] != str(folder) or
            prep['max_network_bytes'] != MAX_BYTES or prep['environment'] != environment()):
        raise ValueError('Zmenený kontrakt alebo numerické/prenosové prostredie.')
    if (prep['content']['allow_mpo_primary'] is not False or prep['content']['phash_bits'] != 64 or
            prep['content']['max_hamming_distance'] != 4 or not prep['content']['compare_rejected_downloads']):
        raise ValueError('Zmenený obsahový filter.')
    check_files(folder, prep['files'])
    for name, checksum in prep['implementation_files'].items():
        if sha(name) != checksum:
            raise ValueError('Zmenený dátový kód po začatí prípravy: ' + name)
    design_path = Path(prep['design_path'])
    if sha(design_path / 'protocol.json') != prep['design_sha256']:
        raise ValueError('Zmenený predtestový návrh.')
    design, meta, legacy = verify_sources(design_path, folder)
    if (sha(Path(prep['metadata_plan']) / 'protocol.json') != prep['metadata_protocol_sha256'] or
            design['metadata_plan'] != prep['metadata_plan'] or meta['targets'] != prep['targets']):
        raise ValueError('Zmenený metadátový výber.')
    audit_module = load_module('_ablation_content_audit', ROOT / 'content_audit.py', prep['implementation_files'][str(ROOT / 'content_audit.py')])
    return prep, legacy, audit_module


def prepare(folder=DEFAULT_DATA, design_path=DEFAULT_DESIGN):
    folder = allowed(folder)
    if (folder / 'preparation.json').exists():
        return verify_preparation(folder)[0]
    if folder.exists() and any(folder.iterdir()):
        raise ValueError('Neznámy alebo neúplne pripravený výstup; nič sa neprepisuje.')
    design, meta, legacy = verify_sources(design_path, folder)
    packages = environment()['packages']
    if any(packages[k] != design['environment']['packages'][k] for k in ('numpy', 'scipy', 'Pillow')):
        raise ValueError('Použite pôvodné numerické prostredie návrhu.')
    folder.mkdir(parents=True, exist_ok=True)
    for name in ('images', 'receipts', 'access', 'transfer_events', 'code'):
        (folder / name).mkdir()
    implementations = {str(ROOT / name): sha(ROOT / name) for name in ('download_data.py', 'content_audit.py')}
    for name in ('download_data.py', 'content_audit.py'):
        shutil.copyfile(ROOT / name, folder / 'code' / name)
    prep = {'version': VERSION, 'created_utc': now(), 'destination': str(folder),
            'design_path': str(Path(design_path).resolve()), 'design_sha256': sha(Path(design_path) / 'protocol.json'),
            'metadata_plan': design['metadata_plan'], 'metadata_protocol_sha256': sha(Path(design['metadata_plan']) / 'protocol.json'),
            'targets': meta['targets'], 'max_network_bytes': MAX_BYTES, 'environment': environment(),
            'implementation_files': implementations,
            'files': {'code/' + name: checksum for name, checksum in
                      ((Path(p).name, value) for p, value in implementations.items())},
            'selection': 'Pevné candidates.jsonl poradie; výmena iba po pôvodovom/obsahovom audite. Žiadne skóre ani zmena kvót.',
            'grouping': 'Komponenty pôvodu a sampling vrátane známych susedov skutočne načítaných Parquet blokov.',
            'content': {'fingerprint': legacy.FINGERPRINT_VERSION, 'allow_mpo_primary': False,
                        'phash_bits': 64, 'max_hamming_distance': 4,
                        'compare_rejected_downloads': True},
            'ready_for_inference': False, 'external_inference_performed': False}
    save(folder / 'preparation.json', prep)
    atomic(folder / 'preparation.sha256', (sha(folder / 'preparation.json') + '\n').encode('ascii'))
    return prep


class Budget:
    """Nemenné účtovné udalosti pred požiadavkou; súčet prežije pád procesu."""
    def __init__(self, folder, preparation_sha, *, writable=False, limit=MAX_BYTES):
        self.folder = Path(folder)
        self.preparation_sha = preparation_sha
        self.limit = limit
        self.writable = writable
        self.used = 0
        self.last = '0' * 64
        self.events = 0
        self.hashes = {}
        prefixes = []
        for index, path in enumerate(sorted((self.folder / 'transfer_events').glob('*.json'))):
            if path.name != f'{index:08d}.json':
                raise ValueError('Medzera v účtovnom journal-e.')
            value = read(path)
            event = value['event']
            if (value['sha256'] != digest(event) or event['previous_sha256'] != self.last or
                    event['preparation_sha256'] != preparation_sha or event['index'] != index or
                    type(event['reserved_bytes']) is not int or event['reserved_bytes'] <= 0):
                raise ValueError('Zmenený účtovný journal.')
            self.used += event['reserved_bytes']
            self.last = value['sha256']; self.events += 1
            self.hashes['transfer_events/' + path.name] = sha(path)
            prefixes.append({'events': self.events, 'reserved_bytes': self.used, 'last_event_sha256': self.last,
                             'preparation_sha256': self.preparation_sha})
        if self.used > self.limit:
            raise ValueError('Prenos prekročil nemenný limit.')
        head_path = self.folder / 'transfer_head.json'
        if head_path.exists():
            head = read(head_path)
            index = head.get('events', 0) - 1
            if index < 0 or index >= len(prefixes) or head != prefixes[index] or len(prefixes) - index > 2:
                raise ValueError('Účtovný journal bol skrátený alebo zmenený.')
        elif self.events > 1:
            raise ValueError('Chýba priebežná kotva prenosového journalu.')
        if not self.events and any((self.folder / 'images').glob('*.img')):
            raise ValueError('Stiahnuté obrázky bez prenosového journalu.')

    def charge(self, amount, reason='range'):
        if not self.writable:
            raise RuntimeError('Read-only budget nesmie účtovať.')
        if type(amount) is not int or amount <= 0 or self.used + amount > self.limit:
            raise RuntimeError('Dosiahnutý kumulatívny limit 40 GiB vrátane rezervovaných retry požiadaviek.')
        event = {'index': self.events, 'previous_sha256': self.last,
                 'preparation_sha256': self.preparation_sha, 'reserved_bytes': amount,
                 'reason': reason, 'created_utc': now()}
        path = self.folder / 'transfer_events' / f'{self.events:08d}.json'
        if path.exists():
            raise ValueError('Účtovná udalosť sa nesmie prepísať.')
        save(path, {'event': event, 'sha256': digest(event)})
        self.used += amount; self.last = digest(event); self.events += 1
        self.hashes['transfer_events/' + path.name] = sha(path)
        save(self.folder / 'transfer_head.json', {'events': self.events, 'reserved_bytes': self.used,
             'last_event_sha256': self.last, 'preparation_sha256': self.preparation_sha})

    def summary(self):
        return {'reserved_bytes': self.used, 'max_bytes': self.limit, 'events': self.events,
                'last_event_sha256': self.last, 'files': self.hashes,
                'semantics': 'Každý HTTP send (vrátane retry/redirect) rezervuje Range +64 KiB pred sieťou; zdrojové metadáta 2 MiB +64 KiB. Horný účtovaný odhad, nie presne nameraný sieťový traffic.'}


class Transport:
    def __init__(self, budget, legacy):
        import requests
        self.budget, self.legacy = budget, legacy
        self.current = self.block = self.remote = None
        class MeteredSession(requests.Session):
            def send(session, request, **kwargs):
                value = request.headers.get('Range', '')
                match = re.fullmatch(r'bytes=(\d+)-(\d+)', value)
                if match:
                    first, last = map(int, match.groups())
                    if last < first or last - first + 1 > 64 * 1024**2:
                        raise ValueError('Neplatný Range.')
                    budget.charge(last - first + 1 + 64 * 1024, 'http-range-send')
                elif request.url == 'https://zenodo.org/api/records/15648378':
                    budget.charge(2 * 1024**2 + 64 * 1024, 'zenodo-metadata-send')
                else:
                    raise ValueError('Neúčtovaný request bez Range nie je dovolený.')
                return super().send(request, **kwargs)
        self.session = MeteredSession()
        self.session.max_redirects = 5

    def fetch(self, row):
        legacy = self.legacy
        if row['dataset'] == 'OpenFake':
            key = row['sampling_group']
            if self.current != key:
                import pyarrow.parquet as pq
                class AlreadyCharged:
                    def charge(self, amount):
                        pass  # MeteredSession rezervovala pred HTTP send.
                logical = max(256 * 1024**2, row['row_group_image_bytes'] + 128 * 1024)
                # RemoteParquet.close vyčistí spojenia; requests Session možno znovu použiť.
                with legacy.of.RemoteParquet(row['shard'], row['shard_size'], AlreadyCharged(),
                        max_logical_read=logical, session=self.session) as remote:
                    parquet = pq.ParquetFile(remote)
                    group = parquet.metadata.row_group(row['row_group'])
                    size = sum(group.column(i).total_compressed_size for i in range(group.num_columns)
                               if group.column(i).path_in_schema.startswith('image.'))
                    if size != row['row_group_image_bytes']:
                        raise ValueError('Zmenený obrazový stĺpec zdroja.')
                    self.block = parquet.read_row_group(row['row_group'], columns=[
                        'image', 'prompt', 'label', 'model', 'type', 'row_id', 'image_sha256', 'prompt_kind'],
                        use_threads=False).to_pylist()
                self.current = key
            raw = self.block[row['row_index']]
            if any(raw[k] != row[k] for k in ('label', 'model', 'type', 'row_id', 'image_sha256', 'prompt_kind')):
                raise ValueError('Zmenené zdrojové metadáta.')
            if (hashlib.sha256(str(raw.get('prompt') or '').encode()).hexdigest() != row['prompt_sha256'] or
                    legacy.of.prompt_hash(raw.get('prompt')) != row['prompt_normalized_sha256']):
                raise ValueError('Zmenený zdrojový prompt.')
            body = raw['image']['bytes']
        elif row['dataset'] == 'DeepShield':
            if self.remote is None:
                with self.session.get('https://zenodo.org/api/records/15648378', stream=True, timeout=(15, 60)) as response:
                    response.raise_for_status(); body = bytearray()
                    for piece in response.iter_content(65536):
                        body.extend(piece)
                        if len(body) > 2 * 1024**2:
                            raise ValueError('Priveľké metadáta archívu.')
                entries = json.loads(body).get('files', [])
                if (len(entries) != 1 or entries[0].get('size') != legacy.ds.ARCHIVE_SIZE or
                        entries[0].get('checksum') != 'md5:' + legacy.ds.ARCHIVE_MD5):
                    raise ValueError('Zmenený archív DeepShield.')
                from v4_deepshield_remote_zip import RemoteZip
                self.remote = RemoteZip(legacy.ds.ARCHIVE_URL, legacy.ds.ARCHIVE_SIZE,
                                        max_total_bytes=self.budget.limit, transport=self.session)
            body = self.remote.read_member(row['entry'])
        else:
            raise ValueError('Nepovolený dataset.')
        validate_raw(row, body)
        return body

    def close(self):
        if self.remote:
            self.remote.close()
        self.session.close()


def validate_raw(row, body):
    if not isinstance(body, bytes):
        raise ValueError('Zdroj nevrátil obrazové bajty.')
    if row['dataset'] == 'OpenFake':
        if hashlib.sha256(body).hexdigest() != row['image_sha256']:
            raise ValueError('Zdrojový SHA OpenFake nesedí.')
    elif len(body) != row['entry']['size'] or (zlib.crc32(body) & 0xffffffff) != row['entry']['crc']:
        raise ValueError('CRC alebo veľkosť položky DeepShield nesedí.')


def candidate_access(folder, row, prep_sha, *, writable):
    key = digest(row['original_id'])
    path = folder / 'access' / (key + '.json')
    expected = {'candidate_sha256': digest(row), 'original_id': row['original_id'],
                'dataset': row['dataset'], 'sampling_group': row['sampling_group'],
                'preparation_sha256': prep_sha}
    if path.exists():
        record = read(path)
        if record['record'] != expected or record['sha256'] != digest(expected):
            raise ValueError('Zmenená evidencia pristúpenia k zdroju.')
    elif writable:
        save(path, {'record': expected, 'sha256': digest(expected)})
    else:
        raise ValueError('Obrázok nemá evidované pristúpenie k zdroju.')
    return 'access/' + path.name


def manifest_record(row, path, fp):
    label = int(row['label'] == 'fake') if row['dataset'] == 'OpenFake' else int(row['label'])
    dependencies = sorted(set(row['dependency_keys']) | {row['source_key'], row['sampling_group']})
    return {'sample_id': 'ablation-confirmation/' + row['original_id'], 'original_id': row['original_id'],
            'dataset': row['dataset'], 'bucket': row['bucket'], 'label': label,
            'source_key': row['source_key'], 'sampling_group': row['sampling_group'],
            'dependency_keys': dependencies, 'path': str(path.resolve()),
            'input_spec': {'version': 1, 'id': 'clean', 'operations': []}, **fp}


def replay(folder, prep, legacy, audit_module, context, *, transport=None, max_new=None):
    """Jediný deterministický replay používa download aj read-only finálny check."""
    prep_sha = sha(folder / 'preparation.json')
    audit = audit_module.Replay(context)
    counts = Counter(); records = []; excluded = []; receipts = {}; accessed = {}; images = set()
    blocks = set(); processed = set(); new = 0; stopped = False
    candidates = list(lines(Path(prep['metadata_plan']) / 'candidates.jsonl'))
    targets = Counter(prep['targets'])
    for ordinal, row in enumerate(candidates):
        if row['candidate_order'] != ordinal:
            raise ValueError('Zmenené poradie kandidátov.')
        if counts[row['bucket']] >= targets[row['bucket']]:
            continue
        key = digest(row['original_id'])
        receipt_path = folder / 'receipts' / (key + '.json')
        image_path = folder / 'images' / (key + '.img')
        access_path = folder / 'access' / (key + '.json')
        previous = None
        if receipt_path.exists():
            previous = read(receipt_path)
            if previous.get('record_sha256') != digest(previous.get('record')):
                raise ValueError('Zmenený kandidátsky receipt.')
            previous = previous['record']
        elif transport is None:
            stopped = True
            break
        elif max_new is not None and new >= max_new:
            stopped = True
            break
        reasons = audit.before_download(row)
        fp = None; raw_sha = None; invalid_detail = None
        if not reasons:
            access_name = candidate_access(folder, row, prep_sha, writable=transport is not None)
            accessed[access_name] = sha(folder / access_name)
            if row['dataset'] == 'OpenFake':
                blocks.add(row['sampling_group'])
            if not image_path.exists():
                if previous is not None or transport is None:
                    raise ValueError('Chýbajú už potvrdené obrazové bajty.')
                atomic(image_path, transport.fetch(row))
            body = image_path.read_bytes()
            validate_raw(row, body)
            raw_sha = hashlib.sha256(body).hexdigest()
            del body
            try:
                fp = legacy.fingerprint(image_path)
            except (ValueError, OSError) as error:
                invalid_detail = str(error)
            reasons = audit.reasons(row, raw_sha, fp, invalid_detail)
            images.add(image_path.resolve())
        elif image_path.exists() or access_path.exists():
            raise ValueError('Zmenená história pre už pristúpeného kandidáta.')
        result = {'preparation_sha256': prep_sha, 'candidate_sha256': digest(row),
                  'original_id': row['original_id'], 'decision': 'excluded' if reasons else 'accepted',
                  'image_sha256': raw_sha, 'fingerprint': fp, 'reasons': reasons}
        if previous is not None:
            if previous != result:
                raise ValueError('Receipt sa nezhoduje s opakovaným auditom: ' + row['original_id'])
        else:
            save(receipt_path, {'record': result, 'record_sha256': digest(result)})
            new += 1
        receipts['receipts/' + receipt_path.name] = sha(receipt_path)
        processed.add(row['original_id'])
        audit.apply(row, raw_sha, fp, reasons)
        if reasons:
            excluded.append({**result, 'dataset': row['dataset'], 'bucket': row['bucket'],
                             'source_key': row['source_key'], 'sampling_group': row['sampling_group'],
                             'dependency_keys': row['dependency_keys']})
        else:
            records.append(manifest_record(row, image_path, fp))
            counts[row['bucket']] += 1
        if transport is not None and len(receipts) % 100 == 0:
            save(folder / 'progress.json', {'accepted': sum(counts.values()), 'counts': dict(counts),
                 'required': prep['targets'], 'new_candidates': new, 'updated_utc': now(), 'ready_for_inference': False})
            print(f'Dáta: {sum(counts.values())}/7500; nové rozhodnutia {new}', flush=True)
    actual_receipts = {'receipts/' + p.name for p in (folder / 'receipts').glob('*.json')}
    if actual_receipts != set(receipts):
        raise ValueError('Receipts nezodpovedajú deterministickému prefixu výberu.')
    # Pri technickom zastavení môže existovať najviac aktuálny orphan access/image
    # po páde pred receipt. Na kompletnom datasete je každý súbor vysvetlený.
    complete = counts == targets
    if complete:
        if {'access/' + p.name for p in (folder / 'access').glob('*.json')} != set(accessed):
            raise ValueError('Osirelé pristúpenie k zdroju.')
        if {p.resolve() for p in (folder / 'images').glob('*.img')} != images:
            raise ValueError('Osirelý obrazový súbor.')
    if not complete and not stopped:
        raise ValueError('Obsahový audit vyčerpal pevné rezervy; kvóty sa nesmú znížiť.')
    manifest, neighbours = context.groups(records, blocks)
    return {'complete': complete, 'manifest': manifest, 'excluded': excluded,
            'receipts': receipts, 'access': accessed, 'neighbours': neighbours,
            'counts': dict(counts), 'blocks': sorted(blocks), 'new_candidates': new}


def expected_audit(prep, result, context, manifest_sha):
    return {'version': VERSION, 'ready_for_inference': True, 'all_images_verified': True,
            'score_independent_selection': True, 'external_inference_performed': False,
            'design_sha256': prep['design_sha256'], 'metadata_protocol_sha256': prep['metadata_protocol_sha256'],
            'manifest_sha256': manifest_sha, 'targets': prep['targets'], 'actual_counts': result['counts'],
            'historical_coverage': context.coverage,
            'phash': {'bits': 64, 'max_hamming_distance': 4, 'compared_with_all_available_history': True},
            'disjointness': {'source_origin': 0, 'sha256': 0, 'rgb_sha256': 0, 'phash_unresolved': 0},
            'downloaded': len(result['access']), 'excluded': len(result['excluded']),
            'raw_only_excluded': sum(bool(r['image_sha256']) and not r['fingerprint'] for r in result['excluded']),
            'accessed_openfake_blocks': result['blocks'], 'protected_block_neighbours': len(result['neighbours']),
            'connected_groups_by_dataset': {name: len({r['group'] for r in result['manifest'] if r['dataset'] == name})
                                           for name in ('DeepShield', 'OpenFake')},
            'limitations': ['Nulové pHash nálezy sa vzťahujú iba na dostupné historické pHash.',
                           'Pôvod a sampling zahŕňa aj známych nevzorkovaných susedov; komponenty sa kvôli významnosti nedelia.',
                           'Oddelené sú prijaté pôvody a historicky načítané bloky. Nový blok môže obsahovať nevybrané historické identity; netvrdíme disjunktnosť celej komponentovej histórie.',
                           'Neznáma identita osôb, sémantické väzby a predtrénovanie zostávajú limitom.']}


def verify_final(folder, artifacts, prep, legacy, audit_module, context):
    protocol = checked_json(artifacts, 'protocol')
    if (protocol['version'] != VERSION or protocol['preparation_sha256'] != sha(folder / 'preparation.json') or
            protocol['design_sha256'] != prep['design_sha256'] or protocol['metadata_protocol_sha256'] != prep['metadata_protocol_sha256'] or
            protocol['ready_for_inference'] is not True or protocol['external_inference_performed'] is not False):
        raise ValueError('Finálny protokol má iné vstupy.')
    check_files(artifacts, protocol['files'])
    budget = Budget(folder, sha(folder / 'preparation.json'))
    if read(artifacts / 'transfer.json') != budget.summary():
        raise ValueError('Finálny prenosový journal sa nezhoduje.')
    result = replay(folder, prep, legacy, audit_module, context)
    if not result['complete']:
        raise ValueError('Chýbajú pevné kvóty.')
    for name, actual in (('manifest.jsonl', result['manifest']),
                         ('candidate_exclusions.jsonl', result['excluded']),
                         ('accessed_block_neighbours.jsonl', result['neighbours'])):
        if list(lines(artifacts / name)) != actual:
            raise ValueError('Nezávislý replay sa nezhoduje: ' + name)
    for name, actual in (('receipts.json', result['receipts']), ('access.json', result['access'])):
        if read(artifacts / name) != actual:
            raise ValueError('Nezodpovedajúce potvrdenia: ' + name)
        check_files(folder, actual)
    audit = read(artifacts / 'audit.json')
    if audit != expected_audit(prep, result, context, sha(artifacts / 'manifest.jsonl')):
        raise ValueError('Finálny audit nesedí.')
    return {'protocol': protocol, 'audit': audit, 'manifest': result['manifest']}


def verify_data(path):
    """Read-only úplné overenie; návrat {protocol, audit, manifest}."""
    folder = allowed(path)
    prep, legacy, audit_module = verify_preparation(folder)
    complete = read(folder / 'complete.json')
    if (complete['protocol_sha256'] != sha(folder / 'protocol.json') or
            complete['ready_for_inference'] is not True or complete['total'] != 7500 or
            complete['external_inference_performed'] is not False):
        raise ValueError('Dáta nemajú platné dokončenie.')
    context = audit_module.Context(Path(prep['metadata_plan']), legacy)
    return verify_final(folder, folder, prep, legacy, audit_module, context)


def download(folder=DEFAULT_DATA, max_new=None):
    folder = allowed(folder)
    if (folder / 'complete.json').exists():
        result = verify_data(folder)
        return {'complete': True, 'total': len(result['manifest']), 'ready_for_inference': True}
    prep, legacy, audit_module = verify_preparation(folder)
    context = audit_module.Context(Path(prep['metadata_plan']), legacy)
    budget = Budget(folder, sha(folder / 'preparation.json'), writable=True)
    transport = Transport(budget, legacy)
    try:
        result = replay(folder, prep, legacy, audit_module, context, transport=transport, max_new=max_new)
        if not result['complete']:
            save(folder / 'progress.json', {'accepted': len(result['manifest']), 'counts': result['counts'],
                 'required': prep['targets'], 'new_candidates': result['new_candidates'],
                 'reserved_bytes': budget.used, 'updated_utc': now(), 'ready_for_inference': False})
            return {'complete': False, 'accepted': len(result['manifest']), 'reason': 'max_new'}
        verify_preparation(folder)  # História musí zostať nemenná počas prenosu.
        stage = folder / ('.finalizing-' + uuid.uuid4().hex)
        stage.mkdir()
        for name, values in (('manifest.jsonl', result['manifest']),
                             ('candidate_exclusions.jsonl', result['excluded']),
                             ('accessed_block_neighbours.jsonl', result['neighbours'])):
            write_lines(stage / name, values)
        for name, value in (('receipts.json', result['receipts']), ('access.json', result['access']),
                            ('transfer.json', budget.summary())):
            save(stage / name, value)
        save(stage / 'audit.json', expected_audit(prep, result, context, sha(stage / 'manifest.jsonl')))
        protocol = {'version': VERSION, 'created_utc': now(),
                    'preparation_sha256': sha(folder / 'preparation.json'),
                    'design_sha256': prep['design_sha256'], 'metadata_protocol_sha256': prep['metadata_protocol_sha256'],
                    'ready_for_inference': True, 'external_inference_performed': False,
                    'files': {p.name: sha(p) for p in sorted(stage.iterdir())}}
        save(stage / 'protocol.json', protocol)
        atomic(stage / 'protocol.sha256', (sha(stage / 'protocol.json') + '\n').encode('ascii'))
        verify_final(folder, stage, prep, legacy, audit_module, context)
        # complete je posledný marker; pred ním nie je inferencia autorizovaná.
        for path in list(stage.iterdir()):
            os.replace(path, folder / path.name)
        stage.rmdir()
        save(folder / 'complete.json', {'version': VERSION, 'protocol_sha256': sha(folder / 'protocol.json'),
             'total': len(result['manifest']), 'ready_for_inference': True, 'external_inference_performed': False})
        return {'complete': True, 'total': len(result['manifest']), 'ready_for_inference': True,
                'groups': read(folder / 'audit.json')['connected_groups_by_dataset'], 'reserved_bytes': budget.used}
    except Exception as error:
        save(folder / 'last_error.json', {'created_utc': now(), 'error': str(error),
                                         'ready_for_inference': False, 'reserved_bytes': budget.used})
        raise
    finally:
        transport.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=DEFAULT_DATA)
    parser.add_argument('--design', type=Path, default=DEFAULT_DESIGN)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--prepare', action='store_true')
    action.add_argument('--download', action='store_true')
    action.add_argument('--check', action='store_true')
    parser.add_argument('--max-new', type=int)
    args = parser.parse_args(argv)
    if args.max_new is not None and (not args.download or args.max_new <= 0):
        parser.error('--max-new musí byť kladné a patrí iba k --download.')
    folder = allowed(args.data)
    if args.prepare:
        prepare(folder, args.design)
        result = {'prepared': True, 'ready_for_inference': False, 'path': str(folder)}
    elif args.check:
        checked = verify_data(folder)
        result = {'verified': True, 'total': len(checked['manifest']), 'ready_for_inference': True}
    else:
        with lock(folder):
            result = download(folder, args.max_new)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


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
