"""Pripraví nezávislé externé obrázky bez spustenia detektorov.

--plan-only uzamkne kvóty, históriu, metadáta a poradie rezerv bez siete.
--download pokračuje iba podľa uzamknutého predvýberu; --check audit zopakuje.
Všetky zápisy patria výhradne novému external_testing/plans/<názov>.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import sys
import uuid

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from data_support import (PROJECT, SEED, TARGETS, canonical, digest, sha, read,
                          collect_history, history_records, historical_index,
                          candidate_plan, connected_groups, cf, of, ds, block_key)
from v4_deepshield_audit import fingerprint, _MemoryIndex, FINGERPRINT_VERSION

DEFAULT_PLAN = ROOT / 'plans/independent_20261027'
MAX_NETWORK_GIB = 40
HELPERS = ('confirmation_v4_data.py', 'external_v4_data.py', 'selector_v5_data.py',
           'external_v5_data.py', 'v4_data_sources.py', 'v4_deepshield_audit.py',
           'v4_deepshield_remote_zip.py')


def now():
    return datetime.now(timezone.utc).isoformat()


def environment():
    return {'python': sys.version, 'fingerprint_algorithm': FINGERPRINT_VERSION,
            'packages': {name: importlib.metadata.version(name)
                         for name in ('numpy', 'scipy', 'Pillow', 'pyarrow', 'requests')}}


def atomic(path, body):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.writing-' + uuid.uuid4().hex)
    with temporary.open('xb') as stream:
        stream.write(body)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def save(path, value):
    atomic(path, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False).encode('utf-8') + b'\n')


def write_lines(path, rows):
    path = Path(path)
    temporary = path.with_name(path.name + '.writing-' + uuid.uuid4().hex)
    with temporary.open('xb') as stream:
        for row in rows:
            stream.write(canonical(row) + b'\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def lines(path):
    with Path(path).open(encoding='utf-8') as stream:
        for line in stream:
            if not line.strip() or not line.endswith('\n'):
                raise ValueError('Neúplný JSONL záznam: ' + str(path))
            yield json.loads(line)


def allowed_plan(path):
    path = Path(path).resolve()
    allowed = (ROOT / 'plans').resolve()
    if path == allowed or not path.is_relative_to(allowed):
        raise ValueError('Cieľ musí byť nový podpriečinok external_testing/plans.')
    return path


@contextmanager
def lock(folder):
    """OS zámok sa uvoľní aj po páde procesu; nepoužíva stale PID výnimky."""
    folder.parent.mkdir(parents=True, exist_ok=True)
    path = folder.parent / ('.' + folder.name + '.lock')
    with path.open('a+b') as stream:
        if stream.tell() == 0:
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


def verify_files(root, files):
    for relative, expected in files.items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root.resolve()) or sha(path) != expected:
            raise ValueError('Zmenený alebo neplatný uzamknutý vstup: ' + str(path))


def verify_preselection(folder):
    guard_other_external_plans(folder)
    path = folder / 'preselection_protocol.json'
    if sha(path) != (folder / 'preselection_protocol.sha256').read_text(encoding='ascii').strip():
        raise ValueError('Zmenený protokol predvýberu.')
    protocol = read(path)
    if protocol['version'] != 'article-external-preselection-v1' or protocol['targets'] != TARGETS or protocol['seed'] != SEED:
        raise ValueError('Nepodporovaný alebo zmenený predvýber.')
    if protocol['max_network_bytes'] != MAX_NETWORK_GIB * 2**30:
        raise ValueError('Zmenený celkový sieťový limit.')
    if protocol['environment'] != environment():
        raise ValueError('Zmenené numerické/prenosové prostredie predvýberu; použite pôvodný interpreter.')
    verify_files(folder, protocol['files'])
    for source, expected in protocol['history_files'].items():
        if sha(source) != expected:
            raise ValueError('Zmenená historická kotva: ' + source)
    for source, expected in protocol['implementation_files'].items():
        if sha(source) != expected:
            raise ValueError('Zmenená implementácia po predvýbere: ' + source)
    return protocol


def guard_other_external_plans(folder):
    """Druhý článkový test vyžaduje rozšírený audit, nikdy tichý nový dátum."""
    base = ROOT / 'plans'
    if not base.exists():
        return
    for previous in base.iterdir():
        if not previous.is_dir() or previous.resolve() == folder.resolve():
            continue
        if ((previous / 'manifest.jsonl').exists() or
                any((previous / 'receipts').glob('*.json')) or
                any((previous / 'images').glob('*'))):
            raise ValueError('Iný článkový externý plán už pristúpil k obrázkom; najprv treba rozšíriť historický audit: ' + str(previous))


def prepare_plan(folder):
    folder = allowed_plan(folder)
    guard_other_external_plans(folder)
    if folder.exists():
        return verify_preselection(folder)
    sources, indexes = collect_history()
    candidates, capacity = candidate_plan(sources, indexes)
    index, coverage = historical_index(sources)
    del index
    temporary = folder.with_name('.' + folder.name + '.planning-' + uuid.uuid4().hex)
    temporary.mkdir(parents=True)
    save(temporary / 'history.json', sources)
    write_lines(temporary / 'candidates.jsonl', candidates)
    def unique_history():
        seen = set()
        for row in history_records(sources):
            checksum = digest(row)
            if checksum not in seen:
                seen.add(checksum)
                yield row
    write_lines(temporary / 'excluded_identities.jsonl', unique_history())
    save(temporary / 'capacity.json', capacity)
    implementation = {str(p): sha(p) for p in [Path(__file__).resolve(), ROOT / 'data_support.py', ROOT / 'freeze.py',
                                               *(PROJECT / name for name in HELPERS)]}
    protocol = {
        'version': 'article-external-preselection-v1', 'created_utc': now(),
        'seed': SEED, 'namespace': 'article-independent-v1', 'targets': TARGETS,
        'class_to_index': {'real': 0, 'fake': 1}, 'project_root': str(PROJECT),
        'destination': str(folder), 'max_network_bytes': MAX_NETWORK_GIB * 2**30,
        'environment': environment(),
        'history_files': {a['path']: a['sha256'] for a in sources['anchors']},
        'history_coverage': coverage, 'implementation_files': implementation,
        'sources': {'CommunityForensics': {'repo': cf.REPO, 'revision': cf.REVISION},
                    'OpenFake': {'repo': of.REPO, 'revision': of.REVISION, 'split': 'core/test'},
                    'DeepShield': {'url': ds.ARCHIVE_URL, 'archive_size': ds.ARCHIVE_SIZE,
                                   'archive_md5': ds.ARCHIVE_MD5, 'platform': 'PreSocial'}},
        'selection': 'Pevné kvóty; deterministické poradie Parquet blokov a riadkov, DeepShield strieda kategórie s 80 % rezervou a globálne odlišnou dependency_key. Výmena iba pri pôvodovom/obsahovom audite, nikdy podľa skóre.',
        'capacity_before_content_audit': capacity,
        'external_inference_performed': False, 'external_metrics_computed': False,
        'limitations': ['Zdroje už boli skúmané; nezávislé sú nové známe skupiny a obrázky, nejde o nové zdroje.',
                       'Vyvážené triedy a pevné kvóty neodhadujú prevalenciu v nasadení.',
                       coverage['phash']['limitation'], coverage['unknown_identity_limitation'],
                       'Zdrojové anotácie sa preberajú bez nezávislej ručnej validácie; parciálny prenos neprepočíta hash celého archívu.'],
        'files': {name: sha(temporary / name) for name in ('history.json', 'candidates.jsonl', 'excluded_identities.jsonl', 'capacity.json')},
    }
    save(temporary / 'preselection_protocol.json', protocol)
    atomic(temporary / 'preselection_protocol.sha256', (sha(temporary / 'preselection_protocol.json') + '\n').encode('ascii'))
    os.rename(temporary, folder)
    print(json.dumps({'plan': str(folder), 'total': sum(TARGETS.values()), 'capacity': capacity,
                      'download_started': False}, ensure_ascii=False, indent=2), flush=True)
    return protocol


class Budget:
    """Kumulatívne rezervované bajty vrátane opakovaní pre všetky obnovenia."""
    def __init__(self, folder, protocol):
        self.path = folder / 'transfer.json'
        self.fingerprint = sha(folder / 'preselection_protocol.json')
        self.limit = protocol['max_network_bytes']
        previous = read(self.path) if self.path.exists() else None
        if previous and previous['preselection_sha256'] != self.fingerprint:
            raise ValueError('Prenosový rozpočet patrí inému predvýberu.')
        self.used = previous['reserved_bytes'] if previous else 0
        if type(self.used) is not int or not 0 <= self.used <= self.limit:
            raise ValueError('Neplatný prenosový rozpočet.')
        self.persist()

    def persist(self):
        save(self.path, {'preselection_sha256': self.fingerprint, 'reserved_bytes': self.used,
                         'max_bytes': self.limit, 'updated_utc': now(),
                         'semantics': 'Horná účtovaná hranica: rezervované Range bajty vrátane neúspešných pokusov; limit platí kumulatívne.'})

    def charge(self, amount):
        amount = int(amount)
        if amount < 0 or self.used + amount > self.limit:
            raise RuntimeError('Dosiahnutý celkový limit 40 GiB; automaticky sa neobnovuje ani nezvyšuje.')
        self.used += amount
        self.persist()


class Transport:
    def __init__(self, budget):
        self.budget = budget
        self.current = self.block = self.remote = None
        self.zip_session = None

    def fetch(self, row):
        dataset = row['dataset']
        if dataset in ('OpenFake', 'CommunityForensics'):
            key = (dataset, row['shard'], row['row_group'])
            if self.current != key:
                self.block = (of if dataset == 'OpenFake' else cf).fetch_group(row, self.budget)
                self.current = key
            raw = self.block[row['row_index']]
            fields = ('label', 'model', 'type', 'row_id', 'image_sha256', 'prompt_kind') if dataset == 'OpenFake' else ('image_name', 'label', 'real_source', 'model_name', 'prompt', 'format', 'resolution')
            if any(raw[field] != row[field] for field in fields):
                raise ValueError('Zdrojové metadáta nezodpovedajú pripnutému indexu.')
            if dataset == 'CommunityForensics':
                return cf.image_bytes(raw['image_data'])
            if hashlib.sha256(str(raw.get('prompt') or '').encode()).hexdigest() != row['prompt_sha256'] or of.prompt_hash(raw.get('prompt')) != row['prompt_normalized_sha256']:
                raise ValueError('Zmenený prompt OpenFake.')
            body = raw['image']['bytes']
            if not isinstance(body, bytes) or hashlib.sha256(body).hexdigest() != row['image_sha256']:
                raise ValueError('Nesúhlasí zdrojový SHA256 OpenFake.')
            return body
        if self.remote is None:
            import requests
            from v4_deepshield_remote_zip import RemoteZip
            self.budget.charge(2 * 1024**2)  # Pevná horná hranica zdrojových metadát.
            ds.verify_source_remote()
            budget = self.budget
            class MeteredSession(requests.Session):
                def get(self, url, **kwargs):
                    value = kwargs.get('headers', {}).get('Range', '')
                    if not value.startswith('bytes='):
                        raise ValueError('DeepShield prenos vyžaduje Range.')
                    first, last = map(int, value[6:].split('-'))
                    budget.charge(last - first + 1)
                    return super().get(url, **kwargs)
            self.zip_session = MeteredSession()
            self.remote = RemoteZip(ds.ARCHIVE_URL, ds.ARCHIVE_SIZE,
                                    max_total_bytes=self.budget.limit, transport=self.zip_session)
        return self.remote.read_member(row['entry'])

    def close(self):
        if self.remote:
            self.remote.close()
        if self.zip_session:
            self.zip_session.close()


def receipt_read(path):
    value = read(path)
    if digest(value['record']) != value['record_sha256']:
        raise ValueError('Poškodené potvrdenie kandidáta.')
    return value['record']


def manifest_record(row, path, fp):
    dataset = row['dataset']
    label = int(row['label'] == 'fake') if dataset == 'OpenFake' else row['label']
    dependencies = {row['source_key'], block_key(dataset, row)}
    if dataset == 'DeepShield':
        dependencies.add(row['dependency_key'])
    return {'sample_id': 'article-external/' + row['original_id'], 'original_id': row['original_id'],
            'dataset': dataset, 'label': label, 'bucket': row['bucket'], 'path': str(path.resolve()),
            'source_key': row['source_key'], 'sampling_group': block_key(dataset, row),
            'dependency_keys': sorted(dependencies), 'input_spec': {'version': 1, 'id': 'clean', 'operations': []},
            **fp}


def finalize(folder, protocol, accepted, excluded, receipts, coverage):
    rows = connected_groups(accepted)
    if Counter(r['bucket'] for r in rows) != Counter(TARGETS):
        raise ValueError('Obsahový audit nepokryl pevné kvóty; nesmú sa znížiť.')
    # Live supervisor sleduje audit.json. Ten sa publikuje až ako posledný
    # marker; nezávislý validátor dovtedy kontroluje izolovaný staging.
    stage = folder / ('.finalizing-' + uuid.uuid4().hex)
    stage.mkdir()
    write_lines(stage / 'manifest.jsonl', rows)
    write_lines(stage / 'candidate_exclusions.jsonl', excluded)
    save(stage / 'receipts.json', receipts)
    copied = ('excluded_identities.jsonl', 'preselection_protocol.json',
              'preselection_protocol.sha256', 'transfer.json')
    for name in copied:
        shutil.copyfile(folder / name, stage / name)
    audit = {'version': 'article-external-data-audit-v1', 'created_utc': now(),
             'ready_for_freeze': True, 'history_inventory_complete': True,
             'all_images_verified': True, 'score_independent_selection': True,
             'manifest_sha256': sha(stage / 'manifest.jsonl'),
             'history_files': protocol['history_files'], 'history_coverage': coverage,
             'phash': {'bits': 64, 'max_hamming_distance': 4, 'compared_with_history': True,
                       'coverage': 'all available historical pHash; missing values reported in history_coverage'},
             'disjointness': dict.fromkeys(('group', 'original_id', 'source_key', 'sha256',
                                          'rgb_sha256', 'dependency_keys', 'phash_unresolved'), 0),
             'targets': TARGETS, 'actual_counts': dict(Counter(r['bucket'] for r in rows)),
             'connected_groups_by_dataset': {d: len({r['group'] for r in rows if r['dataset'] == d})
                                             for d in ('OpenFake', 'DeepShield', 'CommunityForensics')},
             'content_rejections': len(excluded), 'external_inference_performed': False,
             'external_metrics_computed': False}
    save(stage / 'audit.json', audit)
    final = {'version': 'article-independent-external-data-v1', 'created_utc': now(),
             'preselection_protocol_sha256': sha(folder / 'preselection_protocol.json'),
             'targets': TARGETS, 'external_inference_performed': False,
             'external_metrics_computed': False,
             'files': {name: sha(stage / name) for name in ('manifest.jsonl', 'audit.json',
                         'excluded_identities.jsonl', 'candidate_exclusions.jsonl', 'receipts.json',
                         'preselection_protocol.json', 'preselection_protocol.sha256', 'transfer.json')}}
    save(stage / 'protocol.json', final)
    atomic(stage / 'protocol.sha256', (sha(stage / 'protocol.json') + '\n').encode('ascii'))
    from freeze import verify_external_plan
    verify_external_plan(stage, {})
    save(stage / 'complete.json', {'protocol_sha256': sha(stage / 'protocol.json'),
                                  'total': len(rows), 'external_inference_performed': False})
    for name in ('manifest.jsonl', 'candidate_exclusions.jsonl', 'receipts.json',
                 'protocol.json', 'protocol.sha256', 'complete.json'):
        os.replace(stage / name, folder / name)
    os.replace(stage / 'audit.json', folder / 'audit.json')
    # Iba jednotlivé vytvorené kópie v overenom vlastnom staging priečinku.
    if not stage.resolve().is_relative_to(folder.resolve()):
        raise ValueError('Neplatný staging priečinok.')
    for name in copied:
        (stage / name).unlink()
    stage.rmdir()
    return audit


def download(folder, max_new=None):
    protocol = verify_preselection(folder)
    if (folder / 'complete.json').exists() and (folder / 'audit.json').exists():
        return check(folder)
    historical, coverage = historical_index(read(folder / 'history.json'))
    internal = _MemoryIndex()
    used, blocked = set(), set()
    counts = Counter()
    accepted, excluded, receipt_hashes = [], [], {}
    budget = Budget(folder, protocol)
    transport = Transport(budget)
    new = 0
    def progress(error=None):
        save(folder / 'progress.json', {'updated_utc': now(), 'accepted': dict(counts),
                    'required': TARGETS, 'new_candidates_this_run': new, 'reserved_transfer_bytes': budget.used,
                    'complete': counts == Counter(TARGETS), 'error': error})
    try:
        for row in lines(folder / 'candidates.jsonl'):
            if counts[row['bucket']] >= TARGETS[row['bucket']]:
                continue
            key = digest(row['original_id'])
            path = folder / 'images' / (key + '.img')
            receipt = folder / 'receipts' / (key + '.json')
            if receipt.exists():
                result = receipt_read(receipt)
                if result['candidate_sha256'] != digest(row) or result['preselection_sha256'] != budget.fingerprint:
                    raise ValueError('Potvrdenie nepatrí uzamknutému kandidátovi.')
                if result['image_sha256'] and (not path.is_file() or sha(path) != result['image_sha256']):
                    raise ValueError('Zmenený už načítaný obrázok.')
            else:
                if max_new is not None and new >= max_new:
                    progress()
                    return {'complete': False, 'accepted': sum(counts.values()), 'reason': 'max_new'}
                fp, reasons = None, []
                if row['source_key'] in used or row['source_key'] in blocked:
                    reasons = [{'kind': 'known_origin', 'source_key': row['source_key']}]
                else:
                    atomic(path, transport.fetch(row))
                    try:
                        fp = fingerprint(path)
                    except (ValueError, OSError) as error:
                        reasons = [{'kind': 'invalid_image', 'detail': str(error)}]
                    if fp:
                        reasons = [{**m, 'index': 'history'} for m in historical.matches(fp)] + [{**m, 'index': 'new_external'} for m in internal.matches(fp)]
                result = {'candidate_sha256': digest(row), 'preselection_sha256': budget.fingerprint,
                          'original_id': row['original_id'], 'decision': 'excluded' if reasons else 'accepted',
                          'fingerprint': fp, 'reasons': reasons, 'created_utc': now(),
                          'image_sha256': sha(path) if path.exists() else None}
                save(receipt, {'record': result, 'record_sha256': digest(result)})
                new += 1
            receipt_hashes['receipts/' + receipt.name] = sha(receipt)
            if result['decision'] == 'accepted':
                fp = result['fingerprint']
                if (row['source_key'] in used or row['source_key'] in blocked or
                        not fp or fp['sha256'] != result['image_sha256'] or
                        historical.matches(fp) or internal.matches(fp)):
                    raise ValueError('Potvrdený výber nezodpovedá obsahovému auditu.')
                # Na obnovenom behu sa preverí aj dekódovaný obsah a pHash.
                if fingerprint(path) != fp:
                    raise ValueError('Zmenený dekódovaný obrázok alebo algoritmus odtlačku.')
                used.add(row['source_key'])
                internal.add(fp, row['original_id'])
                accepted.append(manifest_record(row, path, fp))
                counts[row['bucket']] += 1
                if len(accepted) % 100 == 0:
                    progress()
                    print(f'Externé dáta: {len(accepted)}/{sum(TARGETS.values())}; rezervovaný prenos {budget.used/2**30:.2f} GiB', flush=True)
            elif result['decision'] == 'excluded':
                blocked.add(row['source_key'])
                excluded.append({'original_id': row['original_id'], 'source_key': row['source_key'],
                                 'bucket': row['bucket'], 'reasons': result['reasons'], 'fingerprint': result['fingerprint']})
            else:
                raise ValueError('Neplatné rozhodnutie potvrdenia.')
        progress()
        if counts != Counter(TARGETS):
            save(folder / 'shortfall.json', {'accepted': dict(counts), 'required': TARGETS})
            raise ValueError('Po obsahovom audite chýbajú kandidáti; kvóty sa automaticky nemenia.')
        # Uzavretie proti zmene histórie počas dlhého prenosu.
        verify_preselection(folder)
        return finalize(folder, protocol, accepted, excluded, receipt_hashes, coverage)
    except Exception as error:
        progress(str(error))
        raise
    finally:
        transport.close()


def check(folder):
    protocol = verify_preselection(folder)
    if not (folder / 'complete.json').exists():
        raise ValueError('Dáta ešte nie sú dokončené; predvýber je overený.')
    if sha(folder / 'protocol.json') != (folder / 'protocol.sha256').read_text(encoding='ascii').strip():
        raise ValueError('Zmenený finálny protokol.')
    final = read(folder / 'protocol.json')
    complete = read(folder / 'complete.json')
    if (complete['protocol_sha256'] != sha(folder / 'protocol.json') or
            final['preselection_protocol_sha256'] != sha(folder / 'preselection_protocol.json') or
            final['targets'] != TARGETS or complete['external_inference_performed'] is not False):
        raise ValueError('Dokončenie nepatrí uzamknutému externému protokolu.')
    verify_files(folder, final['files'])
    receipts = read(folder / 'receipts.json')
    verify_files(folder, receipts)
    historical, coverage = historical_index(read(folder / 'history.json'))
    internal = _MemoryIndex()
    rows, origins = [], set()
    candidates = {row['original_id']: row for row in lines(folder / 'candidates.jsonl')}
    for row in lines(folder / 'manifest.jsonl'):
        candidate = candidates.get(row['original_id'])
        if candidate is None:
            raise ValueError('Externý manifest obsahuje obrázok mimo predvýberu.')
        receipt_path = 'receipts/' + digest(row['original_id']) + '.json'
        if receipt_path not in receipts:
            raise ValueError('Externému manifestu chýba potvrdenie kandidáta.')
        receipt = receipt_read(folder / receipt_path)
        if (receipt['candidate_sha256'] != digest(candidate) or receipt['decision'] != 'accepted' or
                receipt['preselection_sha256'] != final['preselection_protocol_sha256']):
            raise ValueError('Potvrdenie a manifest nepatria rovnakému výberu.')
        fp = fingerprint(Path(row['path']))
        expected_path = folder / 'images' / (digest(row['original_id']) + '.img')
        expected = manifest_record(candidate, expected_path, fp)
        if (any(row.get(k) != v for k, v in expected.items()) or receipt['fingerprint'] != fp or
                receipt['image_sha256'] != fp['sha256'] or historical.matches(fp) or internal.matches(fp)):
            raise ValueError('Nevyhovujúci externý obsah: ' + row['original_id'])
        if row['source_key'] in origins:
            raise ValueError('Opakovaný pôvod v externom teste.')
        origins.add(row['source_key'])
        internal.add(fp, row['original_id'])
        rows.append(row)
    if Counter(r['bucket'] for r in rows) != Counter(TARGETS) or rows != connected_groups(rows):
        raise ValueError('Nesúhlasia kvóty alebo spojené skupiny.')
    audit = read(folder / 'audit.json')
    if audit['manifest_sha256'] != sha(folder / 'manifest.jsonl') or audit['history_coverage'] != coverage:
        raise ValueError('Audit nepatrí manifestu alebo aktuálnej histórii.')
    # Nezávislý validátor identít a odtlačkov z modulu vlastníka freeze.
    from freeze import verify_external_plan
    verify_external_plan(folder, {})
    return {'complete': True, 'total': len(rows), 'ready_for_freeze': True,
            'external_inference_performed': False, 'preselection_sha256': sha(folder / 'preselection_protocol.json')}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, default=DEFAULT_PLAN)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--plan-only', action='store_true')
    action.add_argument('--download', action='store_true')
    action.add_argument('--check', action='store_true')
    parser.add_argument('--max-new', type=int, help='Počet nových kandidátov v tomto spustení; checkpoint je obnoviteľný.')
    args = parser.parse_args(argv)
    if args.max_new is not None and (not args.download or args.max_new <= 0):
        parser.error('--max-new musí byť kladný a patrí iba k --download.')
    folder = allowed_plan(args.plan)
    with lock(folder):
        result = prepare_plan(folder) if args.plan_only else download(folder, args.max_new) if args.download else check(folder)
    if not args.plan_only:
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
