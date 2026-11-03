"""Reprodukovateľný metadátový predvýber; žiadna sieť, inferencia ani učenie.

Staré dáta a kód sa iba čítajú. Predvýber nie je obsahovo overený dataset.
Štatistický protokol a pravidlo adopcie selektora tento skript neurčuje.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
PROJECT = Path(__file__).resolve().parents[1]
NAMESPACE = 'ablation-confirmation-v1'
SEED = 20261028
DS_TARGETS = {'FFHQ': 150, 'FORLAB': 600, 'StyleGAN': 75, 'StyleGAN2': 75,
              'StyleGAN3': 75, 'FLUX.1': 105, 'StableDiffusion1.5': 105,
              'StableDiffusion2': 105, 'StableDiffusion3': 105,
              'StableDiffusionXL': 105}
OF_TARGETS = {'real/docci': 3000, 'fake/flux.2-klein-9b': 600,
              'fake/gpt-image-1.5': 600, 'fake/illustrious': 600,
              'fake/midjourney-7': 600, 'fake/z-image-turbo': 600}
TARGETS = {**{'OpenFake/' + k: v for k, v in OF_TARGETS.items()},
           **{'DeepShield/' + ('real/' if k in ('FFHQ', 'FORLAB') else 'fake/') + k: v
              for k, v in DS_TARGETS.items()}}
OLD_PLAN = 'article_detector_experiment/external_testing/plans/independent_20261027'
VERSION = 'article-ablation-candidate-plan-v1'


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(',', ':'), allow_nan=False).encode('utf-8')


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


def anchor(path, anchors, expected=None):
    path = Path(path).resolve()
    actual = sha(path)
    if expected is not None and actual != expected:
        raise ValueError('Zmenený historický vstup: ' + str(path))
    if str(path) in anchors and anchors[str(path)] != actual:
        raise ValueError('Vstup sa zmenil počas auditu: ' + str(path))
    anchors[str(path)] = actual
    return actual


def verify_file_map(folder, mapping, anchors):
    folder = Path(folder).resolve()
    for name, expected in mapping.items():
        path = (folder / name).resolve()
        if not path.is_relative_to(folder):
            raise ValueError('Referencia mimo protokolu: ' + name)
        anchor(path, anchors, expected)


def sidecar(folder, name, anchors):
    expected = (folder / (name + '.sha256')).read_text(encoding='ascii').strip()
    actual = anchor(folder / (name + '.json'), anchors, expected)
    anchor(folder / (name + '.sha256'), anchors)
    return read(folder / (name + '.json')), actual


def source_inventory(project, target):
    """Nová neznáma kohorta nesmie byť potichu vynechaná z histórie."""
    inventory = []
    old_root = project / 'article_detector_experiment/external_testing/plans'
    for folder in sorted(old_root.iterdir()):
        if not folder.is_dir():
            continue
        touched = (folder / 'manifest.jsonl').exists() or (folder / 'receipts.json').exists()
        touched |= any((folder / 'images').glob('*')) or any((folder / 'receipts').glob('*.json'))
        if touched and folder.resolve() != (project / OLD_PLAN).resolve():
            raise ValueError('Nový externý plán vyžaduje rozšíriť audit: ' + str(folder))
        if touched:
            inventory.append(str(folder.resolve()))
    own = project / 'article_ablation_confirmation'
    for folder in own.glob('data_plan_*'):
        if folder.resolve() == target.resolve():
            continue
        if any((folder / 'images').glob('*')) or any((folder / 'receipts').glob('*')):
            raise ValueError('Predchádzajúci ablačný download vyžaduje nový audit: ' + str(folder))
    for path in sorted((project / 'article_detector_experiment/runs').glob('*/plans/*.jsonl')):
        inventory.append(str(path.resolve()))
    for path in sorted((project / 'article_reference_experiment/runs').glob('*/plan.jsonl')):
        inventory.append(str(path.resolve()))
    return inventory


def load_helpers(project, preselection, anchors):
    # Import sa smie vykonať až po kontrole zmrazených súčtov legacy kódu.
    for name, expected in preselection['implementation_files'].items():
        anchor(name, anchors, expected)
    helper = project / 'article_detector_experiment/external_testing/data_support.py'
    if str(helper) not in preselection['implementation_files']:
        raise ValueError('Pomocný modul nie je uzamknutý.')
    spec = importlib.util.spec_from_file_location('_ablation_metadata_legacy', helper)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verified_history(project, anchors):
    old = project / OLD_PLAN
    protocol, protocol_sha = sidecar(old, 'protocol', anchors)
    preselection, pre_sha = sidecar(old, 'preselection_protocol', anchors)
    if protocol['preselection_protocol_sha256'] != pre_sha:
        raise ValueError('Nesúlad predvýberu a finálneho protokolu.')
    verify_file_map(old, protocol['files'], anchors)
    verify_file_map(old, preselection['files'], anchors)
    complete = read(old / 'complete.json')
    anchor(old / 'complete.json', anchors)
    if complete['protocol_sha256'] != protocol_sha or complete['total'] != 6800:
        raise ValueError('Historický článkový dataset nie je dokončený.')
    for mapping in (preselection['history_files'], read(old / 'audit.json')['history_files']):
        for name, expected in mapping.items():
            anchor(name, anchors, expected)
    history = read(old / 'history.json')
    for item in history['anchors']:
        anchor(item['path'], anchors, item['sha256'])
    helpers = load_helpers(project, preselection, anchors)
    return history, helpers, preselection


def add_identity(row, state, helpers):
    for name in ('sha256', 'image_sha256'):
        if row.get(name):
            state['hashes'].add(row[name])
    if row.get('original_id') or row.get('sample_id'):
        state['original_ids'].add(row.get('original_id') or row['sample_id'])
    for name in ('source_key', 'group'):
        if row.get(name):
            group = helpers.normalized_group(row[name])
            state['groups'].add(group)
            if group.startswith('prompt/'):
                state['prompts'].add(group.split('/', 1)[1])
    for name in ('prompt_normalized_sha256',):
        if row.get(name):
            state['prompts'].add(row[name])
    if row.get('fingerprint'):
        add_identity(row['fingerprint'], state, helpers)


def new_state(history):
    fields = {'groups': 'protected_groups', 'prompts': 'protected_prompt_hashes',
              'hashes': 'protected_image_hashes', 'original_ids': 'protected_original_ids',
              'deep_groups': 'protected_deepshield_groups',
              'deep_dependencies': 'protected_deepshield_dependencies'}
    result = {key: set(history[value]) for key, value in fields.items()}
    result['of_blocks'] = {tuple(x) for x in history['protected_openfake_blocks']}
    result['cf_blocks'] = {tuple(x) for x in history['protected_cf_blocks']}
    return result


def audit_csv_and_reused_plans(project, history, state, helpers, anchors):
    known = set()
    coverage = Counter()
    for name in history['csv_inventories']:
        if str(Path(name).resolve()) not in anchors:
            raise ValueError('Historický CSV nemá kotvu: ' + name)
        with Path(name).open(encoding='utf-8-sig', newline='') as stream:
            for row in csv.DictReader(stream):
                coverage['rows'] += 1
                for field in ('sha256', 'rgb_sha256', 'phash'):
                    coverage[field] += bool(row.get(field))
                if row.get('sha256'):
                    known.add(row['sha256'])
                add_identity(row, state, helpers)
    for row in history['extra_fingerprints']:
        add_identity(row, state, helpers)
    plans = []
    for pattern in ('article_detector_experiment/runs/*/plans/*.jsonl',
                    'article_reference_experiment/runs/*/plan.jsonl'):
        for path in sorted(project.glob(pattern)):
            expected = anchors.get(str(path.resolve()))
            if 'article_reference_experiment' in path.parts:
                parent = path.parent
                protocol, protocol_sha = sidecar(parent, 'protocol', anchors)
                expected = protocol['files']['plan.jsonl']
                if (parent / 'complete.json').exists():
                    complete = read(parent / 'complete.json')
                    if not complete.get('complete') or complete['inputs']['protocol_sha256'] != protocol_sha:
                        raise ValueError('Neplatný dokončený retrospektívny beh.')
                    anchor(parent / 'complete.json', anchors)
            elif expected is None:
                raise ValueError('Neznámy článkový vývojový plán: ' + str(path))
            anchor(path, anchors, expected)
            count = 0
            for row in lines(path):
                if row['sha256'] not in known:
                    raise ValueError('Článkový beh obsahuje novú neauditovanú vzorku: ' + str(path))
                add_identity(row, state, helpers)
                count += 1
            plans.append({'path': str(path), 'rows': count, 'all_sha256_in_historical_csv': True})
    return dict(coverage), plans


def audit_receipts(project, state, helpers, anchors):
    old = project / OLD_PLAN
    pre_sha = anchors[str((old / 'preselection_protocol.json').resolve())]
    candidates = {r['original_id']: r for r in lines(old / 'candidates.jsonl')}
    accepted = {r['original_id']: r for r in lines(old / 'manifest.jsonl')}
    excluded = {r['original_id']: r for r in lines(old / 'candidate_exclusions.jsonl')}
    if set(accepted) & set(excluded):
        raise ValueError('Vzorka je súčasne prijatá aj vyradená.')
    decisions = {**accepted, **excluded}
    receipt_files = read(old / 'receipts.json')
    actual_files = {str(p.relative_to(old)).replace('\\', '/') for p in (old / 'receipts').glob('*.json')}
    if actual_files != set(receipt_files):
        raise ValueError('Chýbajúci alebo osirelý historický receipt.')
    touched = []
    seen = set()
    image_paths = set()
    counts = Counter()
    for name, expected in sorted(receipt_files.items()):
        path = old / name
        anchor(path, anchors, expected)
        receipt = read(path)
        record = receipt['record']
        if digest(record) != receipt['record_sha256']:
            raise ValueError('Zmenený receipt record: ' + name)
        identity = record['original_id']
        if identity in seen or identity not in decisions or identity not in candidates:
            raise ValueError('Receipt nemá jednoznačnú rozhodnutú vzorku.')
        candidate = candidates[identity]
        if (record['candidate_sha256'] != digest(candidate) or
                record['preselection_sha256'] != pre_sha or
                name != 'receipts/' + digest(identity) + '.json'):
            raise ValueError('Receipt nie je viazaný na uzamknutého kandidáta.')
        is_accepted = identity in accepted
        if record['decision'] != ('accepted' if is_accepted else 'excluded'):
            raise ValueError('Rozpor rozhodnutia a manifestu.')
        fp = record.get('fingerprint')
        stored_fp = ({k: decisions[identity][k] for k in fp} if is_accepted and fp else decisions[identity].get('fingerprint'))
        if fp != stored_fp:
            raise ValueError('Rozpor odtlačkov receipt a rozhodnutia.')
        if not is_accepted and record['reasons'] != decisions[identity]['reasons']:
            raise ValueError('Rozpor dôvodov vyradenia.')
        raw_sha = record.get('image_sha256')
        raw_path = old / 'images' / (digest(identity) + '.img')
        if raw_sha:
            anchor(raw_path, anchors, raw_sha)
            image_paths.add(raw_path.resolve())
            counts['downloaded'] += 1
            if not fp:
                counts['downloaded_without_rgb_phash'] += 1
            if fp and fp['sha256'] != raw_sha:
                raise ValueError('Rozpor raw SHA a dekódovaného odtlačku.')
        elif raw_path.exists() or is_accepted:
            raise ValueError('Stiahnutý obrázok nemá raw SHA.')
        add_identity(candidate, state, helpers)
        add_identity(record, state, helpers)
        if candidate['dataset'] == 'DeepShield':
            state['deep_groups'].add(candidate['original_group'])
            state['deep_dependencies'].add(candidate['dependency_key'])
        else:
            # Celý blok už bol použitý aj pri odmietnutí pred dekódovaním.
            key = 'of_blocks' if candidate['dataset'] == 'OpenFake' else 'cf_blocks'
            state[key].add((candidate['shard'], candidate['row_group']))
        touched.append({'original_id': identity, 'dataset': candidate['dataset'],
                        'source_key': candidate['source_key'],
                        'dependency_keys': [candidate['source_key'], candidate['sampling_group']],
                        'decision': record['decision'], 'fingerprint': fp,
                        'image_sha256': raw_sha, 'path': str(raw_path) if raw_sha else None,
                        'candidate_sha256': record['candidate_sha256'],
                        'receipt_path': str(path), 'receipt_sha256': expected,
                        'record_sha256': receipt['record_sha256']})
        seen.add(identity)
        counts[record['decision']] += 1
    if seen != set(decisions):
        raise ValueError('Rozhodnutie bez receipt.')
    if image_paths != {p.resolve() for p in (old / 'images').glob('*') if p.is_file()}:
        raise ValueError('Neauditovaný historický obrazový súbor.')
    expected = {'accepted': 6800, 'excluded': 169, 'downloaded': 6926,
                'downloaded_without_rgb_phash': 100}
    if any(counts[k] != value for k, value in expected.items()):
        raise ValueError('Historická kohorta sa zmenila: ' + str(dict(counts)))
    return touched, dict(counts)


def metadata_closure(indexes, state, helpers):
    """Bloky expandujú na susedov; identity uzatvárame do pevného bodu.

    Iný blok sa nestane historicky načítaným iba preto, že obsahuje jednu
    historickú identitu. Taká identita sa však nikdy nesmie vybrať v ňom.
    """
    rows = []
    for dataset, module, block_field in (
            ('OpenFake', helpers.of, 'of_blocks'),
            ('CommunityForensics', helpers.cf, 'cf_blocks')):
        for row in indexes[dataset]['rows']:
            group = helpers.normalized_group(module.metadata_group(row))
            prompt = row.get('prompt_normalized_sha256') if dataset == 'OpenFake' else helpers.of.prompt_hash(row.get('prompt'))
            identity = ('openfake/' + helpers.of.REVISION + '/' + row['row_id']
                        if dataset == 'OpenFake' else helpers.cf.row_identity(row))
            rows.append((group, prompt, row.get('image_sha256'), identity,
                         (row['shard'], row['row_group']) in state[block_field]))
    protected_rows = set()
    iterations = 0
    while True:
        before = tuple(len(state[k]) for k in ('groups', 'prompts', 'hashes', 'original_ids'))
        iterations += 1
        for index, (group, prompt, raw_sha, identity, touched_block) in enumerate(rows):
            if not (touched_block or group in state['groups'] or
                    (prompt and prompt in state['prompts']) or
                    (raw_sha and raw_sha in state['hashes']) or identity in state['original_ids']):
                continue
            protected_rows.add(index)
            state['groups'].add(group)
            state['original_ids'].add(identity)
            if prompt:
                state['prompts'].add(prompt)
            if raw_sha:
                state['hashes'].add(raw_sha)
        if before == tuple(len(state[k]) for k in ('groups', 'prompts', 'hashes', 'original_ids')):
            break
    return {'fixed_point_iterations': iterations, 'protected_metadata_rows': len(protected_rows),
            'scope': 'Celé historicky načítané bloky → všetci susedia; tranzitívne group/prompt/raw SHA/original_id naprieč OF a CF. Blok obsahujúci historickú identitu nie je automaticky označený ako načítaný; samotná identita je vylúčená všade.'}


def make_candidates(indexes, state, helpers):
    rows = []
    capacity = defaultdict(set)
    for raw in helpers.of.make_plan(indexes['OpenFake'], state['prompts'], SEED):
        if ((raw['shard'], raw['row_group']) in state['of_blocks'] or
                raw['group'] in state['groups'] or raw['image_sha256'] in state['hashes'] or
                raw['sample_id'] in state['original_ids']):
            continue
        rows.append({**raw, 'dataset': 'OpenFake', 'bucket': 'OpenFake/' + raw['bucket'],
                     'original_id': raw['sample_id'], 'source_key': raw['group'],
                     'dependency_keys': [raw['group'], raw['sampling_group']]})
        capacity['OpenFake/' + raw['bucket']].add(raw['group'])
    rows.sort(key=lambda r: (digest([NAMESPACE, SEED, 'block', r['shard'], r['row_group']]),
                             digest([NAMESPACE, SEED, 'row', r['row_id']])))
    sources = {'protected_deepshield_groups': sorted(state['deep_groups']),
               'protected_deepshield_dependencies': sorted(state['deep_dependencies'])}
    deep = helpers.confirmation.make_deepshield_plan(
        indexes['DeepShield'], sources, seed=[NAMESPACE, SEED], targets=DS_TARGETS, reserve=1.25)
    for entry in indexes['DeepShield']:
        row = helpers.ds.metadata(entry)
        if (row and row['generator'] in DS_TARGETS and row['group'] not in state['deep_groups'] and
                row['dependency_key'] not in state['deep_dependencies']):
            capacity['DeepShield/' + ('fake/' if row['label'] else 'real/') + row['generator']].add(row['dependency_key'])
    for raw in deep:
        identity = 'deepshield-v4/' + raw['original_group']
        rows.append({**raw, 'dataset': 'DeepShield', 'original_id': identity,
                     'source_key': raw['group'], 'dependency_keys': [raw['group'], raw['dependency_key']]})
    counts = Counter()
    used = set()
    selected = []
    reserves = []
    all_blocks = {}
    picked_blocks = {}
    for index, row in enumerate(rows):
        row['candidate_order'] = index
        row['content_verified'] = False
        row['input_spec'] = {'version': 1, 'id': 'clean', 'operations': []}
        key = row['source_key']
        choose = counts[row['bucket']] < TARGETS[row['bucket']] and key not in used
        row['metadata_role'] = 'quota_candidate' if choose else 'ordered_reserve'
        row['metadata_eligible_before_content_audit'] = True
        if choose:
            counts[row['bucket']] += 1
            used.add(key)
            selected.append(row)
        else:
            reserves.append(row)
        if row['dataset'] == 'OpenFake':
            all_blocks[row['sampling_group']] = int(row['row_group_image_bytes'])
            if choose:
                picked_blocks[row['sampling_group']] = int(row['row_group_image_bytes'])
    if counts != Counter(TARGETS) or len(deep) != 1880:
        raise ValueError('Predvýber nespĺňa pevné kvóty alebo rezervu.')
    if len(all_blocks) != 242:
        raise ValueError('Zmenila sa kapacita nových OpenFake blokov: ' + str(len(all_blocks)))
    deep_bytes = sum(r['entry']['compressed'] + 1024 for r in selected if r['dataset'] == 'DeepShield')
    deep_pool_bytes = sum(r['entry']['compressed'] + 1024 for r in deep)
    blocks = [{'sampling_group': k, 'compressed_image_column_bytes': value,
               'metadata_initially_selected': k in picked_blocks}
              for k, value in sorted(all_blocks.items(), key=lambda pair: next(r['candidate_order'] for r in rows if r.get('sampling_group') == pair[0]))]
    statistics = {'available_unique_origins_by_bucket': {k: len(v) for k, v in sorted(capacity.items())},
                  'quota_counts': dict(sorted(counts.items())), 'candidate_rows': len(rows),
                  'selected_rows': len(selected), 'ordered_reserve_rows': len(reserves),
                  'deepshield_pool_candidates': len(deep),
                  'deepshield_selected_unique_dependencies': len({r['dependency_key'] for r in selected if r['dataset'] == 'DeepShield'}),
                  'openfake_reserve_blocks': len(all_blocks), 'openfake_initial_blocks': len(picked_blocks),
                  'transfer_estimates_bytes': {'openfake_initial_columns': sum(picked_blocks.values()),
                                               'openfake_all_reserve_columns': sum(all_blocks.values()),
                                               'deepshield_initial_members': deep_bytes,
                                               'deepshield_all_reserve_members': deep_pool_bytes},
                  'estimate_note': 'Komprimované stĺpce/členy ZIP + 1024 bajtov na člen DS; bez ďalších metadát, opakovaní a réžie. Nie je to príkaz na stiahnutie všetkých rezerv.'}
    return rows, selected, reserves, blocks, statistics


def write_json(path, value):
    with Path(path).open('xb') as stream:
        stream.write(canonical(value) + b'\n')


def write_lines(path, values):
    with Path(path).open('xb') as stream:
        for value in values:
            stream.write(canonical(value) + b'\n')


def prepare(project, destination):
    project = Path(project).resolve()
    destination = Path(destination).resolve()
    allowed = project / 'article_ablation_confirmation'
    if destination.parent != allowed or not destination.name.startswith('data_plan_'):
        raise ValueError('Výstup musí byť nový article_ablation_confirmation/data_plan_* priečinok.')
    if destination.exists():
        raise FileExistsError('Existujúci plán sa neprepisuje: ' + str(destination))
    inventory = source_inventory(project, destination)
    anchors = {}
    history, helpers, preselection = verified_history(project, anchors)
    state = new_state(history)
    csv_coverage, reused = audit_csv_and_reused_plans(project, history, state, helpers, anchors)
    touched, receipt_counts = audit_receipts(project, state, helpers, anchors)
    indexes = {'OpenFake': helpers.of.load_source_index(project / helpers.of.INDEX_FILE),
               'CommunityForensics': helpers.cf.load_source_index(project / helpers.cf.INDEX_FILE),
               'DeepShield': helpers.ds.load_index(project / helpers.ds.INDEX_FILE)}
    for name in (helpers.of.INDEX_FILE, helpers.cf.INDEX_FILE, helpers.ds.INDEX_FILE):
        if str(project / name) not in anchors:
            raise ValueError('Zdrojový index nemá historickú kotvu: ' + name)
    closure = metadata_closure(indexes, state, helpers)
    rows, selected, reserves, blocks, capacity = make_candidates(indexes, state, helpers)
    audit = {'version': VERSION, 'metadata_only': True, 'ready_for_inference': False,
             'ready_for_freeze': False, 'all_images_verified': False, 'content_verified': False,
             'score_independent_selection': True, 'new_images_downloaded': 0,
             'new_inference_performed': False, 'statistical_protocol_locked': False,
             'adoption_rule_locked': False, 'targets': TARGETS, 'capacity': capacity,
             'history': {'base_plan': str(project / OLD_PLAN), 'csv_coverage': csv_coverage,
                         'latest_article_receipts': receipt_counts, 'reused_article_plans': reused,
                         'protected_counts': {k: len(v) for k, v in state.items()},
                         'dependency_closure': closure,
                         'phash': {'historical_csv_available': csv_coverage['phash'],
                                   'historical_csv_total': csv_coverage['rows'],
                                   'historical_csv_missing': csv_coverage['rows'] - csv_coverage['phash'],
                                   'latest_downloaded_with_full_fingerprint': receipt_counts['downloaded'] - receipt_counts['downloaded_without_rgb_phash'],
                                   'latest_downloaded_without_rgb_phash': receipt_counts['downloaded_without_rgb_phash'],
                                   'limitation': 'CSV počty sú riadky inventárov, nie jedinečné obrázky. Chýbajúce historické RGB/pHash sa nedopočítali; 100 historicky odmietnutých viacstranových/animovaných obrázkov je chránených raw SHA a pôvodom. Noví kandidáti ešte nemajú overené SHA/RGB/pHash. Budúci audit musí porovnať všetky dostupné hodnoty vrátane pHash Hamming <= 4.'}},
             'cohort_composition': {'DeepShield': 'Nová kohorta: real FFHQ/FORLAB 20/80 %, fake GAN/difúzia 30/70 %. Nie je to presná replikácia starých 10000 (30/70 % a 54/46 %).',
                                    'OpenFake': '3000 DOCCI real + 600 z každého z piatich generátorov; triedy 50/50 %.'},
             'limitations': ['Metadátová kapacita nie je zárukou prijatia po obsahovom audite.',
                             'OF obrázky v rovnakých blokoch alebo pôvodových komponentoch nie sú nezávislé replikácie; skupiny sa musia uzavrieť pre párovú analýzu.',
                             'DS päť difúznych generátorov zdieľa identity: kapacitu dependency union nemožno násobiť piatimi.',
                             'Ide o nové vzorky zo známych zdrojov, nie o doposiaľ nevidené generátory.',
                             'Neznáme identity osôb, sémantické väzby a predtrénovacie dáta sa týmto auditom nevylučujú.',
                             'Modely, endpointy, štatistika a pravidlo adopcie nie sú súčasťou tohto metadátového predvýberu.']}
    # Recheck pred publikovaním, vrátane všetkých raw súborov a receipts.
    for name, expected in list(anchors.items()):
        anchor(name, anchors, expected)
    if inventory != source_inventory(project, destination):
        raise ValueError('Počas auditu vznikla nová história.')
    implementation = Path(__file__).resolve()
    anchor(implementation, anchors)
    destination.mkdir(parents=False, exist_ok=False)
    (destination / 'code').mkdir()
    (destination / 'code/prepare_candidate_plan.py').write_bytes(implementation.read_bytes())
    write_lines(destination / 'candidates.jsonl', rows)
    write_lines(destination / 'selected_metadata.jsonl', selected)
    write_lines(destination / 'ordered_reserves.jsonl', reserves)
    write_lines(destination / 'latest_history_receipts.jsonl', touched)
    write_json(destination / 'openfake_blocks.json', blocks)
    write_json(destination / 'history_protection.json', {k: sorted(v) for k, v in state.items()})
    write_json(destination / 'audit.json', audit)
    write_json(destination / 'history_anchors.json', dict(sorted(anchors.items())))
    files = {str(path.relative_to(destination)).replace('\\', '/'): sha(path)
             for path in sorted(destination.rglob('*')) if path.is_file()}
    protocol = {'version': VERSION, 'created_utc': datetime.now(timezone.utc).isoformat(),
                'namespace': NAMESPACE, 'seed': SEED, 'metadata_only': True,
                'ready_for_inference': False, 'all_images_verified': False,
                'statistical_protocol_locked': False, 'adoption_rule_locked': False,
                'targets': TARGETS, 'source_revisions': preselection['sources'],
                'selected_source_datasets': ['DeepShield', 'OpenFake'],
                'history_source_inventory': inventory,
                'selection': {'OpenFake': 'SHA256([namespace,seed,block,shard,row_group]), potom SHA256([namespace,seed,row,row_id]); prvý dovolený jedinečný pôvod do kvóty, všetkých 242 blokov v pevnom rezervnom poradí.',
                              'DeepShield': 'Zmrazený make_deepshield_plan: seed=[namespace,seed], striedanie kategórií, globally unique dependency, ceil(1.25*kvóta) na generátor =1880 kandidátov.',
                              'replacement': 'Iba v pevnom poradí pri budúcom pôvodovom alebo obsahovom vyradení; nikdy podľa detektorového/skórového výsledku. Zmena kvót/rezerv/poradia vyžaduje nový protokol.',
                              'content_audit_pending': 'SHA256, dekódované RGB SHA256, pHash <=4 voči celej dostupnej histórii aj prijatým novým vzorkám. Bez tohto auditu inferencia zakázaná.'},
                'files': files}
    write_json(destination / 'protocol.json', protocol)
    (destination / 'protocol.sha256').write_text(sha(destination / 'protocol.json') + '\n', encoding='ascii')
    # Posledný marker značí iba dokončený metadátový predvýber.
    write_json(destination / 'metadata_complete.json', {'version': VERSION, 'metadata_only': True,
               'protocol_sha256': sha(destination / 'protocol.json'), 'ready_for_inference': False,
               'all_images_verified': False, 'selected_rows': len(selected)})
    return {'path': str(destination), 'metadata_only': True, 'selected_rows': len(selected),
            'candidates': len(rows), 'openfake_blocks': len(blocks), 'ready_for_inference': False,
            'protocol_sha256': sha(destination / 'protocol.json')}


def check(destination):
    destination = Path(destination).resolve()
    anchors = {}
    protocol, actual = sidecar(destination, 'protocol', anchors)
    marker = read(destination / 'metadata_complete.json')
    if (marker['protocol_sha256'] != actual or not marker['metadata_only'] or
            marker['ready_for_inference'] or marker['all_images_verified'] or
            not protocol['metadata_only'] or protocol['ready_for_inference']):
        raise ValueError('Neplatný marker metadátového plánu.')
    verify_file_map(destination, protocol['files'], anchors)
    for name, expected in read(destination / 'history_anchors.json').items():
        anchor(name, anchors, expected)
    selected = list(lines(destination / 'selected_metadata.jsonl'))
    if Counter(r['bucket'] for r in selected) != Counter(protocol['targets']):
        raise ValueError('Metadátové kvóty nesedia.')
    if len({r['source_key'] for r in selected}) != len(selected):
        raise ValueError('Opakovaná pôvodová identita vo výbere.')
    if any(r['content_verified'] for r in selected):
        raise ValueError('Metadátový plán predstiera obsahové overenie.')
    return {'path': str(destination), 'verified_metadata': True, 'selected_rows': len(selected),
            'ready_for_inference': False, 'protocol_sha256': actual}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, default=PROJECT)
    parser.add_argument('--plan', type=Path)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    destination = args.plan or args.project_root / 'article_ablation_confirmation/data_plan_20261028'
    result = check(destination) if args.check else prepare(args.project_root, destination)
    print(json.dumps(result, ensure_ascii=False, indent=2))


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
