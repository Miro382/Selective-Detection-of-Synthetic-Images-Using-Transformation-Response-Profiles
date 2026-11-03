"""Pripnuté zdroje a normalizovaná história samostatného externého testu.

Staršie experimenty sú iba vstupy. Tento modul nesmie volať ich zapisujúce
prepare/download entrypointy ani čítať predikcie alebo výsledkové metriky.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[2]
sys.dont_write_bytecode = True
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

import confirmation_v4_data as confirmation
import external_v4_data as cf4
import selector_v5_data as cf
import external_v5_data as of
import v4_data_sources as ds
from v4_deepshield_audit import ExistingFingerprintIndex, _MemoryIndex

SEED = 20261027
DS_TARGETS = {'FFHQ': 200, 'FORLAB': 800, 'StyleGAN': 100,
              'StyleGAN2': 100, 'StyleGAN3': 100,
              **{name: 140 for name in ds.DIFFUSION}}
CF_TARGETS = {'real/LAION': 380, 'real/coco': 20, 'fake/DFGAN': 10,
              'fake/Imagen3': 5, 'fake/Dalle3': 60, 'fake/FLUX-dev': 60,
              'fake/FLUX-schnell': 60, 'fake/Firefly_Image3': 65,
              'fake/IdeogramV2': 65, 'fake/MidjourneyV6_1': 75}
TARGETS = {
    **{'OpenFake/' + k: v for k, v in of.TARGETS.items()},
    **{'DeepShield/' + ('real/' if k in ('FFHQ', 'FORLAB') else 'fake/') + k: v
       for k, v in DS_TARGETS.items()},
    **{'CommunityForensics/' + k: v for k, v in CF_TARGETS.items()},
}
PHASH_LIMITATION = (
    'pHash sa porovnáva so všetkými dostupnými historickými hodnotami. '
    'Chýbajúce historické pHash sa nedopočítavajú; nulový počet nevyriešených '
    'zhôd neznamená úplné perceptuálne pokrytie histórie. Počty sú riadky '
    'inventárov, nie počet jedinečných obrázkov.')


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(',', ':'), allow_nan=False).encode('utf-8')


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def sha(path):
    return cf.sha256(Path(path))


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def normalized_group(value):
    return confirmation.canonical_group(str(value))


def block_key(dataset, row):
    if dataset == 'OpenFake':
        return of.sampling_group(row)
    if dataset == 'CommunityForensics':
        return 'cf/parquet/' + row['shard'] + '/' + str(row['row_group'])
    return 'deepshield/' + row['dependency_key']


def collect_history(project=PROJECT):
    """Overí uložený predchádzajúci audit a rozšíri ho o celé potvrdenie V4.

    Historický protocol obsahuje zmrazený rekurzívny audit V1–V5. Jeho
    kontrolné súčty sa znovu overia, takže netreba opakovať dekódovanie už
    auditovaných vyradených súborov. Najnovší confirmation nie je subset.
    """
    project = Path(project).resolve()
    base = project / 'external_data_v4_confirmation'
    protocol = confirmation.verify_protocol(base)
    complete = read(base / 'complete.json')
    if complete['protocol_sha256'] != sha(base / 'protocol.json'):
        raise ValueError('Zmenený protokol historického potvrdenia.')
    # Identita historicky použitého obsahu pochádza zo zmrazených inventárov;
    # kontrola neotvára 20000 starých obrázkov ani ich jednotlivé receipts.
    for name in ('manifest.csv', 'exclusions.json', 'plan.json', 'receipts.json'):
        if complete['files'].get(name) != sha(base / name):
            raise ValueError('Zmenený historický inventár: ' + name)
    sources = protocol['historical_sources']
    cf.verify_anchors(sources['anchors'])
    sources['csv_inventories'] = sorted(set(sources['csv_inventories']) | {str(base / 'manifest.csv')})
    accepted = confirmation.load_manifest(base / 'manifest.csv')
    excluded = read(base / 'exclusions.json')
    sources['extra_fingerprints'].extend(r['fingerprint'] for r in excluded if r.get('fingerprint'))
    anchors = {a['path']: a['sha256'] for a in sources['anchors']}
    for name in ('protocol.json', 'protocol.sha256', 'complete.json', 'manifest.csv',
                 'exclusions.json', 'plan.json', 'receipts.json'):
        anchors[str(base / name)] = sha(base / name)
    comparison = project / 'selective_experiments/selector_comparison_bahat/protocol.json'
    compared = read(comparison)
    comparison_external = project / 'selective_experiments/external_v4_confirmation_reduced_20261017'
    if (Path(compared['external']).resolve() != comparison_external.resolve() or
            Path(compared['source']).resolve() != (project / 'selective_experiments/experiment_v4/selector_profiles_v4').resolve()):
        raise ValueError('Comparison používa iné dáta; historický audit treba rozšíriť.')
    comparison_manifest = comparison_external / 'snapshot/reduced_manifest.csv'
    known = {r['sha256'] for r in accepted}
    with comparison_manifest.open(encoding='utf-8-sig', newline='') as stream:
        comparison_rows = list(csv.DictReader(stream))
    if any(r['sha256'] not in known for r in comparison_rows):
        raise ValueError('Comparison obsahuje neznáme externé obrázky.')
    anchors[str(comparison)] = sha(comparison)
    anchors[str(comparison_manifest)] = sha(comparison_manifest)
    sources['comparison_coverage'] = {'protocol': str(comparison), 'external_subset_rows': len(comparison_rows),
                                      'all_external_sha256_in_full_confirmation': True,
                                      'development_source': compared['source']}
    groups = {normalized_group(g) for g in sources['protected_groups']}
    prompts = set(sources['protected_prompt_hashes'])
    hashes = set(sources['protected_image_hashes'])
    deep_groups = set(sources['protected_deepshield_groups'])
    deep_dependencies = set(sources['protected_deepshield_dependencies'])
    cf_blocks = set()
    of_blocks = {tuple(x) for x in sources['protected_openfake_blocks']}
    for module, relative in ((cf4, 'external_data_v4/CommunityForensics'),
                             (cf, 'selector_data_v5/CommunityForensics')):
        old = project / relative
        touched = {r['sample_id'] for r in module.load_manifest(old / 'manifest.csv') + read(old / 'exclusions.json')}
        cf_blocks.update((r['shard'], r['row_group']) for r in read(old / 'plan.json') if r['sample_id'] in touched)
    touched = {r['sample_id'] for r in accepted + excluded}
    original_ids = set(touched)
    for row in read(base / 'plan.json'):
        if row['sample_id'] not in touched:
            continue
        groups.add(normalized_group(row['group']))
        if row['cohort'] == 'DeepShield':
            deep_groups.add(row['original_group'])
            deep_dependencies.add(row['dependency_key'])
        else:
            of_blocks.add((row['shard'], row['row_group']))
            if row['prompt_normalized_sha256']:
                prompts.add(row['prompt_normalized_sha256'])
            hashes.add(row['image_sha256'])
    indexes = {'CommunityForensics': cf.load_source_index(project / cf.INDEX_FILE),
               'OpenFake': of.load_source_index(project / of.INDEX_FILE),
               'DeepShield': ds.load_index(project / ds.INDEX_FILE)}
    # Celé už načítané Parquet bloky vrátane susedov sú historicky chránené.
    for dataset, module, used_blocks in (('CommunityForensics', cf, cf_blocks), ('OpenFake', of, of_blocks)):
        for row in indexes[dataset]['rows']:
            if (row['shard'], row['row_group']) not in used_blocks:
                continue
            groups.add(normalized_group(module.metadata_group(row)))
            prompt = row.get('prompt_normalized_sha256') if module is of else of.prompt_hash(row.get('prompt'))
            if prompt:
                prompts.add(prompt)
            if row.get('image_sha256'):
                hashes.add(row['image_sha256'])
            original_ids.add('openfake/' + of.REVISION + '/' + row['row_id'] if module is of else cf.row_identity(row))
    for filename in (cf.INDEX_FILE, cf4.INDEX_FILE, of.INDEX_FILE, ds.INDEX_FILE):
        anchors[str(project / filename)] = sha(project / filename)
    # Všetky lokálne článkové plány: kalibrácia, učenie aj výber, bez skóre.
    article_plans = []
    for path in sorted((project / 'article_detector_experiment/runs').glob('*/plans/*.jsonl')):
        anchors[str(path)] = sha(path)
        article_plans.append(str(path))
    for path in sources['csv_inventories']:
        anchors[path] = sha(path)
    sources.update(protected_groups=sorted(groups), protected_prompt_hashes=sorted(prompts),
                   protected_image_hashes=sorted(hashes), protected_deepshield_groups=sorted(deep_groups),
                   protected_deepshield_dependencies=sorted(deep_dependencies),
                   protected_cf_blocks=sorted([list(x) for x in cf_blocks]),
                   protected_openfake_blocks=sorted([list(x) for x in of_blocks]),
                   protected_original_ids=sorted(original_ids), article_plans=article_plans,
                   anchors=[{'path': p, 'sha256': h} for p, h in sorted(anchors.items())])
    return sources, indexes


def normalize_record(row):
    result = {k: str(row[k]) for k in ('sha256', 'rgb_sha256', 'phash', 'path') if row.get(k)}
    if not result.get('path') and row.get('absolute_path'):
        result['path'] = str(row['absolute_path'])
    if row.get('original_id') or row.get('sample_id'):
        result['original_id'] = str(row.get('original_id') or row['sample_id'])
    dependencies = set(row.get('dependency_keys') or [])
    for key in ('group', 'dependency_key', 'sampling_group', 'source_key'):
        if row.get(key):
            dependencies.add(str(row[key]))
            dependencies.add(normalized_group(row[key]))
    if row.get('group'):
        result['group'] = normalized_group(row['group'])
        result['source_key'] = result['group']
    if dependencies:
        result['dependency_keys'] = sorted(dependencies)
    return result


def history_records(sources):
    """Normalizované identity pre nezávislý freeze audit; žiadne skóre."""
    for filename in sources['csv_inventories']:
        with Path(filename).open(encoding='utf-8-sig', newline='') as stream:
            for row in csv.DictReader(stream):
                result = normalize_record(row)
                if result:
                    yield result
    for row in sources['extra_fingerprints']:
        result = normalize_record(row)
        if result:
            yield result
    for filename in sources['article_plans']:
        with Path(filename).open(encoding='utf-8') as stream:
            for line in stream:
                yield normalize_record(json.loads(line))
    for group in sources['protected_groups']:
        yield {'group': group, 'source_key': group, 'dependency_keys': [group]}
    for value in sources['protected_original_ids']:
        yield {'original_id': value}
    for value in sources['protected_prompt_hashes']:
        yield {'source_key': 'prompt/' + value, 'dependency_keys': ['prompt/' + value]}
    for value in sources['protected_image_hashes']:
        yield {'sha256': value}
    for value in sources['protected_deepshield_dependencies']:
        yield {'source_key': 'deepshield/' + value, 'dependency_keys': [value, 'deepshield/' + value]}
    for value in sources['protected_deepshield_groups']:
        yield {'original_id': 'deepshield-v4/' + value, 'dependency_keys': [value]}
    for dataset, key in (('CommunityForensics', 'protected_cf_blocks'), ('OpenFake', 'protected_openfake_blocks')):
        for shard, rg in sources[key]:
            yield {'dependency_keys': [block_key(dataset, {'shard': shard, 'row_group': rg})]}


def historical_index(sources):
    index = ExistingFingerprintIndex([Path(p) for p in sources['csv_inventories']])
    extras = Counter()
    for i, row in enumerate(sources['extra_fingerprints']):
        index._index.add(row, 'historical-exclusion/' + str(i))
        extras.update(k for k in ('sha256', 'rgb_sha256', 'phash') if row.get(k))
    article_count = 0
    for filename in sources['article_plans']:
        with Path(filename).open(encoding='utf-8') as stream:
            for line in stream:
                row = json.loads(line)
                index._index.add(row, row['path'])
                article_count += 1
    total = index.summary['rows_read']
    available = index.summary['field_counts']['phash']
    coverage = {'csv_inventory': index.summary,
                'phash': {'available': available, 'total': total, 'missing': total - available,
                          'limitation': PHASH_LIMITATION},
                'extra_fingerprint_records': len(sources['extra_fingerprints']),
                'extra_fingerprint_fields': dict(extras), 'article_plan_rows': article_count,
                'comparison': sources['comparison_coverage'],
                'identity_scope': 'Dokumentované inventáre V1–V5, celé potvrdenie V4, rezervy V3/V4, vyradené kandidáty a celé už použité Parquet bloky; článkové plány.',
                'identity_counts': {k: len(v) for k, v in sources.items() if k.startswith('protected_')},
                'unknown_identity_limitation': 'Neznáme identity osôb FFHQ, nedokumentované sémantické väzby a pôvod predtrénovania sa týmto auditom nevylučujú.'}
    return index, coverage


def candidate_plan(sources, indexes):
    groups = set(sources['protected_groups'])
    prompts = set(sources['protected_prompt_hashes'])
    hashes = set(sources['protected_image_hashes'])
    cf_blocks = {tuple(x) for x in sources['protected_cf_blocks']}
    of_blocks = {tuple(x) for x in sources['protected_openfake_blocks']}
    plan = []
    for row in of.make_plan(indexes['OpenFake'], prompts, SEED):
        if (row['shard'], row['row_group']) in of_blocks or row['group'] in groups or row['image_sha256'] in hashes:
            continue
        plan.append({**row, 'dataset': 'OpenFake', 'source_key': row['group'],
                     'original_id': row['sample_id'], 'bucket': 'OpenFake/' + row['bucket']})
    cf_pool = []
    all_fake = set()
    original_fake = set()
    for raw in indexes['CommunityForensics']['rows']:
        group = normalized_group(cf.metadata_group(raw))
        if (raw['shard'], raw['row_group']) in cf_blocks or group in groups or of.prompt_hash(raw.get('prompt')) in prompts:
            continue
        if raw['label']:
            all_fake.add(group)
            if raw['model_name'] in cf4.FAKE_MODELS:
                original_fake.add(group)
        bucket = cf.bucket(raw)
        if bucket not in CF_TARGETS:
            continue
        cf_pool.append({**raw, 'dataset': 'CommunityForensics', 'source_key': group,
                        'original_id': cf.row_identity(raw), 'sample_id': cf.row_identity(raw),
                        'sampling_group': block_key('CommunityForensics', raw),
                        'group': group, 'bucket': 'CommunityForensics/' + bucket})
    cf_pool.sort(key=lambda r: (digest([SEED, 'cf-block', r['shard'], r['row_group']]), digest([SEED, r['sample_id']])))
    plan.extend(cf_pool)
    deep = confirmation.make_deepshield_plan(indexes['DeepShield'], sources, seed=SEED, targets=DS_TARGETS, reserve=1.8)
    for row in deep:
        original = 'deepshield-v4/' + row['original_group']
        plan.append({**row, 'dataset': 'DeepShield', 'source_key': row['group'],
                     'original_id': original, 'sample_id': original})
    used = set()
    counts = Counter()
    capacity = defaultdict(set)
    selected_blocks, pool_blocks = {}, {}
    selected_deep_bytes = reserve_deep_bytes = 0
    for row in plan:
        capacity[row['bucket']].add(row['source_key'])
        if row['dataset'] != 'DeepShield':
            key = (row['dataset'], row['sampling_group'])
            pool_blocks[key] = int(row['row_group_image_bytes'])
        else:
            reserve_deep_bytes += row['entry']['compressed'] + 1024
        if counts[row['bucket']] >= TARGETS[row['bucket']] or row['source_key'] in used:
            continue
        counts[row['bucket']] += 1
        used.add(row['source_key'])
        if row['dataset'] == 'DeepShield':
            selected_deep_bytes += row['entry']['compressed'] + 1024
        else:
            selected_blocks[(row['dataset'], row['sampling_group'])] = int(row['row_group_image_bytes'])
    if counts != Counter(TARGETS):
        raise ValueError('Metadátový plán nepokrýva pevné kvóty: ' + str(dict(counts)))
    capacity_report = {'unique_origins_by_bucket': {k: len(v) for k, v in sorted(capacity.items())},
                       'cf_fake_origin_union_original_8': len(original_fake),
                       'cf_fake_origin_union_all_12': len(all_fake),
                       'metadata_selected_counts': dict(counts),
                       'estimated_selected_gib': (sum(selected_blocks.values()) + selected_deep_bytes) / 2**30,
                       'all_reserve_estimated_gib': (sum(pool_blocks.values()) + reserve_deep_bytes) / 2**30,
                       'selected_parquet_blocks': len(selected_blocks),
                       'note': 'Komprimované stĺpce/členy ZIP, bez réžie a opakovaní. Obsahový audit môže čerpať rezervy, kvóty nesmie znížiť. Počet pôvodov nie je počet nezávislých sampling skupín.'}
    return plan, capacity_report


def connected_groups(rows):
    """Celé komponenty pôvod ∪ sampling blok; deterministické identifikátory."""
    parent = {}
    def find(key):
        parent.setdefault(key, key)
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key
    def union(left, right):
        left, right = find(left), find(right)
        if left != right:
            parent[max(left, right)] = min(left, right)
    for row in rows:
        keys = row['dependency_keys']
        for key in keys:
            union(keys[0], key)
    components = defaultdict(list)
    for key in parent:
        components[find(key)].append(key)
    names = {root: 'article-external/component/' + digest(sorted(keys)) for root, keys in components.items()}
    return [{**r, 'group': names[find(r['dependency_keys'][0])]} for r in rows]
