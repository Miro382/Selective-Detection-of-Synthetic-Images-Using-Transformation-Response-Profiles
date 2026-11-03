"""Obsah a pôvod nového holdoutu; bez siete a bez modelových skóre."""
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def lines(path):
    with Path(path).open(encoding='utf-8') as stream:
        for line in stream:
            if not line.strip() or not line.endswith('\n'):
                raise ValueError('Neúplný JSONL: ' + str(path))
            yield json.loads(line)


class Union:
    def __init__(self):
        self.parent = {}

    def find(self, key):
        self.parent.setdefault(key, key)
        while self.parent[key] != key:
            self.parent[key] = self.parent[self.parent[key]]
            key = self.parent[key]
        return key

    def join(self, keys):
        keys = list(keys)
        if not keys:
            raise ValueError('Prázdna množina identít.')
        root = self.find(keys[0])
        for key in keys[1:]:
            other = self.find(key)
            if root != other:
                self.parent[max(root, other)] = min(root, other)
                root = min(root, other)
        return root


class Context:
    def __init__(self, metadata, legacy):
        self.metadata = Path(metadata)
        self.legacy = legacy
        self.helpers = legacy.of, legacy.cf, legacy.ds
        self.history = legacy._MemoryIndex()
        self.aliases = Union()
        self.neighbours = defaultdict(list)
        self.coverage = {'csv': Counter(), 'extra': Counter(), 'latest': Counter()}
        base = read(self.metadata / 'audit.json')['history']['base_plan']
        historical = read(Path(base) / 'history.json')
        for filename in sorted(historical['csv_inventories']):
            with Path(filename).open(encoding='utf-8-sig', newline='') as stream:
                for index, row in enumerate(csv.DictReader(stream)):
                    self.add_history(row, filename + ':' + str(index), 'csv')
        for index, row in enumerate(historical['extra_fingerprints']):
            self.add_history(row, 'historical-excluded/' + str(index), 'extra')
        for filename in historical['article_plans']:
            for row in lines(filename):
                self.add_history(row, row['path'], 'extra')
        for row in lines(self.metadata / 'latest_history_receipts.jsonl'):
            # Aj invalid_image s fingerprint=null má záväzný surový SHA.
            self.add_history({**(row.get('fingerprint') or {}),
                              'sha256': row.get('image_sha256')},
                             row['original_id'], 'latest')
        self.protection = read(self.metadata / 'history_protection.json')
        for checksum in self.protection['hashes']:
            self.history.add({'sha256': checksum}, 'protected-metadata/' + checksum)
        self._metadata_aliases(legacy)
        self.protected_origins = set()
        for key in self.protection['groups']:
            self.protected_origins.add(self.aliases.find('origin/' + key))
        for key in self.protection['original_ids']:
            self.protected_origins.add(self.aliases.find('id/' + key))
        for key in self.protection['hashes']:
            self.protected_origins.add(self.aliases.find('sha/' + key))
        for key in self.protection['deep_dependencies']:
            self.protected_origins.add(self.aliases.find('origin/deepshield/' + key))
        self.coverage = {kind: dict(counts) for kind, counts in self.coverage.items()}
        self.coverage['limitations'] = read(self.metadata / 'audit.json')['history']['phash']

    def add_history(self, row, identity, kind):
        self.coverage[kind]['rows'] += 1
        for field in ('sha256', 'rgb_sha256', 'phash'):
            self.coverage[kind][field] += bool(row.get(field))
        self.history.add(row, identity)

    def _metadata_aliases(self, legacy):
        # Pôvodové aliasy sa spoja globálne, sampling bloky až po ich použití.
        from data_support import normalized_group
        for dataset, module in (('OpenFake', legacy.of), ('CommunityForensics', legacy.cf)):
            index = module.load_source_index(legacy.PROJECT / module.INDEX_FILE)
            for row in index['rows']:
                origin = normalized_group(module.metadata_group(row))
                identity = ('openfake/' + module.REVISION + '/' + row['row_id']
                            if dataset == 'OpenFake' else module.row_identity(row))
                keys = ['origin/' + origin, 'id/' + identity]
                raw_sha = row.get('image_sha256')
                if raw_sha:
                    keys.append('sha/' + raw_sha)
                self.aliases.join(keys)
                block = legacy.block_key(dataset, row)
                self.neighbours[block].append({'dataset': dataset, 'original_id': identity,
                                               'source_key': origin, 'image_sha256': raw_sha,
                                               'sampling_group': block, 'alias_keys': keys})

    def origin_keys(self, row, fingerprint=None, raw_sha=None):
        keys = ['origin/' + row['source_key'], 'id/' + row['original_id']]
        raw_sha = raw_sha or (fingerprint or {}).get('sha256') or row.get('image_sha256')
        if raw_sha:
            keys.append('sha/' + raw_sha)
        return {self.aliases.find(key) for key in keys}

    def groups(self, accepted, accessed_blocks):
        """Komponenty zahŕňajú aj metadátových susedov použitých blokov."""
        graph = Union()
        neighbours = []
        for block in sorted(accessed_blocks):
            if block not in self.neighbours:
                raise ValueError('Neznámy použitý Parquet blok: ' + block)
            for row in self.neighbours[block]:
                roots = {self.aliases.find(key) for key in row['alias_keys']}
                graph.join(['sampling/' + block, *roots])
                neighbours.append({k: v for k, v in row.items() if k != 'alias_keys'})
        row_keys = []
        for row in accepted:
            keys = ['sampling/' + row['sampling_group'], *self.origin_keys(row, row)]
            graph.join(keys)
            row_keys.append(keys)
        components = defaultdict(list)
        for key in graph.parent:
            components[graph.find(key)].append(key)
        names = {root: 'ablation-confirmation/component/' + digest(sorted(keys))
                 for root, keys in components.items()}
        result = [{**row, 'group': names[graph.find(keys[0])]}
                  for row, keys in zip(accepted, row_keys)]
        return result, neighbours


class Replay:
    """Rozhodnutie každého kandidáta sa pri resume vypočíta znovu."""
    def __init__(self, context):
        self.context = context
        self.seen = context.legacy._MemoryIndex()
        self.used_origins = set()
        self.blocked_origins = set()

    def before_download(self, row):
        origins = self.context.origin_keys(row)
        if origins & self.context.protected_origins:
            return [{'kind': 'historical_origin'}]
        if origins & self.used_origins:
            return [{'kind': 'origin_already_selected'}]
        if origins & self.blocked_origins:
            return [{'kind': 'blocked_origin'}]
        return []

    def reasons(self, row, raw_sha, fingerprint, invalid_detail=None):
        reasons = self.before_download(row)
        if reasons:
            return reasons
        origins = self.context.origin_keys(row, fingerprint, raw_sha)
        if origins & self.context.protected_origins:
            reasons.append({'kind': 'historical_content_origin'})
        if origins & self.used_origins:
            reasons.append({'kind': 'selected_content_origin'})
        if origins & self.blocked_origins:
            reasons.append({'kind': 'blocked_content_origin'})
        content = fingerprint or {'sha256': raw_sha}
        reasons.extend({**r, 'index': 'history'} for r in self.context.history.matches(content))
        reasons.extend({**r, 'index': 'new_downloads'} for r in self.seen.matches(content))
        if invalid_detail is not None:
            reasons.append({'kind': 'invalid_image', 'detail': invalid_detail})
        return reasons

    def apply(self, row, raw_sha, fingerprint, reasons):
        origins = self.context.origin_keys(row, fingerprint, raw_sha)
        if reasons:
            # Opakovaný už prijatý pôvod nevyradí prvú prijatú reprezentáciu.
            if not any(r['kind'] == 'origin_already_selected' for r in reasons):
                self.blocked_origins.update(origins)
        else:
            if not fingerprint or raw_sha != fingerprint['sha256']:
                raise ValueError('Prijatie bez overeného obsahu.')
            self.used_origins.update(origins)
        if raw_sha:
            self.seen.add(fingerprint or {'sha256': raw_sha}, row['original_id'])
