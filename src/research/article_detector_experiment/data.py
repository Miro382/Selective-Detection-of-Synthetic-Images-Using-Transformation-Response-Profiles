"""Vopred určený pilot a celý vývojový experiment, bez externého testu."""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
from pathlib import Path
import shutil

from .common import ROOT, PROJECT, SPLITS, digest, environment, output_path, read, records, sha, write, write_records

SEED = 20261027
SOURCE = PROJECT / 'selective_experiments/experiment_v4/selector_profiles_v4'
PILOT_GROUPS = {'selector_train': 64, 'detector_calibration': 128, 'selector_selection': 128}


def validate_partitions(splits):
    seen = {key: {} for key in ('group', 'original_id', 'sha256', 'rgb_sha256', 'path')}
    for split, rows in splits.items():
        ids = set()
        for row in rows:
            if row['sample_id'] in ids or row['split'] != split or row['label'] not in (0, 1):
                raise ValueError('Neplatný alebo duplicitný záznam: ' + row['sample_id'])
            ids.add(row['sample_id'])
            for key in seen:
                value = str(Path(row[key]).resolve()).casefold() if key == 'path' else row[key]
                if not value:
                    raise ValueError('Chýba identita: ' + key)
                previous = seen[key].setdefault(value, split)
                if previous != split:
                    raise ValueError(f'Prienik oddelení ({key}): {previous} / {split}')


def subset_groups(rows, count, split):
    """Výber podľa SHA256 skupiny pred inferenciou, rovnaký pre všetky siete.

    Stratifikácia podľa referenčnej triedy, nikdy podľa správnosti detektora.
    Všetky varianty vybranej skupiny zostávajú spolu.
    """
    groups = defaultdict(list)
    for row in rows:
        groups[row['group']].append(row)
    if count > len(groups) or count < 2 or count % 2:
        raise ValueError('Pilot vyžaduje párny dostupný počet skupín aspoň 2.')
    buckets = defaultdict(list)
    for group, members in groups.items():
        labels = {r['label'] for r in members}
        if len(labels) != 1:
            raise ValueError('Pilotná stratifikácia vyžaduje jedno návestie v skupine.')
        buckets[next(iter(labels))].append(group)
    chosen = set()
    for label in (0, 1):
        ranked = sorted(buckets[label], key=lambda g: digest(['article-pilot', SEED, split, g]))
        if len(ranked) < count // 2:
            raise ValueError('Nedostatok skupín jednej triedy.')
        chosen.update(ranked[:count // 2])
    return [r for r in rows if r['group'] in chosen]


def verified_sources():
    ppath = SOURCE / 'protocol.json'
    if sha(ppath) != (SOURCE / 'protocol.sha256').read_text().strip():
        raise ValueError('Pôvodný dátový protokol bol zmenený.')
    protocol = read(ppath)
    paths = ['plans/selector_train.jsonl', 'plans/selector_selection.jsonl',
             'snapshot/data/detector_calibration.csv', 'snapshot/code/selector_augmentation_v4.py']
    anchors = {str(ppath): sha(ppath)}
    for relative in paths:
        path = SOURCE / relative
        actual = sha(path)
        if actual != protocol['locked_files'][relative]:
            raise ValueError('Zmenený pôvodný súbor: ' + str(path))
        anchors[str(path)] = actual
    splits = {s: records(SOURCE / 'plans' / (s + '.jsonl'))
              for s in ('selector_train', 'selector_selection')}
    with (SOURCE / 'snapshot/data/detector_calibration.csv').open(encoding='utf-8', newline='') as f:
        calibration = list(csv.DictReader(f))
    for row in calibration:
        row['original_id'] = row['sample_id']
        row['group'] = row.get('dependency_key') or row['group']
        row['input_spec'] = {'version': 1, 'id': 'clean', 'operations': []}
    splits['detector_calibration'] = calibration
    for split, rows in splits.items():
        for row in rows:
            row['label'] = int(row['label'])
            row['split'] = split
            row['dataset'] = 'DeepShield-development'
    validate_partitions(splits)
    return splits, anchors


def verify(run, check_code=True):
    run = output_path(run)
    p = read(run / 'protocol.json')
    if sha(run / 'protocol.json') != (run / 'protocol.sha256').read_text().strip():
        raise ValueError('Zmenený protokol nového experimentu.')
    if p['environment'] != environment():
        raise ValueError('Zmenilo sa numerické prostredie; použite nový beh.')
    for relative, expected in p['files'].items():
        if sha(run / relative) != expected:
            raise ValueError('Zmenený uzamknutý vstup: ' + relative)
    if check_code:
        for name, expected in p['code'].items():
            if sha(ROOT / name) != expected:
                raise ValueError('Kód sa od prípravy zmenil; zvoľte nový beh: ' + name)
    for path, expected in p['source_anchors'].items():
        if sha(path) != expected:
            raise ValueError('Zmenený pôvodný vstup: ' + path)
    return p


def prepare(run, mode='pilot'):
    run = output_path(run)
    if (run / 'protocol.json').exists():
        protocol = verify(run)
        if protocol['mode'] != mode:
            raise ValueError('Existujúci beh má iný režim.')
        return protocol
    if run.exists():
        raise ValueError('Neúplný priečinok sa neprepisuje; zvoľte nový názov behu.')
    if mode not in ('pilot', 'development'):
        raise ValueError('Neznámy režim.')
    splits, anchors = verified_sources()
    if mode == 'pilot':
        splits = {s: subset_groups(rows, PILOT_GROUPS[s], s) for s, rows in splits.items()}
    validate_partitions(splits)
    run.mkdir(parents=True)
    for split in SPLITS:
        write_records(run / 'plans' / (split + '.jsonl'), splits[split])
    snapshot = run / 'snapshot'
    snapshot.mkdir()
    shutil.copy2(SOURCE / 'snapshot/code/selector_augmentation_v4.py', snapshot / 'augmentation.py')
    code = {p.name: sha(p) for p in sorted(ROOT.glob('*.py'))}
    (snapshot / 'code').mkdir()
    for name in code:
        shutil.copy2(ROOT / name, snapshot / 'code' / name)
    protocol = {
        'version': 'article-detectors-v1', 'created_utc': datetime.now(timezone.utc).isoformat(),
        'mode': mode, 'seed': SEED, 'environment': environment(), 'source_anchors': anchors, 'code': code,
        'counts': {s: {'profiles': len(rows), 'originals': len({r['original_id'] for r in rows}),
                       'groups': len({r['group'] for r in rows}),
                       'labels': dict(Counter(r['label'] for r in rows))} for s, rows in splits.items()},
        'profile': {'order': ['original', 'jpeg90', 'jpeg75', 'jpeg60', 'resize0.75', 'resize0.50'],
                    'jpeg_subsampling': 0, 'jpeg_optimize': False, 'resize': 'LANCZOS down and restore',
                    'transformation_order': 'outer augmentation, independent profile transforms, native detector preprocessing'},
        'detectors': ['clip', 'resnet50'], 'detector_decision': 'fake iff raw_logit >= 0',
        'selection': 'SHA256 group ranking; equal real/fake group quotas in pilot; all variants retained',
        'selector': {'C': 0.1, 'features': 'V4 symmetric profile', 'temperature_split': 'detector_calibration',
                     'fit_split': 'selector_train', 'evaluation_split': 'selector_selection'},
        'scope': 'Technický pilot na vývojových dátach.' if mode == 'pilot' else 'Celý vývojový beh, nie externý test.',
        'limitations': ['Externé testovacie dáta sa v tomto behu nepoužívajú.',
                       'Vývojové metriky nie sú potvrdením prínosu pre článok ani zárukou rizika.',
                       'Nastavenie sa neladí podľa výsledkov pilota.',
                       'Pôvodné selektorové overenie rizika ostáva nepoužité.',
                       'Verejné detektory sa líšia architektúrou aj tréningom.',
                       'Kontrola SHA256/RGB a zdrojových skupín nenahrádza audit tréningových dát verejných modelov.'],
    }
    protocol['files'] = {p.relative_to(run).as_posix(): sha(p) for p in run.rglob('*') if p.is_file()}
    write(run / 'protocol.json', protocol)
    (run / 'protocol.sha256').write_text(sha(run / 'protocol.json') + '\n', encoding='ascii')
    return protocol
