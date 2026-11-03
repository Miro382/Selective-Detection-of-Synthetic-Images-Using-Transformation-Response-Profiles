"""Zmrazí overené modely a nezávislý externý plán bez testovacej inferencie.

Používa iba štandardnú knižnicu. Nikdy neučí model, neotvára obrazový
detektor, nemení vývojové artefakty a nepočíta externé skóre ani metriky.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
from itertools import zip_longest
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import uuid

ROOT = Path(__file__).resolve().parent
ARTICLE_ROOT = ROOT.parent
DETECTORS = ('clip', 'resnet50')
SPLITS = ('detector_calibration', 'selector_train', 'selector_selection')
EXPECTED_DEVELOPMENT_COUNTS = {'detector_calibration': 1000, 'selector_train': 40000, 'selector_selection': 2000}
DATASETS = ('DeepShield', 'OpenFake', 'CommunityForensics')
FEATURE_COLUMNS = {'lr_full': [0, 1, 2, 3, 4, 5], 'lr_confidence': [0],
                   'lr_resize_only': [0, 4, 5], 'lr_jpeg_only': [0, 1, 2, 3]}
TRANSFORMS = ['original', 'jpeg_q90', 'jpeg_q75', 'jpeg_q60', 'resize_0p75', 'resize_0p5']
IDENTITIES = ('group', 'original_id', 'source_key', 'sha256', 'rgb_sha256', 'path')
HEX64 = re.compile(r'[0-9a-f]{64}')
HEX16 = re.compile(r'[0-9a-f]{16}')


class FreezeError(ValueError):
    pass


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
    def invalid(value):
        raise FreezeError('Nekonečná hodnota JSON: ' + value)
    return json.loads(Path(path).read_text(encoding='utf-8'), parse_constant=invalid)


def lines(path):
    with Path(path).open('rb') as stream:
        for line in stream:
            if not line.endswith(b'\n') or not line.strip():
                raise FreezeError('Nedokončený alebo prázdny riadok: ' + str(path))
            yield json.loads(line)


def checked_file(path, expected, anchors):
    path = Path(path).resolve()
    actual = sha(path)
    if actual != expected:
        raise FreezeError('Nesúhlasí SHA256: ' + str(path))
    anchors[str(path)] = actual
    return path


def checked_protocol(folder, anchors):
    path = Path(folder) / 'protocol.json'
    checked_file(path, (Path(folder) / 'protocol.sha256').read_text(encoding='ascii').strip(), anchors)
    return read(path)


def inside(folder, relative):
    folder = Path(folder).resolve()
    path = (folder / relative).resolve()
    if path == folder or not path.is_relative_to(folder):
        raise FreezeError('Neplatná relatívna cesta v protokole: ' + str(relative))
    return path


def normalized_identity(key, value):
    if key == 'path':
        return os.path.normcase(str(Path(value).resolve()))
    return str(value)


def add_identities(index, row):
    for key in IDENTITIES:
        if row.get(key):
            index[key].add(normalized_identity(key, row[key]))
    for key in row.get('dependency_keys', []):
        index['dependency_keys'].add(str(key))
    # Pôvodné V4 plány už majú normalizovanú skupinu, CSV môže mať aj dependency_key.
    if row.get('dependency_key'):
        index['dependency_keys'].add(str(row['dependency_key']))
    if row.get('group'):
        index['dependency_keys'].add(str(row['group']))
    if row.get('phash'):
        value = str(row['phash']).lower()
        if not HEX16.fullmatch(value):
            raise FreezeError('Neplatný dostupný historický pHash.')
        index['phash'].add(int(value, 16))


class PHashIndex:
    """Presný Hamming <= 4 pomocou piatich disjunktných blokov 64 bitov.

    Pri najviac štyroch rozdielnych bitoch sa aspoň jeden z piatich blokov
    musí zhodovať. Kandidáti sa následne overia presným bit_count, bez
    kvadratického porovnávania každého obrázka s celou históriou.
    """
    SEGMENTS = ((0, 13), (13, 13), (26, 13), (39, 13), (52, 12))

    def __init__(self, values):
        self.tables = [defaultdict(list) for _ in self.SEGMENTS]
        for value in set(values):
            for table, (offset, width) in zip(self.tables, self.SEGMENTS):
                table[(value >> offset) & ((1 << width) - 1)].append(value)

    def contains_near(self, value):
        seen = set()
        for table, (offset, width) in zip(self.tables, self.SEGMENTS):
            for candidate in table.get((value >> offset) & ((1 << width) - 1), ()):
                if candidate not in seen:
                    if (value ^ candidate).bit_count() <= 4:
                        return True
                    seen.add(candidate)
        return False


def validate_selector(model, statistics):
    if model.get('version') != 'article-selector-logistic-v1' or model.get('transforms') != TRANSFORMS:
        raise FreezeError('Nepodporovaná definícia selektora alebo poradie odpovedí.')
    temperature = model.get('temperature')
    if not isinstance(temperature, (int, float)) or isinstance(temperature, bool) or not math.isfinite(temperature) or temperature <= 0:
        raise FreezeError('Neplatná teplota selektora.')
    calibration = model['calibration']
    if calibration['split'] != 'detector_calibration' or calibration['temperature'] != temperature:
        raise FreezeError('Teplota nepochádza z oddelenej kalibrácie.')
    if calibration['count'] != statistics['detector_calibration']['count']:
        raise FreezeError('Nesúhlasí rozsah teplotnej kalibrácie.')
    training = model['training']
    stats = statistics['selector_train']
    expected = {'profiles': stats['count'], 'groups': stats['groups'], 'errors': stats['errors'],
                'split': 'selector_train', 'records_sha256': stats['records_sha256']}
    if any(training.get(key) != value for key, value in expected.items()):
        raise FreezeError('Učený selektor nepatrí k dokončeným tréningovým profilom.')
    if model['calibration_records_sha256'] != statistics['detector_calibration']['records_sha256']:
        raise FreezeError('Selektor má iné kalibračné profily.')
    if not math.isclose(training['weight_sum'], stats['groups'], rel_tol=1e-12, abs_tol=1e-9):
        raise FreezeError('Nesprávne skupinové váhy tréningu.')
    if set(model['models']) != set(FEATURE_COLUMNS):
        raise FreezeError('Selektor neobsahuje presne vopred určené modely.')
    for name, columns in FEATURE_COLUMNS.items():
        parameters = model['models'][name]
        size = 1 if len(columns) == 1 else len(columns) + 3
        if parameters['columns'] != columns or parameters['C'] != 0.1:
            raise FreezeError('Zmenená konfigurácia modelu: ' + name)
        for key in ('coef', 'mean', 'scale'):
            values = parameters[key]
            if len(values) != size or any(not isinstance(v, (int, float)) or not math.isfinite(v) for v in values):
                raise FreezeError('Neplatné parametre selektora: ' + name)
        if any(v <= 0 for v in parameters['scale']) or not math.isfinite(parameters['intercept']):
            raise FreezeError('Neplatná štandardizácia alebo intercept.')


def validate_profiles(path, plan_path, receipt, fingerprint, split, expected_count):
    chain = digest(['article-profiles-v1', fingerprint])
    record_digest = hashlib.sha256()
    record_digest.update(b'[')
    count = errors = 0
    groups = set()
    for row, plan in zip_longest(lines(path), lines(plan_path)):
        if row is None or plan is None:
            raise FreezeError('Neúplný počet profilov: ' + split)
        for key in ('sample_id', 'original_id', 'group', 'label', 'sha256', 'rgb_sha256', 'split', 'input_spec'):
            if row[key] != plan[key]:
                raise FreezeError('Profil nezodpovedá plánu: ' + split + '/' + key)
        if row['split'] != split or row['fingerprint'] != fingerprint:
            raise FreezeError('Zmiešaný detektor alebo oddelenie profilov.')
        margins = row['margins']
        if len(margins) != 6 or any(not isinstance(v, (int, float)) or not math.isfinite(v) for v in margins):
            raise FreezeError('Neplatné logity profilu.')
        error = int(int(margins[0] >= 0) != row['label'])
        if row['error'] != error:
            raise FreezeError('Nesprávne označenie chyby detektora.')
        body = {k: v for k, v in row.items() if k != 'chain'}
        chain = digest([chain, body])
        if row['chain'] != chain:
            raise FreezeError('Poškodený kontrolný reťazec profilov.')
        if count:
            record_digest.update(b',')
        record_digest.update(canonical(row))
        count += 1
        errors += error
        groups.add(row['group'])
    record_digest.update(b']')
    if count != expected_count or receipt['count'] != count or receipt['chain'] != chain or receipt['fingerprint'] != fingerprint:
        raise FreezeError('Nedokončené alebo nezhodné profily: ' + split)
    return {'count': count, 'errors': errors, 'groups': len(groups), 'records_sha256': record_digest.hexdigest()}


def verify_development(development_run):
    run = Path(development_run).resolve()
    anchors = {}
    protocol = checked_protocol(run, anchors)
    if protocol.get('version') != 'article-detectors-v1' or protocol.get('mode') != 'development':
        raise FreezeError('Externý test vyžaduje dokončený plný vývojový beh, nie pilot.')
    if protocol.get('detectors') != list(DETECTORS):
        raise FreezeError('Vývojový protokol neobsahuje oba požadované detektory.')
    for split, count in EXPECTED_DEVELOPMENT_COUNTS.items():
        if protocol['counts'][split]['profiles'] != count:
            raise FreezeError('Nesúhlasí rozsah celého vývojového behu: ' + split)
    for relative, checksum in protocol['files'].items():
        checked_file(inside(run, relative), checksum, anchors)
    for name, checksum in protocol['code'].items():
        checked_file(inside(run, 'snapshot/code/' + name), checksum, anchors)
    for source, checksum in protocol['source_anchors'].items():
        checked_file(source, checksum, anchors)
    index = defaultdict(set)
    for split in SPLITS:
        for row in lines(run / 'plans' / (split + '.jsonl')):
            add_identities(index, row)
    models = {}
    for detector in DETECTORS:
        folder = run / detector
        receipt = read(folder / 'analysis.complete.json')
        anchors[str((folder / 'analysis.complete.json').resolve())] = sha(folder / 'analysis.complete.json')
        inputs = receipt['inputs']
        if inputs['protocol_sha256'] != sha(run / 'protocol.json'):
            raise FreezeError('Analýza patrí k inému vývojovému protokolu.')
        checked_file(folder / 'detector.json', inputs['detector_sha256'], anchors)
        detector_info = read(folder / 'detector.json')
        if detector_info['protocol_sha256'] != inputs['protocol_sha256'] or detector_info['detector'] != detector:
            raise FreezeError('Zmiešaná identita detektora.')
        fingerprint = digest(detector_info)
        statistics = {}
        for split in SPLITS:
            path = folder / (split + '.jsonl')
            complete = read(folder / (split + '.complete.json'))
            anchors[str((folder / (split + '.complete.json')).resolve())] = sha(folder / (split + '.complete.json'))
            if inputs['profiles'][split] != complete['sha256']:
                raise FreezeError('Analýza nepatrí k dokončeným profilom.')
            checked_file(path, complete['sha256'], anchors)
            statistics[split] = validate_profiles(path, run / 'plans' / (split + '.jsonl'), complete,
                                                 fingerprint, split, EXPECTED_DEVELOPMENT_COUNTS[split])
        for name, checksum in receipt['files'].items():
            checked_file(inside(folder, name), checksum, anchors)
        if 'selector.json' not in receipt['files']:
            raise FreezeError('Analýza neuzamkla učený selektor.')
        model = read(folder / 'selector.json')
        if model.get('provenance') != inputs:
            raise FreezeError('Učený selektor nemá platný pôvod.')
        validate_selector(model, statistics)
        # Konkrétne váhy a predspracovanie zostávajú dostupné a zhodné.
        assets = ARTICLE_ROOT / 'assets' / detector
        checked_file(assets / 'manifest.json', detector_info['adapter']['asset_manifest_sha256'], anchors)
        asset_manifest = read(assets / 'manifest.json')
        for name, item in detector_info['adapter']['assets'].items():
            if asset_manifest['files'][name]['sha256'] != item['sha256']:
                raise FreezeError('Zmenený manifest obrazového modelu.')
            checked_file(inside(assets, name), item['sha256'], anchors)
        models[detector] = {'fingerprint': fingerprint, 'temperature': model['temperature'],
                            'selector_sha256': sha(folder / 'selector.json'),
                            'detector_sha256': sha(folder / 'detector.json'), 'statistics': statistics}
    return protocol, models, index, anchors


def verify_external_plan(external_plan_dir, development_index):
    folder = Path(external_plan_dir).resolve()
    anchors = {}
    protocol = checked_protocol(folder, anchors)
    if not isinstance(protocol.get('version'), str) or not protocol['version']:
        raise FreezeError('Externý plán nemá verziu.')
    required = ('manifest.jsonl', 'audit.json', 'excluded_identities.jsonl')
    for name in required:
        if name not in protocol['files']:
            raise FreezeError('Externý protokol neuzamkol súbor: ' + name)
    for name, checksum in protocol['files'].items():
        checked_file(inside(folder, name), checksum, anchors)
    audit = read(folder / 'audit.json')
    if audit.get('version') != 'article-external-data-audit-v1':
        raise FreezeError('Chýba podporovaný audit nezávislosti.')
    for key in ('ready_for_freeze', 'history_inventory_complete', 'all_images_verified', 'score_independent_selection'):
        if audit.get(key) is not True:
            raise FreezeError('Externý plán zatiaľ nie je pripravený: ' + key)
    if audit.get('manifest_sha256') != sha(folder / 'manifest.jsonl'):
        raise FreezeError('Audit nepatrí k aktuálnemu externému manifestu.')
    if not isinstance(audit.get('history_files'), dict) or not audit['history_files']:
        raise FreezeError('Chýbajú kontrolné súčty historických zdrojov.')
    for source, checksum in audit['history_files'].items():
        if not Path(source).is_absolute():
            raise FreezeError('Historický zdroj musí mať absolútnu cestu.')
        checked_file(source, checksum, anchors)
    if not isinstance(audit.get('history_coverage'), dict) or not audit['history_coverage']:
        raise FreezeError('Chýba explicitný rozsah a obmedzenia historického auditu.')
    coverage = audit['history_coverage'].get('phash', {})
    if (any(type(coverage.get(k)) is not int or coverage[k] < 0 for k in ('available', 'total', 'missing'))
            or coverage['available'] + coverage['missing'] != coverage['total']
            or coverage['available'] == 0):
        raise FreezeError('Neplatné číselné pokrytie historických pHash.')
    if coverage['missing'] and not isinstance(coverage.get('limitation'), str):
        raise FreezeError('Chýbajúce historické pHash musia mať explicitne opísané obmedzenie.')
    if coverage['missing'] and not coverage['limitation'].strip():
        raise FreezeError('Obmedzenie chýbajúcich pHash nesmie byť prázdne.')
    for key in ('group', 'original_id', 'source_key', 'sha256', 'rgb_sha256', 'dependency_keys', 'phash_unresolved'):
        if type(audit.get('disjointness', {}).get(key)) is not int or audit['disjointness'][key] != 0:
            raise FreezeError('Nevyriešené alebo neoverené prekryvy: ' + key)
    phash = audit.get('phash', {})
    if phash.get('bits') != 64 or phash.get('max_hamming_distance') != 4 or phash.get('compared_with_history') is not True:
        raise FreezeError('Chýba audit 64-bit pHash s Hammingovou vzdialenosťou <= 4.')
    forbidden = defaultdict(set)
    for key, values in development_index.items():
        forbidden[key].update(values)
    history_count = 0
    for row in lines(folder / 'excluded_identities.jsonl'):
        if not any(row.get(key) for key in IDENTITIES) and not row.get('dependency_keys'):
            raise FreezeError('Historický záznam neobsahuje žiadnu identitu.')
        add_identities(forbidden, row)
        history_count += 1
    if not history_count:
        raise FreezeError('Normalizovaná história je prázdna.')
    phash_index = PHashIndex(forbidden['phash'])
    seen_ids, seen_originals, seen_sha, seen_rgb = set(), set(), set(), set()
    counts = defaultdict(Counter)
    groups = defaultdict(set)
    forbidden_prediction_fields = {'margins', 'scores', 'error', 'prediction', 'detector_error', 'confidence'}
    for row in lines(folder / 'manifest.jsonl'):
        for key in ('sample_id', 'original_id', 'group', 'dataset', 'path', 'source_key', 'sha256', 'rgb_sha256', 'phash'):
            if not isinstance(row.get(key), str) or not row[key]:
                raise FreezeError('Externému obrázku chýba identita: ' + key)
        if row['dataset'] not in DATASETS or type(row.get('label')) is not int or row['label'] not in (0, 1):
            raise FreezeError('Neplatný dataset alebo trieda externého obrázka.')
        if not HEX64.fullmatch(row['sha256']) or not HEX64.fullmatch(row['rgb_sha256']) or not HEX16.fullmatch(row['phash']):
            raise FreezeError('Neplatný odtlačok externého obrázka.')
        if not Path(row['path']).is_absolute():
            raise FreezeError('Externý obraz musí mať absolútnu cestu.')
        if not isinstance(row.get('dependency_keys'), list) or not row['dependency_keys'] or any(not isinstance(k, str) or not k for k in row['dependency_keys']):
            raise FreezeError('Chýbajú normalizované pôvodové a výberové závislosti.')
        if row.get('input_spec') != {'version': 1, 'id': 'clean', 'operations': []}:
            raise FreezeError('Primárny externý plán používa iba pôvodný vstup bez vonkajšej úpravy.')
        if forbidden_prediction_fields & set(row):
            raise FreezeError('Externý plán nesmie byť vybraný podľa uložených výsledkov detektora.')
        for key, seen in (('sample_id', seen_ids), ('original_id', seen_originals), ('sha256', seen_sha), ('rgb_sha256', seen_rgb)):
            if row[key] in seen:
                raise FreezeError('Duplicitný externý obrázok podľa ' + key)
            seen.add(row[key])
        for key in IDENTITIES:
            if row.get(key) and normalized_identity(key, row[key]) in forbidden[key]:
                raise FreezeError('Externý obrázok sa prekrýva s históriou podľa ' + key)
        dependencies = set(row['dependency_keys']) | {row['group']}
        if dependencies & forbidden['dependency_keys']:
            raise FreezeError('Externá pôvodová alebo výberová skupina už bola použitá.')
        if phash_index.contains_near(int(row['phash'], 16)):
            raise FreezeError('Externý obrázok má pHash vzdialenosť <= 4 od dostupnej histórie.')
        checked_file(row['path'], row['sha256'], anchors)
        counts[row['dataset']][row['label']] += 1
        groups[row['dataset']].add(row['group'])
    if set(counts) != set(DATASETS) or any(counts[d][0] < 1 or counts[d][1] < 1 for d in DATASETS):
        raise FreezeError('Externý test potrebuje obe triedy na všetkých troch datasetoch.')
    return protocol, audit, {d: {'real': counts[d][0], 'fake': counts[d][1], 'groups': len(groups[d])} for d in DATASETS}, anchors


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def verify_frozen(output_dir):
    folder = Path(output_dir).resolve()
    anchors = {}
    protocol = checked_protocol(folder, anchors)
    receipt = read(folder / 'complete.json')
    if (receipt.get('protocol_sha256') != sha(folder / 'protocol.json')
            or receipt.get('ready_for_external_inference') is not True
            or receipt.get('external_inference_performed') is not False
            or receipt.get('external_metrics_computed') is not False):
        raise FreezeError('Neplatné potvrdenie zmrazenia.')
    for relative, checksum in protocol['files'].items():
        checked_file(inside(folder, relative), checksum, anchors)
    return protocol


def freeze_protocol(development_run, external_plan_dir, output_dir):
    run, plan, output = map(lambda value: Path(value).resolve(), (development_run, external_plan_dir, output_dir))
    allowed = (ROOT / 'frozen').resolve()
    if output == allowed or not output.is_relative_to(allowed) or output.is_relative_to(run) or output.is_relative_to(plan):
        raise FreezeError('Zmrazený výstup musí byť nový podpriečinok external_testing/frozen.')
    dev, models, development_index, development_anchors = verify_development(run)
    external, audit, counts, external_anchors = verify_external_plan(plan, development_index)
    anchors = {**development_anchors, **external_anchors}
    inputs = {'development_run': str(run), 'external_plan_dir': str(plan),
              'development_protocol_sha256': sha(run / 'protocol.json'),
              'external_protocol_sha256': sha(plan / 'protocol.json'),
              'input_anchors': anchors}
    if output.exists():
        frozen = verify_frozen(output)
        if frozen['inputs'] != inputs:
            raise FreezeError('Existujúci zmrazený test patrí iným vstupom; neprepisuje sa.')
        return frozen
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name('.' + output.name + '.preparing-' + uuid.uuid4().hex)
    temporary.mkdir()
    def copy(source, relative):
        source, target = Path(source), temporary / relative
        expected = sha(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        if sha(source) != expected or sha(target) != expected:
            raise FreezeError('Zdroj sa zmenil pri zmrazovaní: ' + str(source))
    for name in ('manifest.jsonl', 'audit.json', 'excluded_identities.jsonl', 'protocol.json', 'protocol.sha256'):
        copy(plan / name, 'plan/' + name)
    for detector in DETECTORS:
        for name in ('selector.json', 'detector.json', 'analysis.complete.json'):
            copy(run / detector / name, 'models/' + detector + '/' + name)
    copy(run / 'protocol.json', 'development/protocol.json')
    for name in dev['code']:
        copy(run / 'snapshot/code' / name, 'code/' + name)
    copy(run / 'snapshot/augmentation.py', 'code/augmentation.py')
    copy(Path(__file__), 'preparation/freeze.py')
    frozen = {
        'version': 'article-independent-external-freeze-v1', 'created_utc': datetime.now(timezone.utc).isoformat(),
        'inputs': inputs, 'detectors': models, 'counts': counts, 'environment': dev['environment'],
        'profile': dev['profile'], 'decision_rule': 'Syntetický pôvod práve vtedy, keď raw_logit >= 0; selektor nemení predikovanú triedu.',
        'methods': ['confidence', 'tta_mean', *FEATURE_COLUMNS],
        'primary': {'dataset': 'OpenFake', 'contrasts': ['clip:lr_full-confidence', 'resnet50:lr_full-confidence'],
                    'metric': 'Párový rozdiel blokovej AURC; záporná hodnota zvýhodňuje profil.',
                    'pointwise_interval': 0.95,
                    'joint_claim': 'Prínos pre oba detektory sa posudzuje prienikovo-zjednocovacím pravidlom (intersection-union).',
                    'multiplicity': 'Spoločné tvrdenie vyžaduje splnenie oboch vopred určených podmienok; jednotlivé intervaly sú bodové 95 %.',
                    'success_rule': 'Tvrdiť prínos pre oba detektory iba vtedy, ak sú obe horné hranice 95 % intervalov rozdielu pod nulou; vykázať oba výsledky.'},
        'secondary': {'datasets': ['DeepShield', 'CommunityForensics'], 'fixed_coverages': [0.8, 0.9, 0.95],
                      'ablations': ['lr_confidence', 'lr_resize_only', 'lr_jpeg_only'],
                      'interval_policy': 'Opisné bodové 95 % intervaly; žiadny výber podľa výsledkov testu.'},
        'bootstrap': {'repetitions': 5000, 'seed': 20261027, 'unit': 'Celé spojené pôvodové a výberové skupiny.',
                      'pairing': 'Rovnaké žrebované skupiny pre metódy v rámci detektora a datasetu.',
                      'scope': 'Pevné naučené modely; neistota opakovaného učenia nie je zahrnutá.'},
        'ties': {'aurc': 'Celé bloky zhodných skóre.', 'fixed_coverage': 'SHA256 sample_id bez referenčných tried.'},
        'independence': {'history_coverage': audit['history_coverage'],
                         'scope': 'Dosiaľ nepoužité známe skupiny a presný obsah voči auditovanej lokálnej histórii; kontrola blízkych duplicít pokrýva iba dostupné historické pHash.',
                         'limitation': 'Chýbajúce historické percepčné odtlačky a neznámy prekryv s tréningom verejných detektorov neumožňujú univerzálnu záruku neprekryvu.'},
        'prohibitions': ['Na externých triedach neučiť, nemeniť teplotu ani príznaky a neladiť prahy.',
                         'Neprenášať automaticky záruku 2 % rizika pri zmene distribúcie.',
                         'Nevyberať a nenahrádzať vzorky podľa testovacích skóre.',
                         'Netvrdiť čistú kauzalitu architektúry pri rôzne predtrénovaných detektoroch.'],
        'external_inference_performed': False, 'external_metrics_computed': False,
    }
    frozen['files'] = {p.relative_to(temporary).as_posix(): sha(p) for p in temporary.rglob('*') if p.is_file()}
    # Uzavretie kontroly proti súbežnej zmene zdrojových súborov.
    for path, checksum in anchors.items():
        if sha(path) != checksum:
            raise FreezeError('Vstup sa zmenil počas zmrazovania: ' + path)
    write_json(temporary / 'protocol.json', frozen)
    (temporary / 'protocol.sha256').write_text(sha(temporary / 'protocol.json') + '\n', encoding='ascii')
    write_json(temporary / 'complete.json', {'protocol_sha256': sha(temporary / 'protocol.json'),
               'ready_for_external_inference': True, 'external_inference_performed': False,
               'external_metrics_computed': False})
    os.replace(temporary, output)
    return frozen


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--development-run', type=Path, required=True)
    parser.add_argument('--external-plan', type=Path, required=True)
    parser.add_argument('--output', '--output-dir', dest='output', type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = freeze_protocol(args.development_run, args.external_plan, args.output)
    except (FreezeError, OSError, KeyError, TypeError, ValueError) as exc:
        print('Externý test zatiaľ nemožno zmraziť: ' + str(exc), file=sys.stderr, flush=True)
        return 2
    print('Externý plán a oba selektory sú zmrazené: ' + str(args.output.resolve()), flush=True)
    print(json.dumps({'counts': result['counts'], 'external_inference_performed': False}, ensure_ascii=False), flush=True)
    return 0


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
    sys.dont_write_bytecode = True
    raise SystemExit(main())
