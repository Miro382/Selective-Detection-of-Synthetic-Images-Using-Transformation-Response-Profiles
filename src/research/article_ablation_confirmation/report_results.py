"""Slovenský report dokončeného testu; iba uložené výsledky, bez inferencie.

render_report(run) overí receipty a publikuje nový run/report/index.html.
Nevolá štatistický evaluátor a existujúci report nikdy neprepisuje.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import html
import json
import math
import os
from pathlib import Path
import struct
import uuid

VERSION = 'article-ablation-report-v1'
STATISTICS_SHA = 'b2327038f3e33befc9967eb6fac66bda0df436cd3064710e5b153fefceb0ea2f'
DESIGN_SHA = '8b8f293fd8045fb40ada750da667212a547067eda192f822867b31614b66864d'
COHORTS = ('DeepShield', 'OpenFake')
COUNTS = {'DeepShield': 1500, 'OpenFake': 6000}
FRACTIONS = {'DeepShield': (9807, 10000), 'OpenFake': (3639, 4000)}
SECONDARY = {'0.8': (4, 5), '0.9': (9, 10), '0.95': (19, 20)}
REASONS = {
    'missing_cohort': 'Chýbajúca kohorta.',
    'fewer_than_two_groups': 'Menej než dve skupiny.',
    'no_detector_errors': 'Detektor neurobil žiadnu chybu.',
    'no_correct_predictions': 'Všetky predikcie detektora sú chybné.',
    'no_accepted_observations': 'Pri pevnom pokrytí nie je prijatá žiadna vzorka.',
    'fewer_than_ten_matched_error_discordant_groups': 'Menej než 10 skupín s rozdielom prijatých chýb.',
    'fewer_than_5000_bootstrap_repetitions': 'Menej než 5000 bootstrapových replík.',
    'incomplete_bootstrap_aurc': 'Neúplný bootstrap rozdielu AURC.',
    'incomplete_bootstrap_matched': 'Neúplný bootstrap rozdielu rizika.',
    'degenerate_bootstrap_aurc': 'Degenerovaný bootstrap rozdielu AURC.',
    'degenerate_bootstrap_matched': 'Degenerovaný bootstrap rozdielu rizika.',
}


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b''):
            value.update(block)
    return value.hexdigest()


def _read(path):
    def invalid(value):
        raise ValueError('Nekonečná alebo NaN hodnota v JSON: ' + value)
    return json.loads(Path(path).read_text(encoding='utf-8'), parse_constant=invalid)


def _canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')


def _digest(value):
    return hashlib.sha256(_canonical(value)).hexdigest()


def _new_json(path, value):
    with Path(path).open('xb') as stream:
        stream.write(_canonical(value) + b'\n')


def _lines(path):
    with Path(path).open(encoding='utf-8') as stream:
        for line in stream:
            if not line.strip() or not line.endswith('\n'):
                raise ValueError('Neúplný riadok JSONL: ' + str(path))
            yield json.loads(line)


def _check(condition, message):
    if not condition:
        raise ValueError(message)


def _close(left, right):
    return isinstance(left, (int, float)) and math.isfinite(left) and math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-14)


def _integer(value, minimum=0):
    return type(value) is int and value >= minimum


def _file_map(folder, mapping):
    for name, expected in mapping.items():
        path = (folder / name).resolve()
        _check(path.is_relative_to(folder) and sha(path) == expected, 'Nesúhlasí uložený súbor: ' + name)


def validate_results(result, counts=COUNTS):
    """Kontrola uloženého kontraktu; žiadne nové intervaly ani modelové skóre."""
    _check(result['version'] == 'article-ablation-confirmation-statistics-v1', 'Neznáma verzia štatistiky.')
    _check(set(result['cohorts']) == set(COHORTS), 'Výsledok musí obsahovať obe kohorty.')
    bootstrap = result['bootstrap']
    _check(bootstrap['requested_repetitions'] == 5000 and bootstrap['seed'] == 20261028
           and bootstrap['quantiles'] == [.025, .975] and bootstrap['simultaneous_intervals'] is False,
           'Zmenený bootstrapový kontrakt.')
    primary = [name + ':' + metric for name in COHORTS for metric in ('aurc', 'matched')]
    _check(set(result['conditions']) == set(primary) and result['policy']['primary'] == primary,
           'Chýbajúce alebo zmenené primárne podmienky.')
    _check(result['policy']['direction'] == 'resize-minus-full' and result['policy']['noninferiority_margin'] is None
           and result['policy']['allowed_deterioration'] == 0 and result['model_change_performed'] is False,
           'Zmenený význam porovnania alebo neoprávnená zmena modelu.')
    passed = []
    for name in COHORTS:
        cohort = result['cohorts'][name]
        n = cohort['profiles']
        _check(n == counts[name] and _integer(n, 1), 'Nedokončená kohorta: ' + name)
        _check(_integer(cohort['groups'], 1) and cohort['groups'] <= n, 'Neplatný počet skupín.')
        _check(_integer(cohort['detector_errors']) and cohort['detector_errors'] <= n, 'Neplatný počet chýb.')
        _check(cohort['matched_fraction_exact'] == list(FRACTIONS[name])
               and _close(cohort['matched_fraction'], FRACTIONS[name][0] / FRACTIONS[name][1]), 'Zmenené pevné pokrytie.')
        fractions = {'matched': FRACTIONS[name], **SECONDARY}
        for model in ('full', 'resize'):
            values = cohort['models'][model]
            _check(_close(values['aurc'], values['aurc']) and 0 <= values['aurc'] <= 1, 'Neplatná AURC.')
            _check(set(values['risks']) == set(fractions), 'Chýbajúce pokrytie.')
            for key, (numerator, denominator) in fractions.items():
                risk = values['risks'][key]
                k = numerator * n // denominator
                _check(risk['accepted'] == k and _integer(risk['errors']) and risk['errors'] <= k
                       and _close(risk['risk'], risk['errors'] / k) and _close(risk['coverage'], k / n),
                       'Nesúlad počtu prijatých vzoriek/chýb/rizika: ' + name + '/' + key)
        full, resize = cohort['models']['full'], cohort['models']['resize']
        for metric in ('aurc', *fractions):
            expected = (resize['aurc'] - full['aurc'] if metric == 'aurc' else
                        resize['risks'][metric]['risk'] - full['risks'][metric]['risk'])
            _check(_close(cohort['delta_resize_minus_full'][metric], expected), 'Zmenené znamienko alebo rozdiel metrík.')
            interval = cohort['intervals'][metric]
            if interval is not None:
                _check(_close(interval['lower_95'], interval['lower_95'])
                       and _close(interval['upper_95'], interval['upper_95'])
                       and -1 <= interval['lower_95'] <= interval['upper_95'] <= 1
                       and _integer(interval['valid_repetitions'], 1) and interval['valid_repetitions'] <= 5000,
                       'Neplatný uložený interval.')
        discordance = cohort['paired_error_discordance']
        for key in ('removed_errors', 'added_errors', 'discordant_observations', 'discordant_groups'):
            _check(_integer(discordance[key]) and discordance[key] <= n, 'Neplatná párová diagnostika.')
        delta_errors = resize['risks']['matched']['errors'] - full['risks']['matched']['errors']
        _check(cohort['matched_error_count_delta'] == delta_errors == discordance['added_errors'] - discordance['removed_errors']
               and discordance['discordant_observations'] == discordance['added_errors'] + discordance['removed_errors']
               and discordance['discordant_groups'] <= min(cohort['groups'], discordance['discordant_observations']),
               'Nekonzistentné párové počty chýb.')
        adequacy = cohort['adequacy']
        _check(adequacy['minimum_groups'] == 2 and adequacy['minimum_matched_error_discordant_groups'] == 10,
               'Zmenené podmienky dostatočnosti údajov.')
        _check(cohort['status'] in ('ok', 'inadequate') and adequacy['passed'] == (cohort['status'] == 'ok')
               and adequacy['passed'] == (not adequacy['reasons']), 'Rozporný stav dostatočnosti údajov.')
        if adequacy['passed']:
            _check(cohort['groups'] >= 2 and 0 < cohort['detector_errors'] < n and discordance['discordant_groups'] >= 10,
                   'Rozhodnutie obchádza predurčený guard.')
        for metric in ('aurc', 'matched'):
            condition = result['conditions'][name + ':' + metric]
            interval = cohort['intervals'][metric]
            estimate = cohort['delta_resize_minus_full'][metric]
            upper = interval['upper_95'] if interval else None
            expected = bool(adequacy['passed'] and interval and interval['valid_repetitions'] == 5000 and estimate < 0 and upper < 0)
            _check(condition['passed'] is expected and _close(condition['estimate'], estimate)
                   and condition['upper_95'] == upper, 'Uložená primárna podmienka nezodpovedá výsledku.')
            passed.append(expected)
    eligible = all(passed)
    _check(result['all_four_conditions_passed'] is eligible
           and result['decision'] == ('eligible_for_replacement' if eligible else 'retain_full'), 'Nesúhlasí konečné rozhodnutie.')


def verify_run(run):
    run = Path(run).resolve()
    complete = _read(run / 'complete.json')
    _check(complete['complete'] is True, 'Report vyžaduje dokončený beh.')
    required = {'results.json', 'plan.jsonl', 'profiles.jsonl', 'profiles.complete.json'}
    _check(required <= set(complete['files']), 'Chýba uzamknutie výsledkov alebo žurnálu.')
    _file_map(run, complete['files'])
    result, protocol = _read(run / 'results.json'), _read(run / 'run_protocol.json')
    inputs = result['inputs']
    expected = {'run_protocol_sha256': sha(run / 'run_protocol.json'), 'plan_sha256': sha(run / 'plan.jsonl'),
                'profiles_sha256': sha(run / 'profiles.jsonl'), 'profiles_complete_sha256': sha(run / 'profiles.complete.json'),
                'statistics_sha256': STATISTICS_SHA}
    _check(inputs == expected == complete['inputs'], 'Vstupy analýzy a dokončenia sa nezhodujú.')
    design_folder = Path(protocol['design']).resolve()
    _check(protocol['design_sha256'] == DESIGN_SHA == sha(design_folder / 'protocol.json'), 'Zmenený predtestový návrh.')
    _check((design_folder / 'protocol.sha256').read_text(encoding='ascii').strip() == DESIGN_SHA,
           'Nesúhlasí sidecar predtestového návrhu.')
    design = _read(design_folder / 'protocol.json')
    _check(design['files']['code/statistics.py'] == STATISTICS_SHA == sha(design_folder / 'code/statistics.py'),
           'Zmenená zmrazená štatistika.')
    _check(protocol['counts'] == design['counts'] == COUNTS, 'Zmenené počty predtestového návrhu.')
    validate_results(result)
    journal = _read(run / 'profiles.complete.json')
    _check(journal['count'] == sum(COUNTS.values()) and journal['sha256'] == expected['profiles_sha256'],
           'Neúplný alebo zmenený žurnál profilov.')
    _verify_profiles(run, result, journal)
    return result, protocol, {**expected, 'complete_sha256': sha(run / 'complete.json'), 'design_sha256': DESIGN_SHA}


def _verify_profiles(run, result, journal):
    runtime = _read(run / 'runtime.json')
    fingerprint = _digest(runtime)
    _check(journal['runtime_sha256'] == sha(run / 'runtime.json') and journal['fingerprint'] == fingerprint
           and runtime['run_protocol_sha256'] == result['inputs']['run_protocol_sha256'], 'Zmenený runtime receipt.')
    plan = list(_lines(run / 'plan.jsonl'))
    records = list(_lines(run / 'profiles.jsonl'))
    _check(len(records) == len(plan) == sum(COUNTS.values()), 'Neúplné profily alebo plán.')
    _check(len({row['sample_id'] for row in plan}) == len(plan)
           and dict(Counter(row['dataset'] for row in plan)) == COUNTS, 'Duplicitné identity alebo nesprávne kohorty.')
    chain = _digest(['ablation-profiles-v1', fingerprint])
    observed_errors = Counter()
    observed_groups = {name: set() for name in COHORTS}
    for row, record in zip(plan, records):
        body = {key: value for key, value in record.items() if key != 'chain'}
        chain = _digest([chain, body])
        _check(record['chain'] == chain and record['fingerprint'] == fingerprint
               and record['sample_id'] == row['sample_id'] and record['plan_sha256'] == _digest(row),
               'Porušený reťazec, poradie alebo väzba profilu na plán.')
        _check(record['image_sha256'] == row['sha256'] and type(row['label']) is int and row['label'] in (0, 1),
               'Zmenený obrazový vstup alebo neplatná trieda.')
        _check(row['input_spec'] == {'version': 1, 'id': 'clean', 'operations': []}, 'Výsledok nepatrí čistým originálom.')
        logits, margins, calibrated = record['native_logits'], record['raw_margins'], record['calibrated_margins']
        _check(len(logits) == len(margins) == len(calibrated) == 6, 'Neúplný šesťpohľadový profil.')
        for pair, margin, value in zip(logits, margins, calibrated):
            _check(len(pair) == 2 and all(_close(x, x) for x in (*pair, margin, value)), 'Neplatný logit alebo margin.')
            raw = struct.unpack('f', struct.pack('f', pair[1] - pair[0]))[0]
            _check(raw == margin, 'Uložený margin nezodpovedá natívnym FP32 logitom.')
        prediction = int(margins[0] >= 0)
        error = int(prediction != row['label'])
        _check(type(record['prediction']) is int and record['prediction'] == prediction
               and type(record['error']) is int and record['error'] == error, 'Zmenená pôvodná predikcia alebo chyba.')
        _check(all(_close(record[key], record[key]) for key in ('full_score', 'resize_score')), 'Neplatné skóre selektora.')
        observed_errors[row['dataset']] += error
        observed_groups[row['dataset']].add(row['group'])
    _check(journal['chain'] == chain, 'Nesúhlasí koncový reťazec profilov.')
    for name in COHORTS:
        c = result['cohorts'][name]
        _check(c['detector_errors'] == observed_errors[name] and c['groups'] == len(observed_groups[name]),
               'Výsledky nezodpovedajú pôvodným chybám alebo skupinám.')


def _number(value, digits=6, scale=1, signed=False):
    if value is None:
        return 'nevyhodnotiteľné'
    return format(value * scale, ('+' if signed else '') + '.' + str(digits) + 'f').replace('.', ',')


def _ci(value, scale=1, digits=6):
    if value is None:
        return 'nevyhodnotiteľný'
    return '[' + _number(value['lower_95'], digits, scale, True) + '; ' + _number(value['upper_95'], digits, scale, True) + ']'


def _table(headers, rows):
    esc = lambda value: html.escape(str(value))
    return '<div class="table"><table><thead><tr>' + ''.join('<th>' + esc(x) + '</th>' for x in headers) + '</tr></thead><tbody>' + ''.join(
        '<tr>' + ''.join('<td>' + esc(x) + '</td>' for x in row) + '</tr>' for row in rows) + '</tbody></table></div>'


def _group_structure(plan):
    plan = list(plan)
    result = {}
    for name in COHORTS:
        counts = Counter(row['group'] for row in plan if row['dataset'] == name)
        total = sum(counts.values())
        _check(total > 0, 'Chýbajú skupiny kohorty: ' + name)
        largest = max(counts.values())
        result[name] = {'groups': len(counts), 'largest_group_size': largest,
                        'largest_group_share': largest / total, 'profiles': total}
    return result


def _content(result, inputs, group_structure=None):
    eligible = result['decision'] == 'eligible_for_replacement'
    title = 'Podmienky na náhradu sú splnené' if eligible else 'Ponechať plný selektor'
    explanation = ('Obe kohorty preukázali zlepšenie oboch primárnych metrík podľa vopred určeného pravidla. '
                   'Výsledok umožňuje samostatné rozhodnutie o náhrade; žiadny model sa automaticky nezmenil.' if eligible else
                   'Aspoň jedna primárna podmienka alebo podmienka dostatočnosti údajov neprešla. '
                   'Tento výsledok neoprávňuje nahradiť plný selektor. Nejednoznačný výsledok nie je dôkaz rovnocennosti ani nezhoršenia.')
    pieces = ['<header><p class="eyebrow">NOVÝ TEST · ZMRAZENÉ MODELY</p><h1>' + title + '</h1><p>' + explanation + '</p></header>',
              '<section><h2>Štyri primárne podmienky</h2><p>Porovnanie lr_resize_only voči frozen_v4. '
              'Δ = resize − full; záporný rozdiel znamená nižšiu AURC alebo nižšie riziko. '
              'Každá horná hranica 95 % intervalu musí byť striktne pod nulou a musia prejsť podmienky dostatočnosti údajov.</p>']
    rows = []
    for name in COHORTS:
        c = result['cohorts'][name]
        for metric in ('aurc', 'matched'):
            risk = metric == 'matched'
            rows.append([name, 'Riziko pri pevnom q' if risk else 'AURC',
                         _number(c['delta_resize_minus_full'][metric], 4 if risk else 6, 100 if risk else 1, True) + (' p. b.' if risk else ''),
                         _ci(c['intervals'][metric], 100 if risk else 1, 4 if risk else 6) + (' p. b.' if risk else ''),
                         'splnená' if result['conditions'][name + ':' + metric]['passed'] else 'nesplnená'])
    pieces += [_table(['Kohorta', 'Metrika', 'Δ resize − full', 'Bodový 95 % CI', 'Podmienka'], rows),
               '<figure><img src="primary_contrasts.png" alt="Štyri uložené rozdiely a bodové 95-percentné intervaly."/>'
               '<figcaption>Všetky štyri primárne rozdiely. Odhady a intervaly sú prevzaté z results.json; graf ich znovu nepočíta.</figcaption></figure>',
               '<p>Intervaly sú dvojstranné bodové percentilové 95 % CI [2,5 %; 97,5 %], získané z 5000 párových '
               'prevzorkovaní celých spojených pôvodových a výberových skupín; seed 20261028. Spoločné pravidlo je '
               'intersection-union: musia prejsť všetky štyri podmienky. Intervaly nie sú simultánne; nejde o test noninferiority.</p></section>']
    for name in COHORTS:
        c = result['cohorts'][name]
        full, resize = c['models']['full'], c['models']['resize']
        matched = full['risks']['matched']
        q = 100 * c['matched_fraction']
        pieces.append('<section><h2>' + name + '</h2><p>' + str(c['profiles']) + ' originálov v ' + str(c['groups']) +
                      ' skupinách; pôvodný detektor urobil ' + str(c['detector_errors']) + ' chýb. '
                      'Pevné nominálne q = ' + _number(q, 3) + ' %. Oba selektory prijímajú k = floor(q × N) = ' +
                      str(matched['accepted']) + ' vzoriek, teda skutočné pokrytie ' + _number(matched['coverage'], 4, 100) +
                      ' %. Počet prijatých vzoriek nepochádza z nového prahu full.</p>')
        if group_structure is not None:
            structure = group_structure[name]
            pieces.append('<p>Najväčšia skupina obsahuje <strong>' + str(structure['largest_group_size']) +
                          ' z ' + str(structure['profiles']) + ' vzoriek (' +
                          _number(structure['largest_group_share'], 2, 100) + ' % kohorty)</strong>. '
                          'Veľká nevyváženosť skupín obmedzuje spoľahlivosť bootstrapovej inferencie: '
                          'pri prevzorkovaní môže dominantná skupina chýbať alebo sa opakovať. '
                          'Predurčené podmienky dostatočnosti údajov nezaručujú nominálne pokrytie intervalov. '
                          'Skupiny ani rozhodovacie pravidlo sa podľa výsledkov nemenia.</p>')
        pieces.append(_table(['Selektor', 'AURC', 'Prijaté chyby / k', 'Riziko pri pevnom q'], [
            [label, _number(values['aurc']), str(values['risks']['matched']['errors']) + ' / ' + str(values['risks']['matched']['accepted']),
             _number(values['risks']['matched']['risk'], 4, 100) + ' %'] for label, values in [('Plný (frozen_v4)', full), ('Resize (lr_resize_only)', resize)]]))
        d = c['paired_error_discordance']
        pieces.append('<p>Rozdiel počtu prijatých chýb: <strong>' + _number(c['matched_error_count_delta'], 0, signed=True) +
                      '</strong>. Resize prestal prijímať ' + str(d['removed_errors']) + ' chybných predikcií prijatých full a začal prijímať ' +
                      str(d['added_errors']) + ' iných chybných predikcií. Rozdiel sa týka ' + str(d['discordant_groups']) +
                      ' skupín; vopred určené minimum je 10.</p>')
        reasons = c['adequacy']['reasons']
        pieces.append('<p><strong>Dostatočnosť údajov: ' + ('splnená' if c['adequacy']['passed'] else 'nesplnená') +
                      '.</strong> ' + (' '.join(html.escape(REASONS.get(reason, reason)) for reason in reasons) if reasons else
                       'Prešli všetky predurčené kontroly podpory a variability.') +
                      ' Tieto konzervatívne podmienky nezaručujú presné nominálne pokrytie bootstrapových intervalov.</p>')
        rows = []
        for key in SECONDARY:
            f, r = full['risks'][key], resize['risks'][key]
            rows.append([_number(float(key), 0, 100) + ' %', f['accepted'], str(f['errors']) + ' / ' + str(r['errors']),
                         _number(f['risk'], 4, 100) + ' / ' + _number(r['risk'], 4, 100),
                         _number(c['delta_resize_minus_full'][key], 4, 100, True), _ci(c['intervals'][key], 100, 4)])
        pieces += ['<h3>Sekundárne pokrytia</h3><p>Tieto opisné výsledky nemenia primárne rozhodnutie.</p>',
                   _table(['Nominálne pokrytie', 'k', 'Chyby full / resize', 'Riziko full / resize (%)', 'Δ rizika (p. b.)', 'Bodový 95 % CI (p. b.)'], rows), '</section>']
    pieces += ['<section><h2>Rozsah a obmedzenia</h2><p>Oba selektory vyhodnocujú rovnaké chyby pôvodnej deterministickej predikcie EfficientNetu. '
               'Modely ani teplota sa v tomto teste neučia. AURC rešpektuje celé bloky zhodných skóre; pri pevnom pokrytí používa '
               'predurčené SHA256 poradie identít bez závislosti od tried alebo chýb. V každej bootstrapovej replike sa počet prijatých vzoriek '
               'znovu určuje ako floor(q × N_replika).</p><p>Veľkosť vzorky vychádza z dostupnosti nových metadát, nie z preukázanej štatistickej sily. '
               'Zloženie DeepShield sa líši od historickej kohorty. Neúplné historické perceptuálne odtlačky a neznámy prekryv s verejnými '
               'predtrénovacími dátami zostávajú obmedzeniami. Intervaly nezahŕňajú neistotu opakovaného učenia. Ide o interný predtestový plán, '
               'nie verejnú registráciu.</p><p>Rozsah oddelenia od histórie: prijaté obrázky a ich evidované pôvodové identity '
               'sa preverujú voči známej histórii pomocou dostupných identít a obsahových odtlačkov; historicky načítané bloky sa '
               'znovu nepoužívajú. Nový blok môže obsahovať nevybrané historické identity. Skupiny analýzy sa konzervatívne spájajú '
               'aj cez nevzorkovaných susedov týchto blokov. Takéto oddelenie nie je dôkazom úplnej nezávislosti všetkých '
               'tranzitívnych pôvodových a výberových komponentov voči histórii.</p></section>',
               '<section><h2>Overiteľnosť reportu</h2><p>Report vznikol výlučne z dokončeného behu a uložených intervalov. '
               'Kontrolné súčty spájajú výsledky s plánom, žurnálom, predtestovým protokolom a zmrazenou štatistikou. '
               'Nevykonala sa ďalšia inferencia ani opakovanie bootstrapu.</p>',
               _table(['Artefakt', 'SHA256'], sorted(inputs.items())),
               '<p><a href="primary_contrasts.svg">Vektorový graf SVG</a> · <a href="report.complete.json">Receipt reportu</a> · '
               '<a href="../results.json">Zdrojové výsledky JSON</a></p></section>']
    return title, '\n'.join(pieces)


def _plot(result, folder):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout='constrained')
    for ax, metric, title, scale in zip(axes, ('aurc', 'matched'), ('Δ AURC', 'Δ rizika pri pevnom q (p. b.)'), (1, 100)):
        for y, name in enumerate(COHORTS):
            c = result['cohorts'][name]
            point, interval = c['delta_resize_minus_full'][metric] * scale, c['intervals'][metric]
            passed = result['conditions'][name + ':' + metric]['passed']
            color = '#087f5b' if passed else '#b45309'
            if interval:
                low, high = interval['lower_95'] * scale, interval['upper_95'] * scale
                ax.hlines(y, low, high, color=color, linewidth=2.5)
                ax.vlines((low, high), y - .07, y + .07, color=color, linewidth=1.5)
            ax.scatter(point, y, color=color, s=65, zorder=3)
        ax.axvline(0, color='#344054', linestyle='--', linewidth=1)
        ax.set_yticks(range(2), COHORTS)
        ax.set_ylim(1.6, -.6)
        ax.set_title(title, fontsize=13, pad=14)
        ax.set_xlabel('Záporný rozdiel zvýhodňuje resize', fontsize=10)
        ax.grid(axis='x', color='#e5e7eb', linewidth=.7)
        ax.spines[['top', 'right']].set_visible(False)
    fig.suptitle('Primárne kontrasty: uložené odhady a bodové 95 % intervaly', fontsize=14)
    fig.savefig(folder / 'primary_contrasts.png', dpi=180, facecolor='white')
    fig.savefig(folder / 'primary_contrasts.svg', facecolor='white')
    plt.close(fig)


def render_report(run):
    """Vráti Path nového report/index.html; existujúci report odmietne."""
    run = Path(run).resolve()
    destination = run / 'report'
    if destination.exists():
        raise FileExistsError('Existujúci report sa neprepisuje: ' + str(destination))
    result, _, inputs = verify_run(run)
    group_structure = _group_structure(_lines(run / 'plan.jsonl'))
    stage = run / ('.report-staging-' + uuid.uuid4().hex)
    stage.mkdir()
    _plot(result, stage)
    title, content = _content(result, inputs, group_structure)
    style = ('body{margin:0;background:#f5f7fa;color:#16202f;font:16px/1.6 system-ui,sans-serif}'
             'main{max-width:1120px;margin:auto;padding:36px 24px}header,section{background:white;border:1px solid #e2e6ec;border-radius:12px;padding:28px;margin-bottom:24px}'
             'h1{font-size:36px;line-height:1.2;margin:8px 0 20px}h2{font-size:24px;margin:0 0 12px}h3{margin-bottom:8px}'
             '.eyebrow{letter-spacing:.1em;font-size:12px;font-weight:700;color:#526175}.table{overflow:auto}table{border-collapse:collapse;width:100%;font-size:14px}'
             'th,td{border-bottom:1px solid #e2e6ec;padding:10px;text-align:left;vertical-align:top}th{background:#f8fafc}td:last-child{overflow-wrap:anywhere}'
             'figure{margin:24px 0}img{width:100%;height:auto}figcaption{color:#526175;font-size:13px}a{color:#175cd3}'
             '@media print{body{background:white}main{max-width:none;padding:0}header,section{border:0;padding:10px 0}tr,figure{break-inside:avoid}}')
    document = '<!doctype html><html lang="sk"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">' + \
               '<title>' + html.escape(title) + '</title><style>' + style + '</style><main>' + content + '</main></html>\n'
    with (stage / 'index.html').open('x', encoding='utf-8') as stream:
        stream.write(document)
    _, _, latest_inputs = verify_run(run)
    _check(inputs == latest_inputs, 'Vstupy sa počas tvorby reportu zmenili.')
    _new_json(stage / 'report.complete.json', {'version': VERSION, 'complete': True, 'created_utc': datetime.now(timezone.utc).isoformat(),
              'inputs': inputs, 'report_implementation_sha256': sha(__file__), 'inference_performed': False, 'bootstrap_recomputed': False,
              'group_structure': group_structure,
              'files': {p.name: sha(p) for p in sorted(stage.iterdir()) if p.is_file()}})
    if destination.exists():
        raise FileExistsError('Report medzitým vznikol: ' + str(destination))
    os.rename(stage, destination)
    return destination / 'index.html'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    print(render_report(args.run))


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
