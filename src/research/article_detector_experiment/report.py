"""Učenie z hotových profilov a jednoznačne označený vývojový report."""
from __future__ import annotations

import csv
import html
from pathlib import Path

from .common import SPLITS, digest, output_path, read, sha, write
from .data import validate_partitions, verify
from .evaluation import evaluate_records, fit_models, score_records
from .profiles import load_complete


def analyze(run, detector, bootstrap=1000):
    run = output_path(run)
    p = verify(run)
    folder = run / detector
    split_data = {s: load_complete(run, detector, s) for s in SPLITS}
    validate_partitions(split_data)
    inputs = {'protocol_sha256': sha(run / 'protocol.json'),
              'detector_sha256': sha(folder / 'detector.json'),
              'profiles': {s: sha(folder / (s + '.jsonl')) for s in SPLITS},
              'bootstrap': bootstrap, 'seed': p['seed']}
    receipt_path = folder / 'analysis.complete.json'
    if receipt_path.exists():
        receipt = read(receipt_path)
        if receipt['inputs'] != inputs or any(sha(folder / name) != checksum for name, checksum in receipt['files'].items()):
            raise ValueError('Zmenené vstupy alebo dokončené hodnotenie; použite nový beh.')
        return read(folder / 'evaluation.json')
    artifact = fit_models(split_data['selector_train'], split_data['detector_calibration'])
    artifact['provenance'] = inputs
    write(folder / 'selector.json', artifact)
    selection = split_data['selector_selection']
    scores = score_records(selection, artifact)
    evaluation = evaluate_records(selection, scores, bootstrap=bootstrap, seed=p['seed'])
    evaluation['scope'] = p['scope']
    evaluation['interpretation'] = 'Výsledok na vývojovej podmnožine. Nejde o nezávislý externý test ani potvrdenie prínosu metódy.'
    write(folder / 'evaluation.json', evaluation)
    with (folder / 'selection_scores.csv').open('w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['sample_id', 'group', 'label', 'raw_logit', 'error', *scores])
        for i, row in enumerate(selection):
            writer.writerow([row['sample_id'], row['group'], row['label'], row['margins'][0], row['error'],
                             *[float(values[i]) for values in scores.values()]])
    with (folder / 'metrics.csv').open('w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['method', 'aurc', 'delta_aurc_vs_confidence', 'risk_80', 'risk_90', 'risk_95'])
        for method, metric in evaluation['methods'].items():
            delta = evaluation['differences_vs_confidence'].get(method, {}).get('aurc', 0.)
            writer.writerow([method, metric['aurc'], delta,
                             *[metric['risk_at_coverage'][str(q)]['risk'] for q in (.8, .9, .95)]])
    files = ('selector.json', 'evaluation.json', 'selection_scores.csv', 'metrics.csv')
    write(receipt_path, {'inputs': inputs, 'files': {name: sha(folder / name) for name in files}})
    build_report(run)
    return evaluation


def build_report(run):
    run = output_path(run)
    protocol = verify(run)
    sections = []
    summary = {}
    for detector in protocol['detectors']:
        folder = run / detector
        if not (folder / 'analysis.complete.json').exists():
            continue
        receipt = read(folder / 'analysis.complete.json')
        if any(sha(folder / name) != checksum for name, checksum in receipt['files'].items()):
            raise ValueError('Poškodené hodnotenie: ' + detector)
        inputs = receipt['inputs']
        if (inputs['protocol_sha256'] != sha(run / 'protocol.json')
                or inputs['detector_sha256'] != sha(folder / 'detector.json')
                or any(inputs['profiles'][s] != sha(folder / (s + '.jsonl')) for s in SPLITS)):
            raise ValueError('Hodnotenie nepatrí k aktuálnym vstupom: ' + detector)
        result = read(folder / 'evaluation.json')
        artifact = read(folder / 'selector.json')
        timing = read(folder / 'timing.json') if (folder / 'timing.json').exists() else {}
        rows = []
        for name, metrics in result['methods'].items():
            delta = result['differences_vs_confidence'].get(name, {}).get('aurc', 0.)
            risks = ''.join(f'<td>{100*metrics["risk_at_coverage"][str(q)]["risk"]:.2f} %</td>' for q in (.8, .9, .95))
            rows.append(f'<tr><th>{html.escape(name)}</th><td>{metrics["aurc"]:.6f}</td><td>{delta:+.6f}</td>{risks}</tr>')
        wall = timing.get('wall_seconds')
        seconds = f'{wall:.1f} s' if wall is not None else 'nedostupné'
        sections.append(f'''<section><h2>{html.escape(detector)}</h2>
<p>Výberová množina: {result['profiles']} profilov, {result['groups']} skupín.
Chyby detektora: {result['detector']['errors']}; presnosť {100*result['detector']['accuracy']:.2f} %.
Teplota: {artifact['temperature']:.6f}. Tréning selektora: {artifact['training']['profiles']} profilov,
{artifact['training']['groups']} skupín, {artifact['training']['errors']} chybných variantov.</p>
<p>Extrakcia v poslednom meranom pokračovaní: {timing.get('new_profiles', '—')} profilov za {seconds}.
Načítanie modelu a zahriatie sa do tohto času nezapočítavajú.</p>
<table><thead><tr><th>Metóda</th><th>AURC</th><th>ΔAURC voči istote</th><th>Riziko 80 %</th><th>Riziko 90 %</th><th>Riziko 95 %</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
<p><a href="{detector}/metrics.csv">Metriky CSV</a> · <a href="{detector}/evaluation.json">Krivky a skupinové intervaly JSON</a> ·
<a href="{detector}/selection_scores.csv">Jednotlivé skóre CSV</a> · <a href="{detector}/detector.json">Model a predspracovanie</a></p></section>''')
        summary[detector] = {'profiles': result['profiles'], 'detector': result['detector'],
                             'temperature': artifact['temperature'], 'timing': timing,
                             'full_profile_delta_aurc': result['differences_vs_confidence']['lr_full']['aurc']}
    write(run / 'summary.json', {'scope': protocol['scope'], 'detectors': summary,
                                'external_test_performed': False, 'protocol_sha256': sha(run / 'protocol.json')})
    content = f'''<!doctype html><html lang="sk"><meta charset="utf-8"><title>Experiment pre článok – vývojový report</title>
<style>body{{font:16px/1.55 system-ui;max-width:1150px;margin:40px auto;padding:0 24px;color:#172b40}}h1,h2{{line-height:1.2}}section{{margin-top:40px}}table{{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}}th,td{{text-align:right;padding:10px;border-bottom:1px solid #ccd5df}}th:first-child{{text-align:left}}.note{{background:#fff2d2;padding:18px;border-left:4px solid #a26500}}a{{color:#075a91}}</style>
<h1>Samostatný experiment pre článok</h1><p class="note">{html.escape(protocol['scope'])} Výsledky overujú technickú funkčnosť a vývojové správanie. <strong>Nie sú nezávislým potvrdením prínosu profilov.</strong> Externé testy neboli spustené.</p>
<p>Detektory sú zmrazené. Každý používa vlastnú teplotu a samostatne naučený selektor.
Profil: originál, JPEG 90/75/60, zmenšenia 0,75/0,50 s obnovou rozmerov.
Nižšia AURC je priaznivejšia; záporný rozdiel znamená zlepšenie oproti istote toho istého detektora.
Intervaly v JSON sú párové skupinové, bez korekcie viacnásobných porovnaní.</p>
{''.join(sections)}<p><a href="protocol.json">Uzamknutý protokol</a> · <a href="summary.json">Súhrn JSON</a></p></html>'''
    (run / 'report.html').write_text(content, encoding='utf-8')

