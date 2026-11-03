"""Ľudsky čitateľný report iba z nových výsledkov samostatného behu."""
from __future__ import annotations

import html
import json
import os
from pathlib import Path
import uuid

from .common import read, write, digest, sha, output_path

METHOD_LABELS={'confidence':'Raw confidence','tta_mean':'TTA mean',
 'lr_full':'Full profile (main)','lr_confidence':'Learned confidence',
 'lr_jpeg_only':'JPEG-only ablation','lr_resize_only':'Resize-only ablation'}
COLORS=['#777777','#dd9900','#0072b2','#d55e00','#009e73','#aa4499']


def _verified(path):
    result=read(path)
    if result.get('integrity_sha256')!=digest({k:v for k,v in result.items() if k!='integrity_sha256'}):
        raise ValueError('Zmenený výsledok pre report: '+str(path))
    return result


def _text(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name('.'+path.name+'.'+uuid.uuid4().hex+'.tmp')
    with temporary.open('x',encoding='utf-8',newline='\n') as stream:
        stream.write(value)
        stream.flush();os.fsync(stream.fileno())
    os.replace(temporary,path)


def _number(value,scale=1.,digits=6):
    return 'n/a' if value is None else f'{scale*value:.{digits}f}'


def _interval(record,scale=1.,digits=6):
    return ('['+_number(record['lower_95'],scale,digits)+', '+
            _number(record['upper_95'],scale,digits)+']')


def _supported(record):
    return record['upper_95'] is not None and record['estimate']<0 and record['upper_95']<0


def _curve_svg(methods,title):
    width,height=760,310
    left,right,top,bottom=65,20,35,45
    w,h=width-left-right,height-top-bottom
    maxrisk=max(max(info['curve']['risk']) for info in methods.values())
    maxrisk=max(.01,maxrisk*1.04)
    parts=[f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(title)}">',
           f'<title>{html.escape(title)}</title>', '<rect width="100%" height="100%" fill="white"/>']
    for i in range(6):
        x=left+w*i/5;y=top+h-h*i/5
        parts.extend([f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top+h}" stroke="#e4e8ed"/>',
          f'<text x="{x:.2f}" y="{top+h+20}" text-anchor="middle">{20*i}</text>',
          f'<line x1="{left}" y1="{y:.2f}" x2="{left+w}" y2="{y:.2f}" stroke="#e4e8ed"/>',
          f'<text x="{left-8}" y="{y+4:.2f}" text-anchor="end">{100*maxrisk*i/5:.1f}</text>'])
    for (name,info),color in zip(methods.items(),COLORS):
        curve=info['curve'];count=len(curve['coverage'])
        indices=sorted({0,count-1,*[int(i*(count-1)/500) for i in range(501)]})
        points=' '.join(f'{left+w*curve["coverage"][i]:.2f},{top+h-h*curve["risk"][i]/maxrisk:.2f}' for i in indices)
        parts.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2"><title>{html.escape(METHOD_LABELS[name])}</title></polyline>')
    parts.extend([f'<text x="{left+w/2}" y="{height-5}" text-anchor="middle">Coverage (%)</text>',
                  '<text x="17" y="160" transform="rotate(-90 17 160)" text-anchor="middle">Accepted risk (%)</text>',
                  f'<text x="{left}" y="21">{html.escape(title)}</text>','</svg>'])
    legend=' '.join(f'<span style="color:{c};margin-right:1em">&#9632; {html.escape(METHOD_LABELS[n])}</span>' for n,c in zip(methods,COLORS))
    return '\n'.join(parts)+'<div class="legend">'+legend+'</div>'


def _table(headers,rows):
    return '<table><thead><tr>'+''.join('<th>'+html.escape(str(v))+'</th>' for v in headers)+'</tr></thead><tbody>'+''.join(
        '<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in row)+'</tr>' for row in rows)+'</tbody></table>'


def write_summary(run):
    run=output_path(run)
    config=read(run/'protocol.json')['config']
    results={};assessments={};status={};training_info={}
    for name in ('clip','resnet50'):
        folder=run/name
        state='Tréning ešte nie je dokončený.'
        if (folder/'detector'/'complete.json').exists():
            receipt=read(folder/'detector'/'complete.json')
            validation_path=folder/'detector'/'best_validation.json'
            if sha(validation_path)!=receipt['files']['best_validation.json']:
                raise ValueError('Zmenený záznam vybraného validačného checkpointu.')
            validation=read(validation_path)
            epoch=receipt['best_epoch']
            if type(epoch) is not int or epoch<0 or validation['epoch']!=epoch:
                raise ValueError('Nezhodná epocha vybraného checkpointu.')
            training_info[name]={'selected_epoch':epoch,'epochs_completed':receipt['epochs_completed'],
                                 'initial_checkpoint_retained':epoch==0,
                                 'receipt_sha256':sha(folder/'detector'/'complete.json')}
            state='Detektor je dokončený, selektor ešte nie.'
        if (folder/'selector'/'complete.json').exists():
            state='Selektor je zmrazený; retrospektívne vyhodnotenie ešte nie je dokončené.'
            assessments[name]=_verified(folder/'selector'/'assessment.json')
        result_path=folder/'evaluation'/'results.json'
        if result_path.exists():
            result=_verified(result_path)
            if (name not in training_info
                    or result['inputs']['anchors']['detector_complete_sha256']!=training_info[name]['receipt_sha256']):
                raise ValueError('Výsledky a informácia o vybranej epoche patria inému tréningu.')
            for relative,checksum in result['files'].items():
                if sha(folder/'evaluation'/relative)!=checksum:
                    raise ValueError('Zmenený výsledkový vstup reportu: '+relative)
            results[name]=result
            state='Tréning, selektor aj retrospektívne vyhodnotenie sú dokončené.'
        status[name]=state
    policy=('DeepShield je odložený test z domény tréningu. OpenFake je cross-dataset externý test. '
            'Oba testy opätovne používajú pôvodných 14 000 obrázkov EfficientNetu, preto je toto porovnanie retrospektívne. '
            'Modely, teploty, škálovanie a prahy sa určujú pred testovaním; nižšia AURC a nižšie riziko znamenajú lepšie poradie. '
            'Intervaly sú bodové párové 95 % percentilové intervaly celých pôvodových skupín, podmienené konkrétnymi naučenými modelmi. '
            'Nezahŕňajú neistotu opakovaného tréningu ani rodinnú korekciu násobných porovnaní.')
    lines=['SAMOSTATNÝ EXPERIMENT: ADAPTÁCIA CLIP A RESNET-50','',policy,'']
    body=['<h1>Adaptácia CLIP a ResNet-50: selektívna detekcia</h1>',
          '<p>'+html.escape(policy)+'</p>']
    if len(results)!=2:
        body.append('<p><strong>Priebežný report: výsledky oboch detektorov ešte nie sú dokončené.</strong></p>')
        lines.append('PRIEBEŽNÝ REPORT: obidva detektory ešte nie sú dokončené.')
    body.append(_table(['Detektor','Stav','Vybraná epocha'],
                      [[name,state,training_info[name]['selected_epoch'] if name in training_info else 'n/a']
                       for name,state in status.items()]))
    for name,state in status.items():
        lines += [name+': '+state]
        if name in training_info:
            info=training_info[name]
            epoch_text=(f'{name}: vybraná epocha {info["selected_epoch"]}; '
                        f'dokončených tréningových epoch {info["epochs_completed"]}.')
            if info['initial_checkpoint_retained']:
                epoch_text+=(' Vybraný ostal vstupný checkpoint (epocha 0); '
                             'validačné kritérium neuprednostnilo žiadnu dotrénovanú epochu.')
            lines.append(epoch_text)
            body.append('<p>'+html.escape(epoch_text)+'</p>')
    for name,result in results.items():
        body.append('<h2>'+html.escape(name)+'</h2>');lines+=['','DETEKTOR '+name]
        assessment=assessments[name]
        gate=('Samostatná kontrola vývojového rizika: '+('splnená' if assessment['passed'] else 'nesplnená')+
              f'; prijaté {assessment["accepted"]}/{assessment["total"]}, chyby {assessment["errors"]}, '
              'horná hranica '+_number(assessment['risk_upper_bound'],100,3)+' %.')
        body.append('<p>'+html.escape(gate)+'</p>');lines.append(gate)
        for cohort,entry in result['cohorts'].items():
            heading=f'{cohort}: {entry["images"]:,} obrázkov, {entry["groups"]} skupín'
            body.append('<h3>'+html.escape(heading)+'</h3>');lines+=['',heading]
            dm=entry['detector'];change=dm['adapted_minus_initial_accuracy']
            detectorrows=[]
            for key,label in [('initial','Vstupný checkpoint / rovnaký nový preprocessing'),
                              ('adapted',f'Vybraný checkpoint (epocha {training_info[name]["selected_epoch"]})')]:
                item=dm[key]
                detectorrows.append([label,_number(item['accuracy'],100,2),_number(item['balanced_accuracy'],100,2),
                                     _number(item['auroc']),item['errors']])
            body.append(_table(['Checkpoint','Presnosť (%)','Vyvážená presnosť (%)','AUROC','Chyby'],detectorrows))
            change_text=('Zmena presnosti vybraný mínus vstupný checkpoint: '+_number(change['estimate'],100,3)+
                         ' percentuálneho bodu, 95 % interval '+_interval(change,100,3)+'.')
            body.append('<p>'+html.escape(change_text)+'</p>');lines.append(change_text)
            body.append('<p>Porovnávanie checkpointov používa tie isté obrázky a ten istý nový preprocessing. '
                        'Nejde o porovnanie agregátov s pôvodným článkovým záťažovým testom.</p>')
            methodrows=[]
            for method,metrics in entry['methods'].items():
                counts=[metrics['risk_at_coverage'][key]['errors'] for key in ('0.8','0.9','0.95','original')]
                methodrows.append([METHOD_LABELS[method],_number(metrics['aurc']),*counts])
                lines.append(METHOD_LABELS[method]+': AURC='+_number(metrics['aurc'])+
                             '; chyby80/90/95/pôvodné_q='+str(counts))
            body.append(_table(['Selektor','AURC','Chyby 80 %','Chyby 90 %','Chyby 95 %','Chyby pôvodné q'],methodrows))
            ratio=entry['coverage_fractions']['original']
            body.append('<p>Pôvodné porovnateľné pokrytie q='+str(ratio[0])+'/'+str(ratio[1])+
                        '; v každej replike obidve metódy prijmú floor(q × N). '
                        'Riziko pri tomto pokrytí je odlišná otázka od znovu aplikovaného zmrazeného prahu.</p>')
            comparisons=[]
            for contrast,metrics in entry['contrasts'].items():
                for key in ('aurc','risk_original','risk_0.8','risk_0.9','risk_0.95'):
                    interval=metrics[key];scale=1 if key=='aurc' else 100
                    unit='AURC' if key=='aurc' else 'riziko (p. b.) '+key.removeprefix('risk_')
                    comparisons.append([contrast,unit,_number(interval['estimate'],scale),_interval(interval,scale),
                                        'áno' if _supported(interval) else 'nie'])
                main=metrics['aurc']
                lines.append(contrast+': ΔAURC='+_number(main['estimate'])+'; 95 % CI '+_interval(main)+
                             '; interval podporuje plný profil: '+('áno' if _supported(main) else 'nie'))
            body.append(_table(['Kontrast','Metrika','Rozdiel','95 % interval','Podpora nižšej hodnoty plného profilu'],comparisons))
            body.append(_curve_svg(entry['methods'],name+' / '+cohort))
            body.append('<p class="small">Graf je vizuálne vzorkovaný na najviac 501 bodov na krivku. '
                        'AURC aj intervaly sa počítajú zo všetkých uložených odpovedí.</p>')
            wp=entry['frozen_working_point']
            paragraph=(f'Zmrazený výskumný prah prijíma {wp["accepted"]}/{entry["images"]} obrázkov s {wp["errors"]} chybami '
                       f'(riziko {_number(wp["risk"],100,3)} %). Nasadzovací prah je {wp["deployment_threshold"]}; '
                       'pri neúspešnom samostatnom posúdení odmieta všetky vstupy. Poradie pri pevnom pokrytí sa vyhodnocuje samostatne.')
            body.append('<p>'+html.escape(paragraph)+'</p>');lines.append(paragraph)
    body.append('<h2>Interpretácia</h2><p>Uvádzajte absolútnu presnosť detektora spolu s prínosom selektora. '
                'Lepšie poradie nad slabým detektorom samo osebe nepreukazuje prakticky použiteľný detektor. '
                'Plný profil zostáva vopred určeným hlavným selektorom; výsledky ablácií automaticky nemenia model. '
                'Pri AURC a konkrétnom pokrytí môžu vyjsť odlišní víťazi.</p>')
    document='''<!doctype html><html lang="sk"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Dotrénovanie detektorov a selektor</title><style>
body{font:16px/1.5 system-ui,sans-serif;color:#172a3a;max-width:1150px;margin:40px auto;padding:0 22px;background:#fafbfd}
h1,h2,h3{line-height:1.2;color:#075a85}h2{margin-top:2.5em;border-top:2px solid #b6c8d4;padding-top:.8em}
table{border-collapse:collapse;width:100%;font-size:13px;background:white;margin:18px 0;display:block;overflow-x:auto}
th,td{padding:8px 10px;border-bottom:1px solid #dce4eb;text-align:right;white-space:nowrap}th:first-child,td:first-child{text-align:left}
th{background:#e7f0f6}svg{width:100%;background:white;border:1px solid #dce4eb}svg text{font:12px sans-serif;fill:#243b4a}
.legend,.small{font-size:12px}.legend{line-height:2;margin:8px 0 14px}
</style><body>'''+''.join(body)+'</body></html>\n'
    report_dir=run/'report'
    _text(report_dir/'index.html',document)
    _text(report_dir/'summary_SK.txt','\n'.join(lines)+'\n')
    write(report_dir/'summary.json',{'version':'article-adaptation-report-v1','status':status,
          'complete_detectors':list(results),'all_detectors_complete':len(results)==2,
          'checkpoint_selection':training_info,'bootstrap':config['evaluation'],
          'results_sha256':{name:sha(run/name/'evaluation'/'results.json') for name in results},
          'report_sha256':sha(report_dir/'index.html')})
    print('Report: '+str(report_dir/'index.html'),flush=True)
    return report_dir/'index.html'
