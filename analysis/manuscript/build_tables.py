"""Rebuild manuscript tables from audited aggregate evidence. No inference or fitting."""
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'tables'
OUT.mkdir(exist_ok=True)
new = json.loads((ROOT/'evidence/new_experiments.json').read_text(encoding='utf-8'))
core = json.loads((ROOT/'evidence/core_evidence.json').read_text(encoding='utf-8'))
names = {'confidence':'Confidence','tta_mean':'TTA mean','lr_confidence':'Learned confidence',
         'energy_t1':'Energy ($T_E=1$)','frozen_v4':'Full profile',
         'mc_entropy_30':'MC dropout, 30 draws','mc_entropy_6':'MC dropout, 6 draws',
         'mlp_bahat_logits_s17':'Bahat-style MLP, seed 17',
         'mlp_bahat_logits_s29':'Bahat-style MLP, seed 29',
         'mlp_bahat_logits_s43':'Bahat-style MLP, seed 43'}

def table(filename, caption, label, spec, header, rows, wide=True):
    env = 'table*' if wide else 'table'
    text = '\\begin{'+env+'}[t]\n\\caption{'+caption+'}\n\\label{'+label+'}\n'
    text += '\\centering\\small\n\\begin{tabular}{'+spec+'}\n\\toprule\n'
    text += header + ' \\\\\n\\midrule\n'
    rows = [row.rstrip().rstrip('\\').rstrip() + ' ' + chr(92)*2 for row in rows]
    text += '\n'.join(rows) + '\n\\bottomrule\n\\end{tabular}\n\\end{'+env+'}\n'
    (OUT/filename).write_text(text,encoding='utf-8')

rows=[]
for dataset in ['DeepShield','OpenFake']:
    c=core['original_external_confirmation']['cohorts'][dataset]
    p=c['policies']['selector_v4']['research']
    conf=c['matched_confidence']['selector_v4']['research']['confidence']
    b=c['bootstrap']['group']['statistics']
    a=b['delta_aurc_selector_v4_minus_confidence']
    rows.append(f"{dataset} & {p['accepted']:,}/{p['total']:,} & {p['errors']} / {conf['errors']} & "
                f"{100*p['risk']:.3f} / {100*conf['risk']:.3f} & ${a['estimate']:.6f}$ & "
                f"$[{a['ci_95'][0]:.6f}, {a['ci_95'][1]:.6f}]$ \\")
table('original_external.tex','Original frozen EfficientNet selector versus confidence. Accepted errors and risk show full / confidence at the same acceptance count. The AURC difference is full minus confidence; intervals use the original 2,000-replicate protocol.','tab:original','lrrrrl','Cohort & Accepted / total & Errors & Risk (\\%) & $\\Delta$AURC & 95\\% interval',rows)

rr=new['tables']['efficientnet_reference_all_methods']
rows=[]
order=['confidence','lr_confidence','tta_mean','energy_t1','mc_entropy_6','mc_entropy_30',
       'mlp_bahat_logits_s17','mlp_bahat_logits_s29','mlp_bahat_logits_s43','frozen_v4']
for method in order:
    selected=[next(x for x in rr if x['cohort']==d and x['method']==method) for d in ['development','DeepShield','OpenFake']]
    row=names[method]+' & '+' & '.join(f"{x['aurc']:.6f}" for x in selected)
    row+=' & '+' & '.join(str(x['risk_at_coverage']['0.95']['errors']) for x in selected[1:])+' \\\\'
    rows.append(row)
table('references.tex','EfficientNet selection references. Development is the reused 2,000-input selection set; DS/OF are the retrospective 10,000/4,000-image cohorts. AURC is unscaled and lower is better. Final columns give errors at 95\\% coverage (9,500 DS and 3,800 OF acceptances). All Bahat-style seeds are shown.','tab:references','lrrrrr','Method & Dev. AURC & DS AURC & OF AURC & DS errors & OF errors',rows)

rows=[]
for d in ['DeepShield','OpenFake']:
    for m in ['mc_entropy_30','energy_t1','mlp_bahat_logits_s43']:
        x=next(x for x in rr if x['cohort']==d and x['method']==m)
        delta=x['profile_minus_method']; ci=delta['intervals']['aurc']
        rows.append(f"{d} & {names[m]} & ${delta['aurc']:.6f}$ & $[{ci['lower_95']:.6f}, {ci['upper_95']:.6f}]$ \\")
table('reference_contrasts.tex','Selected retrospective AURC contrasts: full profile minus reference. Pointwise 95\\% paired group-bootstrap intervals (5,000 replicates) are descriptive. Positive values favor the reference.','tab:referenceci','llrl','Cohort & Reference & $\\Delta$AURC & 95\\% interval',rows)

pp=[x for x in new['tables']['public_detector_all_methods'] if x['scope']=='external']
rows=[]
for det in ['clip','resnet50']:
    for d,short in [('DeepShield','DS'),('OpenFake','OF'),('CommunityForensics','CF')]:
        items=[next(x for x in pp if x['detector']==det and x['dataset']==d and x['method']==m)
               for m in ['confidence','tta_mean','lr_confidence','lr_full','lr_jpeg_only','lr_resize_only']]
        rows.append(('CLIP' if det=='clip' else 'ResNet-50')+f" & {short} & {100*items[0]['detector_accuracy']:.2f} & "+' & '.join(f"{x['aurc']:.5f}" for x in items)+' \\\\')
table('public_detectors.tex','Fixed-checkpoint stress tests: unconditional accuracy (Acc.) and unscaled AURC for every evaluated selection method. DS is held out from the selector-fitting domain; OF and CF are cross-dataset samples. These stress-test cohorts differ from the principal evaluation cohorts in main-article Table~\\ref{main-tab:cohorts}. JPEG-only and resize-only are ablations.','tab:public','llrrrrrrr','Detector & Data & Acc. (\\%) & Conf. & TTA & Learned conf. & Full & JPEG & Resize',rows)

ab=new['tables']['independent_ablation']
rows=[]
for d in ['DeepShield','OpenFake']:
    f=next(x for x in ab if x['cohort']==d and x['method']=='full')
    r=next(x for x in ab if x['cohort']==d and x['method']=='resize')
    risk=f['risks']['matched']; ri=r['risks']['matched']
    rows.append(f"{d} & {risk['accepted']:,}/{f['n']:,} & {f['aurc']:.6f} & {r['aurc']:.6f} & {risk['errors']} / {ri['errors']} & {100*risk['risk']:.3f} / {100*ri['risk']:.3f} \\")
table('ablation.tex','Independent ablation point estimates. Both methods accept exactly the same number of inputs. Accepted errors and risk are full / resize-only. These point improvements alone do not satisfy the locked adoption rule.','tab:ablation','lrrrrr','Cohort & Accepted / total & Full AURC & Resize AURC & Errors & Risk (\\%)',rows)

rows=[]
for d in ['DeepShield','OpenFake']:
    f=next(x for x in ab if x['cohort']==d and x['method']=='full')
    for metric,title,scale,fmt in [('aurc','AURC',1,'.6f'),('matched','Matched risk (pp)',100,'.4f')]:
        ci=f['paired_ci95_resize_minus_full'][metric]; delta=f['delta_resize_minus_full'][metric]
        success=(d=='OpenFake' and metric=='aurc')
        reason='Pass' if success else ('No: CI touches 0; guard fails' if d=='DeepShield' and metric=='matched' else ('No: CI crosses 0; guard fails' if d=='DeepShield' else 'No: CI crosses 0'))
        rows.append(f"{d} & {title} & ${delta*scale:{fmt}}$ & $[{ci['lower_95']*scale:{fmt}}, {ci['upper_95']*scale:{fmt}}]$ & {reason} \\")
table('ablation_decision.tex','The four locked replacement conditions. Differences are resize-only minus full. Risk differences are percentage points (pp); AURC differences are unscaled. CI denotes confidence interval. Only one condition passes, so the full selector is retained.','tab:ablationdecision','llrll','Cohort & Outcome & Difference & 95\\% interval & Condition',rows)
ext = json.loads((ROOT/'evidence/extended_ablation_table.json').read_text(encoding='utf-8'))
en = {'frozen_v4':'Full profile (frozen LR)','lr_no_summaries':'LR without summaries',
      'lr_no_orientation':'LR without class orientation','lr_no_confidence':'LR without original confidence',
      'lr_summaries_only':'LR confidence + summaries','lr_jpeg_only':'LR JPEG views only',
      'lr_resize_only':'LR resize views only','lr_without_resize_0p5':'LR without half-size view',
      'lr_balanced_no_summaries':'Error-balanced LR, margins','mlp_ours_logits':'MLP, our-view logits',
      'mlp_bahat_logits':'MLP, Bahat-view logits','mlp_margins':'MLP, oriented margins'}
rows=[]
for m in ext['main_table_methods']:
    items=[x for x in ext['all_methods'] if x['method']==m or x['method'].startswith(m+'_s')]
    assert len(items)==(3 if m.startswith('mlp_') else 1)
    row=en[m]+f" & {items[0]['dimension']}"
    for d in ['DeepShield','OpenFake']:
        vals=[x['cohorts'][d] for x in items]
        av=sum(v['aurc'] for v in vals)/len(vals)
        errs=[v['matched_frozen_coverage']['errors'] for v in vals]
        er=str(errs[0]) if len(errs)==1 else f'{min(errs)}--{max(errs)}'
        row+=f' & {av:.6f} & {er}'
    rows.append(row)
table('extended_ablation.tex','Selected exploratory controls on the reused EfficientNet DS/OF evaluation cohorts. AURC is unscaled; errors use fixed counts $k_{\\rm DS}=9{,}807$ and $k_{\\rm OF}=3{,}639$. LR rows report individual fits. MLP rows report mean AURC and the range of accepted errors across all three seeds (17, 29, 43), not an ensemble or a confidence interval. The supplementary aggregate file retains all 26 methods, individual seed values, and every stored original paired AURC interval (1,000 replicates). The main model remains frozen throughout.','tab:extendedablation','lrrrrr','Method & Features & DS AURC & DS errors & OF AURC & OF errors',rows)
print('Generated 7 tables from audited JSON; cohort inventory is authored in experiments.tex.')
