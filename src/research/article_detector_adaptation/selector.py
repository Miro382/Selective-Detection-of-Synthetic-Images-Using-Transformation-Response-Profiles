"""Nové profily, selektor a retrospektívne párové porovnanie adaptovaných sietí.

Modul má iba explicitné funkcie; samotný import nič nespúšťa. Všetky nové
výstupy sú pod odovzdaným samostatným behom. Pôvodné obrázky sa iba čítajú.
"""
from __future__ import annotations

from datetime import datetime, timezone
from fractions import Fraction
import gc
import hashlib
import io
import json
import os
from pathlib import Path
import time
import uuid

from .common import read, write, records, sha, canonical, digest, load_helper, output_path

VERSION = 'article-adapted-selectors-v1'
METHODS = ('confidence', 'tta_mean', 'lr_full', 'lr_confidence', 'lr_jpeg_only', 'lr_resize_only')
CONTRASTS = {'lr_full_minus_confidence': ('lr_full', 'confidence'),
             'lr_full_minus_lr_confidence': ('lr_full', 'lr_confidence')}


def _now():
    return datetime.now(timezone.utc).isoformat()


def _sealed(body):
    return {**body, 'integrity_sha256': digest(body)}


def _checked(path):
    obj = read(path)
    body = {k: v for k, v in obj.items() if k != 'integrity_sha256'}
    if obj.get('integrity_sha256') != digest(body):
        raise ValueError('Zmenený artefakt: ' + str(path))
    return obj


def _configuration(run):
    cfg = read(Path(run) / 'protocol.json')['config']
    if cfg['selector']['C'] != .1:
        raise ValueError('Zmrazená implementácia selektora vyžaduje C=0.1.')
    if (cfg['evaluation']['coverages'] != [.8,.9,.95]
            or cfg['evaluation']['primary_cohort'] != 'OpenFake'
            or cfg['evaluation']['primary_contrasts'] != list(CONTRASTS)
            or cfg['evaluation']['original_fractions'] != {'DeepShield':[9807,10000],'OpenFake':[3639,4000]}
            or cfg['evaluation']['scope'] != 'retrospective_reuse_of_original_EfficientNet_cohorts'):
        raise ValueError('Štatistický plán sa líši od podporovaného porovnania.')
    for key in ('bootstrap','seed'):
        if type(cfg['evaluation'][key]) is not int or cfg['evaluation'][key]<1:
            raise ValueError('Bootstrap a seed musia byť kladné celé čísla.')
    for key in ('selection_risk','assessment_risk','delta','min_coverage'):
        if not 0 < cfg['selector'][key] < 1:
            raise ValueError('Neplatný parameter selektora: '+key)
    if type(cfg['selector']['min_accepted']) is not int or cfg['selector']['min_accepted'] < 1:
        raise ValueError('Minimálny počet prijatí musí byť kladné celé číslo.')
    return cfg


def _anchor(run, name):
    folder = Path(run) / name / 'detector'
    if not (folder / 'complete.json').is_file() or not (folder / 'best.pt').is_file():
        raise ValueError('Najprv dokončite tréning detektora: ' + name)
    return {'protocol_sha256': sha(Path(run) / 'protocol.json'),
            'best_checkpoint_sha256': sha(folder / 'best.pt'),
            'detector_complete_sha256': sha(folder / 'complete.json')}


def _variants(image):
    """Päť nezávislých zmien z toho istého originálu, bez ukladania obrázkov."""
    from PIL import Image
    image = image.convert('RGB')
    yield image.copy()
    for quality in (90, 75, 60):
        with io.BytesIO() as buffer:
            image.save(buffer, format='JPEG', quality=quality, subsampling=0, optimize=False)
            buffer.seek(0)
            with Image.open(buffer) as decoded:
                yield decoded.convert('RGB').copy()
    for scale in (.75, .5):
        size = tuple(max(1, round(v * scale)) for v in image.size)
        yield image.resize(size, Image.Resampling.LANCZOS).resize(image.size, Image.Resampling.LANCZOS)


def _input(plan, augmentation):
    from PIL import Image
    data = Path(plan['path']).read_bytes()
    if hashlib.sha256(data).hexdigest() != plan['sha256']:
        raise ValueError('Obrázok sa zmenil oproti plánu: ' + plan['sample_id'])
    with Image.open(io.BytesIO(data)) as image:
        image.load()
        rgb = image.convert('RGB')
        # Pôvodné rgb_sha256 obsahuje veľkosť + dekódované RGB bajty podľa
        # pôvodného auditu; obsahová integrita sa tu overuje presným raw SHA.
        return augmentation.apply_spec(rgb, plan['input_spec'])


def _scan(path, plans, fingerprint, width, repair):
    import numpy as np
    rows = []
    chain = digest(['adapted-profile-journal-v1', fingerprint])
    if not path.exists():
        return rows, chain
    with path.open('r+b' if repair else 'rb') as stream:
        while True:
            position = stream.tell()
            line = stream.readline()
            if not line:
                break
            if not line.endswith(b'\n'):
                if repair:
                    stream.truncate(position)
                    stream.flush()
                    os.fsync(stream.fileno())
                    break
                raise ValueError('Nedokončený riadok v hotovom žurnále: ' + str(path))
            row = json.loads(line)
            body = {k: v for k, v in row.items() if k != 'chain'}
            expected = digest([chain, body])
            if row.get('chain') != expected or len(rows) >= len(plans):
                raise ValueError('Poškodený alebo nadbytočný profilový záznam.')
            plan = plans[len(rows)]
            if any(row.get(k) != v for k, v in plan.items()):
                raise ValueError('Profil nezodpovedá celému zmrazenému plánu.')
            margins = np.asarray(row['margins'], dtype=float)
            if (row.get('fingerprint') != fingerprint or margins.shape != (width,)
                    or not np.isfinite(margins).all()
                    or row['error'] != int(int(margins[0] >= 0) != plan['label'])):
                raise ValueError('Nesúhlasí model, počet odpovedí alebo chyba profilu.')
            chain = expected
            rows.append(row)
    return rows, chain


def _extract(run, name, split, adapter, *, initial=False):
    import numpy as np
    run = output_path(run)
    plans = records(run / 'plans' / (split + '.jsonl'))
    width = 1 if initial else 6
    folder = run / name / ('evaluation' if initial else 'selector') / 'profiles'
    stem = 'initial_test' if initial else split
    metadata_path = folder / (stem + '.metadata.json')
    journal = folder / (stem + '.jsonl')
    receipt_path = folder / (stem + '.complete.json')
    metadata = {'version': VERSION, 'anchors': _anchor(run, name),
                'plan_sha256': sha(run / 'plans' / (split + '.jsonl')),
                'checkpoint': 'initial' if initial else 'best', 'views': width,
                'adapter': adapter.metadata,
                'order': ['original'] if initial else ['original','jpeg90','jpeg75','jpeg60','resize0.75','resize0.50']}
    fingerprint = digest(metadata)
    if metadata_path.exists():
        if read(metadata_path) != metadata:
            raise ValueError('Zmenené nastavenie profilov; zvoľte nový beh: ' + stem)
    else:
        if journal.exists() or receipt_path.exists():
            raise ValueError('Profilom chýba identita modelu.')
        write(metadata_path, metadata)
    receipt = _checked(receipt_path) if receipt_path.exists() else None
    if receipt and (receipt['fingerprint'] != fingerprint or receipt['sha256'] != sha(journal)):
        raise ValueError('Zmenený dokončený profilový súbor.')
    rows, chain = _scan(journal, plans, fingerprint, width, repair=receipt is None)
    if receipt:
        if receipt['count'] != len(plans) or len(rows) != len(plans) or receipt['chain'] != chain:
            raise ValueError('Neúplné potvrdené profily.')
        print(f'{name}: {stem} hotové {len(rows)}/{len(plans)}', flush=True)
        return rows
    augmentation = load_helper(run, 'selector_augmentation_v4')
    print(f'{name}: {stem} pokračovanie {len(rows)}/{len(plans)}', flush=True)
    started = time.perf_counter()
    before_count = len(rows)
    with journal.open('ab') as stream:
        for plan in plans[len(rows):]:
            image = _input(plan, augmentation)
            images = [image] if initial else list(_variants(image))
            margins = np.asarray(adapter.predict_logits(images), dtype=float)
            if margins.shape != (width,) or not np.isfinite(margins).all():
                raise ValueError('Detektor nevrátil konečné logity v správnom poradí.')
            body = {**plan, 'margins': margins.tolist(),
                    'error': int(int(margins[0] >= 0) != plan['label']), 'fingerprint': fingerprint}
            row = {**body, 'chain': digest([chain, body])}
            payload = canonical(row)
            if isinstance(payload, str):
                payload = payload.encode('utf-8')
            stream.write(payload + b'\n')
            stream.flush()
            rows.append(row)
            chain = row['chain']
            if len(rows) % 100 == 0:
                os.fsync(stream.fileno())
                added = len(rows) - before_count
                rate = (time.perf_counter() - started) / max(1, added)
                print(f'{name}: {stem} {len(rows)}/{len(plans)} ({100*len(rows)/len(plans):.1f} %), {rate:.3f} s/profil', flush=True)
        os.fsync(stream.fileno())
    write(receipt_path, _sealed({'count': len(rows), 'sha256': sha(journal),
          'fingerprint': fingerprint, 'chain': chain, 'completed_utc': _now()}))
    return rows


def _profile_anchors(run, name, splits):
    folder = Path(run) / name / 'selector' / 'profiles'
    return {split: {'sha256': sha(folder / (split+'.jsonl')),
                    'receipt_sha256': sha(folder / (split+'.complete.json')),
                    'metadata_sha256':sha(folder / (split+'.metadata.json'))} for split in splits}


def _frozen(run, name):
    frozen = _checked(Path(run) / name / 'selector' / 'frozen.json')
    if frozen['version'] != VERSION or frozen['anchors'] != _anchor(run, name):
        raise ValueError('Zmrazený selektor patrí inému detektoru alebo protokolu.')
    if frozen['profile_anchors'] != _profile_anchors(run, name, ('detector_calibration','selector_train','selector_selection')):
        raise ValueError('Vývojové profily sa zmenili po zmrazení selektora.')
    return frozen


def fit_selector(run, name):
    """Fit a prah sa ukončia pred prvým otvorením risk-assessment profilov."""
    import numpy as np
    from .training import load_adapter
    run = output_path(run)
    cfg = _configuration(run)
    folder = run / name / 'selector'
    evaluation = load_helper(run, 'evaluation')
    math_module = load_helper(run, 'selector_v4_math')
    adapter = load_adapter(run, name, checkpoint='best')
    freeze_path = folder / 'frozen.json'
    if freeze_path.exists():
        frozen = _frozen(run, name)
    else:
        calibration = _extract(run, name, 'detector_calibration', adapter)
        train = _extract(run, name, 'selector_train', adapter)
        selection = _extract(run, name, 'selector_selection', adapter)
        artifact = evaluation.fit_models(train, calibration)
        scores = evaluation.score_records(selection, artifact)
        errors = np.asarray([row['error'] for row in selection], dtype=np.int64)
        threshold = math_module.choose_threshold(errors, scores['lr_full'], np.ones(len(errors)),
                     cfg['selector']['selection_risk'], cfg['selector']['min_coverage'])
        frozen = _sealed({'version': VERSION, 'created_utc': _now(), 'anchors': _anchor(run,name),
             'profile_anchors': _profile_anchors(run,name,('detector_calibration','selector_train','selector_selection')),
             'selector': artifact, 'main_method': 'lr_full', 'threshold_selection': threshold,
             'selection_diagnostics': evaluation.evaluate_records(selection, scores, bootstrap=0),
             'policy': 'Fixed full profile C=.1; no family selection. Temperature on detector_calibration; weights/scaler on selector_train; threshold on selector_selection only.'})
        write(freeze_path, frozen)
        print(f'{name}: selektor a prah zmrazené pred overením rizika.', flush=True)
    # No assessment/test labels entered the fitting and freezing above.
    frozen = _frozen(run, name)
    assessment_path = folder / 'assessment.json'
    if assessment_path.exists():
        assessment = _checked(assessment_path)
        if (assessment['frozen_sha256'] != sha(freeze_path)
                or assessment['profile_anchors'] != _profile_anchors(run,name,('selector_calibration',))):
            raise ValueError('Overenie rizika patrí inému selektoru/profilom.')
    else:
        rows = _extract(run,name,'selector_calibration',adapter)
        if len({r['group'] for r in rows}) != len(rows):
            raise ValueError('Binomické posúdenie vyžaduje jeden vstup na skupinu.')
        scores = evaluation.score_records(rows, frozen['selector'])['lr_full']
        accepted = scores <= frozen['threshold_selection']['tau']
        n = int(accepted.sum())
        k = int(np.asarray([r['error'] for r in rows])[accepted].sum())
        upper = math_module.clopper_pearson_upper(k,n,cfg['selector']['delta'])
        passed = n >= cfg['selector']['min_accepted'] and upper <= cfg['selector']['assessment_risk']
        assessment = _sealed({'version':VERSION, 'completed_utc':_now(), 'frozen_sha256':sha(freeze_path),
            'profile_anchors':_profile_anchors(run,name,('selector_calibration',)),
            'accepted':n,'total':len(rows),'errors':k,'risk':k/n if n else None,
            'risk_upper_bound':upper,'confidence':1-cfg['selector']['delta'],
            'target_risk':cfg['selector']['assessment_risk'],'min_accepted':cfg['selector']['min_accepted'],
            'passed':passed,'research_threshold':frozen['threshold_selection']['tau'],
            'deployment_threshold':frozen['threshold_selection']['tau'] if passed else -1.,
            'scope':'One fixed rule on a separate previously collected development sample; conditional IID binomial assessment, not a guarantee under distribution shift.'})
        write(assessment_path,assessment)
    complete_path=folder/'complete.json'
    complete_body={'version':VERSION,'anchors':_anchor(run,name),'frozen_sha256':sha(freeze_path),
                   'assessment_sha256':sha(assessment_path)}
    if complete_path.exists():
        prior=_checked(complete_path)
        if any(prior.get(k)!=v for k,v in complete_body.items()):
            raise ValueError('Zmenené dokončené učenie selektora.')
    else:
        write(complete_path,_sealed({**complete_body,'completed_utc':_now()}))
    print(f'{name}: selektor dokončený; assessment accepted={assessment["accepted"]}, errors={assessment["errors"]}, passed={assessment["passed"]}',flush=True)
    return {'frozen':frozen,'assessment':assessment}


def _development_ready(run):
    """Both detector/selector choices must precede any test extraction."""
    run=output_path(run)
    for name in ('clip','resnet50'):
        _frozen(run,name)
        folder=run/name/'selector'
        complete=_checked(folder/'complete.json')
        assessment=_checked(folder/'assessment.json')
        if (complete['anchors']!=_anchor(run,name)
                or complete['frozen_sha256']!=sha(folder/'frozen.json')
                or complete['assessment_sha256']!=sha(folder/'assessment.json')
                or assessment['frozen_sha256']!=sha(folder/'frozen.json')
                or assessment['profile_anchors']!=_profile_anchors(run,name,('selector_calibration',))):
            raise ValueError('Najprv dokončite a zmrazte oba selektory.')


def _fractions(cfg, cohort):
    fractions={str(q):Fraction(str(q)) for q in cfg['evaluation']['coverages']}
    a,b=cfg['evaluation']['original_fractions'][cohort]
    fractions['original']=Fraction(a,b)
    if any(not 0 < q <= 1 for q in fractions.values()):
        raise ValueError('Neplatné pevné pokrytie.')
    return fractions


def _prefix(errors, order, target, weights):
    import numpy as np
    if target==0:
        return {'accepted':0,'errors':0,'risk':None}
    cumulative=np.cumsum(weights[order])
    cumulative_errors=np.cumsum(weights[order]*errors[order])
    j=int(np.searchsorted(cumulative,target))
    prior_count=int(cumulative[j-1]) if j else 0
    prior_errors=int(cumulative_errors[j-1]) if j else 0
    mistakes=prior_errors+(target-prior_count)*int(errors[order[j]])
    return {'accepted':int(target),'errors':int(mistakes),'risk':mistakes/target}


def _statistics(errors, order, ends, weights, fractions):
    import numpy as np
    cw=np.cumsum(weights[order]); ce=np.cumsum(weights[order]*errors[order])
    counts=cw[ends]; keep=counts>0
    coverage=counts[keep]/cw[-1]
    risk=ce[ends][keep]/counts[keep]
    result=[float(np.diff(np.r_[0.,coverage])@risk)]
    total=int(cw[-1])
    for q in fractions.values():
        target=(q.numerator*total)//q.denominator
        result.append(_prefix(errors,order,target,weights)['risk'] if target else np.nan)
    return result


def _npz(path, **arrays):
    import numpy as np
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name('.'+path.name+'.'+uuid.uuid4().hex+'.tmp')
    with temporary.open('xb') as stream:
        np.savez_compressed(stream,**arrays)
        stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary,path)


def _cohort(run,name,cohort,rows,initial,artifact,assessment,cfg,evaluation):
    import numpy as np
    from sklearn.metrics import roc_auc_score,average_precision_score
    if [r['sample_id'] for r in rows]!=[r['sample_id'] for r in initial]:
        raise ValueError('Adaptovaný a pôvodný checkpoint nemajú totožné testovacie vstupy.')
    scores=evaluation.score_records(rows,artifact)
    labels=np.asarray([r['label'] for r in rows],dtype=np.int64)
    errors=np.asarray([r['error'] for r in rows],dtype=np.int64)
    initial_errors=np.asarray([r['error'] for r in initial],dtype=np.int64)
    margins=np.asarray([r['margins'] for r in rows],dtype=np.float64)
    initial_logits=np.asarray([r['margins'][0] for r in initial],dtype=np.float64)
    ids=[r['sample_id'] for r in rows]
    group_names,inverse=np.unique([r['group'] for r in rows],return_inverse=True)
    fractions=_fractions(cfg,cohort)
    metric_names=['aurc',*['risk_'+k for k in fractions]]
    one=np.ones(len(rows),dtype=np.int64)
    methods={}; prepared={}
    for method in METHODS:
        values=scores[method]
        order=evaluation.order_with_ties(values,ids)
        ordered=values[order]
        ends=np.r_[np.flatnonzero(ordered[1:]!=ordered[:-1]),len(ordered)-1]
        prepared[method]=(order,ends)
        curve=evaluation.risk_coverage_curve(errors,values)
        methods[method]={'aurc':curve['aurc'],'curve':curve,
            'risk_at_coverage':{key:_prefix(errors,order,(q.numerator*len(rows))//q.denominator,one)
                                for key,q in fractions.items()}}
    research_tau=assessment['research_threshold']
    accepted=scores['lr_full']<=research_tau
    k=int(accepted.sum())
    fixed_errors=int(errors[accepted].sum())
    frozen_working_point={'threshold':research_tau,'accepted':k,'errors':fixed_errors,
        'risk':fixed_errors/k if k else None,'coverage':k/len(rows),'assessment_passed':assessment['passed'],
        'deployment_threshold':assessment['deployment_threshold'],
        'deployment_accepted':k if assessment['passed'] else 0,
        'matched_references':{method:_prefix(errors,prepared[method][0],k,one) for method in METHODS if method!='lr_full'}}
    reps=int(cfg['evaluation']['bootstrap']); seed=int(cfg['evaluation']['seed'])
    if reps < 1 or len(group_names)<2:
        raise ValueError('Párové intervaly vyžadujú kladný počet replík a aspoň dve skupiny.')
    sampled_methods=['lr_full','confidence','lr_confidence']
    samples=np.empty((reps,len(sampled_methods),len(metric_names)),dtype=np.float64)
    accuracy_change=np.empty(reps,dtype=np.float64)
    rng=np.random.default_rng(seed)
    for b in range(reps):
        weights=np.bincount(rng.integers(0,len(group_names),len(group_names)),minlength=len(group_names))[inverse]
        for j,method in enumerate(sampled_methods):
            order,ends=prepared[method]
            samples[b,j]=_statistics(errors,order,ends,weights,fractions)
        accuracy_change[b]=float(weights@(initial_errors-errors)/weights.sum())
        if (b+1)%500==0 or b+1==reps:
            print(f'{name}: {cohort} párový bootstrap {b+1}/{reps}',flush=True)
    differences={}; paired=[]
    for key,(left,right) in CONTRASTS.items():
        li,ri=sampled_methods.index(left),sampled_methods.index(right)
        values=samples[:,li]-samples[:,ri]; paired.append(values)
        point=_statistics(errors,*prepared[left],one,fractions)
        baseline=_statistics(errors,*prepared[right],one,fractions)
        intervals={}
        for j,metric in enumerate(metric_names):
            valid=values[np.isfinite(values[:,j]),j]
            intervals[metric]={'estimate':float(point[j]-baseline[j]),
                'lower_95':float(np.quantile(valid,.025)) if len(valid) else None,
                'upper_95':float(np.quantile(valid,.975)) if len(valid) else None,
                'valid_repetitions':int(len(valid))}
        differences[key]=intervals
    folder=Path(run)/name/'evaluation'/cohort
    _npz(folder/'scores.npz',sample_ids=np.asarray(ids),groups=np.asarray([r['group'] for r in rows]),
         labels=labels,errors=errors,margins=margins,initial_logits=initial_logits,initial_errors=initial_errors,
         method_names=np.asarray(METHODS),scores=np.column_stack([scores[m] for m in METHODS]))
    _npz(folder/'bootstrap_draws.npz',method_names=np.asarray(sampled_methods),metric_names=np.asarray(metric_names),
         method_values=samples,contrast_names=np.asarray(list(CONTRASTS)),differences=np.stack(paired,axis=1),
         adapted_minus_initial_accuracy=accuracy_change,seed=np.asarray(seed),group_names=group_names)
    def detector_metrics(logits,err):
        byclass={str(c):{'count':int((labels==c).sum()),'accuracy':float(1-err[labels==c].mean())} for c in (0,1)}
        return {'accuracy':float(1-err.mean()),'balanced_accuracy':float(np.mean([v['accuracy'] for v in byclass.values()])),
                'errors':int(err.sum()),'class_metrics':byclass,'auroc':float(roc_auc_score(labels,logits)),
                'average_precision':float(average_precision_score(labels,logits))}
    output={'cohort':cohort,'role':'in-domain held-out' if cohort=='DeepShield' else 'cross-dataset external',
        'primary_cohort':cohort==cfg['evaluation']['primary_cohort'],
        'sample_status':'retrospective reuse of the original EfficientNet evaluation images',
        'images':len(rows),'groups':len(group_names),'methods':methods,'contrasts':differences,
        'coverage_fractions':{key:[q.numerator,q.denominator] for key,q in fractions.items()},
        'frozen_working_point':frozen_working_point,
        'detector':{'adapted':detector_metrics(margins[:,0],errors),'initial':detector_metrics(initial_logits,initial_errors),
             'adapted_minus_initial_accuracy':{'estimate':float(initial_errors.mean()-errors.mean()),
                    'lower_95':float(np.quantile(accuracy_change,.025)),
                    'upper_95':float(np.quantile(accuracy_change,.975))},
             'comparison_scope':'Same images and newly fixed detector-specific preprocessing for both checkpoints; not a reproduction of the old public-checkpoint stress-test aggregates.'},
        'bootstrap':{'repetitions':reps,'seed':seed,'unit':'whole connected provenance groups',
            'pairing':'identical group multiplicities for all scores and both detector checkpoints',
            'interval':'pointwise percentile 95%, no familywise correction',
            'scope':'conditional on fitted models and stored responses; no retraining uncertainty',
            'seed_scope':'RNG restarted per cohort and detector','aurc':'whole exact-score tie blocks',
            'fixed_coverage':'exact rational floor(numerator*N_b/denominator)',
            'tie_namespace':'article-detector-tie-v1/'},
        'files':{'scores.npz':sha(folder/'scores.npz'),'bootstrap_draws.npz':sha(folder/'bootstrap_draws.npz')}}
    write(folder/'results.json',_sealed(output))
    return output


def evaluate_selector(run,name):
    """Test je retrospektívny; oba selektory musia byť hotové už pred ním."""
    from .training import load_adapter
    run=output_path(run); cfg=_configuration(run)
    _development_ready(run)
    frozen=_frozen(run,name)
    folder=run/name/'evaluation'
    result_path=folder/'results.json'
    identity={'anchors':_anchor(run,name),'selector_sha256':sha(run/name/'selector'/'frozen.json'),
              'assessment_sha256':sha(run/name/'selector'/'assessment.json'),
              'test_plan_sha256':sha(run/'plans'/'test.jsonl')}
    if result_path.exists():
        result=_checked(result_path)
        if result['inputs']!=identity:
            raise ValueError('Hotové výsledky patria inému modelu alebo plánu.')
        if result['adapted_profiles']!=_profile_anchors(run,name,('test',)):
            raise ValueError('Zmenené adaptované testovacie profily.')
        for relative,checksum in result['files'].items():
            if sha(folder/relative)!=checksum:
                raise ValueError('Zmenený výsledok alebo uložené skóre: '+relative)
        return result
    assessment=_checked(run/name/'selector'/'assessment.json')
    adapter=load_adapter(run,name,checkpoint='best')
    adapted=_extract(run,name,'test',adapter)
    del adapter
    gc.collect()
    initial_adapter=load_adapter(run,name,checkpoint='initial')
    initial=_extract(run,name,'test',initial_adapter,initial=True)
    del initial_adapter
    gc.collect()
    evaluation=load_helper(run,'evaluation')
    cohorts={}
    for cohort in ('DeepShield','OpenFake'):
        left=[r for r in adapted if r['dataset']==cohort]
        right=[r for r in initial if r['dataset']==cohort]
        cohorts[cohort]=_cohort(run,name,cohort,left,right,frozen['selector'],assessment,cfg,evaluation)
    files={}
    for cohort in cohorts:
        for filename in ('results.json','scores.npz','bootstrap_draws.npz'):
            files[cohort+'/'+filename]=sha(folder/cohort/filename)
    files['profiles/initial_test.jsonl']=sha(folder/'profiles'/'initial_test.jsonl')
    files['profiles/initial_test.complete.json']=sha(folder/'profiles'/'initial_test.complete.json')
    files['profiles/initial_test.metadata.json']=sha(folder/'profiles'/'initial_test.metadata.json')
    # Adapted profiles remain under selector/profiles to use one cache format.
    result=_sealed({'version':VERSION,'completed_utc':_now(),'detector':name,'inputs':identity,
        'adapted_profiles':_profile_anchors(run,name,('test',)),
        'cohorts':cohorts,'files':files,
        'interpretation':'Report absolute detector accuracy alongside selector improvements. Positive ranking effects alone do not establish useful detector accuracy or universal superiority.'})
    write(result_path,result)
    return result


def write_summary(run):
    from .reporting import write_summary as render_summary
    return render_summary(run)
