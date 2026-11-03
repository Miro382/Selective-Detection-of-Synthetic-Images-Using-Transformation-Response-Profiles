"""Check published numerical outcomes against the packaged detector outputs.

This command performs no image access, inference, training, or new bootstrap
sampling. It reconstructs point metrics and checks stored bootstrap percentiles.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    path = Path(path)
    raw = path.read_bytes()
    if path.suffix == '.gz':
        raw = gzip.decompress(raw)
    return json.loads(raw)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def equal(observed, expected, context):
    if expected is None:
        assert observed is None, context
    elif isinstance(expected, (int, bool)):
        assert observed == expected, (context, observed, expected)
    else:
        assert np.isclose(observed, expected, rtol=0, atol=2e-12), (context, observed, expected)


def curve_area(errors, scores):
    order = np.argsort(scores, kind='stable')
    ordered = scores[order]
    ends = np.r_[np.flatnonzero(ordered[1:] != ordered[:-1]), len(order)-1]
    count = ends + 1
    mistakes = np.cumsum(errors[order], dtype=np.int64)[ends]
    return float(np.diff(np.r_[0., count / len(order)]) @ (mistakes / count))


def fixed_count(errors, scores, tie_key, count):
    assert type(count) is int and 0 <= count <= len(errors)
    accepted = np.lexsort((tie_key, scores))[:count]
    mistakes = int(errors[accepted].sum())
    return {'accepted':count, 'errors':mistakes,
            'risk':mistakes/count if count else None, 'coverage':count/len(errors)}


def compare_point(observed, expected, context):
    for key in ('accepted','errors','risk','coverage'):
        if key in expected:
            equal(observed[key], expected[key], context+'/'+key)


def verify_extra_scores(data):
    logits = data['native_logits'].astype(np.float64)
    mc = data['mc_logits'].astype(np.float64)
    assert mc.shape == (len(logits), 30, 2)
    maximum = logits.max(axis=1)
    energy = -maximum - np.log(np.exp(logits-maximum[:,None]).sum(axis=1))
    assert np.allclose(energy, data['score__energy_t1'], rtol=0, atol=1e-14)
    probability = np.exp(mc-mc.max(axis=-1,keepdims=True))
    probability /= probability.sum(axis=-1,keepdims=True)
    for draws in (6,30):
        mean = probability[:,:draws].mean(axis=1)
        logs = np.zeros_like(mean)
        np.log(mean, out=logs, where=mean>0)
        entropy = -(mean*logs).sum(axis=1)
        assert np.allclose(entropy, data['score__mc_entropy_'+str(draws)],rtol=0,atol=1e-14)


def auroc(labels, scores):
    order = np.argsort(scores, kind='stable')
    values = scores[order]
    starts = np.r_[0, np.flatnonzero(values[1:] != values[:-1])+1]
    ends = np.r_[starts[1:], len(order)]
    ranks = np.empty(len(order), dtype=np.float64)
    for start,end in zip(starts,ends):
        ranks[order[start:end]] = (start+1+end)/2
    positives = labels==1
    n_pos = int(positives.sum()); n_neg = len(labels)-n_pos
    return float((ranks[positives].sum()-n_pos*(n_pos+1)/2)/(n_pos*n_neg))


def check_interval(values, expected, context):
    assert np.isfinite(values).all() and len(values) == 5000, context
    low,high=np.quantile(values,[.025,.975])
    equal(float(low),expected['lower_95'],context+'/lower')
    equal(float(high),expected['upper_95'],context+'/upper')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check',action='store_true',help='Accepted for consistency; checking is always read-only.')
    parser.parse_args()
    index=read(ROOT/'evidence/datasets.json')
    methods_checked=points_checked=intervals_checked=0
    headline={}
    for item in index['datasets']:
        name=item['name']; path=ROOT/item['scores']; expected_path=ROOT/item['expected']
        assert digest(path)==item['sha256'], name+'/input checksum'
        assert digest(expected_path)==item['expected_sha256'], name+'/expected checksum'
        with np.load(path,allow_pickle=False) as z:
            data={k:z[k].copy() for k in z.files}
        expected=read(expected_path)
        errors=data['errors']; n=len(errors)
        assert n>0 and np.isin(errors,[0,1]).all(), name
        assert len(data['tie_key'])==n and len(np.unique(data['tie_key']))==n
        assert len(data['group_index'])==n and np.issubdtype(data['group_index'].dtype,np.integer)
        if 'prediction' in data:
            assert np.array_equal(errors,data['prediction']!=data['labels']),name+'/error definition'
        methods=expected['models'] if item['kind']=='ablation' else expected['methods']
        assert set(methods)=={k[7:] for k in data if k.startswith('score__')}, name+'/method inventory'
        areas={}
        for method, target in methods.items():
            scores=data['score__'+method]
            assert scores.shape==(n,) and np.isfinite(scores).all(),name+'/'+method
            area=curve_area(errors,scores);areas[method]=area
            equal(area,target['aurc'],name+'/'+method+'/AURC')
            methods_checked+=1
            for point in target.get('risk_at_coverage',target.get('risks',{})).values():
                compare_point(fixed_count(errors,scores,data['tie_key'],point['accepted']),point,name+'/'+method)
                points_checked+=1
            if 'matched_frozen_coverage' in target:
                point=target['matched_frozen_coverage']
                compare_point(fixed_count(errors,scores,data['tie_key'],point['accepted']),point,name+'/'+method+'/matched')
                points_checked+=1
            if 'fixed_threshold' in target:
                point=target['fixed_threshold'];tau=point['threshold']
                mask=np.zeros(n,dtype=bool) if tau is None else scores<=tau
                count=int(mask.sum());mistakes=int(errors[mask].sum())
                compare_point({'accepted':count,'errors':mistakes,'risk':mistakes/count if count else None,'coverage':count/n},point,name+'/'+method+'/threshold')
                points_checked+=1
        if item['kind']=='references':
            verify_extra_scores(data)
        if item['kind']=='adapted':
            detector=expected['detector']
            for key,err,score in [('adapted',errors,data['margins'][:,0]),('initial',data['initial_errors'],data['initial_logits'])]:
                assert np.array_equal(err,(score>=0)!=data['labels'])
                equal(1-float(err.mean()),detector[key]['accuracy'],name+'/'+key+'/accuracy')
                equal(int(err.sum()),detector[key]['errors'],name+'/'+key+'/errors')
                equal(auroc(data['labels'],score),detector[key]['auroc'],name+'/'+key+'/AUROC')
            draw_path=ROOT/item['bootstrap']
            assert digest(draw_path)==item['bootstrap_sha256']
            with np.load(draw_path,allow_pickle=False) as z:
                for i,contrast in enumerate(z['contrast_names']):
                    for j,metric in enumerate(z['metric_names']):
                        check_interval(z['differences'][:,i,j],expected['contrasts'][str(contrast)][str(metric)],name+'/'+str(contrast)+'/'+str(metric))
                        intervals_checked+=1
                check_interval(z['adapted_minus_initial_accuracy'],detector['adapted_minus_initial_accuracy'],name+'/accuracy gain')
                intervals_checked+=1
            headline[name]={'initial_accuracy':detector['initial']['accuracy'],'adapted_accuracy':detector['adapted']['accuracy'],
                            'relative_AURC_reduction_percent':100*(areas['confidence']-areas['lr_full'])/areas['confidence']}
        elif item['kind']=='comparison':
            headline[name]={'relative_AURC_reduction_percent':100*(areas['confidence']-areas['frozen_v4'])/areas['confidence'],
                'accepted_errors':{key:methods[key]['matched_frozen_coverage']['errors'] for key in ('frozen_v4','confidence')}}
    # Percentiles archived for matched-coverage Bahat and the earlier CLIP stress test.
    report=read(ROOT/'evidence/reports/matched_bahat_and_initial_clip.json')
    with np.load(ROOT/'evidence/bootstrap/matched_bahat_and_initial_clip.npz',allow_pickle=False) as z:
        for cohort, record in report['comparisons'].items():
            for method, contrast in record['differences'].items():
                check_interval(z[cohort+'__'+method],contrast['interval'],cohort+'/'+method)
                intervals_checked+=1
    ablation=read(ROOT/'evidence/reports/independent_ablation.json')
    assert ablation['decision']=='retain_full' and ablation['all_four_conditions_passed'] is False
    assert sum(value['passed'] for value in ablation['conditions'].values())==1
    print(json.dumps({'status':'passed','score_bundles':len(index['datasets']),'method_AURCs':methods_checked,
        'accepted_count_outcomes':points_checked,'stored_percentile_intervals':intervals_checked,
        'headlines':headline,'independent_replacement_conditions_passed':1,
        'scope':'Stored outputs only; no new resampling, inference, fitting, or image access.'},indent=2))


if __name__=='__main__':
    main()
