"""Skóre a párové vyhodnotenie nového retrospektívneho článkového porovnania.

Vyššie skóre vždy znamená väčšiu neistotu; prijímajú sa najnižšie hodnoty.
`evaluate(y, ...)` prijíma indikátor CHYBY pôvodného deterministického
detektora (0 = správne, 1 = nesprávne), nie referenčnú triedu real/fake.
Modul nečíta súbory, nemení modely a nevyberá hyperparametre.
"""
from __future__ import annotations

import hashlib
from collections.abc import Mapping

import numpy as np

VERSION = 'article-reference-scores-v1'
COVERAGES = (0.8, 0.9, 0.95)
TIE_NAMESPACE = 'article-reference-tie-v1/'
PRIMARY_BASELINES = ('mc_entropy_30', 'energy_t1')


def _finite_array(value, description):
    try:
        result = np.asarray(value, dtype=np.float64)
    except (ValueError, TypeError) as error:
        raise ValueError(description + ': očakávané číselné hodnoty.') from error
    if not np.isfinite(result).all():
        raise ValueError(description + ': hodnoty musia byť konečné.')
    return result


def _softmax(logits):
    maximum = np.max(logits, axis=-1, keepdims=True)
    # Rozdiel dvoch konečných extrémov môže pretekať iba smerom k -inf;
    # jeho exponenciála je správne nula. Maximalizujúci člen zostáva jedna.
    with np.errstate(over='ignore', under='ignore'):
        weights = np.exp(logits - maximum)
    return weights / np.sum(weights, axis=-1, keepdims=True)


def _entropy(probabilities):
    log_probabilities = np.zeros_like(probabilities)
    np.log(probabilities, out=log_probabilities, where=probabilities > 0)
    return -np.sum(probabilities * log_probabilities, axis=-1)


def new_scores(logits, mc_logits):
    """Natívne logits[N,2] a MC logits[N,30,2], vždy teplota T=1.

    Vracia JSON-serializovateľný slovník `scores` s mc_entropy_30,
    mc_entropy_6 a energy_t1; prefix 6 sa berie z tých istých 30 priechodov.
    `diagnostics` obsahuje iba samostatnú zmenu MC-priemernej triedy oproti
    pôvodnej triede (fake pri logit_fake >= logit_real). Predikciu nemení.
    Energy zachováva spoločnú úroveň pôvodných logitov; nesmie sa počítať
    z umelo centrovaného páru ani z rekonštruovaných pravdepodobností.
    """
    logits = _finite_array(logits, 'Natívne logity')
    mc_logits = _finite_array(mc_logits, 'MC logity')
    if logits.ndim != 2 or logits.shape[0] < 1 or logits.shape[1] != 2:
        raise ValueError('Natívne logity musia mať tvar N × 2 a N > 0.')
    if mc_logits.shape != (logits.shape[0], 30, 2):
        raise ValueError('MC logity musia mať tvar N × 30 × 2.')
    maximum = logits.max(axis=1)
    with np.errstate(over='ignore', under='ignore'):
        energy = -maximum - np.log(np.exp(logits - maximum[:, None]).sum(axis=1))
    probabilities = _softmax(mc_logits)
    deterministic_class = (logits[:, 1] >= logits[:, 0]).astype(np.int64)
    scores = {'energy_t1': energy.tolist()}
    diagnostics = {'deterministic_class': deterministic_class.tolist(),
                   'class_order': {'real': 0, 'fake': 1}, 'temperature': 1.0,
                   'prediction_changed': False, 'mc_mean_class': {}, 'class_flips': {},
                   'policy': 'Skóre hodnotia pôvodnú deterministickú predikciu; MC triedy sú iba diagnostika.'}
    for repetitions in (30, 6):
        mean_probability = probabilities[:, :repetitions, :].mean(axis=1)
        score = _entropy(mean_probability)
        if not np.isfinite(score).all():
            raise ValueError('MC entropia musí byť konečná.')
        scores['mc_entropy_' + str(repetitions)] = score.tolist()
        mean_class = (mean_probability[:, 1] >= mean_probability[:, 0]).astype(np.int64)
        flips = mean_class != deterministic_class
        diagnostics['mc_mean_class'][str(repetitions)] = mean_class.tolist()
        diagnostics['class_flips'][str(repetitions)] = {'count': int(flips.sum()), 'rate': float(flips.mean())}
    return {'scores': scores, 'diagnostics': diagnostics}


def _ids(values, length, name, unique=False):
    values = list(values)
    if len(values) != length or any(not isinstance(value, str) or not value for value in values):
        raise ValueError(name + ': očakávané neprázdne textové identity dĺžky N.')
    if unique and len(set(values)) != length:
        raise ValueError(name + ': opakovaná identita.')
    return values


def _prepared(errors, scores, ids):
    hashes = np.asarray([hashlib.sha256((TIE_NAMESPACE + identity).encode('utf-8')).hexdigest()
                         for identity in ids])
    order = np.lexsort((hashes, scores))
    sorted_scores = scores[order]
    ends = np.r_[np.flatnonzero(sorted_scores[1:] != sorted_scores[:-1]), len(scores) - 1]
    return order, errors[order], ends


def _statistics(prepared, weights):
    order, errors, ends = prepared
    weights = weights[order]
    cumulative_n = np.cumsum(weights, dtype=np.int64)
    cumulative_e = np.cumsum(weights * errors, dtype=np.int64)
    block_n, block_e = cumulative_n[ends], cumulative_e[ends]
    present = block_n > 0
    block_n, block_e = block_n[present], block_e[present]
    coverage = block_n / cumulative_n[-1]
    risk = block_e / block_n
    result = {'aurc': float(np.diff(np.r_[0.0, coverage]) @ risk), 'risk_at_coverage': {}}
    for fraction in COVERAGES:
        accepted = int(np.floor(cumulative_n[-1] * fraction))
        if accepted == 0:
            point = {'accepted': 0, 'errors': 0, 'risk': None, 'coverage': 0.0}
        else:
            index = int(np.searchsorted(cumulative_n, accepted, side='left'))
            before_n = int(cumulative_n[index - 1]) if index else 0
            before_e = int(cumulative_e[index - 1]) if index else 0
            mistakes = before_e + (accepted - before_n) * int(errors[index])
            point = {'accepted': accepted, 'errors': mistakes, 'risk': mistakes / accepted,
                     'coverage': accepted / int(cumulative_n[-1])}
        result['risk_at_coverage'][str(fraction)] = point
    return result


def _difference(profile, baseline):
    result = {'aurc': profile['aurc'] - baseline['aurc'], 'risk_at_coverage': {}}
    for fraction in COVERAGES:
        key = str(fraction)
        left, right = profile['risk_at_coverage'][key]['risk'], baseline['risk_at_coverage'][key]['risk']
        result['risk_at_coverage'][key] = None if left is None or right is None else left - right
    return result


def evaluate(y, score_map, ids, groups, repetitions=5000, seed=20261028, profile='frozen_v4'):
    """Bloková AURC a párový bootstrap celých skupín pre pevné modely.

    y je indikátor nesprávnej PÔVODNEJ deterministickej predikcie.
    Všetky metódy dostávajú ten istý y a tie isté bootstrapové násobnosti.
    Rozdiely sú vždy `profile - baseline`: záporné hodnoty zvýhodňujú profil.
    AURC rešpektuje celé bloky presne rovnakých skóre. Pevné pokrytie používa
    floor(N*q) a pri zhode skóre SHA256 sample_id bez znalosti triedy/chyby.
    Percentilové intervaly sú bodové 95 %, opisné a bez korekcie násobnosti.
    """
    errors = _finite_array(y, 'Indikátory chýb')
    if errors.ndim != 1 or len(errors) < 1 or not np.isin(errors, (0, 1)).all():
        raise ValueError('y musí byť neprázdny vektor indikátorov chýb 0/1, nie real/fake tried.')
    errors = errors.astype(np.int64)
    n = len(errors)
    ids = _ids(ids, n, 'sample_id', unique=True)
    groups = _ids(groups, n, 'Skupiny')
    if not isinstance(score_map, Mapping) or len(score_map) < 2 or profile not in score_map:
        raise ValueError('Chýba profil alebo aspoň jedna referenčná metóda.')
    if type(repetitions) is not int or repetitions < 0 or type(seed) is not int or seed < 0:
        raise ValueError('Počet bootstrapov a seed musia byť nezáporné celé čísla.')
    prepared = {}
    for name, value in score_map.items():
        if not isinstance(name, str) or not name:
            raise ValueError('Metóda musí mať neprázdny textový názov.')
        values = _finite_array(value, 'Skóre metódy ' + name)
        if values.shape != errors.shape:
            raise ValueError('Skóre metódy ' + name + ' musí mať tvar N.')
        prepared[name] = _prepared(errors, values, ids)
    weights = np.ones(n, dtype=np.int64)
    methods = {name: _statistics(value, weights) for name, value in prepared.items()}
    differences = {name: {**_difference(methods[profile], result),
                           'role': 'primary_descriptive' if name in PRIMARY_BASELINES else 'secondary',
                           'intervals': {}}
                   for name, result in methods.items() if name != profile}
    unique_groups, inverse = np.unique(groups, return_inverse=True)
    draws = {name: {'aurc': [], **{str(q): [] for q in COVERAGES}} for name in differences}
    if repetitions and len(unique_groups) >= 2:
        random = np.random.default_rng(seed)
        for _ in range(repetitions):
            selected = random.integers(0, len(unique_groups), len(unique_groups))
            weights = np.bincount(selected, minlength=len(unique_groups))[inverse]
            values = {name: _statistics(value, weights) for name, value in prepared.items()}
            for name in differences:
                delta = _difference(values[profile], values[name])
                draws[name]['aurc'].append(delta['aurc'])
                for fraction, difference in delta['risk_at_coverage'].items():
                    if difference is not None:
                        draws[name][fraction].append(difference)
        for name, metrics in draws.items():
            differences[name]['intervals'] = {
                metric: {'lower_95': float(np.quantile(values, .025)),
                         'upper_95': float(np.quantile(values, .975)),
                         'valid_repetitions': len(values)}
                for metric, values in metrics.items() if values}
    status = 'disabled' if not repetitions else 'insufficient_groups' if len(unique_groups) < 2 else 'ok'
    return {
        'version': VERSION, 'profiles': n, 'groups': len(unique_groups),
        'detector': {'errors': int(errors.sum()), 'accuracy': float(1.0 - errors.mean())},
        'methods': methods, 'paired_differences': differences,
        'comparisons': {'profile': profile, 'direction': 'profile-minus-baseline',
                        'primary_descriptive': [name for name in PRIMARY_BASELINES if name in differences],
                        'secondary': [name for name in differences if name not in PRIMARY_BASELINES]},
        'paired_group_bootstrap': {'requested_repetitions': repetitions, 'seed': seed, 'status': status,
                                   'unit': 'Celé spojené pôvodové a výberové skupiny.',
                                   'pairing': 'Rovnaký výber skupín pre všetky metódy v každej replike.',
                                   'interval': 'Percentilový bodový 95 % interval.',
                                   'multiple_comparison_correction': 'none',
                                   'scope': 'Pevné modely; nezahŕňa opakované učenie ani opakované generovanie MC masiek.'},
        'policy': {'interpretation': 'Retrospektívne opisné porovnanie; nie nové nezávislé potvrdenie hypotézy.',
                   'prediction': 'Rovnaké chyby pôvodnej deterministickej predikcie pre všetky metódy.',
                   'aurc_ties': 'Celé bloky presne rovnakých skóre.',
                   'coverage_ties': 'SHA256(' + TIE_NAMESPACE + 'sample_id), bez referenčnej triedy alebo chyby.',
                   'lower_score_accepted_first': True, 'confirmatory_claim': False},
    }
