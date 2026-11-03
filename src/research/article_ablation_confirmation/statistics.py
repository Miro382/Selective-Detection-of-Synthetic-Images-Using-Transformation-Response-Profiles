"""Predurčené párové vyhodnotenie zmrazeného resize selektora proti full.

Bez I/O, učenia, inferencie a automatickej zmeny modelu. y/errors znamená
chybu pôvodnej deterministickej predikcie, nie referenčnú triedu real/fake.
"""
from __future__ import annotations

import hashlib

import numpy as np

VERSION = 'article-ablation-confirmation-statistics-v1'
REPETITIONS = 5000
SEED = 20261028
COHORTS = ('DeepShield', 'OpenFake')
MATCHED_FRACTIONS = {'DeepShield': (9807, 10000), 'OpenFake': (3639, 4000)}
SECONDARY_FRACTIONS = {'0.8': (4, 5), '0.9': (9, 10), '0.95': (19, 20)}
MIN_DISCORDANT_GROUPS = 10
TIE_NAMESPACE = 'article-ablation-confirmation-tie-v1/'


def _array(value, name):
    try:
        values = np.asarray(value, dtype=np.float64)
    except (ValueError, TypeError) as error:
        raise ValueError(name + ': očakávané číselné hodnoty.') from error
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError(name + ': očakávaný konečný jednorozmerný vektor.')
    return values


def _strings(values, count, name):
    values = list(values)
    if len(values) != count or any(not isinstance(v, str) or not v for v in values):
        raise ValueError(name + ': očakávané neprázdne textové identity dĺžky N.')
    return values


def _prepare(errors, scores, ids):
    ties = np.asarray([hashlib.sha256((TIE_NAMESPACE + value).encode('utf-8')).hexdigest() for value in ids])
    order = np.lexsort((ties, scores))
    sorted_scores = scores[order]
    ends = np.r_[np.flatnonzero(sorted_scores[1:] != sorted_scores[:-1]), len(scores) - 1]
    return order, errors[order], ends


def _metrics(prepared, multiplicity, fractions):
    order, errors, ends = prepared
    weights = multiplicity[order]
    counts = np.cumsum(weights, dtype=np.int64)
    mistakes = np.cumsum(weights * errors, dtype=np.int64)
    total = int(counts[-1])
    block_counts, block_mistakes = counts[ends], mistakes[ends]
    present = block_counts > 0
    block_counts, block_mistakes = block_counts[present], block_mistakes[present]
    coverage = block_counts / total
    result = {'aurc': float(np.diff(np.r_[0., coverage]) @ (block_mistakes / block_counts)), 'risks': {}}
    for key, (numerator, denominator) in fractions.items():
        # Presné floor(q*N_replika), bez zaokrúhľovacej chyby desatinného q.
        accepted = numerator * total // denominator
        if not accepted:
            result['risks'][key] = {'accepted': 0, 'errors': 0, 'risk': None, 'coverage': 0.}
            continue
        at = int(np.searchsorted(counts, accepted, side='left'))
        previous_n = int(counts[at - 1]) if at else 0
        previous_e = int(mistakes[at - 1]) if at else 0
        selected_errors = previous_e + (accepted - previous_n) * int(errors[at])
        result['risks'][key] = {'accepted': accepted, 'errors': selected_errors,
                               'risk': selected_errors / accepted, 'coverage': accepted / total}
    return result


def _differences(full, resize):
    result = {'aurc': resize['aurc'] - full['aurc']}
    for key in full['risks']:
        left, right = resize['risks'][key]['risk'], full['risks'][key]['risk']
        result[key] = None if left is None or right is None else left - right
    return result


def _interval(values):
    if not values:
        return None
    return {'lower_95': float(np.quantile(values, .025)), 'upper_95': float(np.quantile(values, .975)),
            'valid_repetitions': len(values)}


def _variable(values):
    if not values or not np.isfinite(values).all():
        return False
    # Odmietnuť aj variabilitu na úrovni chyby aritmetiky, nie zaviesť
    # toleranciu zhoršenia: adopčná hranica intervalu zostáva striktne nula.
    scale = max(1., float(np.max(np.abs(values))))
    return bool(np.ptp(values) > 16 * np.finfo(np.float64).eps * scale)


def _cohort(errors, full_scores, resize_scores, ids, groups, name, repetitions, seed):
    n = len(errors)
    fractions = {'matched': MATCHED_FRACTIONS[name], **SECONDARY_FRACTIONS}
    prepared = {'full': _prepare(errors, full_scores, ids), 'resize': _prepare(errors, resize_scores, ids)}
    point = {key: _metrics(value, np.ones(n, dtype=np.int64), fractions) for key, value in prepared.items()}
    delta = _differences(point['full'], point['resize'])
    accepted = point['full']['risks']['matched']['accepted']
    masks = {}
    for model, (order, _, _) in prepared.items():
        mask = np.zeros(n, dtype=bool)
        mask[order[:accepted]] = True
        masks[model] = mask & (errors == 1)
    removed = masks['full'] & ~masks['resize']
    added = masks['resize'] & ~masks['full']
    discordance = removed | added
    discordant_groups = len({group for group, different in zip(groups, discordance) if different})
    unique, inverse = np.unique(groups, return_inverse=True)
    draws = {'aurc': [], **{key: [] for key in fractions}}
    if repetitions and len(unique) >= 2:
        rng = np.random.default_rng(seed)
        for _ in range(repetitions):
            selected = rng.integers(0, len(unique), len(unique))
            multiplicity = np.bincount(selected, minlength=len(unique))[inverse]
            values = {key: _metrics(value, multiplicity, fractions) for key, value in prepared.items()}
            difference = _differences(values['full'], values['resize'])
            for key, value in difference.items():
                if value is not None:
                    draws[key].append(value)
    intervals = {key: _interval(values) for key, values in draws.items()}
    reasons = []
    if len(unique) < 2:
        reasons.append('fewer_than_two_groups')
    if errors.sum() == 0:
        reasons.append('no_detector_errors')
    if errors.sum() == n:
        reasons.append('no_correct_predictions')
    if accepted == 0:
        reasons.append('no_accepted_observations')
    if discordant_groups < MIN_DISCORDANT_GROUPS:
        reasons.append('fewer_than_ten_matched_error_discordant_groups')
    if repetitions < REPETITIONS:
        reasons.append('fewer_than_5000_bootstrap_repetitions')
    for key in ('aurc', 'matched'):
        if len(draws[key]) != repetitions or not draws[key]:
            reasons.append('incomplete_bootstrap_' + key)
        elif not _variable(draws[key]):
            reasons.append('degenerate_bootstrap_' + key)
    return {'status': 'ok' if not reasons else 'inadequate', 'profiles': n, 'groups': len(unique),
            'detector_errors': int(errors.sum()), 'matched_fraction': fractions['matched'][0] / fractions['matched'][1],
            'matched_fraction_exact': list(fractions['matched']), 'models': point,
            'delta_resize_minus_full': delta, 'intervals': intervals,
            'matched_error_count_delta': int(added.sum()) - int(removed.sum()),
            'paired_error_discordance': {'removed_errors': int(removed.sum()), 'added_errors': int(added.sum()),
                                         'discordant_observations': int(discordance.sum()),
                                         'discordant_groups': discordant_groups,
                                         'definition': 'Rozdiel indikátorov prijatá-a-chybná predikcia pri rovnakom k.'},
            'adequacy': {'passed': not reasons, 'reasons': reasons,
                         'minimum_groups': 2, 'minimum_matched_error_discordant_groups': MIN_DISCORDANT_GROUPS,
                         'guard_scope': 'Konzervatívne predurčené stabilitné guardy; nezaručujú nominálne pokrytie bootstrapových intervalov.'}}


def _decision(cohorts):
    conditions = {}
    for name in COHORTS:
        cohort = cohorts[name]
        for metric in ('aurc', 'matched'):
            interval = cohort.get('intervals', {}).get(metric)
            estimate = cohort.get('delta_resize_minus_full', {}).get(metric)
            passed = bool(cohort['status'] == 'ok' and cohort['adequacy']['passed'] and interval is not None
                          and interval['valid_repetitions'] >= REPETITIONS
                          and estimate is not None and np.isfinite(estimate) and estimate < 0
                          and np.isfinite(interval['upper_95']) and interval['upper_95'] < 0)
            conditions[name + ':' + metric] = {'passed': passed, 'estimate': estimate,
                                               'upper_95': interval['upper_95'] if interval else None}
    eligible = all(condition['passed'] for condition in conditions.values())
    return {'decision': 'eligible_for_replacement' if eligible else 'retain_full', 'conditions': conditions,
            'all_four_conditions_passed': eligible, 'model_change_performed': False}


def evaluate(errors, full_scores, resize_scores, ids, groups, cohorts, repetitions=REPETITIONS, seed=SEED):
    """Samostatne DS/OF; Δ=resize−full. Vyžaduje všetky štyri horné CI<0.

    Chýbajúca kohorta je explicitne nevyhodnotiteľná a vždy ponechá full.
    Poškodené/vzájomne nezhodné vstupy vyvolajú ValueError. Menší počet
    replík je dovolený na syntetické/technické overenie, nikdy na adopciu.
    """
    errors, full_scores, resize_scores = (_array(value, name) for value, name in
        ((errors, 'Chyby'), (full_scores, 'Full skóre'), (resize_scores, 'Resize skóre')))
    n = len(errors)
    if len(full_scores) != n or len(resize_scores) != n or not np.isin(errors, (0, 1)).all():
        raise ValueError('Chyby 0/1 a obe skóre musia mať rovnakú dĺžku.')
    ids, groups, cohorts = (_strings(value, n, name) for value, name in
                           ((ids, 'sample_id'), (groups, 'Skupiny'), (cohorts, 'Kohorty')))
    if len(set(ids)) != n:
        raise ValueError('Opakované sample_id.')
    if set(cohorts) - set(COHORTS):
        raise ValueError('Povolené kohorty sú presne DeepShield a OpenFake.')
    if type(repetitions) is not int or repetitions < 0 or type(seed) is not int or seed < 0:
        raise ValueError('Počet bootstrapov a seed musia byť nezáporné celé čísla.')
    errors = errors.astype(np.int64)
    result = {}
    for name in COHORTS:
        indices = np.flatnonzero(np.asarray(cohorts) == name)
        if len(indices) == 0:
            result[name] = {'status': 'missing', 'profiles': 0, 'groups': 0, 'intervals': {},
                            'delta_resize_minus_full': {},
                            'adequacy': {'passed': False, 'reasons': ['missing_cohort']}}
            continue
        result[name] = _cohort(errors[indices], full_scores[indices], resize_scores[indices],
                               [ids[i] for i in indices], [groups[i] for i in indices], name, repetitions, seed)
    return {'version': VERSION, 'cohorts': result, **_decision(result),
            'bootstrap': {'requested_repetitions': repetitions, 'seed': seed,
                          'unit': 'Celé spojené pôvodové a výberové skupiny; párové pre oba modely.',
                          'interval': 'Dvojstranný bodový percentilový 95 % interval [2,5 %, 97,5 %].',
                          'quantiles': [.025, .975], 'simultaneous_intervals': False,
                          'scope': 'Pevné modely; neistota opakovaného učenia nie je zahrnutá.'},
            'policy': {'primary': ['DeepShield:aurc', 'DeepShield:matched', 'OpenFake:aurc', 'OpenFake:matched'],
                       'secondary_coverages': list(SECONDARY_FRACTIONS), 'direction': 'resize-minus-full',
                       'adoption_rule': 'Všetky štyri horné hranice musia byť striktne pod nulou a všetky guardy musia prejsť.',
                       'test': 'Spoločná intersection-union podmienka superiority; nie noninferiority.',
                       'noninferiority_margin': None, 'allowed_deterioration': 0.0,
                       'matched_count': 'floor(q*N) s pevným q; v každej replike floor(q*N_replika). Nie počet z prahu full.',
                       'ties': 'AURC celé bloky zhodných skóre; pevné pokrytie SHA256(' + TIE_NAMESPACE + 'sample_id).',
                       'decision_scope': 'Spôsobilosť na samostatné rozhodnutie o náhrade; žiadna automatická zmena modelu.'}}
