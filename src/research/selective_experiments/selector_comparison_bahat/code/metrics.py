import hashlib
import numpy as np


def curve(y, scores, weights=None):
    y, scores = np.asarray(y), np.asarray(scores)
    w = np.ones(len(y)) if weights is None else np.asarray(weights)
    order = np.argsort(scores, kind="stable")
    s, e, w = scores[order], y[order], w[order]
    end = np.r_[np.flatnonzero(s[1:] != s[:-1]), len(s)-1]
    count = np.cumsum(w)[end]
    errors = np.cumsum(w * e)[end]
    keep = count > 0
    count, errors, end = count[keep], errors[keep], end[keep]
    coverage = count / count[-1]
    risk = errors / count
    area = float(np.sum(np.diff(np.r_[0., coverage]) * risk))
    return dict(coverage=coverage, risk=risk, score=s[end], count=count, errors=errors, aurc=area)


def threshold(y, scores):
    c = curve(y, scores)
    valid = (c["risk"] <= .01) & (c["coverage"] >= .1) & (c["count"] >= 30)
    if not valid.any():
        return None
    return float(c["score"][np.flatnonzero(valid)[-1]])


def order_with_ties(scores, ids):
    tie = np.array([hashlib.sha256(("comparison-tie-v1/" + str(s)).encode()).hexdigest() for s in ids])
    return np.lexsort((tie, scores))


def risk_at(y, scores, ids, count):
    count = int(count)
    if not count:
        return dict(accepted=0, errors=0, risk=None, coverage=0.)
    chosen = order_with_ties(scores, ids)[:count]
    errors = int(np.asarray(y)[chosen].sum())
    return dict(accepted=count, errors=errors, risk=errors/count, coverage=count/len(y))


def summarize(y, scores, ids, tau, match_count):
    c = curve(y, scores)
    accepted = np.zeros(len(y), bool) if tau is None else scores <= tau
    n, e = int(accepted.sum()), int(np.asarray(y)[accepted].sum())
    return {"aurc": c["aurc"], "risk_at_coverage": {
                str(q): risk_at(y, scores, ids, int(np.floor(len(y)*q))) for q in (.8, .9, .95)},
            "matched_frozen_coverage": risk_at(y, scores, ids, match_count),
            "fixed_threshold": {"threshold": tau, "accepted": n, "errors": e,
                                "coverage": n/len(y), "risk": e/n if n else None}}


def paired_bootstrap(y, score_map, ids, groups, comparisons, repetitions=1000, seed=31019):
    """Skupiny sa žrebujú celé a tie isté počty sa použijú pre všetky metódy."""
    unique, inv = np.unique(groups, return_inverse=True)
    rng = np.random.default_rng(seed)
    orders = {k: order_with_ties(v, ids) for k, v in score_map.items()}
    # Opakované triedenie sa vynechá, zhodné skóre ostáva jedným blokom.
    ends = {k: np.r_[np.flatnonzero(v[o][1:] != v[o][:-1]), len(v)-1]
            for k, (v, o) in ((k, (score_map[k], orders[k])) for k in score_map)}
    draws = {a + "__minus__" + b: {"aurc": [], "risk90": []} for a, b in comparisons}
    y = np.asarray(y)
    for _ in range(repetitions):
        gw = np.bincount(rng.integers(0, len(unique), len(unique)), minlength=len(unique))
        w = gw[inv]
        values = {}
        for k, order in orders.items():
            cw = np.cumsum(w[order]); ce = np.cumsum(w[order]*y[order]); en = ends[k]
            keep = cw[en] > 0
            c = cw[en][keep] / cw[-1]; r = ce[en][keep] / cw[en][keep]
            aurc = float(np.sum(np.diff(np.r_[0., c]) * r))
            # Pevné pokrytie s deterministickým rozriešením zhôd. Posledný
            # opakovaný obrázok možno započítať iba s príslušnou násobnosťou.
            target = max(1, int(np.floor(cw[-1] * .9)))
            j = int(np.searchsorted(cw, target))
            before = ce[j-1] if j else 0
            previous_count = cw[j-1] if j else 0
            risk = float((before + (target-previous_count)*y[order[j]]) / target)
            values[k] = (aurc, risk)
        for a, b in comparisons:
            entry = draws[a + "__minus__" + b]
            entry["aurc"].append(values[a][0]-values[b][0])
            entry["risk90"].append(values[a][1]-values[b][1])
    return {key: {metric: {"lower_95": float(np.quantile(v, .025)),
                           "upper_95": float(np.quantile(v, .975)), "repetitions": repetitions}
                  for metric, v in fields.items()} for key, fields in draws.items()}
