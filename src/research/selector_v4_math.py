"""Numerika selektora V3; JSON modely bez pickle a bez dotyku kalibracnych dat.

Mensie skore vzdy znamena mensie odhadovane riziko chyby. Vystup logistickej
funkcie oznacujeme ako error-score, nie ako overenu kalibrovanu pravdepodobnost.
Regularizacia ma sklearn konvenciu: sum(w * logloss) + ||coef||^2 / (2*C).
Pri fitovani sa nepouziva class_weight: explicitne sample weights uz vyjadruju
cielovu zmes realnych a syntetickych obrazkov.
"""
from __future__ import annotations

import warnings

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import beta
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

SCHEMA = "selector-v3-numerics-v1"
MAX_ITER = 5000


def _matrix(value, name):
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    if array.ndim != 2 or array.shape[1] == 0 or not np.isfinite(array).all():
        raise ValueError(f"{name} musi byt konecna matica s aspon jednym stlpcom.")
    return array


def _vector(value, name, n=None):
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 1 or not np.isfinite(array).all():
        raise ValueError(f"{name} musi byt konecny jednorozmerny vektor.")
    if n is not None and len(array) != n:
        raise ValueError(f"{name}: nespravny pocet riadkov.")
    return array


def _labels_weights(y, weights, n, *, two_classes=False):
    labels = _vector(y, "y", n)
    weight = _vector(weights, "weights", n)
    if n == 0 or not np.isin(labels, [0.0, 1.0]).all():
        raise ValueError("y musi obsahovat aspon jeden binarny stitok 0/1.")
    if (weight < 0).any() or not np.isfinite(weight.sum()) or weight.sum() <= 0:
        raise ValueError("Vahy musia byt nezaporne a ich sucet kladny a konecny.")
    if two_classes and len(np.unique(labels[weight > 0])) != 2:
        raise ValueError("Trening vyzaduje obe triedy s kladnou vahou.")
    return labels, weight


def _regularization(C):
    value = float(C)
    if not np.isfinite(value) or value <= 0:
        raise ValueError("C musi byt kladne konecne cislo.")
    return value


def confidence_error_logit(confidence, eps=1e-8):
    """Priznak baseline: logit(1-confidence), orezany pred logaritmovanim.

    Vracia vektor; fit_logistic/fit_offset akceptuju aj takyto jediny stlpec.
    Confidence ma byt istota povodnej predikovanej triedy, nie p(fake).
    """
    confidence = _vector(confidence, "confidence")
    eps = float(eps)
    if not np.isfinite(eps) or not 0 < eps < 0.5:
        raise ValueError("eps musi byt medzi 0 a 0.5.")
    if ((confidence < 0) | (confidence > 1)).any():
        raise ValueError("Istota musi byt v intervale [0, 1].")
    error = np.clip(1.0 - confidence, eps, 1.0 - eps)
    return np.log(error) - np.log1p(-error)


def _fit_scaler(X, weights):
    # Weighted floating point accumulation can produce a tiny negative variance
    # for a nonzero constant column. Fit exact constants as zeros, then restore
    # their exact mean and unit scale. This also makes transformed constants zero.
    constant = np.all(X == X[0], axis=0)
    safe_X = X.copy() if constant.any() else X
    if constant.any():
        safe_X[:, constant] = 0.0
    scaler = StandardScaler().fit(safe_X, sample_weight=weights)
    scaler.mean_[constant] = X[0, constant]
    scaler.scale_[constant] = 1.0
    if not np.isfinite(scaler.mean_).all() or not np.isfinite(scaler.scale_).all():
        raise ValueError("Nekonecne parametre skalovania; skontrolujte rozsah priznakov.")
    return {"mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist()}


def _transform(X, scaler):
    mean = _vector(scaler["mean"], "scaler.mean", X.shape[1])
    scale = _vector(scaler["scale"], "scaler.scale", X.shape[1])
    if (scale <= 0).any():
        raise ValueError("Skala kazdeho priznaku musi byt kladna.")
    transformed = (X - mean) / scale
    if not np.isfinite(transformed).all():
        raise ValueError("Skalovanie vytvorilo nekonecne hodnoty.")
    return transformed


def _linear_logit(payload, X):
    X = _matrix(X, "X")
    coef = _vector(payload["coef"], "coef", X.shape[1])
    intercept = float(payload["intercept"])
    if not np.isfinite(intercept):
        raise ValueError("Intercept musi byt konecny.")
    logits = _transform(X, payload["scaler"]) @ coef + intercept
    if not np.isfinite(logits).all():
        raise ValueError("Model vytvoril nekonecny logit.")
    return logits


def _check_payload(payload, model_type):
    if payload.get("schema") != SCHEMA or payload.get("model_type") != model_type:
        raise ValueError("Neznamy alebo nekompatibilny format modelu.")


def fit_logistic(X, y, weights, C):
    """Weighted StandardScaler + LR. Vahy dodava volajuci (real=5, fake=5/9).

    X, y a weights musia obsahovat iba treningove data aktualneho foldu.
    Vracia plne JSON-serializovatelny model s explicitnymi parametrami.
    """
    X = _matrix(X, "X")
    y, weights = _labels_weights(y, weights, len(X), two_classes=True)
    if (weights <= 0).any():
        raise ValueError("Treningove vahy musia byt striktne kladne.")
    C = _regularization(C)
    scaler = _fit_scaler(X, weights)
    clf = LogisticRegression(C=C, solver="lbfgs", max_iter=MAX_ITER, tol=1e-9)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        try:
            clf.fit(_transform(X, scaler), y, sample_weight=weights)
        except ConvergenceWarning as exc:
            raise RuntimeError("Logisticka regresia nekonvergovala.") from exc
    if clf.n_iter_.max() >= MAX_ITER:
        raise RuntimeError("Logisticka regresia dosiahla limit iteracii.")
    if not np.isfinite(clf.coef_).all() or not np.isfinite(clf.intercept_).all():
        raise RuntimeError("Nekonecne parametre logistickej regresie.")
    return {
        "schema": SCHEMA, "model_type": "weighted_logistic", "C": C,
        "score_semantics": "error_score_not_calibration_guarantee",
        "scaler": scaler, "coef": clf.coef_[0].tolist(),
        "intercept": float(clf.intercept_[0]),
        "convergence": {"success": True, "iterations": int(clf.n_iter_[0])},
    }


def predict_logistic(payload, X):
    """Vrati 1D error-score v [0, 1]; mensie je bezpecnejsie."""
    _check_payload(payload, "weighted_logistic")
    return expit(_linear_logit(payload, X))


def fit_offset(base_X, profile_X, y, weights, C):
    """Zmrazeny baseline + regularizovana korekcia jeho logitu bez interceptu.

    base_X je uz baseline priznak (napr. confidence_error_logit(confidence)),
    nie surova istota. Baseline aj scaler sa fituju iba na tychto riadkoch.
    Korekcia pouziva profil standardizovany treningovymi vahami. Jej nulovy
    koeficient presne obnovi baseline; volny korekcny intercept sa nepouziva.
    C riadi baseline aj korekciu rovnakou sum-weighted-loss konvenciou.
    """
    base_X = _matrix(base_X, "base_X")
    profile_X = _matrix(profile_X, "profile_X")
    if len(base_X) != len(profile_X):
        raise ValueError("base_X a profile_X musia mat rovnaky pocet riadkov.")
    y, weights = _labels_weights(y, weights, len(base_X), two_classes=True)
    if (weights <= 0).any():
        raise ValueError("Treningove vahy musia byt striktne kladne.")
    C = _regularization(C)
    baseline = fit_logistic(base_X, y, weights, C)
    base_logit = _linear_logit(baseline, base_X)
    scaler = _fit_scaler(profile_X, weights)
    profile = _transform(profile_X, scaler)
    mass = float(weights.sum())

    def objective(coef):
        logits = base_logit + profile @ coef
        loss = np.dot(weights, np.logaddexp(0.0, logits) - y * logits)
        loss = loss / mass + 0.5 * np.dot(coef, coef) / (C * mass)
        gradient = profile.T @ (weights * (expit(logits) - y)) / mass
        gradient += coef / (C * mass)
        return float(loss), gradient

    result = minimize(
        objective, np.zeros(profile.shape[1], dtype=np.float64), jac=True,
        method="L-BFGS-B",
        options={"maxiter": MAX_ITER, "gtol": 1e-9, "ftol": 1e-12, "maxls": 50},
    )
    if not result.success or not np.isfinite(result.x).all() or not np.isfinite(result.fun):
        raise RuntimeError(f"Offset regresia nekonvergovala: {result.message}")
    return {
        "schema": SCHEMA, "model_type": "confidence_offset", "C": C,
        "score_semantics": "error_score_not_calibration_guarantee",
        "baseline": baseline, "scaler": scaler,
        "coef": result.x.tolist(), "intercept": 0.0,
        "convergence": {
            "success": True, "iterations": int(result.nit),
            "objective": float(result.fun), "message": str(result.message),
        },
    }


def predict_offset(payload, base_X, profile_X):
    _check_payload(payload, "confidence_offset")
    base_X = _matrix(base_X, "base_X")
    profile_X = _matrix(profile_X, "profile_X")
    if len(base_X) != len(profile_X):
        raise ValueError("base_X a profile_X musia mat rovnaky pocet riadkov.")
    _check_payload(payload["baseline"], "weighted_logistic")
    logits = _linear_logit(payload["baseline"], base_X) + _linear_logit(payload, profile_X)
    if not np.isfinite(logits).all():
        raise ValueError("Offset model vytvoril nekonecny logit.")
    return expit(logits)


def _risk_curve(y, scores, weights):
    scores = _vector(scores, "scores")
    y, weights = _labels_weights(y, weights, len(scores))
    # Zero-weight samples must not create boundaries or misleading counts.
    positive = weights > 0
    order = np.argsort(scores[positive], kind="stable")
    sorted_scores = scores[positive][order]
    sorted_weights = weights[positive][order]
    sorted_y = y[positive][order]
    ends = np.r_[np.flatnonzero(sorted_scores[1:] != sorted_scores[:-1]), len(order) - 1]
    accepted_mass = np.cumsum(sorted_weights)[ends]
    error_mass = np.cumsum(sorted_weights * sorted_y)[ends]
    coverage = accepted_mass / float(weights.sum())
    return {
        "thresholds": sorted_scores[ends], "coverage": coverage,
        "risk": error_mass / accepted_mass,
        "accepted_weight": accepted_mass, "error_weight": error_mass,
        "accepted_positive_weight_rows": ends + 1,
        "total_weight": float(weights.sum()),
    }


def weighted_aurc(y, scores, w):
    """Plocha pod empirickou krivkou risk-coverage, cele bloky rovnakych skore.

    Konvencia: sum_j (coverage_j - coverage_{j-1}) * risk_j (pravy krok).
    Ties sa nikdy neradia podla y ani poradia vstupu. Pri rovnakych skore je
    AURC rovne vazenej prevalencii chyby. Ide o empiricku metriku, nie garanciu.
    """
    curve = _risk_curve(y, scores, w)
    return float(np.dot(np.diff(np.r_[0.0, curve["coverage"]]), curve["risk"]))


def choose_threshold(y, scores, w, target_risk, min_coverage=0.1):
    """Najvyssie empiricke pokrytie s risk <= target pri accept(score <= tau).

    Povolenymi bodmi su iba cele bloky ties. Nemonotonna empiricka krivka sa
    prehlada cela. Toto je treningovy vyber kandidata, nie risk certifikacia
    ani opravnenie optimalizovat na kalibracnych/testovacich stitkoch.
    Pri neuspechu je tau=-1, co pre error-score v [0,1] odmieta vsetko.
    """
    target_risk, min_coverage = float(target_risk), float(min_coverage)
    if not np.isfinite(target_risk) or not 0 <= target_risk <= 1:
        raise ValueError("target_risk musi byt v [0,1].")
    if not np.isfinite(min_coverage) or not 0 <= min_coverage <= 1:
        raise ValueError("min_coverage musi byt v [0,1].")
    scores = _vector(scores, "scores")
    if ((scores < 0) | (scores > 1)).any():
        raise ValueError("Vyber prahu ocakava error-score v [0,1].")
    curve = _risk_curve(y, scores, w)
    feasible = (curve["risk"] <= target_risk) & (curve["coverage"] >= min_coverage)
    indexes = np.flatnonzero(feasible)
    common = {"target_risk": target_risk, "min_coverage": min_coverage,
              "total_weight": curve["total_weight"],
              "criterion": "empirical_weighted_risk_not_certified", "accept_rule": "score <= tau"}
    if len(indexes) == 0:
        return {**common, "feasible": False, "tau": -1.0, "coverage": 0.0,
                "risk": None, "accepted_weight": 0.0, "error_weight": 0.0,
                "accepted_positive_weight_rows": 0}
    index = int(indexes[-1])
    return {**common, "feasible": True, "tau": float(curve["thresholds"][index]),
            "coverage": float(curve["coverage"][index]), "risk": float(curve["risk"][index]),
            "accepted_weight": float(curve["accepted_weight"][index]),
            "error_weight": float(curve["error_weight"][index]),
            "accepted_positive_weight_rows": int(curve["accepted_positive_weight_rows"][index])}


def clopper_pearson_upper(errors, accepted, delta):
    """Jednostranna exact binomicka horna hranica, confidence=1-delta.

    Pouzit iba pre nezavisle Bernoulli pozorovania a vopred zmrazeny prah.
    Nepouzivat vahove pocty ani viac korelovanych variantov jednej predlohy.
    Pri accepted=0 vrati 1 (ziadny dokaz nizkeho rizika), aj ked empiricky
    risk pri prazdnej akceptovanej mnozine nie je definovany.
    """
    if isinstance(errors, (bool, np.bool_)) or isinstance(accepted, (bool, np.bool_)):
        raise ValueError("Pocty musia byt cele cisla, nie bool.")
    try:
        k, n = int(errors), int(accepted)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError("Pocty musia byt konecne cele cisla.") from exc
    if k != errors or n != accepted or not 0 <= k <= n:
        raise ValueError("Vyžaduje sa 0 <= errors <= accepted, oba pocty cele.")
    delta = float(delta)
    if not np.isfinite(delta) or not 0 < delta < 1:
        raise ValueError("delta musi byt v (0,1).")
    if n == 0 or k == n:
        return 1.0
    if k == 0:
        return float(-np.expm1(np.log(delta) / n))
    upper = float(beta.isf(delta, k + 1, n - k))
    if not np.isfinite(upper):
        raise RuntimeError("Nepodarilo sa vypocitat binomicku hranicu.")
    return upper
