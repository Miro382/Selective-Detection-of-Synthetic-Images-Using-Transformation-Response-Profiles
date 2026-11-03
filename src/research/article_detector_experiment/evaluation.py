"""Samostatné učenie a hodnotenie selektora pre článkový experiment.

Modul nečíta ani nemení staré experimenty. Všetky funkcie pracujú s explicitne
odovzdanými záznamami; nižšie skóre vždy znamená skoršie prijatie predikcie.
Pilot overuje funkčnosť, nie nezávislú platnosť výskumnej hypotézy.
"""
from __future__ import annotations

import hashlib
import json
import warnings
from collections import Counter

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import expit
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

VERSION = "article-selector-logistic-v1"
TRANSFORMS = ("original", "jpeg_q90", "jpeg_q75", "jpeg_q60", "resize_0p75", "resize_0p5")
FEATURE_SETS = {
    "lr_full": (0, 1, 2, 3, 4, 5),
    "lr_confidence": (0,),
    "lr_resize_only": (0, 4, 5),
    "lr_jpeg_only": (0, 1, 2, 3),
}
COVERAGES = (.8, .9, .95)


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                   separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _records(records, expected_split=None):
    records = list(records)
    if not records:
        raise ValueError("Množina profilov je prázdna.")
    required = ("sample_id", "original_id", "group", "dataset", "split")
    for row in records:
        if any(not isinstance(row.get(key), str) or not row[key] for key in required):
            raise ValueError("Záznamu chýba neprázdna identita, skupina, dataset alebo oddelenie.")
        if row.get("label") not in (0, 1) or isinstance(row.get("label"), bool):
            raise ValueError("Referenčná trieda musí byť 0 alebo 1.")
        if expected_split is not None and row["split"] != expected_split:
            raise ValueError("Očakávané oddelenie: " + expected_split)
    ids = [row["sample_id"] for row in records]
    if len(set(ids)) != len(ids):
        raise ValueError("Opakované sample_id v množine profilov.")
    identities = {}
    for row in records:
        old = identities.setdefault(row["original_id"], (row["group"], row["label"]))
        if old != (row["group"], row["label"]):
            raise ValueError("Varianty originálu majú odlišnú skupinu alebo referenčnú triedu.")
    try:
        margins = np.asarray([row["margins"] for row in records], dtype=np.float64)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Každý profil musí obsahovať šesť surových logitov.") from error
    if margins.shape != (len(records), 6) or not np.isfinite(margins).all():
        raise ValueError("Očakávané konečné surové logity tvaru N × 6.")
    labels = np.asarray([row["label"] for row in records], dtype=np.int64)
    return records, margins, labels


def _assert_disjoint(train, calibration):
    # Skupina zachováva aj známe závislosti viacerých rôznych originálov.
    for key in ("sample_id", "original_id", "group", "sha256", "rgb_sha256"):
        left = {row[key] for row in train if row.get(key)}
        right = {row[key] for row in calibration if row.get(key)}
        if left & right:
            raise ValueError("Prienik tréningu a kalibrácie podľa " + key)


def group_weights(groups):
    """Každá skupina má súčet váh jedna; počet variantov nemení jej váhu."""
    counts = Counter(groups)
    return np.asarray([1. / counts[group] for group in groups], dtype=np.float64)


def profile_features(margins, temperature=1., columns=(0, 1, 2, 3, 4, 5)):
    """V4 príznaky; ablačné súhrny používajú len ponechané transformácie."""
    raw = np.asarray(margins, dtype=np.float64)
    if raw.ndim != 2 or raw.shape[1] != 6 or not np.isfinite(raw).all():
        raise ValueError("Očakávané konečné surové logity tvaru N × 6.")
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("Teplota musí byť kladná a konečná.")
    columns = tuple(columns)
    if not columns or columns[0] != 0 or len(set(columns)) != len(columns) or any(c not in range(6) for c in columns):
        raise ValueError("Výber príznakov musí začínať originálom bez opakovania stĺpcov.")
    m = raw[:, columns] / temperature
    if columns == (0,):
        return -np.abs(m)
    sign = np.where(m[:, :1] >= 0, 1., -1.)
    p = expit(m)
    return np.concatenate((
        -np.abs(m[:, :1]), sign * (m[:, 1:] - m[:, :1]),
        np.var(p, axis=1)[:, None],
        np.max(np.abs(p[:, 1:] - p[:, :1]), axis=1)[:, None],
        np.mean((m[:, 1:] >= 0) != (m[:, :1] >= 0), axis=1)[:, None],
    ), axis=1)


def fit_temperature(calibration_records):
    """BCE sa optimalizuje iba na origináloch detector_calibration."""
    records, margins, labels = _records(calibration_records, "detector_calibration")
    if len(np.unique(labels)) != 2:
        raise ValueError("Kalibrácia teploty potrebuje obe referenčné triedy.")
    if len({r["group"] for r in records}) != len(records):
        raise ValueError("Kalibrácia teploty vyžaduje jeden reprezentant na skupinu.")
    original = margins[:, 0]

    def nll(log_temperature):
        z = original / np.exp(log_temperature)
        return float(np.mean(np.logaddexp(0., z) - labels * z))

    bounds = (float(np.log(.01)), float(np.log(100.)))
    result = minimize_scalar(nll, bounds=bounds, method="bounded", options={"xatol": 1e-10})
    if not result.success or not np.isfinite(result.fun):
        raise ValueError("Optimalizácia teploty zlyhala.")
    # Vždy zahrnúť T=1 a hranice; ohraničené minimum inak môže byť horšie než T=1.
    candidates = [0., bounds[0], bounds[1], float(result.x)]
    chosen = min(candidates, key=nll)
    return {"temperature": float(np.exp(chosen)), "nll_before": nll(0.),
            "nll_after": nll(chosen), "bounds": [.01, 100.],
            "at_bound": chosen in bounds, "count": len(records),
            "split": "detector_calibration", "scope": "Kladná teplota nemení pôvodnú predikciu."}


def fit_models(train_records, calibration_records):
    train, margins, labels = _records(train_records, "selector_train")
    calibration, _, _ = _records(calibration_records, "detector_calibration")
    _assert_disjoint(train, calibration)
    errors = ((margins[:, 0] >= 0).astype(int) != labels).astype(int)
    if len(np.unique(errors)) != 2:
        raise ValueError("Učenie selektora potrebuje správne aj chybné rozhodnutia detektora; pilot nestačí.")
    temperature_info = fit_temperature(calibration)
    weights = group_weights([r["group"] for r in train])
    models = {}
    for name, columns in FEATURE_SETS.items():
        x = profile_features(margins, temperature_info["temperature"], columns)
        scaler = StandardScaler().fit(x, sample_weight=weights)
        estimator = LogisticRegression(C=.1, solver="lbfgs", max_iter=5000, tol=1e-9)
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            estimator.fit(scaler.transform(x), errors, sample_weight=weights)
        models[name] = {"columns": list(columns), "mean": scaler.mean_.tolist(),
                        "scale": scaler.scale_.tolist(), "coef": estimator.coef_[0].tolist(),
                        "intercept": float(estimator.intercept_[0]), "C": .1,
                        "iterations": int(estimator.n_iter_.max())}
    return {"version": VERSION, "transforms": list(TRANSFORMS),
            "temperature": temperature_info["temperature"], "calibration": temperature_info,
            "models": models, "training": {"profiles": len(train),
                "groups": len({r["group"] for r in train}), "errors": int(errors.sum()),
                "weight_sum": float(weights.sum()), "weighted_errors": float(weights @ errors),
                "split": "selector_train", "records_sha256": _digest(train)},
            "calibration_records_sha256": _digest(calibration),
            "policy": "Samostatný selektor na detektor; C=0,1; skupinové váhy; škálovanie iba z tréningu. Bez výberu pracovného prahu."}


def score_records(records, artifact=None):
    """Bez artefaktu vráti dve nenaučené referencie s T=1 pre krátky pilot."""
    _, margins, _ = _records(records)
    if artifact is not None and artifact.get("version") != VERSION:
        raise ValueError("Nepodporovaná verzia artefaktu selektora.")
    temperature = 1. if artifact is None else artifact["temperature"]
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("Teplota musí byť kladná a konečná.")
    m = margins / temperature
    sign = np.where(m[:, :1] >= 0, 1., -1.)
    result = {"confidence": -np.abs(m[:, 0]), "tta_mean": expit(-sign * m).mean(axis=1)}
    if artifact is not None:
        for name, model in artifact["models"].items():
            if name not in FEATURE_SETS or tuple(model["columns"]) != FEATURE_SETS[name]:
                raise ValueError("Neplatná reprezentácia modelu selektora.")
            x = profile_features(margins, temperature, model["columns"])
            scale = np.asarray(model["scale"], dtype=np.float64)
            if np.any(scale <= 0) or not np.isfinite(scale).all():
                raise ValueError("Neplatná mierka štandardizácie.")
            result[name] = expit(((x - model["mean"]) / scale) @ np.asarray(model["coef"]) + model["intercept"])
    if any(not np.isfinite(values).all() for values in result.values()):
        raise ValueError("Skóre obsahuje nekonečné hodnoty.")
    return result


def _metric_inputs(errors, scores, weights=None):
    errors, scores = np.asarray(errors), np.asarray(scores, dtype=np.float64)
    weights = np.ones(len(errors), dtype=np.int64) if weights is None else np.asarray(weights)
    if errors.ndim != 1 or len(errors) == 0 or scores.shape != errors.shape or weights.shape != errors.shape:
        raise ValueError("Neplatný tvar vstupov metriky.")
    if not np.isin(errors, (0, 1)).all() or not np.isfinite(scores).all():
        raise ValueError("Metriky vyžadujú binárne chyby a konečné skóre.")
    if not np.isfinite(weights).all() or (weights < 0).any() or weights.sum() <= 0:
        raise ValueError("Neplatné váhy metriky.")
    return errors, scores, weights


def risk_coverage_curve(errors, scores, weights=None):
    errors, scores, weights = _metric_inputs(errors, scores, weights)
    order = np.argsort(scores, kind="stable")
    s = scores[order]
    ends = np.r_[np.flatnonzero(s[1:] != s[:-1]), len(s)-1]
    counts = np.cumsum(weights[order])[ends]
    mistakes = np.cumsum(weights[order] * errors[order])[ends]
    keep = counts > 0
    counts, mistakes = counts[keep], mistakes[keep]
    coverage, risk = counts / counts[-1], mistakes / counts
    return {"aurc": float(np.diff(np.r_[0., coverage]) @ risk),
            "coverage": coverage.tolist(), "risk": risk.tolist(),
            "count": counts.tolist(), "errors": mistakes.tolist(), "score": s[ends][keep].tolist()}


def order_with_ties(scores, ids):
    tie = np.asarray([hashlib.sha256(("article-detector-tie-v1/" + str(x)).encode()).hexdigest() for x in ids])
    return np.lexsort((tie, scores))


def risk_at_coverage(errors, scores, ids, coverage, weights=None):
    errors, scores, weights = _metric_inputs(errors, scores, weights)
    if not 0 < coverage <= 1 or len(ids) != len(errors) or not np.equal(weights, np.floor(weights)).all():
        raise ValueError("Pokrytie musí byť v (0,1] a násobnosti musia byť celé.")
    order = order_with_ties(scores, ids)
    cw, ce = np.cumsum(weights[order]), np.cumsum(weights[order] * errors[order])
    target = int(np.floor(cw[-1] * coverage))
    if target == 0:
        return {"accepted": 0, "errors": 0, "risk": None, "coverage": 0.}
    j = int(np.searchsorted(cw, target))
    before_n, before_e = (cw[j-1], ce[j-1]) if j else (0, 0)
    mistakes = int(before_e + (target-before_n) * errors[order[j]])
    return {"accepted": target, "errors": mistakes, "risk": mistakes / target, "coverage": float(target / cw[-1])}


def _bootstrap_statistics(ordered_errors, ordered_weights, ends):
    """Už zoradené hodnoty; každá replika potrebuje iba kumulatívne súčty."""
    cw = np.cumsum(ordered_weights)
    ce = np.cumsum(ordered_weights * ordered_errors)
    block_counts = cw[ends]
    keep = block_counts > 0
    coverages = block_counts[keep] / cw[-1]
    risks = ce[ends][keep] / block_counts[keep]
    result = {"aurc": float(np.diff(np.r_[0., coverages]) @ risks)}
    for q in COVERAGES:
        target = int(np.floor(cw[-1] * q))
        if target == 0:
            result[str(q)] = None
            continue
        j = int(np.searchsorted(cw, target))
        before_n, before_e = (cw[j-1], ce[j-1]) if j else (0, 0)
        result[str(q)] = float((before_e + (target-before_n) * ordered_errors[j]) / target)
    return result


def evaluate_records(records, scores, bootstrap=1000, seed=20261027):
    """Párové rozdiely oproti istote; bootstrap zachováva celé pôvodové skupiny.

    Výstup opisuje pevné modely a zadanú vzorku. Nezahŕňa neistotu opakovaného
    učenia ani korekciu viacnásobných porovnaní; pilot nie je potvrdzujúci test.
    """
    rows, margins, labels = _records(records)
    if len({r["split"] for r in rows}) != 1 or len({r["dataset"] for r in rows}) != 1:
        raise ValueError("Vyhodnocujte osobitne každé oddelenie a každý dataset.")
    if "confidence" not in scores or not isinstance(bootstrap, int) or bootstrap < 0:
        raise ValueError("Chýba referencia confidence alebo je neplatný počet bootstrapov.")
    errors = ((margins[:, 0] >= 0).astype(int) != labels).astype(int)
    ids = [r["sample_id"] for r in rows]
    score_map = {name: _metric_inputs(errors, value)[1] for name, value in scores.items()}
    methods = {}
    for name, values in score_map.items():
        curve = risk_coverage_curve(errors, values)
        methods[name] = {"aurc": curve["aurc"], "curve": curve,
                         "risk_at_coverage": {str(q): risk_at_coverage(errors, values, ids, q) for q in COVERAGES}}
    reference = methods["confidence"]
    differences = {name: {"aurc": data["aurc"] - reference["aurc"],
        "risk_at_coverage": {str(q): (None if data["risk_at_coverage"][str(q)]["risk"] is None else
            data["risk_at_coverage"][str(q)]["risk"] - reference["risk_at_coverage"][str(q)]["risk"]) for q in COVERAGES}}
        for name, data in methods.items() if name != "confidence"}
    groups, inverse = np.unique([r["group"] for r in rows], return_inverse=True)
    intervals = {}
    if bootstrap and len(groups) >= 2:
        draws = {name: {"aurc": [], **{str(q): [] for q in COVERAGES}} for name in differences}
        prepared = {}
        for name, score in score_map.items():
            order = order_with_ties(score, ids)
            sorted_score = score[order]
            ends = np.r_[np.flatnonzero(sorted_score[1:] != sorted_score[:-1]), len(sorted_score)-1]
            prepared[name] = (order, errors[order], ends)
        rng = np.random.default_rng(seed)
        for _ in range(bootstrap):
            multiplicity = np.bincount(rng.integers(0, len(groups), len(groups)), minlength=len(groups))[inverse]
            values = {name: _bootstrap_statistics(ordered_errors, multiplicity[order], ends)
                      for name, (order, ordered_errors, ends) in prepared.items()}
            for name in differences:
                for metric in draws[name]:
                    a, b = values[name][metric], values["confidence"][metric]
                    if a is not None and b is not None:
                        draws[name][metric].append(a-b)
        for name, metrics in draws.items():
            intervals[name] = {metric: {"lower_95": float(np.quantile(v, .025)),
                              "upper_95": float(np.quantile(v, .975)), "valid_repetitions": len(v)}
                               for metric, v in metrics.items() if v}
    class_metrics = {str(label): {"count": int((labels == label).sum()),
        "accuracy": float(1-errors[labels == label].mean()) if (labels == label).any() else None} for label in (0, 1)}
    both_labels = len(np.unique(labels)) == 2
    return {"version": VERSION, "dataset": rows[0]["dataset"], "split": rows[0]["split"],
            "profiles": len(rows), "originals": len({r["original_id"] for r in rows}), "groups": len(groups),
            "detector": {"errors": int(errors.sum()), "accuracy": float(1-errors.mean()),
                "class_metrics": class_metrics,
                "auroc": float(roc_auc_score(labels, margins[:, 0])) if both_labels else None,
                "average_precision": float(average_precision_score(labels, margins[:, 0])) if both_labels else None},
            "methods": methods, "differences_vs_confidence": differences,
            "paired_group_bootstrap": {"requested_repetitions": bootstrap, "seed": seed,
                "status": "ok" if intervals else ("disabled" if not bootstrap else "insufficient_groups"),
                "intervals": intervals, "multiple_comparison_correction": "none"},
            "policy": "Bloková AURC; pevné pokrytia s hashovým riešením zhôd; párový bootstrap celých skupín. Žiadny prah sa nenastavuje na hodnotených štítkoch."}
