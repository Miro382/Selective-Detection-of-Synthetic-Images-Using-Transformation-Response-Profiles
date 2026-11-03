"""Selektor V4: presné logity, rovnaká informácia pre všetky referencie.

Rodiny používajú iba odpovede detektora. Skutočná trieda, zdroj a generátor
nikdy nie sú vstupnými príznakmi. Nižšie skóre znamená menšie odhadované riziko.
"""
from __future__ import annotations

import math
from collections import Counter
import numpy as np
from scipy.special import expit
from selector_v4_math import fit_logistic, predict_logistic

VERSION = "selective-selector-v4-model-1"
PROFILE_CONFIG = {"version": "degradation-profile-v1", "jpeg_qualities": [90, 75, 60], "resize_scales": [.75, .5]}
VARIANTS = ("jpeg_q90", "jpeg_q75", "jpeg_q60", "resize_0p75", "resize_0p5")
BASE_NAMES = ["negative_absolute_calibrated_margin"]
PROFILE_NAMES = ["oriented_margin_delta_" + name for name in VARIANTS] + ["prediction_variance", "max_abs_shift", "flip_fraction"]
BASELINES = ("confidence", "tta_mean", "max_shift")
LEARNED = ("confidence_class", "symmetric_profile", "class_profile")


def feature_names(family):
    if family in BASELINES:
        return [family]
    if family == "confidence_class":
        return BASE_NAMES + ["predicted_fake", "negative_absolute_margin_x_predicted_fake"]
    if family == "symmetric_profile":
        return BASE_NAMES + PROFILE_NAMES
    if family == "class_profile":
        names = BASE_NAMES + PROFILE_NAMES
        return names + ["predicted_fake"] + [n + "_x_predicted_fake" for n in names]
    raise ValueError("Neznáma rodina V4: " + str(family))


def profile_arrays(records):
    if not records:
        raise ValueError("Chýbajú profily V4.")
    margins, probabilities = [], []
    for row in records:
        p = row["profile"]
        t = row["temperature"]
        if isinstance(t, bool) or not isinstance(t, (int, float)) or not math.isfinite(t) or t <= 0:
            raise ValueError("Neplatná teplota detektora.")
        if not isinstance(row["raw_margins"], list) or any(type(v) not in (float, int) for v in row["raw_margins"]):
            raise ValueError("Pôvodné logity musia byť číselný zoznam bez logických hodnôt.")
        m = np.asarray(row["raw_margins"], dtype=np.float64)
        if m.shape != (6,) or not np.isfinite(m).all():
            raise ValueError("Profil musí obsahovať šesť konečných pôvodných logitových rozdielov.")
        if p["config"] != PROFILE_CONFIG or p["decision_threshold"] != .5:
            raise ValueError("Nesprávna konfigurácia profilu V4.")
        if tuple(v["name"] for v in p["variants"]) != VARIANTS:
            raise ValueError("Nesprávne poradie degradácií.")
        ps = np.asarray([p["original_probability"], *[v["probability"] for v in p["variants"]]], dtype=np.float64)
        if not np.isfinite(ps).all() or not np.allclose(ps, expit(m / t), rtol=1e-10, atol=1e-12):
            raise ValueError("Pravdepodobnosti nezodpovedajú pôvodným logitom a teplote.")
        deltas = ps[1:] - ps[0]
        flips = (m[1:] >= 0) != (m[0] >= 0)
        derived = {"original_probability": ps[0], "original_confidence": max(ps[0], 1 - ps[0]),
                   "prediction_variance": ps.var(), "max_abs_shift": np.abs(deltas).max(), "flip_fraction": flips.mean()}
        derived.update({"delta_" + name: deltas[i] for i, name in enumerate(VARIANTS)})
        expected_names = set(derived) | {"log_pixel_count", "log_aspect_ratio", "log_laplacian_variance"}
        if set(p["features"]) != expected_names:
            raise ValueError("Nesprávna schéma profilových príznakov.")
        for name, value in p["features"].items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError("Neplatná hodnota profilového príznaku.")
            if name in derived and not math.isclose(value, derived[name], rel_tol=1e-9, abs_tol=1e-12):
                raise ValueError("Nezhodný odvodený príznak: " + name)
            if name not in derived and value < 0:
                raise ValueError("Neplatný príznak kvality.")
        for j, v in enumerate(p["variants"]):
            if not math.isclose(v["delta"], deltas[j], rel_tol=1e-9, abs_tol=1e-12) or type(v["flipped"]) is not bool or v["flipped"] != bool(flips[j]):
                raise ValueError("Nesúlad odchýlky alebo zmeny rozhodnutia.")
        margins.append(m / t)
        probabilities.append(ps)
    margins = np.asarray(margins)
    ps = np.asarray(probabilities)
    predicted = (margins[:, 0] >= 0).astype(float)
    sign = 2 * predicted - 1
    # Exact error-logit equals -|margin|. No probability clipping/reconstruction.
    base = -np.abs(margins[:, :1])
    delta = ps[:, 1:] - ps[:, :1]
    shifts = np.abs(delta).max(axis=1)
    flips = ((margins[:, 1:] >= 0) != (margins[:, :1] >= 0)).mean(axis=1)
    oriented = np.column_stack((sign[:, None] * (margins[:, 1:] - margins[:, :1]), ps.var(axis=1), shifts, flips))
    class_error_probs = expit(-sign[:, None] * margins)
    return {"base": base, "oriented": oriented, "predicted": predicted,
            "confidence": expit(base[:, 0]), "tta_mean": class_error_probs.mean(axis=1), "max_shift": shifts}


def matrix(arrays, family):
    base, predicted = arrays["base"], arrays["predicted"][:, None]
    if family == "confidence_class":
        return np.column_stack((base, predicted, base * predicted))
    symmetric = np.column_stack((base, arrays["oriented"]))
    if family == "symmetric_profile":
        return symmetric
    if family == "class_profile":
        return np.column_stack((symmetric, predicted, symmetric * predicted))
    raise ValueError("Rodina nemá učenú maticu príznakov.")


def group_weights(records):
    counts = Counter(str(row["group"]) for row in records)
    # Every independent group contributes exactly one total loss unit.
    return np.asarray([1.0 / counts[str(row["group"])] for row in records])


def candidate_grid():
    return ([{"id": f, "family": f, "C": None} for f in BASELINES] +
            [{"id": f + "_C" + str(c).replace(".", "p"), "family": f, "C": c} for f in LEARNED for c in (.01, .1)])


def fit_candidate(candidate, arrays, errors, weights):
    family = candidate["family"]
    model = {"version": VERSION, "family": family, "feature_names": feature_names(family), "C": candidate["C"], "parameters": None}
    if family in LEARNED:
        model["parameters"] = fit_logistic(matrix(arrays, family), errors, weights, C=candidate["C"])
    return model


def predict_candidate(model, arrays):
    if model.get("version") != VERSION or model.get("feature_names") != feature_names(model["family"]):
        raise ValueError("Neplatný formát selektora V4.")
    family = model["family"]
    scores = arrays[family] if family in BASELINES else predict_logistic(model["parameters"], matrix(arrays, family))
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 1 or not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("Neplatné skóre selektora.")
    return scores


def decide_record(artifact, record, *, research=False):
    if artifact.get("version") != "selective-selector-v4-artifact-1":
        raise ValueError("Neplatný artefakt V4.")
    if record["model_fingerprint"] != artifact["model_fingerprint"] or record["temperature"] != artifact["temperature"]:
        raise ValueError("Profil patrí inému detektoru alebo kalibrácii skóre.")
    if record["profile"]["config"] != artifact["profile_config"] or record["profile"]["decision_threshold"] != artifact["detector_threshold"]:
        raise ValueError("Profil má inú konfiguráciu.")
    tau = artifact["research_threshold" if research else "accept_threshold"]
    if isinstance(tau, bool) or not isinstance(tau, (float, int)) or not math.isfinite(tau) or not (tau == -1 or 0 <= tau <= 1):
        raise ValueError("Neplatný prah selektora.")
    if not research and tau != -1:
        c = artifact["calibration_metadata"]
        if c.get("status") != "passed" or tau != artifact["research_threshold"] or not 0 <= c["risk_upper_bound"] <= c["target_risk"]:
            raise ValueError("Nasadzovaný prah nemá úspešné nezávislé overenie.")
    score = float(predict_candidate(artifact["model"], profile_arrays([record]))[0])
    return {"status": "accepted" if score <= tau else "review", "error_score": score, "threshold": tau, "research_only": research}
