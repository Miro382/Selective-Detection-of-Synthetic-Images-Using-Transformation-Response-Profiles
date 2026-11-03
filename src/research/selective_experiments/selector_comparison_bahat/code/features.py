"""Nižšie skóre vždy znamená predpokladanú menšiu pravdepodobnosť chyby."""
import numpy as np
from scipy.special import expit

TRANSFORMS = ("jpeg_q90", "jpeg_q75", "jpeg_q60", "resize_0p75", "resize_0p5")


def profile_features(margins, columns=(0, 1, 2, 3, 4, 5), oriented=True, summaries=True):
    # Súhrny sa vždy prepočítajú LEN z ponechaných transformácií.
    m = np.asarray(margins, dtype=np.float64)[:, columns]
    if columns[0] != 0 or not np.isfinite(m).all():
        raise ValueError("Profil musí začínať pôvodnou predikciou a byť konečný.")
    sign = np.where(m[:, :1] >= 0, 1., -1.)
    delta = m[:, 1:] - m[:, :1]
    if oriented:
        delta *= sign
    blocks = [-np.abs(m[:, :1]), delta]
    if summaries:
        p = expit(m)
        blocks += [np.var(p, axis=1)[:, None], np.max(np.abs(p - p[:, :1]), axis=1)[:, None],
                   np.mean((m[:, 1:] >= 0) != (m[:, :1] >= 0), axis=1)[:, None]]
    return np.concatenate(blocks, axis=1)


def representation(m, name):
    if name == "confidence":
        return -np.abs(m[:, :1])
    if name == "full":
        return profile_features(m)
    if name == "no_summaries":
        return profile_features(m, summaries=False)
    if name == "no_orientation":
        return profile_features(m, oriented=False)
    if name == "no_confidence":
        return profile_features(m)[:, 1:]
    if name == "summaries_only":
        x = profile_features(m)
        return np.c_[x[:, 0], x[:, -3:]]
    if name == "jpeg_only":
        return profile_features(m, (0, 1, 2, 3))
    if name == "resize_only":
        return profile_features(m, (0, 4, 5))
    if name.startswith("without_"):
        j = TRANSFORMS.index(name[len("without_"):]) + 1
        return profile_features(m, tuple(i for i in range(6) if i != j))
    raise ValueError("Neznáma reprezentácia: " + name)


def baseline(m, name):
    if name == "confidence":
        return expit(-np.abs(m[:, 0]))
    if name == "tta_mean":
        sign = np.where(m[:, :1] >= 0, 1., -1.)
        return expit(-sign * m).mean(axis=1)
    raise ValueError(name)


def sorted_logits(pairs):
    """Spoločné zoradenie podľa pôvodnej triedy; NIE triedenie každej transformácie."""
    pairs = np.asarray(pairs, dtype=np.float64)
    if pairs.ndim != 3 or pairs.shape[1:] != (6, 2) or not np.isfinite(pairs).all():
        raise ValueError("Očakávané logity N × 6 × 2.")
    fake = pairs[:, 0, 1] >= pairs[:, 0, 0]
    result = pairs.copy()
    result[fake] = result[fake, :, ::-1]
    return result.reshape(len(result), 12)


def logistic_score(model, x):
    return expit(((x - np.asarray(model["scaler"]["mean"])) / np.asarray(model["scaler"]["scale"]))
                 @ np.asarray(model["coef"]) + model["intercept"])


def weights(groups):
    _, inv, counts = np.unique(groups, return_inverse=True, return_counts=True)
    return 1. / counts[inv]


def methods(suite):
    result = [dict(id="frozen_v4", kind="frozen", features="full"),
              dict(id="confidence", kind="baseline"), dict(id="tta_mean", kind="baseline")]
    names = ["confidence", "full", "no_summaries", "no_orientation", "no_confidence",
             "summaries_only", "jpeg_only", "resize_only"] + ["without_" + n for n in TRANSFORMS]
    result += [dict(id="lr_" + n, kind="lr", features=n, C=.1) for n in names]
    result.append(dict(id="lr_balanced_no_summaries", kind="lr", features="no_summaries", C=.1, balance_classes=True))
    # Rovnaká reprezentácia v LR aj MLP izoluje zmenu pomocného klasifikátora.
    for seed in (17, 29, 43):
        result.append(dict(id=f"mlp_margins_s{seed}", kind="mlp", features="no_summaries", seed=seed))
    if suite == "full":
        for family in ("ours_logits", "bahat_logits"):
            for seed in (17, 29, 43):
                result.append(dict(id=f"mlp_{family}_s{seed}", kind="mlp", features=family, seed=seed))
    return result
