"""Zmrazená definícia profilu degradácií V1, zachovaná aj vo V4.

Modul iba počíta profil. Učenie modelu chyby, výber prahu na výberovej
množine a jednorazové kalibračné overenie rieši train_selector_v4.py.
Stabilita odpovedí sama osebe nezaručuje správnosť rozhodnutia detektora.
"""
from __future__ import annotations

import io
import json
import math
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from threading import Event
from typing import Callable, Sequence

try:
    import numpy as np
except ImportError:  # Preserve the application's dependency diagnostic UI.
    np = None
try:
    from PIL import Image
except ImportError:
    Image = None

def check_cancelled(cancel):
    if cancel is not None and cancel.is_set():
        raise InterruptedError('Výpočet profilu bol zastavený.')


PROFILE_VERSION = "degradation-profile-v1"
ARTIFACT_VERSION = "selective-error-logistic-v1"
THRESHOLD_GRID = tuple(i / 100 for i in range(101))


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{name}: očakáva sa konečné číslo.")
    return float(value)


def _probability(value: object, name: str) -> float:
    result = _finite(value, name)
    if not 0 <= result <= 1:
        raise ValueError(f"{name}: hodnota musí byť v intervale [0, 1].")
    return result


@dataclass(frozen=True)
class ProfileConfig:
    jpeg_qualities: tuple[int, ...] = (90, 75, 60)
    resize_scales: tuple[float, ...] = (0.75, 0.5)
    version: str = PROFILE_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "jpeg_qualities", tuple(self.jpeg_qualities))
        object.__setattr__(self, "resize_scales", tuple(self.resize_scales))
        if self.version != PROFILE_VERSION:
            raise ValueError("Nepodporovaná verzia profilu degradácií.")
        if not 1 <= len(self.jpeg_qualities) + len(self.resize_scales) <= 32:
            raise ValueError("Profil musí obsahovať 1 až 32 degradácií.")
        if any(type(q) is not int or not 1 <= q <= 100 for q in self.jpeg_qualities):
            raise ValueError("Kvalita JPEG musí byť celé číslo od 1 do 100.")
        if any(not 0 < _finite(s, "Mierka") < 1 for s in self.resize_scales):
            raise ValueError("Mierka zmenšenia musí byť medzi 0 a 1.")
        if len(set(self.names)) != len(self.names):
            raise ValueError("Degradácie v profile sa nesmú opakovať.")

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(f"jpeg_q{q}" for q in self.jpeg_qualities) + tuple(
            f"resize_{s:.6g}".replace(".", "p") for s in self.resize_scales
        )

    @property
    def feature_names(self) -> tuple[str, ...]:
        return ("original_probability", "original_confidence") + tuple(
            f"delta_{name}" for name in self.names
        ) + ("prediction_variance", "max_abs_shift", "flip_fraction",
             "log_pixel_count", "log_aspect_ratio", "log_laplacian_variance")

    def to_dict(self) -> dict:
        return {"version": self.version, "jpeg_qualities": list(self.jpeg_qualities),
                "resize_scales": list(self.resize_scales)}

    @classmethod
    def from_dict(cls, value: dict) -> "ProfileConfig":
        if not isinstance(value, dict) or set(value) != {"version", "jpeg_qualities", "resize_scales"}:
            raise ValueError("Neplatná konfigurácia profilu.")
        return cls(tuple(value["jpeg_qualities"]), tuple(value["resize_scales"]), value["version"])


@dataclass(frozen=True)
class DegradationVariant:
    name: str
    probability: float
    delta: float
    flipped: bool

    def to_dict(self) -> dict:
        return {"name": self.name, "probability": self.probability,
                "delta": self.delta, "flipped": self.flipped}


def _prediction_features(original: float, threshold: float,
                         probabilities: Sequence[float], names: Sequence[str]) -> dict[str, float]:
    predicted_fake = original >= threshold
    deltas = [p - original for p in probabilities]
    return {
        "original_probability": original,
        "original_confidence": original if predicted_fake else 1.0 - original,
        **{f"delta_{name}": delta for name, delta in zip(names, deltas)},
        "prediction_variance": float(np.var([original, *probabilities])),
        "max_abs_shift": max(abs(d) for d in deltas),
        "flip_fraction": sum((p >= threshold) != predicted_fake for p in probabilities) / len(probabilities),
    }


@dataclass(frozen=True)
class DegradationProfile:
    config: ProfileConfig
    original_probability: float
    decision_threshold: float
    variants: list[DegradationVariant]
    features: dict[str, float]

    def validate(self) -> None:
        original = _probability(self.original_probability, "Pôvodné skóre")
        threshold = _probability(self.decision_threshold, "Prah detektora")
        if tuple(v.name for v in self.variants) != self.config.names:
            raise ValueError("Varianty nezodpovedajú konfigurácii profilu.")
        for variant in self.variants:
            p = _probability(variant.probability, "Skóre degradácie")
            if not math.isclose(_finite(variant.delta, "Zmena skóre"), p - original, abs_tol=1e-12):
                raise ValueError("Nesúlad zmeny skóre v profile.")
            if type(variant.flipped) is not bool or variant.flipped != ((p >= threshold) != (original >= threshold)):
                raise ValueError("Nesúlad zmeny rozhodnutia v profile.")
        if set(self.features) != set(self.config.feature_names):
            raise ValueError("Nesprávna schéma príznakov profilu.")
        for name, value in self.features.items():
            _finite(value, name)
        derived = _prediction_features(original, threshold, [v.probability for v in self.variants], self.config.names)
        if any(not math.isclose(self.features[name], value, rel_tol=1e-10, abs_tol=1e-12)
               for name, value in derived.items()):
            raise ValueError("Príznaky nezodpovedajú skóre degradácií.")
        if any(self.features[name] < 0 for name in ("log_pixel_count", "log_aspect_ratio", "log_laplacian_variance")):
            raise ValueError("Neplatné príznaky kvality obrázka.")

    def to_dict(self) -> dict:
        self.validate()
        return {"config": self.config.to_dict(), "original_probability": self.original_probability,
                "decision_threshold": self.decision_threshold,
                "variants": [v.to_dict() for v in self.variants], "features": dict(self.features)}

    @classmethod
    def from_dict(cls, value: dict) -> "DegradationProfile":
        if not isinstance(value, dict):
            raise ValueError("Profil musí byť objekt JSON.")
        try:
            result = cls(ProfileConfig.from_dict(value["config"]), value["original_probability"],
                         value["decision_threshold"],
                         [DegradationVariant(**v) for v in value["variants"]], dict(value["features"]))
            result.validate()
            return result
        except (KeyError, TypeError, AttributeError) as error:
            raise ValueError("Neplatný profil degradácií.") from error


def build_profile(image: Image.Image, predict: Callable[[Image.Image], float], *,
                  original_probability: float, decision_threshold: float = 0.5,
                  config: ProfileConfig | None = None,
                  progress: Callable[[str], None] | None = None,
                  cancel: Event | None = None) -> DegradationProfile:
    """Vyhodnotí degradácie postupne; nikdy nemení vstupný obrázok ani súbor.

    predict musí používať rovnaký detektor a predspracovanie ako pôvodné skóre.
    JPEG používa subsampling=0. Zmenšenie aj návrat na pôvodné rozmery používa
    Lanczos. Verzia profilu zahŕňa tieto voľby aj definície príznakov.
    """
    check_cancelled(cancel)
    if np is None or Image is None:
        raise RuntimeError("Profil citlivosti vyžaduje nainštalované balíky numpy a Pillow.")
    config = config or ProfileConfig()
    original = _probability(original_probability, "Pôvodné skóre")
    threshold = _probability(decision_threshold, "Prah detektora")
    source = image.convert("RGB").copy()
    width, height = source.size
    if width < 1 or height < 1:
        raise ValueError("Obrázok nemá platné rozmery.")
    variants: list[DegradationVariant] = []
    specifications = [("jpeg", q) for q in config.jpeg_qualities] + [("resize", s) for s in config.resize_scales]
    for index, (name, (kind, parameter)) in enumerate(zip(config.names, specifications), 1):
        check_cancelled(cancel)
        if progress:
            progress(f"Kontrola spoľahlivosti: variant {index}/{len(specifications)} ({name}).")
        if kind == "jpeg":
            buffer = io.BytesIO()
            source.save(buffer, format="JPEG", quality=parameter, subsampling=0, optimize=False)
            buffer.seek(0)
            with Image.open(buffer) as decoded:
                transformed = decoded.convert("RGB").copy()
        else:
            small_size = (max(1, round(width * parameter)), max(1, round(height * parameter)))
            transformed = source.resize(small_size, Image.Resampling.LANCZOS).resize(source.size, Image.Resampling.LANCZOS)
        check_cancelled(cancel)
        probability = _probability(predict(transformed), "Skóre degradácie")
        check_cancelled(cancel)
        variants.append(DegradationVariant(name, probability, probability - original,
                                           (probability >= threshold) != (original >= threshold)))
    features = _prediction_features(original, threshold, [v.probability for v in variants], config.names)
    # Normalized luminance; capped analysis resolution makes this proxy bounded
    # in cost. It is a texture/blur proxy, not a ground-truth image-quality score.
    quality_image = source.copy()
    quality_image.thumbnail((512, 512), Image.Resampling.LANCZOS)
    gray = np.asarray(quality_image.convert("L"), dtype=np.float64) / 255.0
    padded = np.pad(gray, 1, mode="edge")
    laplacian = padded[:-2, 1:-1] + padded[2:, 1:-1] + padded[1:-1, :-2] + padded[1:-1, 2:] - 4 * gray
    features.update(log_pixel_count=math.log1p(width * height),
                    log_aspect_ratio=abs(math.log(width / height)),
                    log_laplacian_variance=math.log1p(float(laplacian.var())))
    check_cancelled(cancel)
    result = DegradationProfile(config, original, threshold, variants, features)
    result.validate()
    return result


