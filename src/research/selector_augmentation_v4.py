"""Zmrazená politika vonkajších degradácií V2/V3 použitá aj vo V4.

Tieto operácie menia vstup *pred* výpočtom pôvodného skóre a profilu.
Štítok chyby preto treba určiť z rozhodnutia detektora nad týmto vstupom.
Nesmie sa preberať rozhodnutie nad čistým obrázkom. Kalibrácia používa
jeden variant na nezávislú pôvodnú skupinu; osem tréningových variantov
nie je osem nezávislých kalibračných pozorovaní.
"""
from __future__ import annotations

import hashlib
import io
import json
import math
import random
import re
from typing import Any

from PIL import Image, ImageEnhance


VERSION = 1
TRAINING_IDS = (
    "clean", "jpeg_mild", "jpeg_strong", "resize_mild", "resize_strong",
    "brightness", "contrast", "combined",
)
AUGMENTATION_POLICY = {
    "version": VERSION,
    "seed_derivation": "SHA256(JSON([namespace, version, seed, identity, variant]))",
    "parameter_generator": "Python random.Random with independent SHA256 seed per variant",
    "training_identity": "lowercase SHA256 of original file; no path or label",
    "calibration_identity": "original group identifier; no path or label",
    "training_variants_per_original": 8,
    "training_ids": list(TRAINING_IDS),
    "jpeg_mild_quality_inclusive": [70, 95],
    "jpeg_strong_quality_inclusive": [35, 60],
    "resize_mild_scale_inclusive": [0.65, 0.90],
    "resize_strong_scale_inclusive": [0.40, 0.60],
    "resize_scale_step": 0.01,
    "brightness_factor_inclusive": [0.85, 1.15],
    "contrast_factor_inclusive": [0.85, 1.15],
    "enhancement_factor_step": 0.01,
    "combined_order": ["brightness", "contrast", "resize", "jpeg"],
    "combined_brightness_factor_inclusive": [0.90, 1.10],
    "combined_contrast_factor_inclusive": [0.90, 1.10],
    "combined_resize_scale_inclusive": [0.50, 0.85],
    "combined_jpeg_quality_inclusive": [50, 85],
    "calibration_variants_per_group": 1,
    "calibration_mixture": {
        "clean": 0.25, "calibration_jpeg": 0.25,
        "calibration_resize": 0.25, "brightness": 0.125, "contrast": 0.125,
    },
    "calibration_jpeg_quality_inclusive": [35, 95],
    "calibration_resize_scale_inclusive": [0.40, 0.90],
    "calibration_mixture_note": "Independent sampling per group, not exact quotas or label stratification",
    "jpeg_subsampling": 0,
    "jpeg_optimize": False,
    "resize_resampling": "LANCZOS for downsampling and restoring original dimensions",
    "resize_rounding": "max(1, round(original_side * scale))",
    "output_mode": "RGB",
    "output_dimensions": "original dimensions",
}


def _rng(identity: str, seed: int, variant: str, namespace: str) -> random.Random:
    if not isinstance(identity, str) or not identity.strip():
        raise ValueError("Identifikátor obrázka alebo skupiny musí byť neprázdny text.")
    if type(seed) is not int:
        raise ValueError("Seed musí byť celé číslo, nie logická hodnota.")
    payload = json.dumps([namespace, VERSION, seed, identity, variant],
                         ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return random.Random(int.from_bytes(hashlib.sha256(payload).digest(), "big"))


def _spec(variant: str, rng: random.Random) -> dict:
    operations: list[dict] = []
    if variant == "clean":
        pass
    elif variant in {"jpeg_mild", "jpeg_strong", "calibration_jpeg"}:
        low, high = {"jpeg_mild": (70, 95), "jpeg_strong": (35, 60),
                     "calibration_jpeg": (35, 95)}[variant]
        operations = [{"kind": "jpeg", "quality": rng.randint(low, high)}]
    elif variant in {"resize_mild", "resize_strong", "calibration_resize"}:
        low, high = {"resize_mild": (65, 90), "resize_strong": (40, 60),
                     "calibration_resize": (40, 90)}[variant]
        operations = [{"kind": "resize", "scale": rng.randint(low, high) / 100}]
    elif variant in {"brightness", "contrast"}:
        operations = [{"kind": variant, "factor": rng.randint(85, 115) / 100}]
    elif variant == "combined":
        operations = [
            {"kind": "brightness", "factor": rng.randint(90, 110) / 100},
            {"kind": "contrast", "factor": rng.randint(90, 110) / 100},
            {"kind": "resize", "scale": rng.randint(50, 85) / 100},
            {"kind": "jpeg", "quality": rng.randint(50, 85)},
        ]
    else:
        raise ValueError(f"Neznámy variant augmentácie: {variant}")
    return {"version": VERSION, "id": variant, "operations": operations}


def training_specs(sha256: str, seed: int) -> list[dict]:
    """Osem variantov s parametrami nezávislými od poradia dát a ich umiestnenia."""
    if not isinstance(sha256, str) or re.fullmatch(r"[0-9a-fA-F]{64}", sha256) is None:
        raise ValueError("sha256 musí obsahovať presne 64 hexadecimálnych znakov.")
    identity = sha256.lower()
    return [_spec(variant, _rng(identity, seed, variant, "selector-v2-train"))
            for variant in TRAINING_IDS]


def calibration_spec(group: str, seed: int) -> dict:
    """Jeden variant na skupinu, zmes 25/25/25/12,5/12,5 percenta.

    Ide o pravdepodobnosti, nie o presné počty v konečnej vzorke. Viaceré
    súbory z jednej skupiny sa nesmú počítať ako nezávislé kalibračné body.
    """
    choice = _rng(group, seed, "mixture-choice", "selector-v2-calibration").randrange(8)
    variant = ("clean", "clean", "calibration_jpeg", "calibration_jpeg",
               "calibration_resize", "calibration_resize", "brightness", "contrast")[choice]
    return _spec(variant, _rng(group, seed, variant, "selector-v2-calibration"))


def _number(value: Any, low: float, high: float, name: str) -> None:
    if (type(value) not in {int, float} or not math.isfinite(value)
            or not low <= value <= high):
        raise ValueError(f"{name} musí byť konečné číslo v rozsahu {low} až {high}.")


def _validate_spec(spec: dict) -> list[dict]:
    if not isinstance(spec, dict) or set(spec) != {"version", "id", "operations"}:
        raise ValueError("Špecifikácia musí obsahovať iba version, id a operations.")
    if type(spec["version"]) is not int or spec["version"] != VERSION:
        raise ValueError("Nepodporovaná verzia augmentácie.")
    variant = spec["id"]
    layouts = {
        "clean": [], "jpeg_mild": ["jpeg"], "jpeg_strong": ["jpeg"],
        "resize_mild": ["resize"], "resize_strong": ["resize"],
        "brightness": ["brightness"], "contrast": ["contrast"],
        "combined": ["brightness", "contrast", "resize", "jpeg"],
        "calibration_jpeg": ["jpeg"], "calibration_resize": ["resize"],
    }
    if not isinstance(variant, str) or variant not in layouts:
        raise ValueError("Neplatné id augmentácie.")
    operations = spec["operations"]
    if not isinstance(operations, list) or any(not isinstance(op, dict) for op in operations):
        raise ValueError("operations musí byť zoznam objektov.")
    if [op.get("kind") for op in operations] != layouts[variant]:
        raise ValueError("Operácie alebo ich poradie nezodpovedajú variantu.")
    for operation in operations:
        kind = operation["kind"]
        parameter = {"jpeg": "quality", "resize": "scale",
                     "brightness": "factor", "contrast": "factor"}[kind]
        if set(operation) != {"kind", parameter}:
            raise ValueError("Neplatné polia operácie augmentácie.")
        value = operation[parameter]
        if kind == "jpeg":
            low, high = {"jpeg_mild": (70, 95), "jpeg_strong": (35, 60),
                         "calibration_jpeg": (35, 95), "combined": (50, 85)}[variant]
            if type(value) is not int:
                raise ValueError("Kvalita JPEG musí byť celé číslo.")
        elif kind == "resize":
            low, high = {"resize_mild": (0.65, 0.90), "resize_strong": (0.40, 0.60),
                         "calibration_resize": (0.40, 0.90), "combined": (0.50, 0.85)}[variant]
        else:
            low, high = (0.90, 1.10) if variant == "combined" else (0.85, 1.15)
        _number(value, low, high, parameter)
        if kind != "jpeg" and not math.isclose(value * 100, round(value * 100), abs_tol=1e-9):
            raise ValueError("Mierka alebo faktor musí používať krok 0,01.")
    return operations


def apply_spec(image: Image.Image, spec: dict) -> Image.Image:
    """Vráti samostatný RGB obrázok; nemení vstup, rozmery ani špecifikáciu."""
    operations = _validate_spec(spec)
    if not isinstance(image, Image.Image) or min(image.size) < 1:
        raise ValueError("Vstup musí byť neprázdny obrázok Pillow.")
    output = image.convert("RGB").copy()
    for operation in operations:
        kind = operation["kind"]
        if kind == "jpeg":
            with io.BytesIO() as buffer:
                output.save(buffer, format="JPEG", quality=operation["quality"],
                            subsampling=0, optimize=False)
                buffer.seek(0)
                with Image.open(buffer) as decoded:
                    output = decoded.convert("RGB").copy()
        elif kind == "resize":
            original_size = output.size
            small = tuple(max(1, round(side * operation["scale"])) for side in original_size)
            output = output.resize(small, Image.Resampling.LANCZOS).resize(
                original_size, Image.Resampling.LANCZOS)
        elif kind == "brightness":
            output = ImageEnhance.Brightness(output).enhance(operation["factor"])
        else:
            output = ImageEnhance.Contrast(output).enhance(operation["factor"])
    return output
