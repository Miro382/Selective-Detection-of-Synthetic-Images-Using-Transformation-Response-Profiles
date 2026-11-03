"""Audit lokálne stiahnutej podmnožiny DeepShield bez inferencie modelov.

Presné aj dekódované odtlačky sú kompatibilné s external_selector_data.py.
pHash je konzervatívny heuristický filter, nie dôkaz totožnosti predlohy.
Neprítomnosť zhody s dostupnými inventármi negarantuje nezávislosť od
nezdokumentovaných tréningových dát alebo od iných transformácií obrázkov.
"""
from __future__ import annotations

import csv
import hashlib
import struct
import warnings
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath, PureWindowsPath


FINGERPRINT_VERSION = "sha256-rgb8-wh-v1-phash-dct32x32-8x8-median-all-v1"
MAX_PIXELS = 128_000_000
PHASH_DISTANCE = 4
MAX_MATCHES_PER_KIND = 5
_BLOCKS = ((0, 13), (13, 13), (26, 13), (39, 13), (52, 12))
_HASH_WIDTHS = {"sha256": 64, "rgb_sha256": 64, "phash": 16}


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint(path: Path, *, allow_mpo_primary: bool = False) -> dict:
    """Vypočíta odtlačky pôvodných bajtov a RGB bez aplikovania EXIF rotácie.

    Neplatný obrázok, viac než 128 miliónov pixelov alebo zmena súboru počas
    čítania vyvolajú výnimku. Volajúci nesmie taký obrázok zaradiť do testu.
    Veľké bajtové reťazce RGB nevytvárame; hash sa aktualizuje po riadkoch.
    """
    import numpy as np
    from PIL import Image
    from scipy.fft import dctn

    path = Path(path)
    before = path.stat()
    if not path.is_file():
        raise ValueError("Cesta nie je súbor obrázka.")
    sha256 = _file_sha256(path)
    # Pillow má nižší predvolený varovný limit; vlastný pevný limit sa overí
    # pred dekódovaním. Jeho globálne nastavenie sa nikdy nevypína ani nemení.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", Image.DecompressionBombWarning)
        with Image.open(path) as opened:
            width, height = opened.size
            if width <= 0 or height <= 0 or width * height > MAX_PIXELS:
                raise ValueError(f"Obrázok {width} × {height} prekračuje limit {MAX_PIXELS} pixelov.")
            if getattr(opened, "n_frames", 1) != 1 and not (allow_mpo_primary and opened.format == "MPO"):
                raise ValueError("Animovaný alebo viacstranový obrázok nie je podporovaný.")
            opened.load()
            rgb = opened.convert("RGB")
            digest = hashlib.sha256(struct.pack("<QQ", width, height))
            for top in range(0, height, 128):
                digest.update(rgb.crop((0, top, width, min(height, top + 128))).tobytes())
            gray = rgb.convert("L").resize((32, 32), Image.Resampling.LANCZOS)
            low = dctn(np.asarray(gray, dtype=np.float64), type=2, norm="ortho")[:8, :8]
            bits = (low > np.median(low)).reshape(-1)
            perceptual = 0
            for bit in bits:
                perceptual = (perceptual << 1) | int(bit)
    after = path.stat()
    if (after.st_size, after.st_mtime_ns) != (before.st_size, before.st_mtime_ns):
        raise RuntimeError("Súbor sa počas výpočtu odtlačku zmenil.")
    return {"sha256": sha256, "rgb_sha256": digest.hexdigest(),
            "phash": f"{perceptual:016x}", "width": width, "height": height,
            "size_bytes": before.st_size}


def _hash_value(value, field: str, context: str = "odtlačok") -> str | None:
    if value is None or str(value).strip() == "":
        return None
    value = str(value).strip().lower()
    if len(value) != _HASH_WIDTHS[field] or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"Neplatný {field}: {context}.")
    return value


class _MemoryIndex:
    """Ohraničené záznamy zhôd a päťblokový index 64-bitového pHash.

    Pri Hammingovej vzdialenosti najviac štyri musí byť aspoň jeden z piatich
    disjunktných blokov nezmenený. Index preto nevynecháva kandidátov filtra.
    Ukladá iba jedinečné hodnoty pHash a najviac päť ciest na jednu hodnotu.
    """

    def __init__(self):
        self.exact = {field: defaultdict(list) for field in ("sha256", "rgb_sha256")}
        self.perceptual: dict[int, list[str]] = {}
        self.blocks = [defaultdict(list) for _ in _BLOCKS]

    @staticmethod
    def _add_path(bucket: list[str], path: str):
        if len(bucket) < MAX_MATCHES_PER_KIND and path not in bucket:
            bucket.append(path)

    def add(self, record: dict, path: str):
        for field, table in self.exact.items():
            value = _hash_value(record.get(field), field, path)
            if value is not None:
                self._add_path(table[value], path)
        phash = _hash_value(record.get("phash"), "phash", path)
        if phash is None:
            return
        value = int(phash, 16)
        if value not in self.perceptual:
            self.perceptual[value] = []
            for table, (shift, width) in zip(self.blocks, _BLOCKS):
                table[(value >> shift) & ((1 << width) - 1)].append(value)
        self._add_path(self.perceptual[value], path)

    def matches(self, record: dict) -> list[dict]:
        result = []
        for field, table in self.exact.items():
            value = _hash_value(record.get(field), field)
            if value is not None:
                result.extend({"kind": field, "other_path": path, "distance": None}
                              for path in table.get(value, ()))
        phash = _hash_value(record.get("phash"), "phash")
        if phash is None:
            return result
        value = int(phash, 16)
        candidates = set()
        for table, (shift, width) in zip(self.blocks, _BLOCKS):
            candidates.update(table.get((value >> shift) & ((1 << width) - 1), ()))
        # Výstup nie je závislý od interného poradia hash tabuliek.
        count = 0
        for other in sorted(candidates):
            distance = (other ^ value).bit_count()
            if distance <= PHASH_DISTANCE:
                for path in self.perceptual[other]:
                    result.append({"kind": "phash", "other_path": path, "distance": distance})
                    count += 1
                    if count == MAX_MATCHES_PER_KIND:
                        return result
        return result


class ExistingFingerprintIndex:
    """Index dostupných historických CSV; obsah CSV overuje hlavný protokol.

    Neotvára historické obrázky ani nevykonáva sieťové požiadavky. Prázdne
    jednotlivé hash polia sú dovolené; neplatný neprázdny hash audit zastaví.
    Riadky s explicitným chybným statusom sa preskočia a uvedú v súhrne.
    """

    def __init__(self, paths: list[Path]):
        self._index = _MemoryIndex()
        self.summary = {"csv_files": [], "rows_read": 0, "rows_indexed": 0,
                        "rows_without_hashes": 0, "rows_with_error_status": 0,
                        "field_counts": dict.fromkeys(_HASH_WIDTHS, 0)}
        for source in sorted({str(Path(p).resolve()) for p in paths}):
            path = Path(source)
            self.summary["csv_files"].append(source)
            with path.open("r", encoding="utf-8-sig", newline="") as stream:
                reader = csv.DictReader(stream)
                if not reader.fieldnames or not set(reader.fieldnames) & set(_HASH_WIDTHS):
                    raise ValueError(f"CSV neobsahuje odtlačky sha256/rgb_sha256/phash: {path}")
                for line_number, row in enumerate(reader, 2):
                    self.summary["rows_read"] += 1
                    status = (row.get("status") or "").strip().lower()
                    if status not in ("", "ok", "success"):
                        self.summary["rows_with_error_status"] += 1
                        continue
                    other_path = (row.get("path") or row.get("absolute_path")
                                  or row.get("relative_path") or f"{source}:riadok_{line_number}")
                    clean = {field: _hash_value(row.get(field), field, f"{source}:{line_number}")
                             for field in _HASH_WIDTHS}
                    if not any(clean.values()):
                        self.summary["rows_without_hashes"] += 1
                        continue
                    self._index.add(clean, other_path)
                    self.summary["rows_indexed"] += 1
                    for field, value in clean.items():
                        if value is not None:
                            self.summary["field_counts"][field] += 1

    def matches(self, fingerprint: dict) -> list[dict]:
        """Najviac päť zhôd každého druhu; vzdialenosť iba pri pHash."""
        return self._index.matches(fingerprint)


def _local_path(root: Path, relative_path: str) -> Path:
    if not isinstance(relative_path, str) or not relative_path.strip():
        raise ValueError("Prázdna relatívna cesta obrázka.")
    posix = PurePosixPath(relative_path.replace("\\", "/"))
    win = PureWindowsPath(relative_path)
    if (posix.is_absolute() or win.drive or win.root or ".." in posix.parts
            or any(":" in part for part in posix.parts)):
        raise ValueError("Cesta obrázka musí zostať v adresári podmnožiny.")
    candidate = (root / Path(*posix.parts)).resolve()
    if not candidate.is_relative_to(root) or candidate == root:
        raise ValueError("Cesta obrázka opúšťa adresár podmnožiny.")
    return candidate


def audit_downloads(rows: list[dict], root: Path, index: ExistingFingerprintIndex) -> dict:
    """Vylúči celé predlohové skupiny a vráti odtlačky aj dôvody vylúčenia.

    Skupiny sa spracujú lexikograficky. Interné porovnanie používa len už
    prijaté skupiny; prvá vyhovujúca skupina zostane. Social verzie jednej
    predlohy sa nikdy nevylúčia pre ich vzájomnú podobnosť. Vyhodnotenie
    samotného klasifikátora ani selektora sa nevykonáva.
    """
    root = Path(root).resolve()
    if not root.is_dir():
        raise ValueError("Adresár stiahnutých obrázkov neexistuje.")
    grouped = defaultdict(list)
    seen_paths = set()
    required = ("relative_path", "group", "label", "generator", "platform", "role")
    for row in rows:
        if any(field not in row for field in required):
            raise ValueError("Záznam manifestu nemá všetky povinné polia.")
        if not isinstance(row["group"], str) or not row["group"].strip():
            raise ValueError("Prázdny identifikátor skupiny.")
        normalized = str(row["relative_path"]).replace("\\", "/").casefold()
        if normalized in seen_paths:
            raise ValueError("Rovnaká lokálna cesta je v manifeste viackrát.")
        seen_paths.add(normalized)
        grouped[row["group"]].append(dict(row))

    accepted, excluded = [], []
    internal = _MemoryIndex()
    internal_groups = {}
    group_reasons = Counter()
    fingerprint_errors = 0
    missing_records = 0
    missing_groups = set()
    for group in sorted(grouped):
        records, reasons = [], []
        for row in sorted(grouped[group], key=lambda item: item["relative_path"]):
            record = dict(row)
            try:
                path = _local_path(root, row["relative_path"])
                record.update(fingerprint(path))
                record["fingerprint_status"] = "ok"
            except (ImportError, MemoryError):
                # Chýbajúca knižnica alebo pamäť nie sú dôvod vyradiť obrázok.
                # Zastavenie umožní opravu prostredia a opakovanie toho istého
                # vopred zvoleného panelu bez dátovo závislých vylúčení.
                raise
            except Exception as error:
                fingerprint_errors += 1
                missing = isinstance(error, FileNotFoundError)
                if missing:
                    missing_records += 1
                    missing_groups.add(group)
                record["fingerprint_status"] = "missing" if missing else "error"
                record["fingerprint_error"] = f"{type(error).__name__}: {error}"
                reasons.append({"kind": "missing_file" if missing else "invalid_image", "relative_path": row["relative_path"],
                                "error": record["fingerprint_error"]})
                records.append(record)
                continue
            for match in index.matches(record):
                reasons.append({**match, "kind": f"historical_{match['kind']}",
                                "relative_path": row["relative_path"]})
            for match in internal.matches(record):
                reasons.append({**match, "kind": f"internal_{match['kind']}",
                                "relative_path": row["relative_path"],
                                "other_group": internal_groups[match["other_path"]]})
            records.append(record)
        if reasons:
            for kind in {reason["kind"] for reason in reasons}:
                group_reasons[kind] += 1
            for record in records:
                record["exclusion_reasons"] = reasons
                excluded.append(record)
        else:
            for record in records:
                record["exclusion_reasons"] = []
                accepted.append(record)
                path_key = record["relative_path"]
                internal_groups[path_key] = group
                internal.add(record, path_key)

    def counts(records):
        return {field: dict(sorted(Counter(str(r[field]) for r in records).items()))
                for field in ("label", "generator", "platform", "role")}

    return {"accepted": accepted, "excluded": excluded, "summary": {
        "schema_version": 1, "fingerprint_version": FINGERPRINT_VERSION,
        "status": "incomplete_downloads" if missing_records else "available_history_and_internal_audit_completed",
        "independence_guaranteed": False,
        "input_records": len(rows), "input_groups": len(grouped),
        "accepted_records": len(accepted), "excluded_records": len(excluded),
        "accepted_groups": len({r["group"] for r in accepted}),
        "excluded_groups": len({r["group"] for r in excluded}),
        "fingerprint_errors": fingerprint_errors,
        "missing_records": missing_records, "missing_groups": len(missing_groups),
        "group_exclusion_counts": dict(sorted(group_reasons.items())),
        "accepted_counts": counts(accepted), "excluded_counts": counts(excluded),
        "historical_index": index.summary,
        "phash_distance": PHASH_DISTANCE, "max_pixels": MAX_PIXELS,
        "match_examples_per_kind_limit": MAX_MATCHES_PER_KIND,
        "notes": [
            "pHash ≤ 4 je konzervatívny heuristický filter podobnosti, nie dôkaz duplicity.",
            "Vylúčenie sa vzťahuje na celú predlohovú skupinu; súbory sa nemažú.",
            "Pri interných zhodách zostane lexikograficky prvá vyhovujúca skupina.",
            "Social verzie jednej predlohy nie sú nezávislé pozorovania.",
            "Audit pokrýva iba dodané historické inventáre; negarantuje úplnú nezávislosť.",
            "Výber sa nemení podľa skóre detektora ani selektora; audit nevykonáva inferenciu.",
        ]}}
