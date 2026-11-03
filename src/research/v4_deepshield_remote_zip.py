"""Bezpečné čítanie jednotlivých súborov vzdialeného ZIP/ZIP64 cez HTTP Range.

Server musí podporovať presné čiastkové odpovede 206. Odpoveď 200 sa odmietne
pred čítaním tela: tento modul nikdy automaticky nestiahne celý archív.
"""
from __future__ import annotations

import email.utils
import io
import math
import re
import struct
import time
import zipfile
import zlib
from datetime import datetime, timezone

import requests


class RemoteZipError(RuntimeError):
    """Archív alebo odpoveď servera nespĺňajú požadované kontroly."""


def _safe_name(name: str) -> None:
    if not isinstance(name, str) or not name or "\x00" in name:
        raise RemoteZipError("Neplatný názov súboru v archíve.")
    parts = name.rstrip("/").split("/")
    if (name.startswith("/") or "\\" in name or ":" in name
            or any(part in ("", ".", "..") for part in parts)):
        raise RemoteZipError(f"Nebezpečná cesta v archíve: {name!r}")


class _RangeReader(io.RawIOBase):
    """Seekable rozhranie potrebné pre štandardnú podporu ZIP64 v zipfile."""

    def __init__(self, remote):
        super().__init__()
        self.remote = remote
        self.position = 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=io.SEEK_SET):
        if whence == io.SEEK_SET:
            position = offset
        elif whence == io.SEEK_CUR:
            position = self.position + offset
        elif whence == io.SEEK_END:
            position = self.remote.size + offset
        else:
            raise ValueError("Neplatný začiatok posunu v súbore.")
        if not 0 <= position <= self.remote.size:
            raise RemoteZipError("Index ZIP požaduje pozíciu mimo archívu.")
        self.position = position
        return position

    def read(self, size=-1):
        if self.closed:
            raise ValueError("Čítačka archívu je zatvorená.")
        remaining = self.remote.size - self.position
        size = remaining if size is None or size < 0 else min(size, remaining)
        if size > self.remote.max_index_bytes:
            raise RemoteZipError("Požadované čítanie indexu presahuje bezpečnostný limit.")
        result = self.remote._read_at(self.position, size)
        self.position += len(result)
        return result


class RemoteZip:
    """Načíta index ZIP64 a vybrané položky bez stiahnutia celého archívu.

    ``transport`` môže byť requests.Session alebo testovací objekt s metódou
    get(url, headers=..., stream=True, timeout=...). Cudzí transport sa nezatvára.
    Počítadlo bytes_received zahŕňa aj skutočne prečítané neúspešné pokusy.
    max_retries označuje opakovania navyše k prvému pokusu (najviac štyri).
    """

    def __init__(self, url, size, *, max_request_bytes=64 * 1024**2,
                 min_interval=0.7, transport=None, max_total_bytes=8 * 1024**3,
                 max_index_bytes=256 * 1024**2, max_retries=4,
                 timeout=(15, 90)):
        if not isinstance(url, str) or not url.startswith("https://"):
            raise ValueError("Archív musí mať adresu HTTPS.")
        for value in (size, max_request_bytes, max_total_bytes, max_index_bytes):
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError("Veľkosti a prenosové limity musia byť kladné celé čísla.")
        if not math.isfinite(min_interval) or min_interval < 0:
            raise ValueError("Interval požiadaviek musí byť nezáporný a konečný.")
        if not isinstance(max_retries, int) or not 0 <= max_retries <= 4:
            raise ValueError("Povolené sú najviac štyri opakovania požiadavky.")
        self.url, self.size = url, size
        self.max_request_bytes = max_request_bytes
        self.max_total_bytes = max_total_bytes
        self.max_index_bytes = max_index_bytes
        self.min_interval, self.max_retries = min_interval, max_retries
        self.timeout = timeout
        self.transport = transport if transport is not None else requests.Session()
        self._own_transport = transport is None
        self._last_request = None
        self._closed = False
        self.bytes_received = 0
        self.requests_count = 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def close(self):
        if not self._closed and self._own_transport:
            self.transport.close()
        self._closed = True

    @staticmethod
    def _retry_delay(response, attempt):
        value = response.headers.get("Retry-After") if response is not None else None
        delay = min(2**attempt, 16)
        if value:
            try:
                delay = max(0.0, float(value))
            except ValueError:
                try:
                    date = email.utils.parsedate_to_datetime(value)
                    if date.tzinfo is None:
                        date = date.replace(tzinfo=timezone.utc)
                    delay = max(0.0, (date - datetime.now(timezone.utc)).total_seconds())
                except (ValueError, TypeError, OverflowError):
                    pass
        if not math.isfinite(delay) or delay > 60:
            raise RemoteZipError("Server žiada dlhšie čakanie; obnovte sťahovanie neskôr.")
        return delay

    def _range(self, start, length):
        if self._closed:
            raise RemoteZipError("Spojenie s archívom je zatvorené.")
        if not (0 <= start < self.size and 0 < length <= self.max_request_bytes
                and start + length <= self.size):
            raise RemoteZipError("Neplatný alebo príliš veľký rozsah HTTP požiadavky.")
        end = start + length - 1
        for attempt in range(self.max_retries + 1):
            if self.bytes_received + length > self.max_total_bytes:
                raise RemoteZipError("Bol dosiahnutý povolený limit prenesených bajtov.")
            if self._last_request is not None:
                wait = self.min_interval - (time.monotonic() - self._last_request)
                if wait > 0:
                    time.sleep(wait)
            self._last_request = time.monotonic()
            self.requests_count += 1
            response = None
            retry_delay = None
            try:
                response = self.transport.get(
                    self.url, headers={"Range": f"bytes={start}-{end}",
                                       "Accept-Encoding": "identity"},
                    stream=True, timeout=self.timeout)
                status = response.status_code
                if status == 429 or 500 <= status <= 599:
                    if attempt >= self.max_retries:
                        raise RemoteZipError(f"Server opakovane vrátil HTTP {status}.")
                    retry_delay = self._retry_delay(response, attempt)
                else:
                    if status != 206:
                        raise RemoteZipError(
                            f"Server vrátil HTTP {status} namiesto 206; telo sa nestiahne.")
                    content_range = response.headers.get("Content-Range", "")
                    match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", content_range)
                    if not match or tuple(map(int, match.groups())) != (start, end, self.size):
                        raise RemoteZipError("Server vrátil nesprávny rozsah Content-Range.")
                    encoding = response.headers.get("Content-Encoding", "identity")
                    if encoding.lower() not in ("", "identity"):
                        raise RemoteZipError("Server zmenil kódovanie bajtov archívu.")
                    content_length = response.headers.get("Content-Length")
                    if content_length is not None:
                        try:
                            correct_length = int(content_length) == length
                        except ValueError:
                            correct_length = False
                        if not correct_length:
                            raise RemoteZipError("Server ohlásil nesprávnu dĺžku rozsahu.")
                    data = bytearray()
                    for chunk in response.iter_content(chunk_size=64 * 1024):
                        if not chunk:
                            continue
                        self.bytes_received += len(chunk)
                        if len(data) + len(chunk) > length:
                            raise RemoteZipError("Server poslal viac bajtov než vyžiadaný rozsah.")
                        data.extend(chunk)
                    if len(data) != length:
                        raise RemoteZipError("Server poslal neúplný rozsah archívu.")
                    return bytes(data)
            except requests.RequestException as exc:
                if attempt >= self.max_retries:
                    raise RemoteZipError("Sieťové čítanie zlyhalo aj po opakovaní.") from exc
                retry_delay = self._retry_delay(None, attempt)
            finally:
                if response is not None:
                    response.close()
            if retry_delay is not None:
                time.sleep(retry_delay)
        raise RemoteZipError("Čítanie rozsahu sa nepodarilo dokončiť.")

    def _read_at(self, start, length):
        if length == 0:
            return b""
        if start < 0 or length < 0 or start + length > self.size:
            raise RemoteZipError("Čítanie presahuje hranice archívu.")
        chunks = []
        while length:
            take = min(length, self.max_request_bytes)
            chunks.append(self._range(start, take))
            start += take
            length -= take
        return b"".join(chunks)

    def entries(self):
        """Vráti centrálny index; obsah obrázkov sa pri tomto kroku nečíta."""
        try:
            with _RangeReader(self) as reader, zipfile.ZipFile(reader) as archive:
                result = []
                names = set()
                for item in archive.infolist():
                    # ZipInfo na Windows normalizuje spätné lomky a odstraňuje
                    # časť za NUL; kontrolujeme aj pôvodný názov pred normalizáciou.
                    _safe_name(item.orig_filename)
                    _safe_name(item.filename)
                    if item.filename in names:
                        raise RemoteZipError(f"Duplicitný názov v archíve: {item.filename!r}")
                    names.add(item.filename)
                    result.append({"name": item.filename, "size": item.file_size,
                                   "compressed": item.compress_size,
                                   "method": item.compress_type, "crc": item.CRC,
                                   "offset": item.header_offset, "flag_bits": item.flag_bits})
                return result
        except (zipfile.BadZipFile, struct.error, UnicodeError, OverflowError) as exc:
            raise RemoteZipError("Centrálny index ZIP/ZIP64 je neplatný.") from exc

    def read_member(self, entry, *, max_uncompressed=128 * 1024**2):
        """Vráti obsah jednej položky po kontrole lokálnej hlavičky, veľkosti a CRC."""
        try:
            name = entry["name"]
            values = [entry[k] for k in ("size", "compressed", "method", "crc",
                                         "offset", "flag_bits")]
        except (KeyError, TypeError) as exc:
            raise RemoteZipError("Chýbajú údaje položky z centrálneho indexu.") from exc
        _safe_name(name)
        if any(not isinstance(v, int) or isinstance(v, bool) or v < 0 for v in values):
            raise RemoteZipError("Číselné údaje položky archívu nie sú platné.")
        size, compressed, method, crc, offset, flags = values
        if not isinstance(max_uncompressed, int) or max_uncompressed <= 0:
            raise ValueError("Limit rozbalenej položky musí byť kladné celé číslo.")
        if size > max_uncompressed or compressed > max_uncompressed * 101 // 100 + 65536:
            raise RemoteZipError("Položka presahuje povolenú veľkosť obrázka.")
        if name.endswith("/") or method not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED):
            raise RemoteZipError("Položka je priečinok alebo má nepodporovanú kompresiu.")
        if flags & (0x1 | 0x40 | 0x2000):
            raise RemoteZipError("Šifrované položky archívu nie sú podporované.")
        if crc > 0xFFFFFFFF or offset + 30 > self.size:
            raise RemoteZipError("Hlavička položky je mimo archívu alebo má neplatné CRC.")
        guess = 30 + len(name.encode("utf-8")) + 256 + compressed
        data = self._read_at(offset, min(guess, self.size - offset))
        try:
            (signature, _version, local_flags, local_method, _time, _date,
             local_crc, local_compressed, local_size, name_length,
             extra_length) = struct.unpack("<4s5H3I2H", data[:30])
        except struct.error as exc:
            raise RemoteZipError("Lokálna hlavička ZIP je neúplná.") from exc
        if signature != b"PK\x03\x04" or local_flags != flags or local_method != method:
            raise RemoteZipError("Lokálna hlavička nezodpovedá centrálnemu indexu.")
        header_size = 30 + name_length + extra_length
        required = header_size + compressed
        if offset + required > self.size:
            raise RemoteZipError("Obsah položky presahuje koniec archívu.")
        if required > len(data):
            data += self._read_at(offset + len(data), required - len(data))
        try:
            local_name = data[30:30 + name_length].decode("utf-8" if flags & 0x800 else "cp437")
        except UnicodeError as exc:
            raise RemoteZipError("Názov lokálnej položky má neplatné kódovanie.") from exc
        if local_name != name:
            raise RemoteZipError("Názov lokálnej položky nezodpovedá indexu.")
        if not flags & 0x8:
            extra = data[30 + name_length:header_size]
            cursor = 0
            zip64 = None
            while cursor < len(extra):
                if len(extra) - cursor < 4:
                    raise RemoteZipError("Rozšírená hlavička ZIP je neúplná.")
                kind, amount = struct.unpack("<HH", extra[cursor:cursor + 4])
                cursor += 4
                if cursor + amount > len(extra):
                    raise RemoteZipError("Rozšírená hlavička ZIP má nesprávnu dĺžku.")
                if kind == 1:
                    zip64 = extra[cursor:cursor + amount]
                cursor += amount
            for field in ("size", "compressed"):
                value = local_size if field == "size" else local_compressed
                if value == 0xFFFFFFFF:
                    if zip64 is None or len(zip64) < 8:
                        raise RemoteZipError("Chýbajú údaje veľkosti ZIP64.")
                    value = struct.unpack("<Q", zip64[:8])[0]
                    zip64 = zip64[8:]
                    if field == "size":
                        local_size = value
                    else:
                        local_compressed = value
            if (local_crc, local_compressed, local_size) != (crc, compressed, size):
                raise RemoteZipError("Veľkosť alebo CRC lokálnej položky nezodpovedajú indexu.")
        payload = data[header_size:required]
        if method == zipfile.ZIP_STORED:
            result = payload
        else:
            try:
                decoder = zlib.decompressobj(-zlib.MAX_WBITS)
                result = decoder.decompress(payload, min(size, max_uncompressed) + 1)
                if (len(result) > size or not decoder.eof or decoder.unused_data
                        or decoder.unconsumed_tail):
                    raise RemoteZipError("Komprimovaný obsah je neúplný alebo prekračuje deklarovanú veľkosť.")
            except zlib.error as exc:
                raise RemoteZipError("Rozbalenie položky DEFLATE zlyhalo.") from exc
        if len(result) != size:
            raise RemoteZipError("Rozbalená položka nemá deklarovanú veľkosť.")
        if zlib.crc32(result) & 0xFFFFFFFF != crc:
            raise RemoteZipError("Kontrolný súčet CRC položky nesúhlasí.")
        return result
