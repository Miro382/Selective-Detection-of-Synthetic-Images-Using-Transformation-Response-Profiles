"""Pripnuté zdroje a spoločné kontroly dát V4; pôvodné súbory V3 nemení."""
from __future__ import annotations
import csv, gzip, hashlib, io, json, os, subprocess, sys, uuid
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parent
SEED = 20261015
INDEX_FILE = "v4_deepshield_index_v1.json.gz"
INDEX_SHA256 = "81986092ffc6c8e1b76455dad717fc04564389bca4d7d6f8556e0b9bed02d78e"
ARCHIVE_SIZE = 167573270584
ARCHIVE_URL = "https://zenodo.org/api/records/15648378/files/DeepShield_Dataset.zip/content"
ARCHIVE_MD5 = "f99ea49817e1641697166cfb8f53aa1b"
EPOCH8_SHA = "c4b54730991c9615d6e69f9505dd87fda4cb737db0f2f883a17811405b86ad0a"
DIFFUSION = ("FLUX.1", "StableDiffusion1.5", "StableDiffusion2", "StableDiffusion3", "StableDiffusionXL")
TARGETS = {"FFHQ": 6000, "FORLAB": 3000, "StyleGAN": 3000, "StyleGAN2": 3000,
           "StyleGAN3": 3000, **{name: 600 for name in DIFFUSION}}
CODE = ("v4_data_sources.py", "download_training_data_v4.py", "prepare_data_v4.py",
        "v4_deepshield_audit.py", "v4_deepshield_remote_zip.py", INDEX_FILE)

def now(): return datetime.now(timezone.utc).isoformat()
def read_json(path): return json.loads(Path(path).read_text(encoding="utf-8-sig"))
def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for data in iter(lambda: f.read(8*1024**2), b""): h.update(data)
    return h.hexdigest()
def atomic(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name("." + path.name + "." + uuid.uuid4().hex + ".part")
    tmp.write_bytes(data); os.replace(tmp, path)
def save_json(path, value):
    atomic(path, (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n").encode("utf-8"))
def rank(seed, value): return hashlib.sha256(f"v4:{seed}:{value}".encode()).hexdigest()
def ensure_python(project):
    """Use the explicitly selected interpreter in the release environment."""
    return None

def safe_path(root, relative):
    p = PurePosixPath(relative)
    if p.is_absolute() or not p.parts or any(x in (".", "..") or ":" in x or "\\" in x for x in p.parts):
        raise ValueError("Nebezpečná relatívna cesta: " + relative)
    out = Path(root).joinpath(*p.parts).resolve()
    if not out.is_relative_to(Path(root).resolve()): raise ValueError("Cesta opustila dátový adresár.")
    return out
def csv_rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as f: return list(csv.DictReader(f))
def write_csv(path, rows, fields):
    out = io.StringIO(newline=""); w = csv.DictWriter(out, fieldnames=fields, extrasaction="ignore")
    w.writeheader(); w.writerows(rows); atomic(path, out.getvalue().encode("utf-8"))
@contextmanager
def lock(path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    stream=path.open("a+b");locked=False
    try:
        stream.seek(0,os.SEEK_END)
        if stream.tell()==0:stream.write(b" ");stream.flush()
        stream.seek(0)
        try:
            if os.name=="nt":
                import msvcrt
                msvcrt.locking(stream.fileno(),msvcrt.LK_NBLCK,1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
            locked=True
        except OSError as e:raise ValueError(f"Rovnaká operácia už beží: {path}") from e
        stream.seek(0);stream.truncate();stream.write(json.dumps({"pid":os.getpid(),"created_utc":now()}).encode());stream.flush()
        yield
    finally:
        if locked:
            stream.seek(0)
            if os.name=="nt":
                import msvcrt
                msvcrt.locking(stream.fileno(),msvcrt.LK_UNLCK,1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(),fcntl.LOCK_UN)
        stream.close()
def anchor(path, expected=None, role="source"):
    path = Path(path).resolve(); actual = sha256(path)
    if expected is not None and actual != expected: raise ValueError("Zmenený uzamknutý zdroj: " + str(path))
    return {"path":str(path),"sha256":actual,"role":role}
def verify_anchors(anchors):
    for a in anchors: anchor(a["path"], a["sha256"])
def locked_json(path, sidecar):
    if sha256(path) != Path(sidecar).read_text(encoding="ascii").strip(): raise ValueError("Nesúhlasí hash protokolu: " + str(path))
    return read_json(path)
def load_index(path):
    with gzip.open(path,"rb") as f: data=f.read(64*1024**2+1)
    if len(data)>64*1024**2 or hashlib.sha256(data).hexdigest()!=INDEX_SHA256: raise ValueError("Zmenený index DeepShield.")
    entries=json.loads(data)
    if len(entries)!=130157: raise ValueError("Nesprávny index DeepShield.")
    return entries
def metadata(entry):
    p=PurePosixPath(entry["name"]); parts=p.parts
    if entry["name"].endswith("/") or len(parts)<2 or parts[1]!="PreSocial": return None
    if len(parts) not in (5,6) or parts[0]!="DeepShield" or p.suffix.lower() not in (".jpg",".png"): raise ValueError("Neznámy člen archívu.")
    kind,source=parts[2:4]; category=parts[4] if len(parts)==6 else "all"
    label=0 if kind=="Real" and source in ("FFHQ","FORLAB") else 1 if kind=="Fake" and source in TARGETS else None
    if label is None: raise ValueError("Neznáma trieda alebo zdroj.")
    family="real" if label==0 else "gan" if source.startswith("StyleGAN") else "diffusion"
    group="/".join(parts[2:-1])+"/"+p.stem
    dependency=("real/"+source if label==0 else "gan/"+source if family=="gan" else "dm/"+category)+"/"+p.stem
    return {"sample_id":"deepshield-v4/"+group,"group":group,"dependency_key":dependency,"source":"deepshield_v4",
            "generator":source,"category":category,"label":label,"family":family,
            "domain":"portrait" if source=="FFHQ" or family=="gan" or category=="faces" else "general",
            "platform":"PreSocial","relative_path":"images/"+"/".join(parts[1:]),"entry":entry}
def project_sources(project):
    """Chráni celý plán V3 vrátane vylúčených MPO, nielen prijatý test."""
    project=Path(project); ds=project/"deepshield_subset_v3"
    p=locked_json(ds/"protocol.json",ds/"protocol.sha256")
    complete=read_json(ds/"complete.json")
    anchors=[anchor(ds/"complete.json",role="protected_test")]
    for rel,digest in complete["files"].items(): anchors.append(anchor(safe_path(ds,rel),digest,"protected_test"))
    for rel,digest in p["locked_files"].items(): anchors.append(anchor(safe_path(ds,rel),digest,"protected_test"))
    verify_anchors(p["anchors"]); anchors.extend(p["anchors"])
    plan=read_json(ds/"selection.json")
    protected={r["group"] for r in plan}; dependencies={r["dependency_key"] for r in plan}
    parent=project/"selective_experiments/experiment_v3_20261014"
    pp=locked_json(parent/"protocol.json",parent/"protocol.sha256")
    if pp["best_epoch"]!=8 or pp["model_sha256"]!=EPOCH8_SHA: raise ValueError("V4 musí vychádzať zo zmrazeného detektora V3 z 8. epochy.")
    checkpoint=anchor(parent/"baseline/models/image/best_model.pt",EPOCH8_SHA,"initial_checkpoint")
    anchors.append(checkpoint)
    # Aj pôvodne nepodporované MPO chránime cez hlavnú fotografiu. Do nového
    # tréningového výberu sa MPO neprijímajú; toto je výlučne index zákazov.
    from v4_deepshield_audit import fingerprint
    excluded_fingerprints=[]
    for row in read_json(ds/"exclusions.json"):
        if row.get("rgb_sha256") and row.get("phash"):
            excluded_fingerprints.append(row);continue
        receipt=ds/"receipts"/(hashlib.sha256(row["sample_id"].encode()).hexdigest()+".json")
        anchors.append(anchor(receipt,role="protected_test_receipt"))
        recorded=read_json(receipt); path=safe_path(ds,row["relative_path"])
        fp=fingerprint(path,allow_mpo_primary=True)
        if fp["sha256"]!=recorded["sha256"]:raise ValueError("Zmenený vylúčený obrázok testu V3.")
        excluded_fingerprints.append({"sample_id":row["sample_id"],**fp})
    return {"anchors":anchors,"protected_groups":sorted(protected),"protected_dependencies":sorted(dependencies),
            "initial_checkpoint":checkpoint,"deep_v3":str(ds.resolve()),"environment":pp["environment"],
            "protected_exclusion_fingerprints":excluded_fingerprints}
def historical_index(source):
    from v4_deepshield_audit import ExistingFingerprintIndex
    paths=[Path(a["path"]) for a in source["anchors"] if a["role"]=="fingerprint_inventory"]
    paths.append(Path(source["deep_v3"])/"manifest.csv")
    idx=ExistingFingerprintIndex(paths)
    for r in source["protected_exclusion_fingerprints"]:
        if r.get("sha256"): idx._index.add(r,"protected_deepshield_v3/"+r["sample_id"])
    return idx
def make_plan(entries, protected_groups, protected_dependencies, seed=SEED, reserve=1.25):
    pools=defaultdict(list)
    for e in entries:
        r=metadata(e)
        if r and r["group"] not in protected_groups and r["dependency_key"] not in protected_dependencies: pools[r["generator"]].append(r)
    result=[]; used=set(protected_dependencies)
    # V rámci zdroja sa kategórie striedajú; difúzne category/stem sa neopakujú medzi modelmi.
    for source in sorted(TARGETS,key=lambda x:rank(seed,x)):
        cats=defaultdict(list)
        for row in pools[source]: cats[row["category"]].append(row)
        for c in cats: cats[c].sort(key=lambda r:rank(seed,r["group"]))
        chosen=[]; required=int(TARGETS[source]*reserve)
        keys=sorted(cats,key=lambda x:rank(seed,source+x)); positions=dict.fromkeys(keys,0)
        while len(chosen)<required:
            added=False
            for c in keys:
                while positions[c]<len(cats[c]):
                    r=cats[c][positions[c]];positions[c]+=1
                    if r["dependency_key"] in used: continue
                    chosen.append(r);used.add(r["dependency_key"]);added=True;break
                if len(chosen)==required: break
            if not added: raise ValueError("Nedostatok odlišných kandidátov pre "+source)
        result.extend(chosen)
    return result
def verify_source_remote():
    import requests
    with requests.get("https://zenodo.org/api/records/15648378",stream=True,timeout=(15,60)) as response:
        response.raise_for_status(); data=bytearray()
        for part in response.iter_content(65536):
            data.extend(part)
            if len(data)>2*1024**2: raise ValueError("Príliš veľké metadáta Zenodo.")
    doc=json.loads(data); files=doc.get("files",[])
    if len(files)!=1 or files[0].get("size")!=ARCHIVE_SIZE or files[0].get("checksum")!="md5:"+ARCHIVE_MD5:
        raise ValueError("DeepShield archív sa zmenil.")
    return {"url":ARCHIVE_URL,"doi":"10.5281/zenodo.15648378","size":ARCHIVE_SIZE,"md5":ARCHIVE_MD5,"checked_utc":now()}
