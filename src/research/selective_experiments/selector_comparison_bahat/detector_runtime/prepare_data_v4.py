"""Audit, import CelebA a oddelenie dát V4 pred tréningom.

CelebA: pôvodné In-The-Wild Images a oficiálny identity_CelebA.txt získaný
od autorov (https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html). Obrazy nesmú
byť bežné nízkorozlíšené Align&Cropped 178x218. Vyberá sa jeden obrázok
na identitu s oboma rozmermi aspoň 512 px; bez odhadovania identity z tváre.
"""
from __future__ import annotations
import argparse, hashlib, os, sys
from collections import Counter, defaultdict
from pathlib import Path
from v4_data_sources import (ROOT, SEED, CODE, EPOCH8_SHA, TARGETS, atomic,
    read_json, save_json, sha256, rank, safe_path, csv_rows, write_csv,
    anchor, verify_anchors, locked_json, now, historical_index, lock, ensure_python)

FIELDS=["sample_id","path","label","source","generator","group","sha256","rgb_sha256","phash",
        "width","height","size_bytes","mtime_ns","split","family","domain","dependency_key","platform"]
INT_FIELDS=("label","width","height","size_bytes","mtime_ns")
SPLIT_COUNTS={"train":12000,"val":2000,"detector_calibration":1000,"selector_train":5000,
              "selector_selection":2000,"selector_calibration":2000}

def load_manifest(path:Path)->list[dict]:
    rows=csv_rows(path)
    for r in rows:
        if set(r)!=set(FIELDS): raise ValueError("Nesprávna schéma manifestu: "+str(path))
        for f in INT_FIELDS: r[f]=int(r[f])
        if r["label"] not in (0,1) or min(r["width"],r["height"],r["size_bytes"])<=0: raise ValueError("Neplatná trieda alebo rozmery.")
        for f,n in (("sha256",64),("rgb_sha256",64),("phash",16)):
            if f=="phash" and not r[f] and r["source"].startswith("v3/"): continue
            if len(r[f])!=n or any(c not in "0123456789abcdef" for c in r[f]): raise ValueError("Neplatný odtlačok "+f)
        if r["split"] not in SPLIT_COUNTS or r["family"] not in ("real","gan","diffusion","historical_unknown"):
            raise ValueError("Neplatná množina alebo rodina.")
        if r["domain"] not in ("portrait","general","unknown") or not r["dependency_key"]: raise ValueError("Chýbajúca doména alebo závislosť.")
        if not Path(r["path"]).is_absolute(): raise ValueError("Manifest musí obsahovať absolútne cesty.")
    return rows

def check_disjoint(rows):
    ids=set(); paths=set(); groups={}; hashes={"sha256":{},"rgb_sha256":{}}
    for r in rows:
        key=os.path.normcase(os.path.abspath(r["path"]))
        if key in paths or r["sample_id"] in ids: raise ValueError("Opakovaný obrázok alebo sample_id.")
        ids.add(r["sample_id"]); paths.add(key)
        for name in ("group","dependency_key"):
            g=(name,r[name]); prior=groups.setdefault(g,r["split"])
            if prior!=r["split"]: raise ValueError("Závislé predlohy zasahujú do viacerých množín: "+str(g))
        for name,index in hashes.items():
            prior=index.setdefault(r[name],r)
            if prior["label"]!=r["label"]: raise ValueError("Konfliktné triedy rovnakého obsahu.")
            if prior["split"]!=r["split"]: raise ValueError("Presná duplicita medzi dátovými množinami.")

def split_new(rows, counts=None, seed=SEED):
    counts=SPLIT_COUNTS if counts is None else counts
    if len(rows)!=sum(counts.values()): raise ValueError("Počet nových predlôh nezodpovedá protokolu.")
    if len({r["dependency_key"] for r in rows})!=len(rows): raise ValueError("Nový súbor musí obsahovať jednu nezávislú evidovanú predlohu na dependency_key.")
    ordered=sorted(rows,key=lambda r:rank(seed,"split/"+r["dependency_key"]))
    offset=0; result=[]
    for split,n in counts.items():
        result.extend({**r,"split":split} for r in ordered[offset:offset+n]);offset+=n
    return result

def read_identities(path):
    result={}
    for lineno,line in enumerate(Path(path).read_text(encoding="utf-8-sig").splitlines(),1):
        if not line.strip():continue
        parts=line.split()
        if len(parts)!=2 or Path(parts[0]).name!=parts[0] or not parts[1].isdigit() or int(parts[1])<=0:
            raise ValueError(f"Neplatný riadok identít CelebA {lineno}.")
        if parts[0] in result: raise ValueError("Opakovaný názov v anotácii CelebA.")
        result[parts[0]]=parts[1]
    if not result:raise ValueError("Súbor identít CelebA je prázdny.")
    return result

def collect_celeba(folder, identities_file, cache, historical, existing, count=3000, min_side=512):
    from v4_deepshield_audit import fingerprint, _MemoryIndex
    identities=read_identities(identities_file); grouped=defaultdict(list); seen=set()
    for path in sorted(Path(folder).rglob("*")):
        if not path.is_file() or path.suffix.lower() not in (".jpg",".jpeg",".png"):continue
        if path.name not in identities: raise ValueError("Obrázok CelebA nemá známu identitu: "+str(path))
        if path.name in seen: raise ValueError("Opakovaný názov súboru CelebA v dvoch adresároch.")
        seen.add(path.name);grouped[identities[path.name]].append(path.resolve())
    if len(grouped)<count: raise ValueError(f"CelebA má iba {len(grouped)} dostupných identít; vyžaduje sa {count}.")
    index=_MemoryIndex()
    for r in existing:index.add(r,r["sample_id"])
    accepted=[];excluded=[];cache=Path(cache);cache.mkdir(parents=True,exist_ok=True)
    for identity in sorted(grouped,key=lambda x:rank(SEED,"celeba/"+x)):
        for path in sorted(grouped[identity],key=lambda x:rank(SEED,x.name)):
            receipt=cache/(hashlib.sha256(str(path).encode()).hexdigest()+".json")
            try:
                stat=path.stat()
                if receipt.exists():
                    old=read_json(receipt)
                    if old["path"]!=str(path) or old["fingerprint"]["sha256"]!=sha256(path): raise ValueError("Zmenený už auditovaný obrázok CelebA: "+str(path))
                    fp=old["fingerprint"]
                else:
                    try:fp=fingerprint(path)
                    except ValueError as e:
                        excluded.append({"path":str(path),"identity":identity,"reasons":[{"kind":"unsupported_image","detail":str(e)}]});continue
                    save_json(receipt,{"path":str(path),"fingerprint":fp})
                reasons=historical.matches(fp)+index.matches(fp)
                if min(fp["width"],fp["height"])<min_side: reasons.append({"kind":"low_native_resolution","min_side":min_side})
                if reasons:
                    excluded.append({"path":str(path),"identity":identity,"reasons":reasons});continue
                group="celeba/identity/"+identity
                row={"sample_id":"celeba-v4/"+path.name,"path":str(path),"source":"celeba_v4","generator":"CelebA",
                    "group":group,"dependency_key":group,"label":0,"family":"real","domain":"portrait",
                    "platform":"Original","mtime_ns":stat.st_mtime_ns,**fp}
                accepted.append(row);index.add(row,row["sample_id"]);break
            except (OSError,RuntimeError) as e:
                excluded.append({"path":str(path),"identity":identity,"reasons":[{"kind":"invalid_image","detail":str(e)}]})
        if len(accepted)%100==0 and accepted:print(f"CelebA: {len(accepted)}/{count} identít",flush=True)
        if len(accepted)==count:return accepted,excluded
    save_json(cache.parent/"celeba_insufficient.json",{"accepted":len(accepted),"required":count,"exclusions":excluded})
    raise ValueError(f"Po kontrole duplicít a natívneho rozlíšenia >= {min_side}px zostalo iba {len(accepted)}/{count} identít CelebA. Nedopĺňajú sa nízkorozlíšenými orezmi ani náhradným zdrojom.")

def historic_rows(project, sources):
    base=Path(project)/"detector_experiments/detector_v3_20261013/data"
    p=locked_json(base/"data_protocol.json",base/"data_protocol.sha256")
    output=[]
    for split in ("train","val"):
        anchor(base/(split+".csv"),p["manifests"][split+".csv"])
        for r in csv_rows(base/(split+".csv")):
            for f in INT_FIELDS:r[f]=int(r[f])
            original_source=r["source"]
            r.update({"source":"v3/"+original_source,"phash":"","platform":"Original",
                "family":"real" if r["label"]==0 else "historical_unknown" if original_source=="historical" else "diffusion",
                "domain":"unknown" if original_source=="historical" else "general",
                "group":"v3/"+r["group"],"split":split})
            r["dependency_key"]=r["group"];output.append(r)
    return output

def verify_dataset(data_dir:Path, full_hashes=False)->dict:
    data_dir=Path(data_dir)
    p=locked_json(data_dir/"data_protocol.json",data_dir/"data_protocol.sha256")
    if p.get("version")!=4 or p.get("schema_version")!=4:raise ValueError("Vyžaduje sa dátový protokol V4.")
    complete=read_json(data_dir/"complete.json")
    if complete["protocol_sha256"]!=sha256(data_dir/"data_protocol.json"):raise ValueError("Príprava dát nie je dokončená.")
    for name,digest in complete["files"].items():anchor(safe_path(data_dir,name),digest)
    verify_anchors(p["source_anchors"])
    anchor(p["initial_checkpoint"]["path"],EPOCH8_SHA)
    if p["initial_checkpoint"]["sha256"]!=EPOCH8_SHA:raise ValueError("Nesprávny počiatočný checkpoint.")
    all_rows=[]
    for name,digest in p["manifests"].items():
        anchor(safe_path(data_dir,name),digest);rows=load_manifest(data_dir/name)
        if len(rows)!=p["counts"][name[:-4]]["total"]:raise ValueError("Nesúhlasí počet riadkov.")
        for r in rows:
            if r["split"]!=name[:-4]:raise ValueError("Nesprávne označenie splitu v manifeste.")
            stat=Path(r["path"]).stat()
            if stat.st_size!=r["size_bytes"] or stat.st_mtime_ns!=r["mtime_ns"]:raise ValueError("Zmenený obrázok od prípravy: "+r["path"])
            if full_hashes and sha256(r["path"])!=r["sha256"]:raise ValueError("Zmenený obsah obrázka: "+r["path"])
        all_rows.extend(rows)
    check_disjoint(all_rows)
    return p

def prepare(project, cache, output, celeba_dir, celeba_identities):
    from download_training_data_v4 import check_download
    if (output/"complete.json").exists():return verify_dataset(output,full_hashes=True)
    source_protocol,deep_rows=check_download(cache,full_hashes=True)
    if not Path(celeba_dir).is_dir() or not Path(celeba_identities).is_file():
        raise ValueError("Chýba druhý zdroj reálnych portrétov CelebA. Dodajte pôvodné In-The-Wild obrázky (--celeba-dir) a oficiálny identity_CelebA.txt (--celeba-identities). Štandardné 178x218 orezané obrázky nevyhovujú. https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html")
    sources=source_protocol["sources"];historical=historical_index(sources)
    deep=[{**r,"path":str(safe_path(cache,r["relative_path"]))} for r in deep_rows]
    print("Audit druhého zdroja portrétov; jedna fotografia na identitu...",flush=True)
    celeba,excluded=collect_celeba(celeba_dir,celeba_identities,cache/"celeba_fingerprints",historical,deep)
    new=split_new(deep+celeba)
    previous=historic_rows(project,sources);rows=previous+new
    check_disjoint(rows)
    # Pred zmrazením protokolu sa preverí aj obsah všetkých pôvodných tréningových obrázkov.
    for i,r in enumerate(previous,1):
        if i%10000==0:print(f"Kontrola pôvodných tréningových dát V3: {i}/{len(previous)}",flush=True)
        stat=Path(r["path"]).stat()
        if stat.st_size!=r["size_bytes"] or stat.st_mtime_ns!=r["mtime_ns"] or sha256(r["path"])!=r["sha256"]:
            raise ValueError("Zmenený pôvodný obrázok: "+r["path"])
    output.mkdir(parents=True,exist_ok=True)
    counts={};manifests={}
    for split in SPLIT_COUNTS:
        subset=sorted((r for r in rows if r["split"]==split),key=lambda r:r["sample_id"])
        write_csv(output/(split+".csv"),subset,FIELDS);manifests[split+".csv"]=sha256(output/(split+".csv"))
        counts[split]={"total":len(subset),"new":sum(not r["source"].startswith("v3/") for r in subset),
            "class_counts":dict(Counter(str(r["label"]) for r in subset)),"source_counts":dict(Counter(r["generator"] for r in subset))}
    save_json(output/"celeba_exclusions.json",excluded)
    snapshot=output/"code";snapshot.mkdir(exist_ok=True)
    for name in CODE:atomic(snapshot/name,(ROOT/name).read_bytes())
    source_anchors=sources["anchors"]+[anchor(cache/"download_protocol.json"),anchor(cache/"download_complete.json"),
        anchor(cache/"accepted.json"),anchor(celeba_identities,role="celeba_identities")]
    p={"version":4,"schema_version":4,"created_utc":now(),"seed":SEED,"project_root":str(Path(project).resolve()),
       "initial_checkpoint":sources["initial_checkpoint"],"environment":sources["environment"],
       "class_to_index":{"real":0,"fake":1},"manifests":manifests,"counts":counts,"source_anchors":source_anchors,
       "new_split_targets":SPLIT_COUNTS,"new_source_counts":dict(Counter(r["generator"] for r in new)),
       "group_allocation":"Jedna evidovaná predloha na dependency_key; globálna náhodná permutácia pevným seed bez stratifikácie podľa výsledku. CelebA jedna fotografia na identitu.",
       "protected_v3_groups":sources["protected_groups"],"protected_v3_dependencies":sources["protected_dependencies"],
       "test_only":False,"confirmatory":False,"external_test_required":True,
       "training_allowed_splits":["train"],"model_selection_allowed_splits":["val"],
       "full_image_hashes_checked_at_preparation":True,
       "limitations":["DeepShield V3 už motivoval tento vývoj; V4 vyžaduje nový uzamknutý externý test.",
           "Staré train/val zostávajú iba v rovnakých vývojových úlohách; staré selektorové a testovacie dáta sa nepridávajú.",
           "SHA/RGB/pHash<=4 je filter dostupných prekryvov, nie dôkaz nezávislosti od neznámych dát alebo identít.",
           "FFHQ a FORLAB v DeepShield neobsahujú identifikátory známych osôb či fotografických sérií; neznáme závislosti zostávajú možné.",
           "Globálne náhodné rozdelenie približuje zdrojovú zmes; nepreukazuje iid ani univerzálnu platnosť rizikovej záruky.",
           "CelebA sa obmedzuje na natívne rozlíšenie >=512 px a jednu fotografiu na identitu; ide o vybranú podpopuláciu."]}
    save_json(output/"data_protocol.json",p);atomic(output/"data_protocol.sha256",(sha256(output/"data_protocol.json")+"\n").encode())
    files={**manifests,"celeba_exclusions.json":sha256(output/"celeba_exclusions.json"),
           **{"code/"+name:sha256(snapshot/name) for name in CODE}}
    save_json(output/"complete.json",{"version":4,"finished_utc":now(),"protocol_sha256":sha256(output/"data_protocol.json"),"files":files})
    print("Dáta V4 pripravené: 24000 nových predlôh + pôvodné povolené train/val V3.",flush=True)
    return p

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project-root",type=Path,default=ROOT);p.add_argument("--cache-dir",type=Path);p.add_argument("--data-dir",type=Path)
    p.add_argument("--celeba-dir",type=Path);p.add_argument("--celeba-identities",type=Path)
    p.add_argument("--check",action="store_true");p.add_argument("--full-hashes",action="store_true")
    a=p.parse_args();ensure_python(a.project_root);cache=(a.cache_dir or a.project_root/"training_data_v4").resolve()
    output=(a.data_dir or a.project_root/"detector_experiments/detector_v4/data").resolve()
    if a.check:verify_dataset(output,a.full_hashes);print("Dátový protokol V4: OK");return
    celeba=a.celeba_dir or cache/"celeba/img_celeba"; identities=a.celeba_identities or cache/"celeba/identity_CelebA.txt"
    output.mkdir(parents=True,exist_ok=True)
    with lock(output/".prepare.lock"):prepare(a.project_root,cache,output,celeba,identities)
# Release entry point: require an explicit research-job request.
if __name__ == '__main__':
    import sys as _release_sys
    if not any(arg in _release_sys.argv for arg in ('--help', '-h', '--execute-research')):
        print('Research source. Read src/RESEARCH_CODE.txt before running a job. '
              'Use --help to inspect options or --execute-research to run explicitly.')
        raise SystemExit(0)
    if '--execute-research' in _release_sys.argv:
        _release_sys.argv.remove('--execute-research')

if __name__=="__main__":
    try:main()
    except KeyboardInterrupt:print("Prerušené. Auditované odtlačky zostali uložené; zopakujte príkaz.",file=sys.stderr);sys.exit(130)
    except Exception as e:print("Chyba:",e,file=sys.stderr);sys.exit(2)
