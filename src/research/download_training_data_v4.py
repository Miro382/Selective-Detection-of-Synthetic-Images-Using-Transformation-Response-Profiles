"""Stiahne iba vybrané originály DeepShield pre vývoj V4; nikdy celý ZIP.

Run bez parametrov obnoví sťahovanie. --plan-only vytvorí plán bez prenosu
obrázkov; --max-new-files N obmedzí jeden beh. CelebA sa importuje lokálne
v prepare_data_v4.py s oficiálnym súborom identít. Žiadny model sa nespúšťa.
"""
from __future__ import annotations
import argparse, hashlib, os, shutil, sys, zlib
from collections import Counter
from pathlib import Path
from v4_data_sources import (ROOT, CODE, SEED, TARGETS, INDEX_FILE, ARCHIVE_URL,
    ARCHIVE_SIZE, atomic, save_json, read_json, sha256, rank, safe_path, anchor,
    verify_anchors, lock, now, load_index, project_sources, make_plan,
    historical_index, verify_source_remote, ensure_python)

def prepare_plan(project, output, max_gib=80.0):
    output=Path(output).resolve(); output.mkdir(parents=True,exist_ok=True)
    pp=output/"download_protocol.json"; planpath=output/"download_plan.json"
    if pp.exists():
        p=read_json(pp)
        if sha256(pp)!=(output/"download_protocol.sha256").read_text().strip(): raise ValueError("Zmenený plán V4.")
        if p["project_root"]!=str(Path(project).resolve()) or p["max_gib"]!=max_gib: raise ValueError("Parametre uzamknutého plánu nemožno meniť.")
        verify_anchors(p["sources"]["anchors"])
        for name,digest in p["code"].items(): anchor(ROOT/name,digest)
        anchor(planpath,p["plan_sha256"])
        return p,read_json(planpath)
    if any(output.iterdir()): raise ValueError("Priečinok obsahuje súbory bez platného protokolu: "+str(output))
    source=project_sources(project)
    plan=make_plan(load_index(ROOT/INDEX_FILE),set(source["protected_groups"]),set(source["protected_dependencies"]))
    planned=sum(r["entry"]["compressed"] for r in plan)
    if max_gib<=0 or planned>max_gib*2**30: raise ValueError(f"Plán vrátane rezervy potrebuje {planned/2**30:.2f} GiB, viac než --max-gib {max_gib}.")
    save_json(planpath,plan)
    p={"version":4,"created_utc":now(),"project_root":str(Path(project).resolve()),"seed":SEED,
       "targets":TARGETS,"candidate_reserve":1.25,"plan_sha256":sha256(planpath),"max_gib":max_gib,
       "compressed_candidate_bytes":planned,"candidate_disk_bytes":sum(r["entry"]["size"] for r in plan),
       "sources":source,"code":{name:sha256(ROOT/name) for name in CODE},
       "selection":"Deterministické poradie kategórií, prijatie prvých platných jedinečných predlôh až po kvótu. Bez predikcií.",
       "second_portrait_source":{"name":"CelebA","count":3000,"state":"required_local_import_with_identities",
          "official_url":"https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html"},
       "limitations":["V3 DeepShield bol už vyhodnotený; nové dáta z rovnakej kolekcie sú vývojové.",
           "FFHQ a FORLAB nemajú v tomto archíve spoľahlivé identifikátory identity alebo pôvodnej série.",
           "Kontrola SHA/RGB/pHash pokrýva iba dostupné evidencie, nie neznáme predtrénovacie dáta.",
           "Výber je len PreSocial; všetky sociálne varianty chráneného testu V3 zostávajú vylúčené."]}
    save_json(pp,p);atomic(output/"download_protocol.sha256",(sha256(pp)+"\n").encode())
    return p,plan

def receipt_path(output,row): return output/"receipts"/(hashlib.sha256(row["sample_id"].encode()).hexdigest()+".json")

def validate_receipt(output,row,receipt):
    if receipt.get("sample_id")!=row["sample_id"] or receipt.get("entry")!=row["entry"]: raise ValueError("Potvrdenie nesúhlasí s plánom.")
    path=safe_path(output,row["relative_path"])
    if not path.is_file() or path.stat().st_size!=row["entry"]["size"] or sha256(path)!=receipt.get("sha256"):
        raise ValueError("Stiahnutý súbor bol zmenený: "+str(path))
    if receipt.get("decision") not in ("accepted","excluded"): raise ValueError("Neplatná kontrola obrázka.")
    if receipt["decision"]=="accepted" and receipt["fingerprint"]["sha256"]!=receipt["sha256"]: raise ValueError("Nesúhlasí odtlačok.")
    return path

def check_download(output, full_hashes=True):
    output=Path(output);p=read_json(output/"download_protocol.json")
    if sha256(output/"download_protocol.json")!=(output/"download_protocol.sha256").read_text().strip(): raise ValueError("Zmenený protokol sťahovania.")
    verify_anchors(p["sources"]["anchors"])
    anchor(output/"download_plan.json",p["plan_sha256"])
    for name,digest in p["code"].items(): anchor(ROOT/name,digest)
    complete=read_json(output/"download_complete.json")
    if complete["protocol_sha256"]!=sha256(output/"download_protocol.json"): raise ValueError("Neplatné dokončenie sťahovania.")
    for name,digest in complete["files"].items(): anchor(safe_path(output,name),digest)
    rows=read_json(output/"accepted.json")
    if Counter(r["generator"] for r in rows)!=Counter(TARGETS): raise ValueError("Chýba povinný zdroj alebo počet.")
    ids=set();deps=set()
    for r in rows:
        if r["sample_id"] in ids or r["dependency_key"] in deps: raise ValueError("Opakovaná predloha vo výslednom výbere.")
        ids.add(r["sample_id"]);deps.add(r["dependency_key"])
        path=safe_path(output,r["relative_path"]); stat=path.stat()
        if stat.st_size!=r["size_bytes"]: raise ValueError("Zmenená veľkosť obrázka.")
        if full_hashes and sha256(path)!=r["sha256"]: raise ValueError("Zmenený obrázok: "+str(path))
    return p,rows

def download(output, protocol, plan, max_new_files=None):
    from v4_deepshield_remote_zip import RemoteZip
    from v4_deepshield_audit import fingerprint, _MemoryIndex
    output=Path(output)
    if max_new_files is not None and max_new_files<=0: raise ValueError("--max-new-files musí byť kladné.")
    if (output/"download_complete.json").exists(): return check_download(output)
    print("Indexujem historické odtlačky a chránený test V3...",flush=True)
    historical=historical_index(protocol["sources"]); internal=_MemoryIndex()
    accepted=[]; excluded=[];counts=Counter();new=0; remote=None
    if shutil.disk_usage(output).free < 256*1024**2: raise ValueError("Nedostatok voľného miesta.")
    try:
        for row in plan:
            if counts[row["generator"]]>=TARGETS[row["generator"]]: continue
            receiptfile=receipt_path(output,row); path=safe_path(output,row["relative_path"])
            if receiptfile.exists():
                receipt=read_json(receiptfile);validate_receipt(output,row,receipt)
                fp=receipt.get("fingerprint")
            else:
                if max_new_files is not None and new>=max_new_files:
                    print(f"Čiastkový beh: prijaté {len(accepted)}/21000. Opakujte Run pre pokračovanie.",flush=True);return None
                if remote is None:
                    save_json(output/"remote_source.json",verify_source_remote())
                    remote=RemoteZip(ARCHIVE_URL,ARCHIVE_SIZE,max_total_bytes=int(protocol["max_gib"]*2**30))
                if shutil.disk_usage(output).free < row["entry"]["size"]+128*1024**2: raise ValueError("Nedostatok miesta na nasledujúci obrázok.")
                print(f"[{len(accepted)+1}/21000] {row['generator']} {row['group']}",flush=True)
                data=path.read_bytes() if path.is_file() and path.stat().st_size==row["entry"]["size"] else None
                if data is None or zlib.crc32(data)&0xffffffff!=row["entry"]["crc"]:
                    data=remote.read_member(row["entry"]);atomic(path,data)
                fp=None; reasons=[]
                try:
                    fp=fingerprint(path)
                    reasons=historical.matches(fp)+internal.matches(fp)
                except (ValueError,OSError,RuntimeError) as e: reasons=[{"kind":"unsupported_image","detail":str(e)}]
                receipt={"sample_id":row["sample_id"],"entry":row["entry"],"sha256":sha256(path),"fingerprint":fp,
                         "decision":"excluded" if reasons else "accepted","reasons":reasons,"checked_utc":now()}
                save_json(receiptfile,receipt);new+=1
            if receipt["decision"]=="accepted":
                # Kontrola rozhodnutia pri obnove v tom istom deterministickom poradí.
                if historical.matches(fp) or internal.matches(fp): raise ValueError("Obnovené prijatie je v konflikte s odtlačkami.")
                record={**row,**fp,"mtime_ns":path.stat().st_mtime_ns}
                accepted.append(record);internal.add(fp,row["sample_id"]);counts[row["generator"]]+=1
            else: excluded.append({**row,"sha256":receipt["sha256"],"reasons":receipt["reasons"]})
        if counts!=Counter(TARGETS):
            save_json(output/"insufficient_candidates.json",{"accepted":dict(counts),"targets":TARGETS,"excluded":len(excluded)})
            raise ValueError("Rezervní kandidáti nestačili po audite. Experiment sa nevytvorí; treba nový plán, nie dopĺňať podľa výsledkov modelu.")
        save_json(output/"accepted.json",accepted);save_json(output/"exclusions.json",excluded)
        save_json(output/"download_complete.json",{"version":4,"finished_utc":now(),"counts":dict(counts),
            "protocol_sha256":sha256(output/"download_protocol.json"),
            "files":{name:sha256(output/name) for name in ("accepted.json","exclusions.json","download_plan.json")},
            "dataset_ready":False,"remaining":"Import 3000 pôvodných CelebA obrázkov so známou identitou a vytvorenie delenia."})
        print("DeepShield V4: 21000 overených originálov. Nasleduje prepare_data_v4.py s CelebA a identitami.",flush=True)
        return protocol,accepted
    finally:
        if remote is not None: remote.close()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project-root",type=Path,default=ROOT);p.add_argument("--output-dir",type=Path)
    p.add_argument("--plan-only",action="store_true");p.add_argument("--check",action="store_true")
    p.add_argument("--max-new-files",type=int);p.add_argument("--max-gib",type=float,default=80.)
    a=p.parse_args();ensure_python(a.project_root);output=(a.output_dir or a.project_root/"training_data_v4").resolve()
    # Zámok je mimo výstupu, aby chránil aj prvé vytvorenie plánu bez
    # porušenia požiadavky na prázdny cieľový adresár. OS ho uvoľní po páde.
    with lock(output.parent/("."+output.name+".download.lock")):
        protocol,plan=prepare_plan(a.project_root,output,a.max_gib)
        print(f"Plán: {len(plan)} kandidátov vrátane rezervy; najviac {protocol['compressed_candidate_bytes']/2**30:.2f} GiB komprimovaných obrázkov.",flush=True)
        if a.plan_only:return
        if a.check:check_download(output);print("Kontrola sťahovania: OK");return
        download(output,protocol,plan,a.max_new_files)
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
    except KeyboardInterrupt:print("Prerušené. Rovnaký príkaz pokračuje po poslednom uloženom obrázku.",file=sys.stderr);sys.exit(130)
    except Exception as e:print("Chyba:",e,file=sys.stderr);sys.exit(2)
