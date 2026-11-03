"""Uzamknutá externá podmnožina Community Forensics pre zmrazenú V4.

Výber nepoužíva výsledky detektora. RAISE, FFHQ a ImageNet sú vylúčené.
SHA/RGB/pHash audit nepreukazuje nezávislosť od nezdokumentovaných dát.
"""
from __future__ import annotations
import csv, gzip, hashlib, io, json, math, os, re, sys, time, uuid
from collections import Counter, defaultdict
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent
REPO = 'OwensLab/CommunityForensics-Eval'
REVISION = '7d4a74a88d2cac93b513c0853bf92c260eaceea0'
INDEX_FILE = 'external_v4_cf_index.json.gz'
INDEX_SHA256 = '7a8857a5b87fbdb08509d238d22bead517d9fd6fcf097a0cb5244437094e7a85'
SEED = 20261016
FAKE_MODELS = ('DFGAN','Dalle3','FLUX-dev','FLUX-schnell','Firefly_Image3','IdeogramV2','Imagen3','MidjourneyV6_1')
TARGETS = {'real/coco':500,'real/LAION':1500,**{'fake/'+x:250 for x in FAKE_MODELS}}
CODE = ('external_v4_data.py','download_external_data_v4.py',INDEX_FILE,'v4_deepshield_audit.py','v4_data_sources.py')
FIELDS = ['sample_id','path','label','group','source','generator','family','domain','sha256','rgb_sha256','phash',
          'width','height','size_bytes','mtime_ns','bucket','real_source','paired_generator','source_image_name',
          'prompt_sha256','source_revision','shard','row_group','row_index','platform']
INT_FIELDS = ('label','width','height','size_bytes','mtime_ns','row_group','row_index')

def now():return datetime.now(timezone.utc).isoformat()
def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def read_json(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def atomic(path,body):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name('.'+path.name+'.'+uuid.uuid4().hex+'.tmp');tmp.write_bytes(body);os.replace(tmp,path)
def save_json(path,value):atomic(path,(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode())
def save_receipt(path,record):save_json(path,{'record':record,'record_sha256':digest(record)})
def read_receipt(path):
    wrapper=read_json(path)
    if set(wrapper)!={'record','record_sha256'} or digest(wrapper['record'])!=wrapper['record_sha256']:raise ValueError('Poškodené potvrdenie auditu: '+str(path))
    return wrapper['record']
def safe_path(root,relative):
    if not isinstance(relative,str) or not relative or '\\' in relative or ':' in relative:raise ValueError('Neplatná relatívna cesta.')
    p=Path(relative)
    if p.is_absolute() or '..' in p.parts:raise ValueError('Cesta opúšťa dátový priečinok.')
    result=(Path(root)/p).resolve()
    if not result.is_relative_to(Path(root).resolve()):raise ValueError('Cesta opúšťa dátový priečinok.')
    return result
def anchor(path):return {'path':str(Path(path).resolve()),'sha256':sha256(path)}
def verify_anchors(anchors):
    for item in anchors:
        if sha256(item['path'])!=item['sha256']:raise ValueError('Zmenený chránený súbor: '+item['path'])
def verify_migrations(data_dir,protocol):
    """Preserve and verify the old selection contract after a transport-only fix."""
    data_dir=Path(data_dir)
    migrations=protocol.get('transport_migrations',[])
    if not isinstance(migrations,list) or len(migrations)>1:raise ValueError('Nepodporovaná história migrácií prenosu.')
    for entry in migrations:
        if set(entry)!={'record','sha256'} or entry['record']!='transport_migrations/v1/migration.json':raise ValueError('Neplatný záznam migrácie.')
        path=safe_path(data_dir,entry['record'])
        if sha256(path)!=entry['sha256']:raise ValueError('Zmenený záznam migrácie prenosu.')
        record=read_json(path);prefix='transport_migrations/v1/'
        required={prefix+'protocol.json',prefix+'protocol.sha256',prefix+'receipt_hashes.json',prefix+'migration_tool.py',*(prefix+'code/'+n for n in CODE)}
        if record.get('version')!='external-v4-transport-migration-1' or not required<=set(record.get('files',{})):raise ValueError('Neúplný archív migrácie.')
        for name,h in record['files'].items():
            if not name.startswith(prefix) or sha256(safe_path(data_dir,name))!=h:raise ValueError('Zmenený archív migrácie: '+name)
        old_path=data_dir/prefix/'protocol.json';old=read_json(old_path)
        if sha256(old_path)!=(data_dir/prefix/'protocol.sha256').read_text().strip() or sha256(old_path)!=record['old_protocol_sha256']:raise ValueError('Zmenený pôvodný protokol migrácie.')
        if old.get('transport_migrations') or old['plan_sha256']!=protocol['plan_sha256'] or old['plan_sha256']!=record['plan_sha256']:raise ValueError('Migrácia zmenila výberový plán.')
        unchanged=lambda p:{k:v for k,v in p.items() if k not in ('code','transport_migrations','transport_note')}
        if unchanged(old)!=unchanged(protocol):raise ValueError('Migrácia zmenila pôvodný dátový protokol.')
        if set(old['code'])!=set(CODE) or set(protocol['code'])!=set(CODE):raise ValueError('Nesprávny zoznam migrovaného kódu.')
        for name,h in old['code'].items():
            if sha256(data_dir/prefix/'code'/name)!=h:raise ValueError('Zmenená pôvodná snímka kódu.')
            if name!='external_v4_data.py' and protocol['code'][name]!=h:raise ValueError('Migrácia zmenila kód mimo prenosu.')
        if record['old_helper_sha256']!=old['code']['external_v4_data.py'] or record['new_helper_sha256']!=protocol['code']['external_v4_data.py']:raise ValueError('Nesprávne verzie transportného kódu.')
        receipts=read_json(data_dir/prefix/'receipt_hashes.json')
        if not isinstance(receipts,dict) or len(receipts)!=record['receipt_count'] or digest(receipts)!=record['receipt_map_sha256']:raise ValueError('Neúplný archív potvrdení migrácie.')
        for name,h in receipts.items():
            if not re.fullmatch(r'receipts/[0-9a-f]{64}\.json',name) or sha256(safe_path(data_dir,name))!=h:raise ValueError('Migrácia nezachovala pôvodné potvrdenie obrázka.')
def load_manifest(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
    for r in rows:
        if set(r)!=set(FIELDS):raise ValueError('Nesprávna schéma externého manifestu.')
        for key in INT_FIELDS:r[key]=int(r[key])
        if r['label'] not in (0,1) or min(r['width'],r['height'],r['size_bytes'])<=0:raise ValueError('Neplatný riadok manifestu.')
        if not Path(r['path']).is_absolute() or not r['group']:raise ValueError('Chýba cesta alebo skupina.')
        for key,n in (('sha256',64),('rgb_sha256',64),('phash',16)):
            if not re.fullmatch('[0-9a-f]{'+str(n)+'}',r[key]):raise ValueError('Neplatný odtlačok.')
    return rows
def write_manifest(path,rows):
    s=io.StringIO(newline='');w=csv.DictWriter(s,fieldnames=FIELDS,extrasaction='ignore');w.writeheader();w.writerows(rows);atomic(path,s.getvalue().encode())

def load_source_index(path=None):
    path=Path(path or ROOT/INDEX_FILE)
    with gzip.open(path,'rb') as f:raw=f.read(32*1024*1024+1)
    if len(raw)>32*1024*1024 or hashlib.sha256(raw).hexdigest()!=INDEX_SHA256:raise ValueError('Zmenený index externého zdroja.')
    p=json.loads(raw)
    if p['repo']!=REPO or p['revision']!=REVISION or len(p['rows'])!=20008:raise ValueError('Nesprávny externý index.')
    return p

def row_identity(row):
    return f"cf/{REVISION}/{Path(row['shard']).stem}/{row['row_group']}/{row['row_index']}"
def metadata_group(row):
    if row['real_source'] not in ('coco','LAION') or row['model_name'] not in FAKE_MODELS:raise ValueError('Nepovolený zdroj.')
    if row['label'] not in (0,1):raise ValueError('Nepovolená trieda.')
    prompt=' '.join(str(row.get('prompt') or '').casefold().split())
    if row['label']==0:return 'cf/real/'+row['real_source']+'/'+row['image_name'].casefold()
    if prompt:return 'cf/prompt/'+hashlib.sha256(prompt.encode()).hexdigest()
    return 'cf/fake/'+row['model_name']+'/'+row['image_name'].casefold()
def bucket(row):return 'real/'+row['real_source'] if row['label']==0 else 'fake/'+row['model_name']
def make_plan(index,seed=SEED):
    pools=defaultdict(list);seen=set()
    for raw in index['rows']:
        r=dict(raw);r['sample_id']=row_identity(r);r['group']=metadata_group(r);r['bucket']=bucket(r)
        if r['sample_id'] in seen:raise ValueError('Opakovaná zdrojová pozícia.')
        seen.add(r['sample_id']);pools[r['bucket']].append(r)
    plan=[]
    for cell in TARGETS:
        # Náhodné poradie malých Parquet row groups obmedzuje prenos nepotrebných
        # susedných obrázkov. Nejde o jednoduchý náhodný výber jednotlivých riadkov.
        rows=sorted(pools[cell],key=lambda r:(digest([seed,cell,r['shard'],r['row_group']]),digest([seed,r['sample_id']])))
        if len({r['group'] for r in rows})<TARGETS[cell]:raise ValueError('Nedostatok nezávisle evidovaných kandidátov '+cell)
        plan.extend(rows)
    return plan

def historical_sources(project):
    """Existujúce inventáre + všetkých šesť V4 oddelení + chránený V3 test."""
    project=Path(project);download=project/'training_data_v4/download_protocol.json';dp=read_json(download)
    if sha256(download)!=(download.with_suffix('.sha256')).read_text().strip():raise ValueError('Zmenený protokol V4 dát.')
    originals=dp['sources']['anchors'];verify_anchors(originals)
    anchors=[anchor(download),*originals];csvs=[]
    for a in originals:
        if a.get('role')=='fingerprint_inventory':csvs.append(a['path'])
    base=project/'detector_experiments/detector_v4/data';p=read_json(base/'data_protocol.json');complete=read_json(base/'complete.json')
    if sha256(base/'data_protocol.json')!=(base/'data_protocol.sha256').read_text().strip() or complete['protocol_sha256']!=sha256(base/'data_protocol.json'):
        raise ValueError('Nedokončené alebo zmenené dáta V4.')
    anchors.extend([anchor(base/'data_protocol.json'),anchor(base/'complete.json')])
    for name,expected in p['manifests'].items():
        path=safe_path(base,name)
        if sha256(path)!=expected:raise ValueError('Zmenený V4 manifest.')
        anchors.append(anchor(path));csvs.append(str(path))
    deep=project/'deepshield_subset_v3/manifest.csv';anchors.append(anchor(deep));csvs.append(str(deep))
    # Aj pôvodne vyradené MPO sa nesmú vrátiť ako nový testovací obrázok.
    extras=dp['sources']['protected_exclusion_fingerprints']
    # Všetkých 21000 nových obrázkov vrátane prípadných neskôr nepoužitých zdrojov.
    accepted=project/'training_data_v4/accepted.json';download_complete=project/'training_data_v4/download_complete.json'
    dc=read_json(download_complete)
    if dc['protocol_sha256']!=sha256(download) or dc['files']['accepted.json']!=sha256(accepted):raise ValueError('Zmenený zoznam stiahnutých tréningových dát V4.')
    anchors.extend([anchor(accepted),anchor(download_complete)]);extras=extras+read_json(accepted)
    anchors=list({a['path']:a for a in anchors}.values())
    return {'anchors':anchors,'csv_inventories':sorted(set(csvs)),'extra_fingerprints':extras}
def historical_index(sources):
    from v4_deepshield_audit import ExistingFingerprintIndex
    idx=ExistingFingerprintIndex([Path(p) for p in sources['csv_inventories']])
    for i,row in enumerate(sources['extra_fingerprints']):idx._index.add(row,row.get('sample_id','protected/'+str(i)))
    return idx

class TransferBudget:
    def __init__(self,limit):self.limit=int(limit);self.used=0
    def charge(self,n):
        if self.used+n>self.limit:raise RuntimeError('Dosiahnutý limit sieťového prenosu. Výsledok nie je dokončený; pokračujte rovnakým príkazom.')
        self.used+=n

class RemoteParquet(io.RawIOBase):
    """Strict HTTP Range; nikdy neprijme celú odpoveď pri ignorovanom Range."""
    HTTP_CHUNK_BYTES=32*1024*1024
    MAX_LOGICAL_BYTES=1024*1024*1024
    def __init__(self,shard,size,budget,session=None,max_logical_read=256*1024*1024):
        import requests
        self.shard=shard;self.size=int(size);self.pos=0;self.budget=budget;self.session=session or requests.Session();self.cache=[]
        self.max_logical_read=int(max_logical_read)
        if not re.fullmatch(r'data/CompEval-\d{5}-of-00413\.parquet',shard) or self.size<=0:raise ValueError('Neplatný shard.')
        if not 0<self.max_logical_read<=self.MAX_LOGICAL_BYTES:raise ValueError('Nepovolená veľkosť logického Parquet bloku.')
    def readable(self):return True
    def seekable(self):return True
    def tell(self):return self.pos
    def seek(self,offset,whence=0):
        p=offset if whence==0 else self.pos+offset if whence==1 else self.size+offset if whence==2 else -1
        if p<0 or p>self.size:raise ValueError('Seek mimo shardu.')
        self.pos=p;return p
    def read(self,n=-1):
        n=min(self.size-self.pos,n if n>=0 else self.size);start=self.pos
        if n<=0:return b''
        if n>self.max_logical_read:raise ValueError('Parquet blok prekročil veľkosť viazanú na pripnutý index.')
        if n>self.HTTP_CHUNK_BYTES:
            parts=[];remaining=n
            while remaining:
                part=self.read(min(remaining,self.HTTP_CHUNK_BYTES));parts.append(part);remaining-=len(part)
                if n>256*1024*1024:print(f'Prenos veľkého stĺpcového bloku: {(n-remaining)/1024**2:.0f}/{n/1024**2:.0f} MiB',flush=True)
            return b''.join(parts)
        for a,body in self.cache:
            if a<=start and start+n<=a+len(body):self.pos+=n;return body[start-a:start-a+n]
        amount=min(max(n,16384),self.HTTP_CHUNK_BYTES,self.size-start)
        url=f'https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{self.shard}?download=true&range={start}-{amount}'
        for attempt in range(4):
            try:
                with self.session.get(url,headers={'Range':f'bytes={start}-{start+amount-1}','Accept-Encoding':'identity'},stream=True,timeout=(20,90)) as r:
                    r.raise_for_status()
                    expected=f'bytes {start}-{start+amount-1}/{self.size}'
                    if r.status_code!=206 or r.headers.get('Content-Range')!=expected:raise ValueError('Server nepodporil presný Range; celý shard sa nesťahuje.')
                    self.budget.charge(amount);body=bytearray()
                    for block in r.iter_content(1024*1024):
                        body.extend(block)
                        if len(body)>amount:raise ValueError('Range odpoveď prekročila dĺžku.')
                    if len(body)!=amount:raise IOError('Neúplný Range.')
                body=bytes(body)
                if amount<128*1024:self.cache.append((start,body))
                self.pos+=n;return body[:n]
            except (ValueError,RuntimeError):raise
            except Exception:
                if attempt==3:raise
                time.sleep(2**attempt)
    def close(self):
        self.session.close();super().close()

def image_bytes(value):
    if isinstance(value,(bytes,bytearray)):return bytes(value)
    if isinstance(value,list) and all(isinstance(x,int) and 0<=x<=255 for x in value):return bytes(value)
    raise ValueError('Neznámy typ image_data v pripnutom Parquet.')
def fetch_group(row,budget):
    import pyarrow.parquet as pq
    logical=max(256*1024*1024,int(row['row_group_image_bytes'])+128*1024)
    with RemoteParquet(row['shard'],row['shard_size'],budget,max_logical_read=logical) as remote:
        p=pq.ParquetFile(remote);rg=p.metadata.row_group(row['row_group'])
        image_columns=[rg.column(i) for i in range(rg.num_columns) if rg.column(i).path_in_schema=='image_data']
        if len(image_columns)!=1 or image_columns[0].total_compressed_size!=row['row_group_image_bytes']:
            raise ValueError('Veľkosť obrazového stĺpca nezodpovedá pripnutému indexu.')
        table=p.read_row_group(row['row_group'],columns=['image_name','label','real_source','model_name','prompt','format','resolution','image_data'],use_threads=False)
        return table.to_pylist()

def verify_dataset(data_dir,full_hashes=False):
    data_dir=Path(data_dir);p=read_json(data_dir/'protocol.json')
    if sha256(data_dir/'protocol.json')!=(data_dir/'protocol.sha256').read_text().strip():raise ValueError('Zmenený externý protokol.')
    verify_migrations(data_dir,p)
    if p.get('version')!='external-v4-communityforensics-1':raise ValueError('Nesprávna verzia datasetu.')
    c=read_json(data_dir/'complete.json')
    if c['protocol_sha256']!=sha256(data_dir/'protocol.json'):raise ValueError('Neplatné dokončenie externých dát.')
    required={'manifest.csv','summary.json','exclusions.json','plan.json','receipts.json',*('code/'+n for n in CODE)}
    if not required<=set(c.get('files',{})):raise ValueError('Dokončenie nemá všetky povinné súbory auditu.')
    for name,h in c['files'].items():
        if sha256(safe_path(data_dir,name))!=h:raise ValueError('Zmenený súbor externých dát: '+name)
    if sha256(data_dir/'plan.json')!=p.get('plan_sha256',sha256(data_dir/'plan.json')):raise ValueError('Plán nie je naviazaný na protokol.')
    for name,h in p.get('code',{}).items():
        if sha256(safe_path(data_dir,'code/'+name))!=h:raise ValueError('Kód nie je naviazaný na protokol.')
    receipts=read_json(data_dir/'receipts.json')
    if not isinstance(receipts,dict) or not receipts:raise ValueError('Chýbajú potvrdenia obrázkov.')
    for name,h in receipts.items():
        if not re.fullmatch(r'receipts/[0-9a-f]{64}\.json',name) or sha256(safe_path(data_dir,name))!=h:raise ValueError('Zmenené potvrdenie obrázka.')
    verify_anchors(p['historical_sources']['anchors'])
    rows=load_manifest(data_dir/'manifest.csv');groups=set();ids=set();hashes=set();rgb=set();paths=set()
    if Counter(r['bucket'] for r in rows)!=Counter(p['targets']):raise ValueError('Nedokončené cieľové počty.')
    for r in rows:
        path=Path(r['path']).resolve()
        if not path.is_relative_to((data_dir/'images').resolve()) or r['group'] in groups or r['sample_id'] in ids or r['sha256'] in hashes or r['rgb_sha256'] in rgb or str(path) in paths:
            raise ValueError('Opakovaný alebo nesprávny testovací obrázok.')
        if r['real_source'] not in ('coco','LAION') or r['source_revision']!=REVISION:raise ValueError('Nesprávny pôvod testovacieho obrázka.')
        stat=path.stat()
        if stat.st_size!=r['size_bytes'] or stat.st_mtime_ns!=r['mtime_ns'] or (full_hashes and sha256(path)!=r['sha256']):raise ValueError('Zmenený testovací obrázok: '+str(path))
        groups.add(r['group']);ids.add(r['sample_id']);hashes.add(r['sha256']);rgb.add(r['rgb_sha256']);paths.add(str(path))
    return p

def prepare(output,project):
    output=Path(output);project=Path(project)
    if (output/'protocol.json').exists():
        p=read_json(output/'protocol.json')
        if sha256(output/'protocol.json')!=(output/'protocol.sha256').read_text().strip():raise ValueError('Zmenený existujúci protokol.')
        if sha256(output/'plan.json')!=p['plan_sha256']:raise ValueError('Zmenený plán.')
        verify_migrations(output,p)
        verify_anchors(p['historical_sources']['anchors'])
        for name,h in p['code'].items():
            if sha256(project/name)!=h or sha256(output/'code'/name)!=h:raise ValueError('Zmenený prípravný kód: '+name)
        return p,read_json(output/'plan.json')
    if output.exists() and any(output.iterdir()):raise ValueError('Neprázdny adresár bez protokolu sa neprepisuje.')
    index=load_source_index(project/INDEX_FILE);plan=make_plan(index);sources=historical_sources(project)
    destination=output;output=output.with_name('.'+output.name+'.preparing-'+uuid.uuid4().hex)
    output.mkdir(parents=True,exist_ok=False);save_json(output/'plan.json',plan);code={}
    for name in CODE:
        atomic(output/'code'/name,(project/name).read_bytes());code[name]=sha256(project/name)
    p={'version':'external-v4-communityforensics-1','created_utc':now(),'project_root':str(project.resolve()),'seed':SEED,
       'source':{'repo':REPO,'revision':REVISION,'url':'https://huggingface.co/datasets/'+REPO,'license':'CC-BY-NC-SA-4.0; research and education only; original image/model licenses also apply'},
       'targets':TARGETS,'class_to_index':{'real':0,'fake':1},'plan_sha256':sha256(output/'plan.json'),'code':code,'historical_sources':sources,
       'selection':'Fixed seeded ordering of 25-image Parquet row groups per source/class; same order of reserve candidates. No detector outputs or labels of detector errors are used.',
       'group_policy':'One real source/image identifier or one normalized nonempty fake prompt across generators; missing prompt falls back to generator/image identifier; SHA/RGB/pHash<=4 deduplication against all available histories and accepted test items. Any observed content overlap blocks subsequent candidates of that known group.',
       'inference_used_for_selection':False,'test_only':True,'confirmatory':True,
       'limitations':['Authoritative generator labels and source metadata come from the pinned source; real-source ancestry is COCO/LAION, excluding RAISE/FFHQ/ImageNet.',
         'Source class ratio and generator quotas are designed, not estimates of prevalence in application traffic.',
         'Metadata and perceptual hashes do not establish absence of all semantic, identity or unknown pretrained-data overlap.',
         'Missing real-to-prompt links and DFGAN captions prevent full recovery of every source dependency.',
         'Range transport authenticates the pinned repository revision via HTTPS; entire multi-GB Parquet SHA hashes are not recomputed for partial downloads. Downloaded image bytes are locally hashed.',
         'This tests a defined subset of eight generators, not all 21 generators or the published full benchmark.']}
    save_json(output/'protocol.json',p);atomic(output/'protocol.sha256',(sha256(output/'protocol.json')+'\n').encode())
    if destination.exists():
        if any(destination.iterdir()):raise ValueError('Cieľový priečinok sa počas prípravy zmenil.')
        destination.rmdir()  # Only a verified empty destination, never recursive.
    os.rename(output,destination);return p,plan

def download(output,protocol,plan,max_new=None,max_gib=5.0):
    from v4_deepshield_audit import fingerprint,_MemoryIndex
    output=Path(output)
    if (output/'complete.json').exists():verify_dataset(output,full_hashes=True);print('Externé dáta sú už dokončené a overené.');return
    historical=historical_index(protocol['historical_sources']);internal=_MemoryIndex();accepted=[];excluded=[];used=set();blocked=set();counts=Counter();budget=TransferBudget(max_gib*1024**3)
    current_group=None;group_data=None;new=0;receipt_hashes={}
    for row in plan:
        cell=row['bucket']
        if counts[cell]>=protocol['targets'][cell]:continue
        sid=row['sample_id'];key=hashlib.sha256(sid.encode()).hexdigest();receipt=output/'receipts'/(key+'.json')
        extension='.jpg' if row['format']=='JPEG' else '.png';relative='images/'+key+extension;path=output/relative
        if receipt.exists():
            result=read_receipt(receipt)
            if result['sample_id']!=sid or result['plan_row_sha256']!=digest(row):raise ValueError('Zmenený potvrdený kandidát.')
            if result.get('image_sha256') and (not path.is_file() or sha256(path)!=result['image_sha256']):raise ValueError('Zmenený už stiahnutý obrázok.')
            if result['fingerprint'] is not None and result['fingerprint']['sha256']!=result['image_sha256']:raise ValueError('Odtlačok nepatrí stiahnutým bajtom.')
            if result['decision']=='accepted' and fingerprint(path)!=result['fingerprint']:raise ValueError('Zmenený dekódovaný obsah alebo odtlačok obrázka.')
        else:
            if max_new is not None and new>=max_new:
                save_json(output/'progress.json',{'accepted':dict(counts),'required':protocol['targets'],'new_candidates':new,'transferred_bytes':budget.used,'complete':False});print('Dosiahnutý limit kandidátov; rovnaký Run pokračuje.');return
            reasons=[];fp=None
            if row['group'] in blocked:reasons=[{'kind':'known_group_overlap','group':row['group']}]
            elif row['group'] in used:reasons=[{'kind':'repeated_known_group','group':row['group']}]
            else:
                rg=(row['shard'],row['row_group'])
                if rg!=current_group:group_data=fetch_group(row,budget);current_group=rg
                raw=group_data[row['row_index']]
                for field in ('image_name','label','real_source','model_name','prompt','format','resolution'):
                    if raw[field]!=row[field]:raise ValueError('Zdrojové metadáta nesúhlasia s uzamknutým indexom: '+field)
                body=image_bytes(raw['image_data']);atomic(path,body)
                try:fp=fingerprint(path)
                except (ValueError,OSError) as e:reasons=[{'kind':'invalid_image','detail':str(e)}]
                if fp is not None:reasons=[{**r,'index':'historical'} for r in historical.matches(fp)]+[{**r,'index':'internal'} for r in internal.matches(fp)]
            result={'sample_id':sid,'plan_row_sha256':digest(row),'decision':'excluded' if reasons else 'accepted','reasons':reasons,'fingerprint':fp,
                    'image_sha256':sha256(path) if path.exists() else None,'created_utc':now()}
            save_receipt(receipt,result);new+=1
        receipt_hashes['receipts/'+receipt.name]=sha256(receipt)
        if result['decision']=='accepted':
            fp=result['fingerprint']
            if row['group'] in used or row['group'] in blocked or historical.matches(fp) or internal.matches(fp):raise ValueError('Potvrdený výber nezodpovedá zmrazenému auditu.')
            used.add(row['group']);internal.add(fp,sid);stat=path.stat()
            r={'sample_id':sid,'path':str(path.resolve()),'label':int(row['label']),'group':row['group'],
               'source':'CommunityForensics-Eval/'+row['real_source'],'generator':row['model_name'] if row['label'] else row['real_source'],
               'family':'real' if row['label']==0 else 'gan' if row['architecture']=='GAN' else 'diffusion' if row['architecture']=='LatDiff' else 'commercial',
               'domain':'general',**fp,'mtime_ns':stat.st_mtime_ns,'bucket':cell,'real_source':row['real_source'],'paired_generator':row['model_name'],
               'source_image_name':row['image_name'],'prompt_sha256':hashlib.sha256(row['prompt'].encode()).hexdigest() if row['prompt'] else '',
               'source_revision':REVISION,'shard':row['shard'],'row_group':row['row_group'],'row_index':row['row_index'],'platform':'original'}
            accepted.append(r);counts[cell]+=1
            if len(accepted)%100==0:print(f'Prijaté obrázky: {len(accepted)}/4000; prenesené {budget.used/1024**3:.2f} GiB',flush=True)
        elif result['decision']=='excluded':
            excluded.append({'sample_id':sid,'group':row['group'],'bucket':cell,'reasons':result['reasons'],'fingerprint':result['fingerprint']})
            if any(r['kind'] in ('sha256','rgb_sha256','phash','known_group_overlap') for r in result['reasons']):blocked.add(row['group'])
        else:raise ValueError('Neplatné potvrdenie auditu.')
    save_json(output/'exclusions.json',excluded)
    if counts!=Counter(protocol['targets']):
        save_json(output/'shortfall.json',{'accepted':dict(counts),'required':protocol['targets'],'complete':False})
        raise ValueError('Auditované dáta nesplnili pevné kvóty. Pozri shortfall.json; kvóty sa automaticky nemenia.')
    write_manifest(output/'manifest.csv',accepted)
    save_json(output/'receipts.json',receipt_hashes)
    save_json(output/'summary.json',{'total':len(accepted),'class_counts':dict(Counter(str(r['label']) for r in accepted)),
       'source_counts':dict(Counter(r['generator'] for r in accepted)),'targets':dict(counts),'excluded':len(excluded),'transferred_bytes_this_run':budget.used,'audit_inventory':historical.summary})
    files={n:sha256(output/n) for n in ('manifest.csv','summary.json','exclusions.json','plan.json','receipts.json')}
    files.update({'code/'+n:sha256(output/'code'/n) for n in CODE})
    save_json(output/'complete.json',{'completed_utc':now(),'protocol_sha256':sha256(output/'protocol.json'),'files':files,'total':len(accepted)})
    print('Nezávislá externá podmnožina je pripravená: '+str(output/'manifest.csv'),flush=True)
