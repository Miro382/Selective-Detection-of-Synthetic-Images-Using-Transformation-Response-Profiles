"""Nové vývojové dáta selektora V5 pri zmrazenom detektore V4.

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
INDEX_FILE = 'selector_v5_cf_index.json.gz'
INDEX_SHA256 = 'cf468a749759af3e24bbc830a8ddac74fdf8abf5d0b5281659a4c965d10ac58f'
SEED = 202610165
FAKE_MODELS = ('DFGAN','Dalle3','FLUX-dev','FLUX-schnell','Firefly_Image3','IdeogramV2','Imagen3','MidjourneyV6_1','Dalle2','IdeogramV1','Firefly_Image2','MidjourneyV5_2')
TARGETS = {'real/coco':400,'real/LAION':4600,'fake/Firefly_Image3':50,'fake/MidjourneyV6_1':50,'fake/Firefly_Image2':150,'fake/MidjourneyV5_2':150,'fake/DFGAN':600,'fake/Dalle3':600,'fake/FLUX-dev':600,'fake/FLUX-schnell':600,'fake/IdeogramV2':600,'fake/Dalle2':800,'fake/IdeogramV1':800}
OWN_CODE = ('selector_v5_data.py','download_selector_data_v5.py','prepare_selector_data_v5.py',INDEX_FILE)
DEPENDENCY_CODE = ('external_v4_data.py','download_external_data_v4.py','external_v4_cf_index.json.gz','v4_deepshield_audit.py','v4_data_sources.py')
CODE = OWN_CODE + DEPENDENCY_CODE
SPLIT_RATIOS = {'selector_train':0.60,'selector_selection':0.40,'selector_calibration':0.0}
CALIBRATION_BLOCKS = 300
FIELDS = ['sample_id','path','label','group','source','generator','family','domain','sha256','rgb_sha256','phash',
          'width','height','size_bytes','mtime_ns','bucket','real_source','paired_generator','source_image_name',
          'prompt_sha256','source_revision','shard','row_group','row_index','platform']
PREPARED_FIELDS = FIELDS + ['split','validation_domain','sampling_block']
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
    if p['repo']!=REPO or p['revision']!=REVISION or len(p['rows'])!=31836:raise ValueError('Nesprávny externý index.')
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
def make_plan(index,seed=SEED,protected_groups=(),protected_ids=()):
    pools=defaultdict(list);seen=set();protected_groups=set(protected_groups);protected_ids=set(protected_ids)
    for raw in index['rows']:
        r=dict(raw);r['sample_id']=row_identity(r);r['group']=metadata_group(r);r['bucket']=bucket(r)
        if r['sample_id'] in seen:raise ValueError('Opakovaná zdrojová pozícia.')
        seen.add(r['sample_id'])
        if r['bucket'] not in TARGETS or r['group'] in protected_groups or r['sample_id'] in protected_ids:continue
        pools[r['bucket']].append(r)
    plan=[]
    for cell in TARGETS:
        # Náhodné poradie malých Parquet row groups obmedzuje prenos nepotrebných
        # susedných obrázkov. Nejde o jednoduchý náhodný výber jednotlivých riadkov.
        rows=sorted(pools[cell],key=lambda r:(digest([seed,cell,r['shard'],r['row_group']]),digest([seed,r['sample_id']])))
        if len({r['group'] for r in rows})<TARGETS[cell]:raise ValueError('Nedostatok nezávisle evidovaných kandidátov '+cell)
        plan.extend(rows)
    return plan

def historical_sources(project):
    """Všetky staré inventáre a celý už využitý externý CF test sú chránené."""
    import external_v4_data as old
    project=Path(project);sources=old.historical_sources(project)
    ext=project/'external_data_v4/CommunityForensics'
    old.verify_dataset(ext,full_hashes=False)
    accepted=old.load_manifest(ext/'manifest.csv');excluded=read_json(ext/'exclusions.json')
    for name in ('protocol.json','protocol.sha256','complete.json','manifest.csv','exclusions.json','plan.json','receipts.json'):
        sources['anchors'].append(anchor(ext/name))
    sources['csv_inventories'].append(str(ext/'manifest.csv'))
    sources['extra_fingerprints'].extend(r['fingerprint'] for r in excluded if r.get('fingerprint'))
    sources['protected_groups']=sorted({r['group'] for r in accepted+excluded})
    sources['protected_ids']=sorted({r['sample_id'] for r in accepted+excluded})
    sources['protected_external_manifest']=str(ext/'manifest.csv')
    sources['external_results_used']=False
    sources['anchors']=list({r['path']:r for r in sources['anchors']}.values())
    return sources

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

def plan_capacity(plan,targets):
    """Predbežný odhad bez obrazového obsahu; pHash audit môže kvóty znížiť."""
    counts=Counter();used=set();blocks={};pool_blocks={}
    for r in plan:
        key=(r['shard'],r['row_group']);pool_blocks[key]=int(r['row_group_image_bytes'])
        if counts[r['bucket']]>=targets[r['bucket']] or r['group'] in used:continue
        counts[r['bucket']]+=1;used.add(r['group']);blocks[key]=int(r['row_group_image_bytes'])
    return {'selected_counts':dict(counts),'original_groups':len(used),'selected_sampling_blocks':len(blocks),
            'eligible_sampling_blocks':len(pool_blocks),'estimated_image_column_gib':sum(blocks.values())/1024**3,
            'all_reserve_image_column_gib':sum(pool_blocks.values())/1024**3,
            'estimate_note':'Compressed image columns, excluding metadata, retries and rejected candidates; not a download size guarantee.'}

def verify_dataset(data_dir,full_hashes=False):
    data_dir=Path(data_dir);p=read_json(data_dir/'protocol.json')
    if sha256(data_dir/'protocol.json')!=(data_dir/'protocol.sha256').read_text().strip():raise ValueError('Zmenený externý protokol.')
    if p.get('version')!='selector-v5-communityforensics-development-1':raise ValueError('Nesprávna verzia datasetu.')
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
    protected_groups=set(p['historical_sources'].get('protected_groups',[]));protected_ids=set(p['historical_sources'].get('protected_ids',[]))
    if Counter(r['bucket'] for r in rows)!=Counter(p['targets']):raise ValueError('Nedokončené cieľové počty.')
    for r in rows:
        path=Path(r['path']).resolve()
        if r['group'] in protected_groups or r['sample_id'] in protected_ids or not path.is_relative_to((data_dir/'images').resolve()) or r['group'] in groups or r['sample_id'] in ids or r['sha256'] in hashes or r['rgb_sha256'] in rgb or str(path) in paths:
            raise ValueError('Opakovaný alebo nesprávny testovací obrázok.')
        if r['real_source'] not in ('coco','LAION') or r['source_revision']!=REVISION or r['bucket'] not in p['targets']:raise ValueError('Nesprávny pôvod testovacieho obrázka.')
        stat=path.stat()
        if stat.st_size!=r['size_bytes'] or stat.st_mtime_ns!=r['mtime_ns'] or (full_hashes and sha256(path)!=r['sha256']):raise ValueError('Zmenený testovací obrázok: '+str(path))
        groups.add(r['group']);ids.add(r['sample_id']);hashes.add(r['sha256']);rgb.add(r['rgb_sha256']);paths.add(str(path))
    return p

def prepare(output,project,code_root=None):
    output=Path(output);project=Path(project);code_root=Path(code_root or ROOT)
    source_file=lambda n:(code_root/n if n in OWN_CODE else project/n)
    if (output/'protocol.json').exists():
        p=read_json(output/'protocol.json')
        if sha256(output/'protocol.json')!=(output/'protocol.sha256').read_text().strip():raise ValueError('Zmenený existujúci protokol.')
        if sha256(output/'plan.json')!=p['plan_sha256']:raise ValueError('Zmenený plán.')
        verify_anchors(p['historical_sources']['anchors'])
        for name,h in p['code'].items():
            if sha256(source_file(name))!=h or sha256(output/'code'/name)!=h:raise ValueError('Zmenený prípravný kód: '+name)
        return p,read_json(output/'plan.json')
    if output.exists() and any(output.iterdir()):raise ValueError('Neprázdny adresár bez protokolu sa neprepisuje.')
    index=load_source_index(code_root/INDEX_FILE);sources=historical_sources(project)
    plan=make_plan(index,protected_groups=sources['protected_groups'],protected_ids=sources['protected_ids'])
    capacity=plan_capacity(plan,TARGETS)
    if capacity['selected_counts']!=TARGETS:raise ValueError('Kvóty nie je možné dosiahnuť bez opakovania známych skupín.')
    destination=output;output=output.with_name('.'+output.name+'.preparing-'+uuid.uuid4().hex)
    output.mkdir(parents=True,exist_ok=False);save_json(output/'plan.json',plan);code={}
    for name in CODE:
        atomic(output/'code'/name,source_file(name).read_bytes());code[name]=sha256(source_file(name))
    p={'version':'selector-v5-communityforensics-development-1','created_utc':now(),'project_root':str(project.resolve()),'seed':SEED,
       'source':{'repo':REPO,'revision':REVISION,'url':'https://huggingface.co/datasets/'+REPO,'license':'CC-BY-NC-SA-4.0; research and education only; original image/model licenses also apply'},
       'targets':TARGETS,'class_to_index':{'real':0,'fake':1},'plan_sha256':sha256(output/'plan.json'),'code':code,'historical_sources':sources,
       'selection':'Fixed seeded ordering of 25-image Parquet row groups per source/class; same order of reserve candidates. No detector outputs or labels of detector errors are used.',
       'group_policy':'One real source/image identifier or one normalized nonempty fake prompt across generators; missing prompt falls back to generator/image identifier; SHA/RGB/pHash<=4 deduplication against all available histories and accepted test items. Any observed content overlap blocks subsequent candidates of that known group.',
       'inference_used_for_selection':False,'test_only':False,'confirmatory':False,'purpose':'selector-development-only','capacity_before_content_audit':capacity,
       'limitations':['Authoritative generator labels and source metadata come from the pinned source; real-source ancestry is COCO/LAION, excluding RAISE/FFHQ/ImageNet.',
         'Source class ratio and generator quotas are designed, not estimates of prevalence in application traffic.',
         'Metadata and perceptual hashes do not establish absence of all semantic, identity or unknown pretrained-data overlap.',
         'Missing real-to-prompt links and DFGAN captions prevent full recovery of every source dependency.',
         'Range transport authenticates the pinned repository revision via HTTPS; entire multi-GB Parquet SHA hashes are not recomputed for partial downloads. Downloaded image bytes are locally hashed.',
         'Development subset of eleven generators; Imagen3 is omitted because almost all available prompt groups overlap the previous external test. This is not a new independent external test.',
         'The fixed previous CommunityForensics test influenced the V5 design. New disjoint images from this collection support development and within-collection calibration only.']}
    save_json(output/'protocol.json',p);atomic(output/'protocol.sha256',(sha256(output/'protocol.json')+'\n').encode())
    if destination.exists():
        if any(destination.iterdir()):raise ValueError('Cieľový priečinok sa počas prípravy zmenil.')
        destination.rmdir()  # Only a verified empty destination, never recursive.
    os.rename(output,destination);return p,plan

def download(output,protocol,plan,max_new=None,max_gib=12.0):
    from v4_deepshield_audit import fingerprint,_MemoryIndex
    output=Path(output)
    if (output/'complete.json').exists():verify_dataset(output,full_hashes=True);print('Vývojové dáta V5 sú už dokončené a overené.');return
    historical=historical_index(protocol['historical_sources']);internal=_MemoryIndex();accepted=[];excluded=[];used=set();blocked=set(protocol['historical_sources'].get('protected_groups',[]));counts=Counter();budget=TransferBudget(max_gib*1024**3)
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
            if len(accepted)%100==0:print(f'Prijaté obrázky: {len(accepted)}/{sum(protocol['targets'].values())}; prenesené {budget.used/1024**3:.2f} GiB',flush=True)
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
    print('Nové vývojové dáta selektora V5 sú pripravené: '+str(output/'manifest.csv'),flush=True)


def sampling_block(row):
    if not re.fullmatch(r'data/CompEval-\d{5}-of-00413\.parquet',row['shard']) or int(row['row_group'])<0:
        raise ValueError('Neplatný pôvod vzorkovacieho bloku.')
    return 'cf/parquet/'+row['shard']+'/'+str(int(row['row_group']))

def partition_rows(rows,ratios=None,calibration_blocks=CALIBRATION_BLOCKS,excluded_calibration_blocks=()):
    """Celé transportné bloky sa priraďujú bez modelových skóre a bez presunu obrázkov."""
    ratios=dict(ratios or SPLIT_RATIOS);excluded_calibration_blocks=set(excluded_calibration_blocks)
    if set(ratios)!=set(SPLIT_RATIOS) or ratios['selector_train']<=0 or ratios['selector_selection']<=0 or ratios['selector_calibration']!=0 or abs(sum(ratios.values())-1)>1e-9:
        raise ValueError('Neplatné rozdelenie V5.')
    blocks=defaultdict(list);seen_ids=set();seen_groups=set();seen_hashes=set();seen_rgb=set()
    for raw in rows:
        if raw['sample_id'] in seen_ids or raw['group'] in seen_groups or raw['sha256'] in seen_hashes or raw['rgb_sha256'] in seen_rgb:
            raise ValueError('Opakovaná predloha v nových dátach.')
        seen_ids.add(raw['sample_id']);seen_groups.add(raw['group']);seen_hashes.add(raw['sha256']);seen_rgb.add(raw['rgb_sha256'])
        blocks[sampling_block(raw)].append(dict(raw))
    totals=Counter(r['bucket'] for r in rows)
    if not isinstance(calibration_blocks,int) or calibration_blocks<1 or len(blocks)<calibration_blocks+50:
        raise ValueError('Nedostatok disjunktných blokov pre pevnú kalibráciu a vývoj. Kvóta kalibrácie sa neznižuje.')
    if len(set(blocks)-excluded_calibration_blocks)<calibration_blocks:
        raise ValueError('Nedostatok nových kalibračných blokov po vylúčení blokov predchádzajúceho externého testu.')
    remaining=len(blocks)-calibration_blocks
    desired={'selector_selection':max(25,round(remaining*ratios['selector_selection'])),'selector_calibration':calibration_blocks}
    desired['selector_train']=remaining-desired['selector_selection']
    if desired['selector_train']<25:raise ValueError('Nedostatok nových tréningových blokov.')
    effective_ratios={name:n/len(blocks) for name,n in desired.items()}
    counts={name:Counter() for name in ratios};out={name:[] for name in ratios}
    assigned={};assigned_counts=Counter()
    def assign(block,split):
        assigned[block]=split;assigned_counts[split]+=1
        counts[split].update(r['bucket'] for r in blocks[block])
    # Každý zdroj má blok v tréningu, výbere aj kalibrácii. Vzácne zdroje
    # sa pokryjú pred bežnými; pri nemožnosti sa kontrakt nezjemní potichu.
    bucket_blocks={bucket:{b for b,items in blocks.items() if any(r['bucket']==bucket for r in items)} for bucket in totals}
    for bucket in sorted(totals,key=lambda b:(len(bucket_blocks[b]),b)):
        if len(bucket_blocks[bucket])<3:raise ValueError('Zdroj nemá tri bloky na oddelený vývoj a kalibráciu: '+bucket)
        for split in ('selector_train','selector_selection','selector_calibration'):
            if counts[split][bucket]:continue
            candidates=[b for b in bucket_blocks[bucket] if b not in assigned and (split!='selector_calibration' or b not in excluded_calibration_blocks)]
            if not candidates:raise ValueError('Nie je možné oddeliť bloky všetkých zdrojov: '+bucket)
            # Pri tréningu/výbere najprv používame už známe bloky; vzácny
            # nový blok tak zostáva dostupný pre kalibráciu daného zdroja.
            chosen=min(candidates,key=lambda b:(b not in excluded_calibration_blocks,digest([SEED,'source-coverage',bucket,split,b])))
            assign(chosen,split)
    # Veľké bloky najskôr zlepšujú stratifikáciu. SHA určuje stabilné poradie
    # remíz a je nezávislé od pixelov, výstupov detektora a jeho správnosti.
    order=sorted(blocks,key=lambda b:(b not in excluded_calibration_blocks,-len(blocks[b]),digest([SEED,'split-block',b])))
    for block in order:
        if block in assigned:continue
        members=blocks[block];bucket_counts=Counter(r['bucket'] for r in members)
        def cost(name):
            result=0.0
            for bucket,n in bucket_counts.items():
                target=totals[bucket]*effective_ratios[name]
                before=counts[name][bucket]-target
                result+=((before+n)**2-before**2)/max(target,1.0)
            return result,digest([SEED,block,name])
        available=[name for name in ratios if assigned_counts[name]<desired[name] and (name!='selector_calibration' or block not in excluded_calibration_blocks)]
        if not available:raise ValueError('Bloky predchádzajúceho testu nemožno umiestniť mimo kalibrácie pri pevnom rozdelení.')
        split=min(available,key=cost);assign(block,split)
    if dict(assigned_counts)!=desired:raise ValueError('Nesúlad pevného počtu kalibračných blokov.')
    for block,members in blocks.items():
        split=assigned[block]
        for r in sorted(members,key=lambda r:r['sample_id']):
            r.update(split=split,validation_domain='CF/'+r['bucket'],sampling_block=block)
            out[split].append(r)
    for name in out:out[name].sort(key=lambda r:r['sample_id'])
    return out

def write_prepared_manifest(path,rows):
    s=io.StringIO(newline='');writer=csv.DictWriter(s,fieldnames=PREPARED_FIELDS,extrasaction='raise')
    writer.writeheader();writer.writerows(rows);atomic(path,s.getvalue().encode())

def load_splits(data_dir):
    data_dir=Path(data_dir);out={}
    for split in SPLIT_RATIOS:
        with (data_dir/(split+'.csv')).open(encoding='utf-8-sig',newline='') as stream:rows=list(csv.DictReader(stream))
        for r in rows:
            if set(r)!=set(PREPARED_FIELDS):raise ValueError('Neplatná schéma pripravených dát V5.')
            for key in INT_FIELDS:r[key]=int(r[key])
            if r['split']!=split or r['label'] not in (0,1) or r['validation_domain']!='CF/'+r['bucket'] or r['sampling_block']!=sampling_block(r):
                raise ValueError('Zmenená strata alebo členstvo v oddelení.')
        out[split]=rows
    return out

def prepare_splits(data_dir,download_dir,project,code_root=None):
    data_dir=Path(data_dir);download_dir=Path(download_dir).resolve();project=Path(project).resolve();code_root=Path(code_root or ROOT)
    if (data_dir/'complete.json').exists():return verify_prepared(data_dir,True)
    if data_dir.exists() and any(data_dir.iterdir()):raise ValueError('Neprázdnu nedokončenú prípravu neprepisujem.')
    downloaded=verify_dataset(download_dir,True);rows=load_manifest(download_dir/'manifest.csv')
    excluded_blocks=calibration_exclusions(downloaded)
    splits=partition_rows(rows,excluded_calibration_blocks=excluded_blocks)
    if len({r['sampling_block'] for r in splits['selector_calibration']})!=CALIBRATION_BLOCKS:
        raise ValueError('Kalibrácia musí obsahovať presne 300 oddelených vzorkovacích blokov.')
    for name,items in splits.items():
        if {r['label'] for r in items}!={0,1}:raise ValueError('V každom oddelení musia byť obe triedy.')
    destination=data_dir;work=data_dir.with_name('.'+data_dir.name+'.preparing-'+uuid.uuid4().hex);work.mkdir(parents=True,exist_ok=False)
    for name,items in splits.items():write_prepared_manifest(work/(name+'.csv'),items)
    code={}
    for name in OWN_CODE:
        atomic(work/'code'/name,(code_root/name).read_bytes());code[name]=sha256(work/'code'/name)
    protocol={'version':'selector-v5-prepared-1','created_utc':now(),'seed':SEED,'project_root':str(project),
              'download_dir':str(download_dir),'source_protocol_sha256':sha256(download_dir/'protocol.json'),
              'source_manifest_sha256':sha256(download_dir/'manifest.csv'),'source_complete_sha256':sha256(download_dir/'complete.json'),
              'code':code,'split_ratios':SPLIT_RATIOS,'calibration_sampling_blocks':CALIBRATION_BLOCKS,
              'calibration_excluded_sampling_blocks':excluded_blocks,
              'manifests':{name+'.csv':sha256(work/(name+'.csv')) for name in splits},
              'counts':{name:len(items) for name,items in splits.items()},
              'bucket_counts':{name:dict(Counter(r['bucket'] for r in items)) for name,items in splits.items()},
              'sampling_block_counts':{name:len({r['sampling_block'] for r in items}) for name,items in splits.items()},
              'assignment':'Exactly 300 entire Parquet row groups for calibration, excluding every block represented in the prior CommunityForensics external test. Remaining blocks approximately 60% selector training, 40% selection. Source buckets stratified with at least one block per bucket per split. Each original and all transformations remain in one split. No detector outputs are used.',
              'scope':'New disjoint images from CommunityForensics used for selector development and within-collection calibration. This collection has already informed V5 design; it is not a new external confirmatory test.',
              'calibration_unit':'A single preselected representative per sampling block is required for any binomial calibration audit; the raw image count is not the number of independent Bernoulli observations.',
              'detector_training_allowed':False,'external_test_reuse_allowed':False,'inference_used_for_split':False,
              'old_selector_policy':'Only historical V4 selector-training profiles may be reused. Historical selection, calibration and external test images are not V5 calibration data.'}
    save_json(work/'data_protocol.json',protocol);atomic(work/'data_protocol.sha256',(sha256(work/'data_protocol.json')+'\n').encode())
    files={**protocol['manifests'],**{'code/'+name:h for name,h in code.items()}}
    save_json(work/'complete.json',{'version':'selector-v5-prepared-complete-1','protocol_sha256':sha256(work/'data_protocol.json'),'files':files})
    if destination.exists():
        if any(destination.iterdir()):raise ValueError('Cieľ sa počas prípravy zmenil.')
        destination.rmdir()
    os.rename(work,destination)
    return verify_prepared(destination,False)

def verify_prepared(data_dir,full_hashes=False):
    data_dir=Path(data_dir);protocol=read_json(data_dir/'data_protocol.json');complete=read_json(data_dir/'complete.json')
    if protocol.get('version')!='selector-v5-prepared-1' or complete.get('version')!='selector-v5-prepared-complete-1':raise ValueError('Nesprávna verzia prípravy V5.')
    ph=sha256(data_dir/'data_protocol.json')
    if ph!=(data_dir/'data_protocol.sha256').read_text().strip() or ph!=complete['protocol_sha256']:raise ValueError('Zmenený pripravený protokol V5.')
    required={*(name+'.csv' for name in SPLIT_RATIOS),*('code/'+name for name in OWN_CODE)}
    if set(complete['files'])!=required:raise ValueError('Neúplná snímka pripravených dát.')
    for name,h in complete['files'].items():
        if sha256(safe_path(data_dir,name))!=h:raise ValueError('Zmenený súbor prípravy V5: '+name)
    for name,h in protocol['manifests'].items():
        if name not in required or h!=complete['files'].get(name):raise ValueError('Nesúlad odtlačkov manifestov.')
    for name,h in protocol['code'].items():
        if name not in OWN_CODE or h!=complete['files'].get('code/'+name):raise ValueError('Nesúlad snímky kódu.')
    download_dir=Path(protocol['download_dir'])
    for name,key in [('protocol.json','source_protocol_sha256'),('manifest.csv','source_manifest_sha256'),('complete.json','source_complete_sha256')]:
        if sha256(download_dir/name)!=protocol[key]:raise ValueError('Zmenený pôvod nových dát V5.')
    downloaded=verify_dataset(download_dir,full_hashes)
    excluded_blocks=calibration_exclusions(downloaded)
    if protocol['calibration_excluded_sampling_blocks']!=excluded_blocks:
        raise ValueError('Zmenený zoznam blokov predchádzajúceho externého testu.')
    source=load_manifest(download_dir/'manifest.csv');splits=load_splits(data_dir)
    expected=partition_rows(source,protocol['split_ratios'],protocol['calibration_sampling_blocks'],excluded_blocks)
    if splits!=expected:raise ValueError('Rozdelenie nezodpovedá zmrazenému deterministickému pravidlu.')
    blocks=set();groups=set();ids=set()
    for name,rows in splits.items():
        bs={r['sampling_block'] for r in rows};gs={r['group'] for r in rows};ss={r['sample_id'] for r in rows}
        if blocks&bs or groups&gs or ids&ss or len(rows)!=len(gs) or len(rows)!=len(ss):raise ValueError('Únik medzi oddeleniami V5.')
        if len(rows)!=protocol['counts'][name] or len(bs)!=protocol['sampling_block_counts'][name] or dict(Counter(r['bucket'] for r in rows))!=protocol['bucket_counts'][name]:raise ValueError('Nesúlad počtov oddelenia.')
        blocks|=bs;groups|=gs;ids|=ss
    if ids!={r['sample_id'] for r in source}:raise ValueError('Rozdelenie nezachovalo celý nový dataset.')
    return protocol


def calibration_exclusions(download_protocol):
    """Kalibračná jednotka nesmie byť blok, ktorý už ovplyvnil návrh V5."""
    manifest=Path(download_protocol['historical_sources']['protected_external_manifest'])
    anchors=download_protocol['historical_sources']['anchors']
    bound=[a for a in anchors if Path(a['path']).resolve()==manifest.resolve()]
    if len(bound)!=1 or sha256(manifest)!=bound[0]['sha256']:
        raise ValueError('Zoznam blokov starého externého testu nie je naviazaný na nemenný manifest.')
    return sorted({sampling_block(r) for r in load_manifest(manifest)})
