"""Uzamknutý nezávislý test OpenFake: DOCCI a päť obrazových generátorov.

Výber sa robí bez inferencie; žiadne tréningové ani kalibračné úpravy V5.
"""
from __future__ import annotations
import csv, gzip, hashlib, io, json, os, re, time, uuid
from collections import Counter
from pathlib import Path
from selector_v5_data import (now, sha256, digest, read_json, atomic, save_json,
    save_receipt, read_receipt, safe_path, anchor, verify_anchors, TransferBudget)

ROOT=Path(__file__).resolve().parent
REPO='ComplexDataLab/OpenFake'
REVISION='3247b942f18d39d1fd476b1077852af033d35d8c'
INDEX_FILE='external_v5_openfake_index.json.gz'
INDEX_SHA256='b9f95fab6a0bcc78c1bcdf8adea38b894699589a391e8ff24986d4efd3d7dbbd'
SEED=202610166
FAKE_MODELS=('z-image-turbo','flux.2-klein-9b','gpt-image-1.5','midjourney-7','illustrious')
TARGETS={'real/docci':2000,**{'fake/'+model:400 for model in FAKE_MODELS}}
OWN_CODE=('external_v5_data.py','download_external_data_v5.py','prepare_external_data_v5.py',INDEX_FILE)
DEPENDENCY_CODE=('selector_v5_data.py','download_selector_data_v5.py','prepare_selector_data_v5.py','selector_v5_cf_index.json.gz',
    'external_v4_data.py','download_external_data_v4.py','external_v4_cf_index.json.gz','v4_deepshield_audit.py','v4_data_sources.py')
CODE=OWN_CODE+DEPENDENCY_CODE
FIELDS=['sample_id','path','label','group','source','generator','family','domain','sha256','rgb_sha256','phash',
    'width','height','size_bytes','mtime_ns','bucket','real_source','paired_generator','source_image_name',
    'prompt_sha256','source_revision','shard','row_group','row_index','platform','sampling_group','validation_domain']
INT_FIELDS=('label','width','height','size_bytes','mtime_ns','row_group','row_index')
VERSION='external-v5-openfake-1'

def prompt_hash(text):
    text=' '.join(str(text or '').casefold().split())
    return hashlib.sha256(text.encode()).hexdigest() if text else ''

def metadata_group(row):
    return 'prompt/'+row['prompt_normalized_sha256'] if row['prompt_normalized_sha256'] else 'image/'+row['image_sha256']

def sampling_group(row):
    if not re.fullmatch(r'core/test-\d{5}-of-00013\.parquet',row['shard']) or int(row['row_group'])<0:raise ValueError('Neplatný testovací blok OpenFake.')
    return 'openfake/parquet/'+row['shard']+'/'+str(int(row['row_group']))

def load_source_index(path=None):
    with gzip.open(path or ROOT/INDEX_FILE,'rb') as stream:body=stream.read(64*1024**2+1)
    if len(body)>64*1024**2 or hashlib.sha256(body).hexdigest()!=INDEX_SHA256:raise ValueError('Zmenený pripnutý index OpenFake.')
    result=json.loads(body)
    if result['repo']!=REPO or result['revision']!=REVISION or result['split']!='core/test':raise ValueError('Nesprávny zdroj indexu.')
    return result

def make_plan(index,protected_prompt_hashes=(),seed=SEED):
    protected=set(protected_prompt_hashes);rows=[];seen=set()
    for raw in index['rows']:
        r=dict(raw)
        if r['label'] not in ('real','fake') or r['type']=='video':raise ValueError('Neplatná trieda alebo video v obrazovom indexe.')
        r['bucket']=r['label']+'/'+r['model']
        if r['bucket'] not in TARGETS:raise ValueError('Nepovolený testovací zdroj.')
        for field in ('image_sha256','prompt_sha256'):
            if not re.fullmatch('[0-9a-f]{64}',r[field]):raise ValueError('Neplatný zdrojový odtlačok.')
        r['sample_id']='openfake/'+REVISION+'/'+r['row_id'];r['group']=metadata_group(r);r['sampling_group']=sampling_group(r)
        if r['sample_id'] in seen:raise ValueError('Duplicitná zdrojová identita.')
        seen.add(r['sample_id'])
        if r['prompt_normalized_sha256'] in protected:continue
        rows.append(r)
    # Rovnaké náhodné poradie blokov pre všetky zdroje šetrí prenos pri zachovaní
    # pevného výberu; nejde o jednoduchý náhodný výber jednotlivých obrázkov.
    rows.sort(key=lambda r:(digest([seed,'block',r['shard'],r['row_group']]),digest([seed,'row',r['sample_id']])))
    return rows

def plan_capacity(plan,targets=TARGETS):
    counts=Counter();groups=set();blocks={};pool={}
    for r in plan:
        key=r['sampling_group'];pool[key]=r['row_group_image_bytes']
        if counts[r['bucket']]>=targets[r['bucket']] or r['group'] in groups:continue
        counts[r['bucket']]+=1;groups.add(r['group']);blocks[key]=r['row_group_image_bytes']
    return {'selected_counts':dict(counts),'groups':len(groups),'selected_sampling_blocks':len(blocks),
        'estimated_image_column_gib':sum(blocks.values())/1024**3,'all_reserve_image_column_gib':sum(pool.values())/1024**3,
        'note':'Odhad komprimovaných stĺpcov, bez metadát, opakovaní a vyradených kandidátov.'}

def historical_sources(project):
    import selector_v5_data as old
    project=Path(project);sources=old.historical_sources(project)
    data=project/'selector_data_v5/CommunityForensics'
    old.verify_dataset(data,full_hashes=False)
    rows=old.load_manifest(data/'manifest.csv');excluded=read_json(data/'exclusions.json')
    for name in ('protocol.json','protocol.sha256','complete.json','manifest.csv','exclusions.json','plan.json','receipts.json'):
        sources['anchors'].append(anchor(data/name))
    sources['csv_inventories'].append(str(data/'manifest.csv'))
    sources['extra_fingerprints'].extend(r['fingerprint'] for r in excluded if r.get('fingerprint'))
    groups=set(sources.get('protected_groups',[]))|{r['group'] for r in rows+excluded}
    sources['protected_prompt_hashes']=sorted({r.rsplit('/',1)[1] for r in groups if '/prompt/' in r})
    sources['protected_groups']=sorted(groups)
    # Freeze baseline artifacts before accessing test image bytes.
    for relative in ('detector_experiments/detector_v4/training_main/best_model.pt',
        'detector_experiments/detector_v4/training_main/temperature.json',
        'selective_experiments/experiment_v4/selector_training_v4/selector.json',
        'selective_experiments/experiment_v5/selector_training_v5/selector.json',
        'selective_experiments/experiment_v5/selector_training_v5/complete.json'):
        sources['anchors'].append(anchor(project/relative))
    sources['anchors']=list({r['path']:r for r in sources['anchors']}.values())
    sources['csv_inventories']=sorted(set(sources['csv_inventories']))
    sources['external_results_used']=False
    return sources

def historical_index(sources):
    from v4_deepshield_audit import ExistingFingerprintIndex
    result=ExistingFingerprintIndex([Path(p) for p in sources['csv_inventories']])
    for i,row in enumerate(sources['extra_fingerprints']):result._index.add(row,'protected/'+str(i))
    return result

class RemoteParquet(io.RawIOBase):
    """Prijíma iba presný HTTP Range; nikdy nestiahne celý shard omylom."""
    CHUNK=32*1024**2
    def __init__(self,shard,size,budget,max_logical_read=256*1024**2,session=None):
        import requests
        if not re.fullmatch(r'core/test-\d{5}-of-00013\.parquet',shard) or int(size)<=0:raise ValueError('Neplatný shard OpenFake.')
        if not 0<max_logical_read<=1024**3:raise ValueError('Priveľký logický Parquet blok.')
        self.shard=shard;self.size=int(size);self.budget=budget;self.pos=0;self.cache=[];self.max_logical_read=int(max_logical_read);self.session=session or requests.Session()
    def readable(self):return True
    def seekable(self):return True
    def tell(self):return self.pos
    def seek(self,offset,whence=0):
        pos=offset if whence==0 else self.pos+offset if whence==1 else self.size+offset if whence==2 else -1
        if not 0<=pos<=self.size:raise ValueError('Seek mimo súboru.')
        self.pos=pos;return pos
    def read(self,n=-1):
        n=min(self.size-self.pos,n if n>=0 else self.size);start=self.pos
        if n<=0:return b''
        if n>self.max_logical_read:raise ValueError('Neočakávane veľký Parquet blok.')
        if n>self.CHUNK:
            parts=[];left=n
            while left:
                part=self.read(min(left,self.CHUNK));parts.append(part);left-=len(part)
            return b''.join(parts)
        for begin,body in self.cache:
            if begin<=start and start+n<=begin+len(body):self.pos+=n;return body[start-begin:start-begin+n]
        amount=min(max(n,16384),self.CHUNK,self.size-start)
        url=f'https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{self.shard}?download=true&range={start}-{amount}'
        for attempt in range(4):
            try:
                with self.session.get(url,headers={'Range':f'bytes={start}-{start+amount-1}','Accept-Encoding':'identity'},stream=True,timeout=(20,90)) as response:
                    response.raise_for_status()
                    if response.status_code!=206 or response.headers.get('Content-Range')!=f'bytes {start}-{start+amount-1}/{self.size}':raise ValueError('Server nepodporil presný Range; celý shard sa nesťahuje.')
                    self.budget.charge(amount);body=bytearray()
                    for piece in response.iter_content(1024**2):
                        body.extend(piece)
                        if len(body)>amount:raise ValueError('Priveľká Range odpoveď.')
                    if len(body)!=amount:raise IOError('Neúplný prenos.')
                body=bytes(body)
                if amount<128*1024:self.cache.append((start,body))
                self.pos+=n;return body[:n]
            except (ValueError,RuntimeError):raise
            except Exception:
                if attempt==3:raise
                time.sleep(2**attempt)
    def close(self):self.session.close();super().close()

def fetch_group(row,budget):
    import pyarrow.parquet as pq
    logical=max(256*1024**2,int(row['row_group_image_bytes'])+128*1024)
    with RemoteParquet(row['shard'],row['shard_size'],budget,max_logical_read=logical) as remote:
        p=pq.ParquetFile(remote);rg=p.metadata.row_group(row['row_group'])
        size=sum(rg.column(i).total_compressed_size for i in range(rg.num_columns) if rg.column(i).path_in_schema.startswith('image.'))
        if size!=row['row_group_image_bytes']:raise ValueError('Obrazový stĺpec nezodpovedá pripnutému indexu.')
        return p.read_row_group(row['row_group'],columns=['image','prompt','label','model','type','row_id','image_sha256','prompt_kind'],use_threads=False).to_pylist()

def load_manifest(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as stream:rows=list(csv.DictReader(stream))
    for row in rows:
        if set(row)!=set(FIELDS):raise ValueError('Neplatná schéma OpenFake manifestu.')
        for key in INT_FIELDS:row[key]=int(row[key])
        if row['label'] not in (0,1) or min(row['width'],row['height'],row['size_bytes'])<=0:raise ValueError('Neplatný obrazový záznam.')
        for field,length in (('sha256',64),('rgb_sha256',64),('phash',16)):
            if not re.fullmatch('[0-9a-f]{'+str(length)+'}',row[field]):raise ValueError('Neplatný odtlačok manifestu.')
        if row['sampling_group']!=sampling_group(row) or not Path(row['path']).is_absolute():raise ValueError('Neplatná cesta alebo skupina.')
    return rows

def write_manifest(path,rows):
    stream=io.StringIO(newline='');writer=csv.DictWriter(stream,fieldnames=FIELDS,extrasaction='raise');writer.writeheader();writer.writerows(rows);atomic(path,stream.getvalue().encode())

def verify_protocol(output):
    output=Path(output);p=read_json(output/'protocol.json')
    if p.get('version')!=VERSION or sha256(output/'protocol.json')!=(output/'protocol.sha256').read_text().strip():raise ValueError('Zmenený protokol externých dát V5.')
    if p.get('targets')!=TARGETS or p.get('test_only') is not True or p.get('inference_used_for_selection') is not False:raise ValueError('Zmenený testovací kontrakt.')
    if sha256(output/'plan.json')!=p['plan_sha256']:raise ValueError('Zmenený testovací plán.')
    if set(p['code'])!=set(CODE):raise ValueError('Neúplná snímka dátového kódu.')
    for name,h in p['code'].items():
        if sha256(output/'code'/name)!=h:raise ValueError('Zmenený kód auditu: '+name)
    verify_anchors(p['historical_sources']['anchors'])
    return p

def prepare(output,project,code_root=None):
    output=Path(output);project=Path(project);code_root=Path(code_root or ROOT)
    source_file=lambda n:code_root/n if n in OWN_CODE else project/n
    if (output/'protocol.json').exists():
        p=verify_protocol(output)
        for name,h in p['code'].items():
            if sha256(source_file(name))!=h:raise ValueError('Prípravný kód sa po zmrazení zmenil: '+name)
        return p,read_json(output/'plan.json')
    if output.exists() and any(output.iterdir()):raise ValueError('Neprázdny priečinok sa neprepisuje.')
    sources=historical_sources(project);index=load_source_index(code_root/INDEX_FILE)
    plan=make_plan(index,sources['protected_prompt_hashes']);capacity=plan_capacity(plan)
    if capacity['selected_counts']!=TARGETS:raise ValueError('Nedostatok zdrojovo odlíšených kandidátov; kvóty sa nemenia.')
    destination=output;output=output.with_name('.'+output.name+'.preparing-'+uuid.uuid4().hex);output.mkdir(parents=True)
    save_json(output/'plan.json',plan);code={}
    for name in CODE:
        atomic(output/'code'/name,source_file(name).read_bytes());code[name]=sha256(source_file(name))
    p={'version':VERSION,'created_utc':now(),'project_root':str(project.resolve()),'seed':SEED,
       'source':{'repo':REPO,'revision':REVISION,'split':'core/test','url':'https://huggingface.co/datasets/'+REPO,'license':'CC-BY-NC-4.0; original image/model terms also apply'},
       'targets':TARGETS,'class_to_index':{'real':0,'fake':1},'plan_sha256':sha256(output/'plan.json'),'code':code,'historical_sources':sources,
       'test_only':True,'confirmatory':True,'inference_used_for_selection':False,'capacity_before_content_audit':capacity,
       'selection':'Fixed seeded global Parquet-block ordering; one normalized prompt or image identity; quota replacement follows fixed reserve order; no model scores.',
       'group_policy':'One normalized prompt across real/fake and generators, fallback image_sha256; block and origin dependencies must be unioned in bootstrap.',
       'limitations':['New public collection relative to documented V4/V5 experiments; model pretraining provenance cannot be completely audited.',
          'Source labels are authoritative dataset labels, not independently manually adjudicated annotations.',
          'Only DOCCI real images and five fixed generators; equal class ratio and quotas do not estimate deployment prevalence.',
          'SHA/RGB/pHash<=4 filtering excludes known content matches but cannot prove absence of all semantic or identity relations.',
          'Caption equality is only a conservative known-origin proxy; hidden matching prompts/identities can remain.',
          'Partial Range downloads verify image_sha256 from pinned metadata; whole Parquet LFS hashes are not recomputed.',
          'OpenFake core/test is OOD relative to OpenFake own training split; this label alone is not proof that every generator family is unseen by V4.']}
    save_json(output/'protocol.json',p);atomic(output/'protocol.sha256',(sha256(output/'protocol.json')+'\n').encode())
    if destination.exists():
        if any(destination.iterdir()):raise ValueError('Cieľ sa počas prípravy zmenil.')
        destination.rmdir()
    os.rename(output,destination);return p,plan

def download(output,protocol,plan,max_new=None,max_gib=20.0):
    from v4_deepshield_audit import fingerprint,_MemoryIndex
    output=Path(output)
    if (output/'complete.json').exists():verify_dataset(output,full_hashes=True);print('Externé dáta V5 sú už dokončené.');return
    if verify_protocol(output)!=protocol or digest(read_json(output/'plan.json'))!=digest(plan):raise ValueError('Nesúlad plánu pred sťahovaním.')
    historical=historical_index(protocol['historical_sources']);internal=_MemoryIndex();accepted=[];excluded=[];used=set();blocked=set();counts=Counter();budget=TransferBudget(max_gib*1024**3)
    current=None;data=None;new=0;receipt_hashes={}
    for row in plan:
        cell=row['bucket']
        if counts[cell]>=protocol['targets'][cell]:continue
        sid=row['sample_id'];key=hashlib.sha256(sid.encode()).hexdigest();receipt=output/'receipts'/(key+'.json');path=output/'images'/(key+'.img')
        if receipt.exists():
            result=read_receipt(receipt)
            if result['sample_id']!=sid or result['plan_row_sha256']!=digest(row):raise ValueError('Zmenený potvrdený kandidát.')
            if result.get('image_sha256') and (not path.is_file() or sha256(path)!=result['image_sha256']):raise ValueError('Zmenený stiahnutý obrázok.')
            if result['decision']=='accepted' and fingerprint(path)!=result['fingerprint']:raise ValueError('Zmenený dekódovaný obsah.')
        else:
            if max_new is not None and new>=max_new:
                save_json(output/'progress.json',{'accepted':dict(counts),'required':TARGETS,'complete':False,'transferred_bytes':budget.used});print('Limit kandidátov; rovnaký Run pokračuje.');return
            fp=None;reasons=[]
            if row['group'] in used:reasons=[{'kind':'repeated_known_group'}]
            elif row['group'] in blocked:reasons=[{'kind':'known_group_overlap'}]
            else:
                rg=(row['shard'],row['row_group'])
                if rg!=current:data=fetch_group(row,budget);current=rg
                raw=data[row['row_index']]
                for field in ('label','model','type','row_id','image_sha256','prompt_kind'):
                    if raw[field]!=row[field]:raise ValueError('Nesúlad pripnutých metadát: '+field)
                if hashlib.sha256(str(raw.get('prompt') or '').encode()).hexdigest()!=row['prompt_sha256'] or prompt_hash(raw.get('prompt'))!=row['prompt_normalized_sha256']:raise ValueError('Nesúlad zdrojového promptu.')
                body=raw['image']['bytes']
                if not isinstance(body,bytes) or hashlib.sha256(body).hexdigest()!=row['image_sha256']:raise ValueError('Bajty obrázka nesúhlasia s publikovaným SHA256.')
                atomic(path,body)
                try:fp=fingerprint(path)
                except (ValueError,OSError) as e:reasons=[{'kind':'invalid_image','detail':str(e)}]
                if fp is not None:reasons=[{**r,'index':'historical'} for r in historical.matches(fp)]+[{**r,'index':'internal'} for r in internal.matches(fp)]
            result={'sample_id':sid,'plan_row_sha256':digest(row),'decision':'excluded' if reasons else 'accepted','fingerprint':fp,'reasons':reasons,'image_sha256':sha256(path) if path.exists() else None,'created_utc':now()}
            save_receipt(receipt,result);new+=1
        receipt_hashes['receipts/'+receipt.name]=sha256(receipt)
        if result['decision']=='accepted':
            fp=result['fingerprint']
            if row['group'] in used or row['group'] in blocked or historical.matches(fp) or internal.matches(fp):raise ValueError('Potvrdenie nezodpovedá opakovanému auditu.')
            used.add(row['group']);internal.add(fp,sid);label=int(row['label']=='fake')
            accepted.append({'sample_id':sid,'path':str(path.resolve()),'label':label,'group':row['group'],'source':'OpenFake/core/test',
                'generator':row['model'],'family':'real' if not label else 'image-generator','domain':'general',**fp,'mtime_ns':path.stat().st_mtime_ns,
                'bucket':cell,'real_source':'docci' if not label else '', 'paired_generator':'','source_image_name':row['row_id'],'prompt_sha256':row['prompt_normalized_sha256'],
                'source_revision':REVISION,'shard':row['shard'],'row_group':row['row_group'],'row_index':row['row_index'],'platform':'original',
                'sampling_group':row['sampling_group'],'validation_domain':'OpenFake/'+cell})
            counts[cell]+=1
            if len(accepted)%100==0:print(f'Prijaté: {len(accepted)}/4000; prenos {budget.used/1024**3:.2f} GiB',flush=True)
        elif result['decision']=='excluded':
            excluded.append({'sample_id':sid,'group':row['group'],'bucket':cell,'reasons':result['reasons'],'fingerprint':result['fingerprint']})
            if any(r['kind'] in ('sha256','rgb_sha256','phash','known_group_overlap') for r in result['reasons']):blocked.add(row['group'])
        else:raise ValueError('Neplatné rozhodnutie auditu.')
    save_json(output/'exclusions.json',excluded)
    if counts!=Counter(TARGETS):
        save_json(output/'shortfall.json',{'accepted':dict(counts),'required':TARGETS,'complete':False});raise ValueError('Audit nedosiahol kvóty; pozri shortfall.json. Test nie je pripravený.')
    write_manifest(output/'manifest.csv',accepted);save_json(output/'receipts.json',receipt_hashes)
    save_json(output/'summary.json',{'total':len(accepted),'class_counts':dict(Counter(str(r['label']) for r in accepted)),
        'source_counts':dict(counts),'sampling_blocks':len({r['sampling_group'] for r in accepted}),'excluded':len(excluded),'audit_inventory':historical.summary,'transferred_bytes_this_run':budget.used})
    files={n:sha256(output/n) for n in ('manifest.csv','summary.json','exclusions.json','plan.json','receipts.json')}
    files.update({'code/'+n:sha256(output/'code'/n) for n in CODE})
    save_json(output/'complete.json',{'completed_utc':now(),'protocol_sha256':sha256(output/'protocol.json'),'files':files,'total':len(accepted)})
    print('Nezávislé testovacie dáta sú pripravené: '+str(output/'manifest.csv'),flush=True)

def verify_dataset(data_dir,full_hashes=False):
    output=Path(data_dir);p=verify_protocol(output);c=read_json(output/'complete.json')
    required={'manifest.csv','summary.json','exclusions.json','plan.json','receipts.json',*('code/'+n for n in CODE)}
    if c['protocol_sha256']!=sha256(output/'protocol.json') or not required<=set(c['files']):raise ValueError('Neplatné dokončenie dát.')
    for name,h in c['files'].items():
        if sha256(safe_path(output,name))!=h:raise ValueError('Zmenený súbor auditu: '+name)
    rows=load_manifest(output/'manifest.csv');receipts=read_json(output/'receipts.json')
    for name,h in receipts.items():
        if not re.fullmatch(r'receipts/[0-9a-f]{64}\.json',name) or sha256(safe_path(output,name))!=h:raise ValueError('Zmenené potvrdenie obrázka.')
    if len(rows)!=4000 or Counter(r['bucket'] for r in rows)!=Counter(TARGETS):raise ValueError('Nesúlad počtu testovacích obrázkov.')
    for field in ('sample_id','group','sha256','rgb_sha256','path'):
        if len({r[field] for r in rows})!=len(rows):raise ValueError('Duplicitný údaj: '+field)
    for row in rows:
        path=Path(row['path']).resolve();stat=path.stat()
        if not path.is_relative_to((output/'images').resolve()) or row['source_revision']!=REVISION:raise ValueError('Nesprávny pôvod testovacieho obrázka.')
        if row['bucket']!=('fake/' if row['label'] else 'real/')+row['generator']:raise ValueError('Nesúlad triedy a zdroja.')
        if stat.st_size!=row['size_bytes'] or stat.st_mtime_ns!=row['mtime_ns'] or (full_hashes and sha256(path)!=row['sha256']):raise ValueError('Zmenený testovací obrázok: '+str(path))
    return p

def prepare_data(project,data_dir):
    """Overí hotový download; celý manifest zostáva výhradne testovací."""
    p=verify_dataset(data_dir,full_hashes=True)
    if Path(p['project_root']).resolve()!=Path(project).resolve():raise ValueError('Testovacie dáta patria inému projektu.')
    print('4 000 obrázkov: 2 000 DOCCI + 2 000 fake; integrita a oddelenie od vývoja overené.')
    return p
