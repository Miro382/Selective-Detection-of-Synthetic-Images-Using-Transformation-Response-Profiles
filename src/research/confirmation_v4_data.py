"""Nové obrázky zo známych zdrojov pre vopred uzamknuté overenie selektora V4.

Výber zdrojov nadväzuje na exploráciu; výber jednotlivých obrázkov je bez
výstupov modelu. Kvóty sa po audite nesmú znížiť. Nejde o test nových zdrojov.
"""
from __future__ import annotations
import csv, hashlib, io, math, os, re, uuid
from collections import Counter, defaultdict
from pathlib import Path
from selector_v5_data import (now, sha256, digest, read_json, atomic, save_json,
    save_receipt, read_receipt, safe_path, anchor, verify_anchors, TransferBudget)
import external_v5_data as of
import v4_data_sources as ds

ROOT = Path(__file__).resolve().parent
VERSION = 'selector-v4-source-confirmation-data-1'
SEED = 202610171
DS_TARGETS = {'FFHQ':1500, 'FORLAB':3500, 'StyleGAN':900, 'StyleGAN2':900,
              'StyleGAN3':900, **{name:460 for name in ds.DIFFUSION}}
TARGETS = {'OpenFake/real/docci':5000,
    **{'OpenFake/fake/'+name:1000 for name in of.FAKE_MODELS},
    **{'DeepShield/'+('real/' if name in ('FFHQ','FORLAB') else 'fake/')+name:n for name,n in DS_TARGETS.items()}}
OWN_CODE = ('confirmation_v4_data.py', 'download_selector_v4_confirmation_data.py',
            'prepare_selector_v4_confirmation_data.py')
ANALYSIS_CODE = ('confirmation_v4_workflow.py', 'confirmation_v4_metrics.py',
                 'confirmation_v4_report.py', 'test_selector_v4_confirmation.py',
                 'external_v4_workflow.py', 'external_v4_artifacts.py',
                 'external_v4_metrics.py', 'external_v4_report.py')
CODE = tuple(dict.fromkeys(OWN_CODE + ANALYSIS_CODE + of.CODE + ('v4_deepshield_remote_zip.py', ds.INDEX_FILE)))
ANALYSIS_PLAN = {
    'primary':{'cohort':'OpenFake','condition':'clean','contrast':'selector_v4_minus_confidence',
               'metric':'AURC','resampling_unit':'origin_and_sampling_group_union'},
    'conditions':['clean','jpeg_q40','resize_0p5'], 'fixed_coverages':[.75,.90,.95],
    'cohort_targets':{'OpenFake':10000,'DeepShield':10000},
    'bootstrap_replicates':2000,'bootstrap_seed':20261017,
    'success':'paired_group_95_ci_upper_below_zero',
    'other_comparisons':'secondary','outcome_based_selection':False,
    'threshold_policy':'unchanged_frozen_V4_and_original_development_confidence',
    'certification':'none_on_new_test'}
FROZEN_DETECTOR_SHA = '351a6c3347a7acc27f58b6e213128ca65e421dbc227800fe13bbe7ef4a2dfdbd'
FROZEN_SELECTOR_SHA = '4d0469e6495433619066d52430f7d5b30cc89d3a76d557b64dfada187c2cee0d'
FIELDS = of.FIELDS + ['cohort','dependency_key','category']
INT_FIELDS = of.INT_FIELDS
TOTAL = sum(TARGETS.values())

def canonical_group(value):
    return 'prompt/'+value.rsplit('/',1)[1] if '/prompt/' in value else value

def _historical_parquet_rows(project, module, relative):
    """Protect all metadata neighbours in blocks whose images were accessed."""
    base=project/relative; plan=read_json(base/'plan.json')
    manifest=module.load_manifest(base/'manifest.csv'); excluded=read_json(base/'exclusions.json')
    touched_ids={r['sample_id'] for r in manifest+excluded}
    touched={(r['shard'],r['row_group']) for r in plan if r['sample_id'] in touched_ids}
    index=module.load_source_index(project/module.INDEX_FILE)
    neighbours=[r for r in index['rows'] if (r['shard'],r['row_group']) in touched]
    return touched, neighbours

def historical_sources(project):
    project=Path(project); sources=of.historical_sources(project)
    base=project/'external_data_v5/OpenFake'; of.verify_dataset(base,full_hashes=False)
    accepted=of.load_manifest(base/'manifest.csv'); excluded=read_json(base/'exclusions.json')
    for name in ('protocol.json','protocol.sha256','complete.json','manifest.csv','exclusions.json','plan.json','receipts.json'):
        sources['anchors'].append(anchor(base/name))
    sources['csv_inventories'].append(str(base/'manifest.csv'))
    sources['extra_fingerprints'].extend(r['fingerprint'] for r in excluded if r.get('fingerprint'))
    groups={canonical_group(r) for r in sources.get('protected_groups',[])}
    groups.update(canonical_group(r['group']) for r in accepted+excluded)
    prompts=set(sources.get('protected_prompt_hashes',[])); image_hashes=set()
    import external_v4_data, selector_v5_data
    for module,relative in ((external_v4_data,'external_data_v4/CommunityForensics'),
                            (selector_v5_data,'selector_data_v5/CommunityForensics'),
                            (of,'external_data_v5/OpenFake')):
        blocks, neighbours=_historical_parquet_rows(project,module,relative)
        for r in neighbours:
            groups.add(canonical_group(module.metadata_group(r)))
            prompt=r.get('prompt_normalized_sha256') if module is of else of.prompt_hash(r.get('prompt'))
            if prompt:prompts.add(prompt)
            if r.get('image_sha256'):image_hashes.add(r['image_sha256'])
        if module is of:sources['protected_openfake_blocks']=sorted([list(r) for r in blocks])
        sources['anchors'].append(anchor(project/module.INDEX_FILE))
    deep_groups=set(); deep_dependencies=set()
    for relative in ('deepshield_subset_v3/selection.json', 'training_data_v4/download_plan.json',
                     'training_data_v4/exclusions.json'):
        path=project/relative; sources['anchors'].append(anchor(path))
        for r in read_json(path):
            deep_groups.add(r['group']); deep_dependencies.add(r['dependency_key'])
    # Download exclusions can contain a previously unused image. Fingerprint
    # those files as well, not only the accepted training manifest.
    from v4_deepshield_audit import fingerprint
    for r in read_json(project/'training_data_v4/exclusions.json'):
        path=ds.safe_path(project/'training_data_v4',r['relative_path'])
        if not path.is_file() or sha256(path)!=r['sha256']:raise ValueError('Zmenený vylúčený obrázok V4.')
        try:fp=fingerprint(path,allow_mpo_primary=True)
        except (ValueError,OSError):fp={'sha256':r['sha256']}
        sources['extra_fingerprints'].append(fp)
    sources['protected_groups']=sorted(groups)
    sources['protected_prompt_hashes']=sorted(prompts)
    sources['protected_image_hashes']=sorted(image_hashes)
    sources['protected_deepshield_groups']=sorted(deep_groups)
    sources['protected_deepshield_dependencies']=sorted(deep_dependencies)
    sources['anchors']=list({r['path']:r for r in sources['anchors']}.values())
    sources['csv_inventories']=sorted(set(sources['csv_inventories']))
    sources['external_results_used_for_source_choice']=True
    sources['external_results_used_for_image_choice']=False
    sources.pop('external_results_used',None)
    return sources

def make_openfake_plan(index, sources, seed=SEED):
    protected=set(sources['protected_groups']); prompts=set(sources['protected_prompt_hashes'])
    blocks={tuple(r) for r in sources['protected_openfake_blocks']}
    hashes=set(sources['protected_image_hashes']); result=[]
    for r in of.make_plan(index, prompts, seed):
        if (r['shard'],r['row_group']) in blocks or r['group'] in protected or r['image_sha256'] in hashes:continue
        result.append({**r,'cohort':'OpenFake','bucket':'OpenFake/'+r['bucket'], 'dependency_key':r['group'], 'category':'general'})
    return result

def make_deepshield_plan(entries, sources, seed=SEED, targets=None, reserve=1.20):
    targets=DS_TARGETS if targets is None else targets
    protected=set(sources['protected_deepshield_groups']); used=set(sources['protected_deepshield_dependencies'])
    pools=defaultdict(lambda:defaultdict(list))
    for entry in entries:
        row=ds.metadata(entry)
        if row and row['generator'] in targets and row['group'] not in protected and row['dependency_key'] not in used:
            pools[row['generator']][row['category']].append(row)
    plan=[]
    for source in sorted(targets,key=lambda x:digest([seed,x])):
        cats=pools[source]; keys=sorted(cats,key=lambda x:digest([seed,source,x]))
        for key in keys:cats[key].sort(key=lambda r:digest([seed,r['sample_id']]))
        positions=dict.fromkeys(keys,0); count=0; required=math.ceil(targets[source]*reserve)
        while count<required:
            added=False
            for key in keys:
                while positions[key]<len(cats[key]):
                    row=cats[key][positions[key]];positions[key]+=1
                    if row['dependency_key'] in used:continue
                    used.add(row['dependency_key']);count+=1;added=True
                    group='deepshield/'+row['dependency_key']
                    plan.append({**row,'sample_id':'deepshield-confirmation/'+row['entry']['name'],
                        'original_group':row['group'],'group':group,'sampling_group':group,'cohort':'DeepShield',
                        'bucket':'DeepShield/'+('fake/' if row['label'] else 'real/')+source})
                    break
                if count==required:break
            if not added:raise ValueError('Nedostatok nezávislých kandidátov vrátane rezervy: '+source)
    return plan

def capacity(plan):
    counts=Counter(); groups=set(); blocks={}; reserve=Counter(); deep_bytes=0
    for row in plan:
        reserve[row['bucket']]+=1
        if row['cohort']=='DeepShield':deep_bytes+=row['entry']['compressed']+1024
        if counts[row['bucket']]>=TARGETS[row['bucket']] or row['group'] in groups:continue
        groups.add(row['group']);counts[row['bucket']]+=1
        if row['cohort']=='OpenFake':blocks[row['sampling_group']]=row['row_group_image_bytes']
    if counts!=Counter(TARGETS):raise ValueError('Plán nedosahuje uzamknuté kvóty: '+str(dict(counts)))
    return {'counts_before_content_audit':dict(counts),'candidates_by_bucket':dict(reserve),
        'openfake_selected_blocks':len(blocks),'openfake_estimated_selected_columns_gib':sum(blocks.values())/2**30,
        'deepshield_all_reserve_compressed_gib':deep_bytes/2**30,
        'note':'Odhad prenosu bez opakovaní; obsahový audit môže použiť rezervy. Nedostatok zastaví prípravu bez zmeny kvót.'}

def frozen_models(project):
    paths={'detector':project/'detector_experiments/detector_v4/training_main/best_model.pt',
           'temperature':project/'detector_experiments/detector_v4/training_main/temperature.json',
           'selector':project/'selective_experiments/experiment_v4/selector_training_v4/selector.json'}
    result={name:anchor(path) for name,path in paths.items()}; selector=read_json(paths['selector'])
    if result['detector']['sha256']!=selector['model_sha256']:raise ValueError('Selektor nepatrí zmrazenému detektoru V4.')
    if result['detector']['sha256']!=FROZEN_DETECTOR_SHA or result['selector']['sha256']!=FROZEN_SELECTOR_SHA:
        raise ValueError('Potvrdzujúci test vyžaduje presne pôvodné zmrazené artefakty V4.')
    result.update({'research_threshold':selector['research_threshold'],'accept_threshold':selector['accept_threshold'],
                   'temperature_value':selector['temperature'],'profile_config':selector['profile_config']})
    return result

def verify_protocol(output):
    output=Path(output); p=read_json(output/'protocol.json')
    if p.get('version')!=VERSION or sha256(output/'protocol.json')!=(output/'protocol.sha256').read_text().strip():raise ValueError('Zmenený dátový protokol.')
    if p.get('targets')!=TARGETS or not p.get('test_only') or p.get('inference_used_for_selection') is not False:raise ValueError('Zmenený testovací kontrakt.')
    if p.get('analysis_plan')!=ANALYSIS_PLAN:raise ValueError('Zmenený vopred určený plán analýzy.')
    if p['frozen_models']['detector']['sha256']!=FROZEN_DETECTOR_SHA or p['frozen_models']['selector']['sha256']!=FROZEN_SELECTOR_SHA:
        raise ValueError('Dátový protokol odkazuje na iný detektor alebo selektor.')
    if sha256(output/'plan.json')!=p['plan_sha256'] or set(p['code'])!=set(CODE):raise ValueError('Zmenený plán alebo snímka kódu.')
    for name,h in p['code'].items():
        if sha256(output/'code'/name)!=h:raise ValueError('Zmenený kód: '+name)
    verify_anchors(p['historical_sources']['anchors'])
    verify_anchors([p['frozen_models'][name] for name in ('detector','temperature','selector')])
    return p

def prepare(output,project,code_root=None):
    output=Path(output);project=Path(project);code_root=Path(code_root or ROOT)
    source=lambda name:code_root/name if (code_root/name).is_file() else project/name
    code={name:sha256(source(name)) for name in CODE}
    if (output/'protocol.json').exists():
        p=verify_protocol(output)
        if p['code']!=code:raise ValueError('Prípravný kód sa po zmrazení zmenil.')
        return p,read_json(output/'plan.json')
    if output.exists() and any(output.iterdir()):raise ValueError('Neprázdny cieľ sa neprepisuje.')
    sources=historical_sources(project); models=frozen_models(project)
    plan=make_openfake_plan(of.load_source_index(source(of.INDEX_FILE)),sources)
    plan+=make_deepshield_plan(ds.load_index(source(ds.INDEX_FILE)),sources)
    audit=capacity(plan)
    destination=output;output=output.with_name('.'+output.name+'.preparing-'+uuid.uuid4().hex);output.mkdir(parents=True)
    save_json(output/'plan.json',plan)
    for name,h in code.items():
        body=source(name).read_bytes()
        if hashlib.sha256(body).hexdigest()!=h:raise ValueError('Kód sa zmenil počas prípravy.')
        atomic(output/'code'/name,body)
    p={'version':VERSION,'created_utc':now(),'project_root':str(project.resolve()),'seed':SEED,
       'targets':TARGETS,'cohorts':{name:{'images':10000,'class_counts':{'0':5000,'1':5000}} for name in ('OpenFake','DeepShield')},
       'plan_sha256':sha256(output/'plan.json'),'code':code,'historical_sources':sources,'frozen_models':models,
       'analysis_plan':ANALYSIS_PLAN,
       'test_only':True,'confirmatory_within_preselected_sources':True,'inference_used_for_selection':False,
       'capacity_before_content_audit':audit,'class_to_index':{'real':0,'fake':1},
       'sources':{'OpenFake':{'repo':of.REPO,'revision':of.REVISION,'split':'core/test'},
                  'DeepShield':{'doi':'10.5281/zenodo.15648378','url':ds.ARCHIVE_URL,'size':ds.ARCHIVE_SIZE,'archive_md5':ds.ARCHIVE_MD5,'platform':'PreSocial'}},
       'selection':'Pevné kvóty a seed bez modelových skóre. OpenFake: poradie blokov a riadkov; DeepShield: striedanie kategórií, 20 % rezerva a globálne odlišné dependency_key.',
       'exclusions':'Celá história vrátane vývojových a testovacích dát, rezerv V4, V3 DeepShield plánu, použitých OpenFake blokov a známych promptov; SHA256, dekódované RGB a pHash do vzdialenosti 4.',
       'limitations':['Zdroje boli vybrané po explorácii; nové sú jednotlivé obrázky, nie zdroje. Úspech nie je zaručený.',
          'COCO + DFGAN nie sú zahrnuté: nepoužitá verejná zásoba v pripnutom zdroji je vyčerpaná.',
          'Vyvážené triedy a pevné kvóty neodhadujú prevalenciu v nasadení.',
          'Audit známych identít a odtlačkov nezaručuje neprítomnosť skrytých sémantických súvislostí alebo prekryvu s predtrénovaním.',
          'FFHQ nemá v tomto protokole anotácie identity osoby: oddeľujeme známe fotografie a odtlačky, nie nevyhnutne odlišné osoby.',
          'OpenFake bloky sa pri intervaloch neistoty spájajú s pôvodom; DeepShield jednotkou je dependency_key, nie celý ZIP.',
          'Zdrojové anotácie sa preberajú; nie sú nezávisle ručne overené. Parciálny transport neprepočíta hash celého archívu.']}
    save_json(output/'protocol.json',p);atomic(output/'protocol.sha256',(sha256(output/'protocol.json')+'\n').encode())
    if destination.exists():
        if any(destination.iterdir()):raise ValueError('Cieľ sa počas prípravy zmenil.')
        destination.rmdir()
    os.rename(output,destination);return p,plan

class Transport:
    def __init__(self,budget):self.budget=budget;self.current=None;self.block=None;self.remote=None
    def fetch(self,row):
        if row['cohort']=='OpenFake':
            group=(row['shard'],row['row_group'])
            if self.current!=group:self.block=of.fetch_group(row,self.budget);self.current=group
            raw=self.block[row['row_index']]
            for field in ('label','model','type','row_id','image_sha256','prompt_kind'):
                if raw[field]!=row[field]:raise ValueError('Zmenené metadáta OpenFake: '+field)
            if hashlib.sha256(str(raw.get('prompt') or '').encode()).hexdigest()!=row['prompt_sha256'] or of.prompt_hash(raw.get('prompt'))!=row['prompt_normalized_sha256']:raise ValueError('Zmenený prompt OpenFake.')
            body=raw['image']['bytes']
            if not isinstance(body,bytes) or hashlib.sha256(body).hexdigest()!=row['image_sha256']:raise ValueError('Nesúhlasí SHA256 obrázka OpenFake.')
            return body
        from v4_deepshield_remote_zip import RemoteZip
        if self.remote is None:
            ds.verify_source_remote()
            self.remote=RemoteZip(ds.ARCHIVE_URL,ds.ARCHIVE_SIZE,max_total_bytes=self.budget.limit-self.budget.used)
        before=self.remote.bytes_received
        try:return self.remote.read_member(row['entry'])
        finally:self.budget.charge(self.remote.bytes_received-before)
    def close(self):
        if self.remote:self.remote.close()

def _record(row,path,fp):
    label=int(row['label']=='fake') if row['cohort']=='OpenFake' else row['label']
    isof=row['cohort']=='OpenFake'; generator=row['model'] if isof else row['generator']
    return {'sample_id':row['sample_id'],'path':str(path.resolve()),'label':label,'group':row['group'],
        'source':'OpenFake/core/test' if isof else 'DeepShield/PreSocial','generator':generator,
        'family':('real' if not label else 'image-generator') if isof else row['family'],
        'domain':'general' if isof else row['domain'],**fp,'mtime_ns':path.stat().st_mtime_ns,
        'bucket':row['bucket'],'real_source':generator if not label else '', 'paired_generator':'',
        'source_image_name':row['row_id'] if isof else row['entry']['name'],
        'prompt_sha256':row['prompt_normalized_sha256'] if isof else '',
        'source_revision':of.REVISION if isof else 'zenodo.15648378/'+ds.ARCHIVE_MD5,
        'shard':row['shard'] if isof else 'DeepShield_Dataset.zip','row_group':row['row_group'] if isof else -1,
        'row_index':row['row_index'] if isof else -1,'platform':'original' if isof else 'PreSocial',
        'sampling_group':row['sampling_group'],'validation_domain':row['bucket'],
        'cohort':row['cohort'],'dependency_key':row['dependency_key'],'category':row['category']}

def write_manifest(path,rows):
    stream=io.StringIO(newline='');writer=csv.DictWriter(stream,fieldnames=FIELDS,extrasaction='raise')
    writer.writeheader();writer.writerows(rows);atomic(path,stream.getvalue().encode())

def load_manifest(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as stream:rows=list(csv.DictReader(stream))
    for r in rows:
        if set(r)!=set(FIELDS):raise ValueError('Neplatná schéma potvrdenia dát.')
        for field in INT_FIELDS:r[field]=int(r[field])
        if r['label'] not in (0,1) or r['cohort'] not in ('OpenFake','DeepShield') or min(r['width'],r['height'],r['size_bytes'])<=0:raise ValueError('Neplatný obrazový záznam.')
        for field,length in (('sha256',64),('rgb_sha256',64),('phash',16)):
            if not re.fullmatch('[0-9a-f]{'+str(length)+'}',r[field]):raise ValueError('Neplatný odtlačok.')
        if not Path(r['path']).is_absolute() or not r['group'] or not r['sampling_group']:raise ValueError('Chýba cesta alebo pôvod.')
    return rows

def download(output,protocol,plan,max_new=None,max_gib=50.0):
    from v4_deepshield_audit import fingerprint,_MemoryIndex
    output=Path(output)
    if max_new is not None and max_new<=0:raise ValueError('Limit kandidátov musí byť kladný.')
    if not math.isfinite(max_gib) or not 0<max_gib<=200:raise ValueError('Prenosový limit musí byť v (0,200] GiB.')
    if (output/'complete.json').exists():verify_dataset(output,True);print('Dáta už sú dokončené.');return
    if verify_protocol(output)!=protocol or digest(read_json(output/'plan.json'))!=digest(plan):raise ValueError('Zmenený plán.')
    historical=of.historical_index(protocol['historical_sources']);internal=_MemoryIndex()
    accepted=[];excluded=[];used=set();blocked=set();counts=Counter();receipts={};new=0
    budget=TransferBudget(max_gib*2**30);transport=Transport(budget)
    def progress():save_json(output/'progress.json',{'accepted':dict(counts),'required':TARGETS,'complete':False,'transferred_bytes_this_run':budget.used})
    try:
        for row in plan:
            cell=row['bucket']
            if counts[cell]>=TARGETS[cell]:continue
            key=hashlib.sha256(row['sample_id'].encode()).hexdigest();receipt=output/'receipts'/(key+'.json');path=output/'images'/(key+'.img')
            if receipt.exists():
                result=read_receipt(receipt)
                if result['sample_id']!=row['sample_id'] or result['plan_row_sha256']!=digest(row):raise ValueError('Zmenený potvrdený kandidát.')
                if result.get('image_sha256') and (not path.is_file() or sha256(path)!=result['image_sha256']):raise ValueError('Zmenený stiahnutý obrázok.')
                if result.get('fingerprint') and fingerprint(path)!=result['fingerprint']:raise ValueError('Zmenený dekódovaný obsah.')
            else:
                if max_new is not None and new>=max_new:progress();print('Čiastkový beh; rovnaký príkaz pokračuje.');return
                fp=None;reasons=[]
                if row['group'] in used:reasons=[{'kind':'repeated_known_group'}]
                elif row['group'] in blocked:reasons=[{'kind':'known_group_overlap'}]
                else:
                    atomic(path,transport.fetch(row))
                    try:fp=fingerprint(path)
                    except (ValueError,OSError,RuntimeError) as e:reasons=[{'kind':'invalid_image','detail':str(e)}]
                    if fp is not None:reasons=[{**r,'index':'historical'} for r in historical.matches(fp)]+[{**r,'index':'internal'} for r in internal.matches(fp)]
                result={'sample_id':row['sample_id'],'plan_row_sha256':digest(row),'decision':'excluded' if reasons else 'accepted',
                    'fingerprint':fp,'reasons':reasons,'image_sha256':sha256(path) if path.is_file() else None,'created_utc':now()}
                save_receipt(receipt,result);new+=1
            receipts['receipts/'+receipt.name]=sha256(receipt)
            if result['decision']=='accepted':
                fp=result['fingerprint']
                if row['group'] in used or row['group'] in blocked or historical.matches(fp) or internal.matches(fp):raise ValueError('Prijatie nezodpovedá opakovanému auditu.')
                used.add(row['group']);internal.add(fp,row['sample_id']);accepted.append(_record(row,path,fp));counts[cell]+=1
                if len(accepted)%100==0:print(f'Prijaté {len(accepted)}/{TOTAL}; prenos tohto behu {budget.used/2**30:.2f} GiB',flush=True)
            elif result['decision']=='excluded':
                # Re-evaluate exclusions as well: a valid previously rejected
                # candidate cannot silently disappear when dependencies change.
                kinds={r['kind'] for r in result['reasons']}
                if not kinds:raise ValueError('Vylúčenie bez dôvodu.')
                if 'invalid_image' in kinds:
                    try:fingerprint(path)
                    except (ValueError,OSError,RuntimeError):pass
                    else:raise ValueError('Predtým neplatný obrázok už dekóder prijíma; zmenilo sa prostredie.')
                if 'repeated_known_group' in kinds and row['group'] not in used:raise ValueError('Neplatné vylúčenie opakovanej skupiny.')
                if 'known_group_overlap' in kinds and row['group'] not in blocked:raise ValueError('Neplatné vylúčenie chránenej skupiny.')
                if result['fingerprint'] and not (historical.matches(result['fingerprint']) or internal.matches(result['fingerprint'])):raise ValueError('Obsahové vylúčenie sa nepotvrdilo.')
                excluded.append({'sample_id':row['sample_id'],'group':row['group'],'sampling_group':row['sampling_group'],'bucket':cell,'reasons':result['reasons'],'fingerprint':result['fingerprint']})
                if kinds&{'sha256','rgb_sha256','phash','known_group_overlap'}:blocked.add(row['group'])
            else:raise ValueError('Neplatné rozhodnutie auditu.')
        save_json(output/'exclusions.json',excluded)
        if counts!=Counter(TARGETS):
            progress();save_json(output/'shortfall.json',{'accepted':dict(counts),'required':TARGETS})
            raise ValueError('Rezerva nestačí. Kvóty sa neznižujú; test nie je pripravený.')
        write_manifest(output/'manifest.csv',accepted);save_json(output/'receipts.json',receipts)
        save_json(output/'summary.json',{'total':len(accepted),'source_counts':dict(counts),
            'cohort_counts':dict(Counter(r['cohort'] for r in accepted)),
            'sampling_groups':dict(Counter({name:len({r['sampling_group'] for r in accepted if r['cohort']==name}) for name in ('OpenFake','DeepShield')})),
            'excluded':len(excluded),'audit_inventory':historical.summary,'transferred_bytes_last_run':budget.used})
        files={name:sha256(output/name) for name in ('manifest.csv','summary.json','exclusions.json','plan.json','receipts.json')}
        files.update({'code/'+name:sha256(output/'code'/name) for name in CODE})
        save_json(output/'complete.json',{'completed_utc':now(),'protocol_sha256':sha256(output/'protocol.json'),'files':files,'total':len(accepted)})
        print('20 000 nových testovacích obrázkov pripravených: '+str(output/'manifest.csv'),flush=True)
    finally:transport.close()

def verify_dataset(data_dir,full_hashes=False):
    output=Path(data_dir);p=verify_protocol(output);c=read_json(output/'complete.json')
    required={'manifest.csv','summary.json','exclusions.json','plan.json','receipts.json',*('code/'+name for name in CODE)}
    if c['protocol_sha256']!=sha256(output/'protocol.json') or not required<=set(c['files']):raise ValueError('Neplatné dokončenie.')
    for name,h in c['files'].items():
        if sha256(safe_path(output,name))!=h:raise ValueError('Zmenený výstup: '+name)
    rows=load_manifest(output/'manifest.csv');receipts=read_json(output/'receipts.json')
    for name,h in receipts.items():
        if not re.fullmatch(r'receipts/[0-9a-f]{64}\.json',name) or sha256(safe_path(output,name))!=h:raise ValueError('Zmenené potvrdenie obrázka.')
    if len(rows)!=TOTAL or Counter(r['bucket'] for r in rows)!=Counter(TARGETS):raise ValueError('Nesplnené pevné kvóty.')
    for field in ('sample_id','group','sha256','rgb_sha256','path'):
        if len({r[field] for r in rows})!=len(rows):raise ValueError('Duplicitný údaj: '+field)
    protected=p['historical_sources'];blocks={tuple(r) for r in protected['protected_openfake_blocks']}
    for r in rows:
        path=Path(r['path']).resolve();stat=path.stat()
        if not path.is_relative_to((output/'images').resolve()):raise ValueError('Obrázok mimo dátového priečinka.')
        if r['bucket']!=r['cohort']+'/'+('fake/' if r['label'] else 'real/')+r['generator']:raise ValueError('Nesúlad triedy a zdroja.')
        if r['cohort']=='OpenFake':
            if r['source_revision']!=of.REVISION or r['sampling_group']!=of.sampling_group(r) or (r['shard'],r['row_group']) in blocks or r['group'] in protected['protected_groups'] or r['prompt_sha256'] in protected['protected_prompt_hashes']:raise ValueError('Prekryv OpenFake s históriou.')
        elif r['sampling_group']!=r['group'] or r['dependency_key'] in protected['protected_deepshield_dependencies']:raise ValueError('Prekryv DeepShield s históriou.')
        if stat.st_size!=r['size_bytes'] or stat.st_mtime_ns!=r['mtime_ns'] or (full_hashes and sha256(path)!=r['sha256']):raise ValueError('Zmenený obrázok: '+str(path))
    return p

def prepare_data(project,data_dir):
    p=verify_dataset(data_dir,True)
    if Path(p['project_root']).resolve()!=Path(project).resolve():raise ValueError('Dáta patria inému projektu.')
    print('Overených 20 000 nových obrázkov; všetky zostávajú výhradne testovacie.')
    return p
