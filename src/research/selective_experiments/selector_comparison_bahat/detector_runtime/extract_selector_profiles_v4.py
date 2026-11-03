"""V4: obnoviteľná extrakcia profilov zmrazeného detektora a jeho teploty.

Run v PyCharme spustí prípravu a výpočet; nič nemení vo V3. Opakovaný Run
pokračuje od posledného úplného riadka. Selektor sa týmto ešte neučí.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parent
DEFAULT_RUN = ROOT / 'selective_experiments/experiment_v4'
DEFAULT_DETECTOR = ROOT / 'detector_experiments/detector_v4/training_main'
DEFAULT_DATA = ROOT / 'detector_experiments/detector_v4/data'
SEED = 20261014
SPLITS = ('selector_train', 'selector_selection', 'selector_calibration')
ALL_SPLITS = ('train', 'val', 'detector_calibration', *SPLITS)
PROFILE_CONFIG = {'version': 'degradation-profile-v1', 'jpeg_qualities': [90,75,60], 'resize_scales': [.75,.5]}
CODE = ('extract_selector_profiles_v4.py', 'selector_profile_runtime_v4.py',
        'selector_augmentation_v4.py', 'detector_training_v4.py', 'sampler_v4.py',
        'prepare_data_v4.py', 'v4_data_sources.py', 'v4_deepshield_audit.py',
        'v4_deepshield_remote_zip.py', 'calibrate_detector_v4.py')


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def read_json(path):
    def bad(value):
        raise ValueError('Nekonečná hodnota JSON: '+value)
    return json.loads(Path(path).read_text(encoding='utf-8'), parse_constant=bad)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name('.'+path.name+'.'+uuid.uuid4().hex+'.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    os.replace(temporary, path)


def environment():
    return {'python': platform.python_version(), 'packages': {
        n: metadata.version(n) for n in ('torch','torchvision','Pillow','numpy')}}


@contextmanager
def process_lock(path):
    """OS lock sa uvoľní aj pri vypnutí procesu; súbor smie zostať na disku."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+b') as stream:
        stream.seek(0, 2)
        if stream.tell() == 0:
            stream.write(b'0'); stream.flush()
        stream.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError('V tomto priečinku už beží výpočet.') from exc
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == 'nt':
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def canonical_group(row):
    return row.get('dependency_key') or row['group']


def check_splits(splits):
    seen = {key:{} for key in ('group','sample_id','sha256','rgb_sha256','path')}
    for split, rows in splits.items():
        ids = set()
        for row in rows:
            if row['sample_id'] in ids or int(row['label']) not in (0,1) or row['split'] != split:
                raise ValueError('Nesprávny alebo opakovaný riadok manifestu: '+row['sample_id'])
            ids.add(row['sample_id'])
            for key in seen:
                value = canonical_group(row) if key == 'group' else row[key]
                if key == 'path':
                    value = os.path.normcase(str(Path(value).resolve()))
                if not value:
                    raise ValueError('Chýba identita '+key)
                previous = seen[key].setdefault(value, split)
                if previous != split:
                    raise ValueError(f'Prienik {key}: {previous} / {split}')


def make_plans(splits, seed=SEED):
    from selector_augmentation_v4 import training_specs, calibration_spec
    plans, partitions = {}, {}
    for split in SPLITS:
        groups = defaultdict(list)
        for row in splits[split]:
            groups[canonical_group(row)].append(row)
        chosen = []
        for group, members in sorted(groups.items()):
            if split == 'selector_train':
                chosen.extend(members)
            else:
                # Reprezentant sa určí bez skóre aj bez použitia správnej triedy.
                chosen.append(min(members, key=lambda r:digest(['v4-representative',seed,group,r['sample_id']])))
        partitions[split] = [{'original_id':r['sample_id'], 'group':canonical_group(r), 'sha256':r['sha256']}
                             for r in sorted(chosen,key=lambda r:r['sample_id'])]
        if len({r['sha256'] for r in chosen}) != len(chosen) or len({r['rgb_sha256'] for r in chosen}) != len(chosen):
            raise ValueError('Opakovaný obsah v selectorovom oddelení: '+split)
        plan = []
        for row in sorted(chosen,key=lambda r:r['sample_id']):
            specs = training_specs(row['sha256'],seed) if split == 'selector_train' else [calibration_spec(canonical_group(row),seed)]
            for spec in specs:
                plan.append({**row, 'label':int(row['label']), 'group':canonical_group(row),
                    'original_id':row['sample_id'], 'sample_id':row['sample_id']+'::'+spec['id'], 'input_spec':spec})
        plans[split] = plan
    return plans, partitions


def partition_digest(partitions):
    return digest({s:sorted([[r['original_id'],r['group'],r['sha256']] for r in rows])
                   for s,rows in sorted(partitions.items())})


def calibration_digest(rows):
    return digest(sorted(r['sha256'] for r in rows))


def model_fingerprint(info, env):
    return digest({'version':'v4-detector-temperature-profile-1', 'model_sha256':info['model_sha256'],
        'temperature_artifact_sha256':info['temperature_artifact_sha256'], 'temperature':info['temperature'],
        'config':info['config'], 'image_size':info['image_size'], 'detector_threshold':.5,
        'profile_config':PROFILE_CONFIG, 'inference_device':info['device'], 'environment':env})


def prepare(run, detector, data, device):
    from prepare_data_v4 import verify_dataset, load_manifest
    from detector_training_v4 import load_frozen_detector
    from selector_augmentation_v4 import AUGMENTATION_POLICY
    if run.exists():
        raise ValueError('Existujúci experiment sa neprepisuje: '+str(run))
    verify_dataset(data, full_hashes=False)
    splits = {s:load_manifest(data/(s+'.csv')) for s in ALL_SPLITS}
    check_splits(splits)
    plans, partitions = make_plans(splits)
    model, transform, info = load_frozen_detector(detector, device=device, require_temperature=True)
    config_hash = info['config'].get('data_protocol_sha256')
    if config_hash != sha256(data/'data_protocol.json'):
        raise ValueError('Detektor nebol trénovaný podľa tohto dátového protokolu.')
    del model, transform
    env = environment()
    temporary = run.with_name('.'+run.name+'.preparing-'+uuid.uuid4().hex)
    temporary.mkdir(parents=True)
    locked = {}
    def copy(source, relative):
        before = sha256(source)
        target = temporary/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source,target)
        if sha256(target) != before or sha256(source) != before:
            raise ValueError('Súbor sa pri kopírovaní zmenil: '+str(source))
        locked[relative] = before
    for name in CODE:
        copy(ROOT/name, 'snapshot/code/'+name)
    for name in ('data_protocol.json',*[s+'.csv' for s in ALL_SPLITS]):
        copy(data/name, 'snapshot/data/'+name)
    for name in ('config.json','complete.json','temperature.json','temperature.complete.json'):
        copy(detector/name,'snapshot/detector/'+name)
    copy(Path(info['checkpoint_path']),'snapshot/detector/best_model.pt')
    write_json(temporary/'partitions.json',partitions)
    locked['partitions.json'] = sha256(temporary/'partitions.json')
    for split,plan in plans.items():
        relative = 'plans/'+split+'.jsonl'
        (temporary/'plans').mkdir(exist_ok=True)
        with (temporary/relative).open('w',encoding='utf-8',newline='\n') as stream:
            for row in plan:
                stream.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
        locked[relative] = sha256(temporary/relative)
    protocol = {'version':'selector-v4-profiles-1','created_utc':datetime.now(timezone.utc).isoformat(),
        'detector_run':str(detector),'data_dir':str(data),'model_sha256':info['model_sha256'],
        'model_fingerprint':model_fingerprint(info,env),'temperature':info['temperature'],
        'temperature_artifact_sha256':info['temperature_artifact_sha256'], 'profile_config':PROFILE_CONFIG,
        'detector_threshold':.5,'environment':env,'seed':SEED,'locked_files':locked,
        'counts':{s:len(p) for s,p in plans.items()},'partition_file':'partitions.json',
        'partition_digest':partition_digest(partitions),
        'calibration_partition_digest':calibration_digest(partitions['selector_calibration']),
        'augmentation_policy':AUGMENTATION_POLICY,
        'policy':'Train: 8 variantov/originál, váhy podľa skupiny. Selection/calibration: jeden vopred určený reprezentant a variant na skupinu.',
        'scope':'Vývoj a oddelená kalibrácia V4. Nie nový externý test; bez tvrdenia o zaručenej chybe pri zmene distribúcie.'}
    write_json(temporary/'protocol.json',protocol)
    (temporary/'protocol.sha256').write_text(sha256(temporary/'protocol.json')+'\n',encoding='ascii')
    os.rename(temporary,run)
    print(json.dumps({'pripravené':str(run),'counts':protocol['counts']},ensure_ascii=False),flush=True)


def verify(run):
    if sha256(run/'protocol.json') != (run/'protocol.sha256').read_text().strip():
        raise ValueError('Zmenený profilový protokol.')
    p = read_json(run/'protocol.json')
    if p['version'] != 'selector-v4-profiles-1' or p['profile_config'] != PROFILE_CONFIG:
        raise ValueError('Neplatná verzia profilov.')
    for name,expected in p['locked_files'].items():
        if sha256(run/name) != expected:
            raise ValueError('Zmenený uzamknutý súbor: '+name)
    for name in CODE:
        if sha256(ROOT/name) != p['locked_files']['snapshot/code/'+name]:
            raise ValueError('Zmenený kód výpočtu: '+name)
    if environment() != p['environment']:
        raise ValueError('Zmenené knižnice alebo verzia Pythonu.')
    return p


def sigmoid(value):
    if not math.isfinite(value):
        raise ValueError('Nekonečný logit.')
    return 1/(1+math.exp(-value)) if value >= 0 else math.exp(value)/(1+math.exp(value))


def make_record(item, image, predictor, protocol):
    from selector_profile_runtime_v4 import build_profile
    from selector_augmentation_v4 import apply_spec
    margins = []
    def predict(im):
        result = predictor(im)
        raw = float(result['raw_logit'])
        probability = sigmoid(raw/protocol['temperature'])
        if not math.isclose(probability,result['probability_fake'],rel_tol=1e-10,abs_tol=1e-12):
            raise ValueError('Inferencia používa nesprávnu teplotu alebo zaokrúhlené skóre.')
        margins.append(raw)
        return probability
    transformed = apply_spec(image,item['input_spec'])
    original = predict(transformed)
    profile = build_profile(transformed,predict,original_probability=original,decision_threshold=.5)
    row = {key:item[key] for key in ('sample_id','original_id','group','split','label','source','generator','family','domain','sha256','input_spec')}
    row.update(profile=profile.to_dict(),raw_margins=margins,temperature=protocol['temperature'],
        model_fingerprint=protocol['model_fingerprint'],detector_error=int((margins[0]>=0)!=bool(item['label'])))
    validate_record(row,item,protocol)
    return row


def validate_record(row,item,protocol):
    from selector_profile_runtime_v4 import DegradationProfile
    for key in ('sample_id','original_id','group','split','label','source','generator','family','domain','sha256','input_spec'):
        if row[key] != item[key]:
            raise ValueError('Nesúlad profilu a plánu: '+key)
    if row['model_fingerprint'] != protocol['model_fingerprint'] or row['temperature'] != protocol['temperature']:
        raise ValueError('Profil patrí inému modelu alebo teplote.')
    p = DegradationProfile.from_dict(row['profile'])
    if p.config.to_dict() != PROFILE_CONFIG or p.decision_threshold != .5:
        raise ValueError('Nesprávny profil degradácií.')
    raw = row['raw_margins']
    if len(raw) != 6 or any(type(v) not in (float,int) or not math.isfinite(v) for v in raw):
        raise ValueError('Očakáva sa šesť konečných logitov.')
    probs = [p.original_probability,*[v.probability for v in p.variants]]
    if any(not math.isclose(sigmoid(m/row['temperature']),v,rel_tol=1e-10,abs_tol=1e-12) for m,v in zip(raw,probs)):
        raise ValueError('Skóre nezodpovedá logitom.')
    if row['detector_error'] != int((raw[0]>=0)!=bool(row['label'])):
        raise ValueError('Chybný cieľ učenia selektora.')


def scan_prefix(path,plan,protocol,repair=False):
    """Obnoví len nedokončený posledný riadok; inú korupciu neignoruje."""
    if not path.exists():
        return 0
    count = 0
    with path.open('r+b' if repair else 'rb') as stream:
        while True:
            start = stream.tell()
            line = stream.readline()
            if not line:
                break
            if not line.endswith(b'\n'):
                if repair:
                    stream.truncate(start)
                    break
                raise ValueError('Nedokončený posledný riadok; obnov výpočet bez --check.')
            if count >= len(plan):
                raise ValueError('Profil obsahuje priveľa riadkov.')
            row = json.loads(line.decode('utf-8'))
            validate_record(row,plan[count],protocol)
            count += 1
    return count


def extract(run,protocol,device,max_new=None,check_only=False):
    from PIL import Image
    from detector_training_v4 import load_frozen_detector,predict_logits,rgb_image
    plans = {s:[json.loads(line) for line in (run/'plans'/(s+'.jsonl')).read_text(encoding='utf-8').splitlines()] for s in SPLITS}
    outputs = run
    profiles = outputs/'profiles'
    profiles.mkdir(parents=True,exist_ok=True)
    pending = []
    for split in SPLITS:
        path = profiles/(split+'.jsonl')
        receipt = profiles/(split+'.receipt.json')
        if receipt.exists() and read_json(receipt)['sha256'] != sha256(path):
            raise ValueError('Zmenené dokončené profily: '+split)
        count = scan_prefix(path,plans[split],protocol,repair=not check_only)
        print(f'{split}: {count}/{len(plans[split])}',flush=True)
        pending.append((split,count))
    if check_only:
        return
    model = transform = info = None
    added = 0
    for split,count in pending:
        path = profiles/(split+'.jsonl')
        if count < len(plans[split]) and model is None:
            model,transform,info = load_frozen_detector(Path(protocol['detector_run']),device=device,require_temperature=True)
            if model_fingerprint(info,environment()) != protocol['model_fingerprint']:
                raise ValueError('Detektor alebo jeho teplota sa po príprave zmenili.')
        with path.open('a',encoding='utf-8',newline='\n') as stream:
            for item in plans[split][count:]:
                if max_new is not None and added >= max_new:
                    print('Dosiahnutý limit; rovnaký príkaz bezpečne pokračuje.',flush=True)
                    return
                source = Path(item['path'])
                if sha256(source) != item['sha256']:
                    raise ValueError('Zmenený vstupný obrázok: '+str(source))
                with Image.open(source) as im:
                    im.load()
                    image = rgb_image(im)
                if sha256(source) != item['sha256']:
                    raise ValueError('Obrázok sa počas čítania zmenil: '+str(source))
                row = make_record(item,image,lambda im:predict_logits(im,model,transform,info),protocol)
                stream.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
                stream.flush()
                count += 1; added += 1
                if count % 25 == 0 or count == len(plans[split]):
                    os.fsync(stream.fileno())
                    print(f'{split}: {count}/{len(plans[split])}',flush=True)
        write_json(profiles/(split+'.receipt.json'),{'sha256':sha256(path),'count':count,
            'protocol_sha256':sha256(run/'protocol.json'),'model_sha256':protocol['model_sha256'],
            'model_fingerprint':protocol['model_fingerprint']})
    write_json(outputs/'complete.json',{'version':'selector-v4-profiles-complete-1',
        'protocol_sha256':sha256(run/'protocol.json'),
        'files':{s:sha256(profiles/(s+'.jsonl')) for s in SPLITS},'counts':protocol['counts']})
    print('Všetky profily V4 sú hotové. Ďalší krok: train_selector_v4.py',flush=True)


def ensure_python():
    pinned = Path(sys.executable)
    if pinned.exists() and Path(sys.executable).resolve() != pinned.resolve():
        env = dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONIOENCODING='utf-8')
        raise SystemExit(subprocess.call([str(pinned),'-B','-u',str(Path(__file__).resolve()),*sys.argv[1:]],env=env))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,default=DEFAULT_RUN)
    parser.add_argument('--detector-dir',type=Path,default=DEFAULT_DETECTOR)
    parser.add_argument('--data-dir',type=Path,default=DEFAULT_DATA)
    parser.add_argument('--device',choices=('cuda','cpu'),default='cuda')
    parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--check',action='store_true')
    parser.add_argument('--status',action='store_true')
    parser.add_argument('--max-new-profiles',type=int)
    args = parser.parse_args(argv)
    if args.max_new_profiles is not None and args.max_new_profiles < 1:
        parser.error('--max-new-profiles musí byť kladné.')
    experiment = args.run_dir.resolve()
    run = experiment/'selector_profiles_v4'
    with process_lock(experiment.parent/('.'+experiment.name+'.profiles.lock')):
        if not run.exists():
            if args.check or args.status:
                raise FileNotFoundError('Profily V4 ešte nie sú pripravené.')
            if not (args.detector_dir/'temperature.complete.json').exists():
                print('Najprv spusť calibrate_detector_v4.py na dokončenom detektore.',flush=True)
                raise FileNotFoundError('Chýba zmrazená teplotná kalibrácia detektora.')
            prepare(run,args.detector_dir.resolve(),args.data_dir.resolve(),args.device)
        protocol = verify(run)
        if args.prepare_only:
            return 0
        extract(run,protocol,args.device,args.max_new_profiles,args.check or args.status)
    return 0


# Release entry point: require an explicit research-job request.
if __name__ == '__main__':
    import sys as _release_sys
    if not any(arg in _release_sys.argv for arg in ('--help', '-h', '--execute-research')):
        print('Research source. Read src/RESEARCH_CODE.txt before running a job. '
              'Use --help to inspect options or --execute-research to run explicitly.')
        raise SystemExit(0)
    if '--execute-research' in _release_sys.argv:
        _release_sys.argv.remove('--execute-research')

if __name__ == '__main__':
    try:
        ensure_python()
        raise SystemExit(main())
    except KeyboardInterrupt:
        print('Prerušené. Opakovaný Run pokračuje od posledného úplného profilu.',file=sys.stderr)
        raise SystemExit(130)
    except Exception as error:
        print('Chyba: '+str(error),file=sys.stderr)
        raise SystemExit(2)
