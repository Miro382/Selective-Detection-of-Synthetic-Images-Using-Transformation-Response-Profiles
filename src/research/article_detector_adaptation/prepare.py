"""Pri ručnom spustení uzamkne rovnaké vstupy V4 a nový plán porovnania."""
from __future__ import annotations

from collections import Counter
import csv
import os
from pathlib import Path
import shutil

from .common import ROOT, PROJECT, digest, environment, output_path, read, records, sha, utc_now, write, write_records


def config_from(path):
    cfg = read(path)
    if cfg.get('schema') != 'article-detector-adaptation-v1':
        raise ValueError('Neznáma schéma konfigurácie.')
    if cfg.get('detectors') != ['clip', 'resnet50']:
        raise ValueError('Protokol očakáva CLIP a ResNet-50 v tomto poradí.')
    for key, value in cfg['paths'].items():
        source = Path(value)
        cfg['paths'][key] = str((source if source.is_absolute() else PROJECT / source).resolve())
    t = cfg['training']
    for key in ('epochs','batch_size','accumulation_steps','samples_per_epoch','max_group_repeats','patience'):
        if not isinstance(t[key], int) or t[key] < 1:
            raise ValueError('Neplatné kladné tréningové nastavenie: ' + key)
    if t['num_workers'] != 0:
        raise ValueError('Táto verzia používa num_workers=0 pre reprodukovateľnú obnovu.')
    if t['precision'] != 'float32' or t['train_batchnorm'] or t['label_smoothing'] != 0:
        raise ValueError('Táto implementácia vyžaduje FP32, zmrazené BN a smoothing=0.')
    if t['clip_mode'] != 'linear_head_only' or t['resnet50_mode'] != 'full_after_head_warmup':
        raise ValueError('Nepodporovaný rozsah adaptácie.')
    if cfg['selector']['C'] != .1:
        raise ValueError('Zmrazené pomocné učenie selektora používa C=0.1.')
    if cfg['preprocessing'] != {'image_size': 224, 'resize_shorter': 256}:
        raise ValueError('Tento plán používa resize256/crop224 pre oba detektory.')
    if cfg['evaluation']['bootstrap'] < 1:
        raise ValueError('Bootstrap musí mať kladný počet replík.')
    return cfg


def csv_rows(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        row['label'] = int(row['label'])
    return rows


def _checked(path, expected, anchors):
    path = Path(path)
    actual = sha(path)
    if actual != expected:
        raise ValueError('Vstup nezodpovedá pôvodnému záznamu: ' + str(path))
    anchors[str(path.resolve())] = actual
    return path


def _protocol(folder, expected, anchors):
    path = _checked(Path(folder) / 'protocol.json', expected, anchors)
    if (Path(folder) / 'protocol.sha256').read_text(encoding='ascii').strip() != expected:
        raise ValueError('Neplatný pôvodný uzáver protokolu: ' + str(folder))
    return read(path)


def origin_keys(row):
    # Normalize the documented DeepShield namespaces, never rebuild OF clusters.
    values = {str(row.get(k, '')) for k in ('group','dependency_key','origin_group','sampling_group')}
    return {v.removeprefix('deepshield/') for v in values if v}


def audit_disjoint(splits):
    seen = {k: {} for k in ('sample_id','path','sha256','rgb_sha256','origin_group')}
    for split, rows in splits.items():
        for row in rows:
            if row['label'] not in (0, 1) or not row.get('sha256') or not row.get('rgb_sha256'):
                raise ValueError('Chýba trieda alebo obsahový odtlačok: ' + str(row.get('sample_id')))
            tokens = {'sample_id': {str(row.get('original_id') or row['sample_id']).split('::')[0]},
                      'path': {str(Path(row['path']).resolve()).casefold()},
                      'sha256': {row['sha256']}, 'rgb_sha256': {row['rgb_sha256']},
                      'origin_group': origin_keys(row)}
            for key, values in tokens.items():
                for value in values:
                    previous = seen[key].setdefault(value, split)
                    if previous != split:
                        raise ValueError(f'Prienik oddelení {previous}/{split} podľa {key}: {value}')
    return {'roles': {k: len(v) for k, v in splits.items()},
            'checks': list(seen), 'known_identifiers_disjoint': True,
            'scope': 'Exact recorded identity/content and documented source groups; hidden semantic/pretraining overlap is not excluded.'}


def verify(run, cfg=None):
    run = output_path(run)
    protocol = read(run / 'protocol.json')
    if sha(run / 'protocol.json') != (run / 'protocol.sha256').read_text(encoding='ascii').strip():
        raise ValueError('Zmenený protokol nového behu.')
    if cfg is not None and digest(cfg) != digest(protocol['config']):
        raise ValueError('Zmenená konfigurácia; zvoľte nový --run, pôvodný beh sa nemení.')
    if protocol['environment'] != environment():
        raise ValueError('Zmenilo sa prostredie tohto nového behu. Použite jeho Python/verzie alebo nový --run.')
    for name, expected in protocol['code'].items():
        _checked(ROOT / name, expected, {})
    for relative, expected in protocol['files'].items():
        target = (run / relative).resolve()
        if not target.is_relative_to(run):
            raise ValueError('Neplatná cesta protokolu.')
        _checked(target, expected, {})
    for name, expected in protocol['source_anchors'].items():
        _checked(name, expected, {})
    return protocol


def prepare_run(run, cfg):
    run = output_path(run)
    if (run / 'protocol.json').exists():
        return verify(run, cfg)
    run.mkdir(parents=True, exist_ok=True)
    # An interrupted preparation may be rebuilt, but never detector outputs.
    allowed = {'.run.lock', 'plans', 'snapshot', 'input_audit.json', 'protocol.sha256'}
    if any(p.name not in allowed for p in run.iterdir()):
        raise ValueError('Priečinok bez protokolu obsahuje iné výstupy; použite nový --run.')
    paths = {k: Path(v) for k,v in cfg['paths'].items()}
    anchors = {}
    data_protocol_path = _checked(paths['data']/'data_protocol.json', cfg['expected']['data_protocol'], anchors)
    if (paths['data']/'data_protocol.sha256').read_text().strip() != cfg['expected']['data_protocol']:
        raise ValueError('Zmenený dátový protokol V4.')
    data_protocol = read(data_protocol_path)
    v4_config_path = _checked(paths['efficientnet_run']/'config.json', cfg['expected']['efficientnet_config'], anchors)
    v4_config = read(v4_config_path)
    selection_file = _checked(paths['efficientnet_run']/'validation_selection.json', cfg['expected']['validation_selection'], anchors)
    source_manifests = {}
    for name, expected in data_protocol['manifests'].items():
        if Path(name).name != name:
            raise ValueError('Neplatný pôvodný manifest.')
        source_manifests[Path(name).stem] = csv_rows(_checked(paths['data']/name, expected, anchors))
    audit = audit_disjoint(source_manifests)
    train = source_manifests['train']
    selected_ids = read(selection_file)['sample_ids']
    val_by_id = {row['sample_id']: row for row in source_manifests['val']}
    if len(selected_ids) != 4880 or len(set(selected_ids)) != 4880 or not set(selected_ids) <= set(val_by_id):
        raise ValueError('Pôvodný validačný panel musí obsahovať 4 880 jedinečných uložených ID.')
    validation = [val_by_id[sample_id] for sample_id in selected_ids]
    if len(train) != 104706 or len({r.get('dependency_key') or r['group'] for r in train}) != 100224:
        raise ValueError('Tréning už nie je pôvodný manifest V4 (104 706 / 100 224).')

    sp = _protocol(paths['selector_profiles'], cfg['expected']['selector_protocol'], anchors)
    plans = {}
    for split, count in [('selector_train',40000),('selector_selection',2000),('selector_calibration',2000)]:
        relative = 'plans/' + split + '.jsonl'
        plans[split] = records(_checked(paths['selector_profiles']/relative, sp['locked_files'][relative], anchors))
        if len(plans[split]) != count:
            raise ValueError('Nesprávny počet profilov: ' + split)
        for row in plans[split]:
            row.update(dataset='DeepShield-development', split=split, label=int(row['label']))
        origins = {r['original_id'] for r in plans[split]}
        if origins != {r['sample_id'] for r in source_manifests[split]}:
            raise ValueError('Profilový plán nezodpovedá základnému manifestu: ' + split)
    plans['detector_calibration'] = []
    for row in source_manifests['detector_calibration']:
        plans['detector_calibration'].append(dict(row, original_id=row['sample_id'],
            group=row.get('dependency_key') or row['group'], dataset='DeepShield-development',
            split='detector_calibration', input_spec={'version':1,'id':'clean','operations':[]}))
    if len(plans['detector_calibration']) != 1000:
        raise ValueError('Kalibrácia detektora musí mať 1 000 vstupov.')

    ep = _protocol(paths['external_run'], cfg['expected']['external_protocol'], anchors)
    test_plan_path = _checked(paths['external_run']/'plan.jsonl', ep['files']['plan.jsonl'], anchors)
    manifest_rel = 'snapshot/reduced_manifest.csv'
    manifest_rows = csv_rows(_checked(paths['external_run']/manifest_rel, ep['files'][manifest_rel], anchors))
    manifest = {r['sample_id']:r for r in manifest_rows}
    plans['test'] = records(test_plan_path)
    for row in plans['test']:
        original = manifest[row['original_id']]
        for key in ('sha256','path','label'):
            if str(row[key]) != str(original[key]):
                raise ValueError('Testovací plán a manifest sa líšia: '+row['sample_id'])
        row['rgb_sha256'] = original['rgb_sha256']
        row['dataset'] = row['cohort']
        row['label'] = int(row['label'])
        if row['condition'] != 'clean' or row['input_spec']['operations']:
            raise ValueError('Hlavné porovnanie obsahuje iba pôvodné clean vstupy.')
    for cohort,n,groups in [('DeepShield',10000,10000),('OpenFake',4000,213)]:
        rows = [r for r in plans['test'] if r['dataset']==cohort]
        if len(rows)!=n or len({r['group'] for r in rows})!=groups or Counter(r['label'] for r in rows)!={0:n//2,1:n//2}:
            raise ValueError('Zmenená hlavná testovacia kohorta: '+cohort)
    if len(plans['test']) != 14000:
        raise ValueError('Nesprávna veľkosť testovacieho plánu.')
    audit['test_disjointness'] = audit_disjoint(dict(source_manifests, test=plans['test']))
    # Metadata preflight does not decode/test any image. Bytes are hashed on use.
    missing = []
    checked_paths = set()
    for rows in list(source_manifests.values()) + [plans['test']]:
        for row in rows:
            path = Path(row['path'])
            if str(path) in checked_paths: continue
            checked_paths.add(str(path))
            if not path.is_file(): missing.append(str(path))
            elif row.get('size_bytes') and path.stat().st_size != int(row['size_bytes']):
                raise ValueError('Zmenená veľkosť vstupného obrázka: '+str(path))
    if missing:
        raise FileNotFoundError(f'Chýba {len(missing)} obrázkov, napr. {missing[:5]}. Automatické sťahovanie je vypnuté.')
    audit['existing_image_paths'] = len(checked_paths)

    helpers = {
        'sampler_v4.py': (paths['efficientnet_run']/'snapshot/sampler_v4.py', v4_config['source_sha256']['sampler_v4.py']),
        'selector_augmentation_v4.py': (paths['selector_profiles']/'snapshot/code/selector_augmentation_v4.py', sp['locked_files']['snapshot/code/selector_augmentation_v4.py']),
        'evaluation.py': (PROJECT/'article_detector_experiment/runs/development_20261027/snapshot/code/evaluation.py', '3e939ff77551c7dcb3615203719bbeaffb6ab15220814f967d3627fa84ff89f2'),
        'selector_v4_math.py': (PROJECT/'selector_v4_math.py', 'bdc64e27ff3c5da2a8bc6fc8587e07ebddd77b966a0e897ea954f21e62f40a7f')
    }
    for name in cfg['detectors']:
        asset_dir = paths['assets']/name
        asset_manifest = asset_dir/'manifest.json'
        anchors[str(asset_manifest)] = sha(asset_manifest)
        for filename,item in read(asset_manifest)['files'].items():
            if Path(filename).name != filename:
                raise ValueError('Neplatná cesta verejného aktíva.')
            _checked(asset_dir/filename,item['sha256'],anchors)
    (run/'plans').mkdir(exist_ok=True)
    for filename,rows in [('detector_train.csv',train),('detector_validation.csv',validation)]:
        with (run/'plans'/filename).open('w',encoding='utf-8',newline='') as stream:
            writer = csv.DictWriter(stream,fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
    for split,rows in plans.items(): write_records(run/'plans'/f'{split}.jsonl',rows)
    (run/'snapshot/helpers').mkdir(parents=True,exist_ok=True)
    for name,(source,expected) in helpers.items():
        shutil.copy2(_checked(source,expected,anchors),run/'snapshot/helpers'/name)
    (run/'snapshot/code').mkdir(exist_ok=True)
    code = {p.name:sha(p) for p in ROOT.glob('*.py')}
    for name in code: shutil.copy2(ROOT/name,run/'snapshot/code'/name)
    write(run/'input_audit.json',audit)
    protocol = {'version':cfg['schema'],'created_utc':utc_now(),'config':cfg,
        'environment':environment(),'source_anchors':anchors,'code':code,
        'counts':{'detector_train':len(train),'detector_validation':len(validation),
                  **{split:len(rows) for split,rows in plans.items()}},
        'roles':{'DeepShield':'in-domain held-out','OpenFake':'cross-dataset external; historically known; retrospective reuse'},
        'training_scope':'Same V4 manifest, locked validation IDs and sampler policy; CLIP head only, ResNet-50 full after warmup; 224px vs EfficientNet 512px.',
        'selector_scope':'Refit on new detector errors; never reuse old correctness labels or fitted coefficients.',
        'decision_scope':'Full-versus-raw and full-versus-learned AURC on OpenFake, per detector; pointwise paired intervals; no guaranteed success or new independent confirmation.',
        'files':{p.relative_to(run).as_posix():sha(p) for base in ('plans','snapshot') for p in (run/base).rglob('*') if p.is_file()}}
    protocol['files']['input_audit.json'] = sha(run/'input_audit.json')
    write(run/'protocol.json',protocol)
    (run/'protocol.sha256').write_text(sha(run/'protocol.json')+'\n',encoding='ascii')
    print('Pripravený a uzamknutý nový beh:',run,flush=True)
    return protocol
