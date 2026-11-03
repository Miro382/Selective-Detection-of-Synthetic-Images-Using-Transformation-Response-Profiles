"""Uzamkne interný predtestový plán; neoznačuje metadáta za overené obrázky."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import uuid

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))
from article_ablation_confirmation import freeze_models, prepare_candidate_plan

VERSION = 'article-ablation-pretest-design-v1'
DEFAULT_MODELS = ROOT / 'frozen_models/resize_vs_full_20261028'
DEFAULT_DATA = ROOT / 'data_plan_20261028'
DEFAULT_OUTPUT = ROOT / 'protocols/strict_20261028'
COUNTS = {'DeepShield': 1500, 'OpenFake': 6000}
MATCHED = {'DeepShield': 0.9807, 'OpenFake': 0.90975}


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            value.update(block)
    return value.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_new(path, value):
    with Path(path).open('xb') as stream:
        stream.write((json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + '\n').encode('utf-8'))
        stream.flush()
        os.fsync(stream.fileno())


def allowed_output(path):
    path = Path(path).resolve()
    if path.parent != (ROOT / 'protocols').resolve() or path.name.startswith('.'):
        raise ValueError('Výstup musí byť nový article_ablation_confirmation/protocols/<názov>.')
    return path


def validate_plan(folder):
    prepare_candidate_plan.check(folder)
    from collections import Counter
    with (folder / 'selected_metadata.jsonl').open(encoding='utf-8') as stream:
        rows = [json.loads(line) for line in stream]
    counts = Counter(row['dataset'] for row in rows)
    if dict(counts) != COUNTS:
        raise ValueError('Počet plánovaných vzoriek nezodpovedá dohodnutému návrhu.')
    return counts


def verify(output):
    output = allowed_output(output)
    protocol = read(output / 'protocol.json')
    expected = (output / 'protocol.sha256').read_text(encoding='ascii').strip()
    receipt = read(output / 'design.complete.json')
    if (sha(output / 'protocol.json') != expected or receipt['protocol_sha256'] != expected
            or protocol['version'] != VERSION or receipt['ready_for_inference'] is not False
            or protocol['ready_for_inference'] is not False):
        raise ValueError('Poškodený predtestový návrh alebo zmenený význam dokončenia.')
    for name, checksum in protocol['files'].items():
        path = (output / name).resolve()
        if not path.is_relative_to(output) or sha(path) != checksum:
            raise ValueError('Zmenená uzamknutá kópia: ' + name)
    for name, checksum in protocol['anchors'].items():
        if sha(name) != checksum:
            raise ValueError('Zmenený predtestový zdroj: ' + name)
    freeze_models.verify(Path(protocol['frozen_models']))
    validate_plan(Path(protocol['metadata_plan']))
    return protocol


def freeze(output=DEFAULT_OUTPUT, models=DEFAULT_MODELS, data=DEFAULT_DATA):
    output = allowed_output(output)
    models, data = Path(models).resolve(), Path(data).resolve()
    if output.exists():
        protocol = verify(output)
        if protocol['frozen_models'] != str(models) or protocol['metadata_plan'] != str(data):
            raise ValueError('Existujúci návrh patrí iným zdrojom.')
        return protocol
    model_info = freeze_models.verify(models)
    counts = validate_plan(data)
    from article_ablation_confirmation import statistics
    # Implementácia musí zodpovedať vopred oznámenému primárnemu pravidlu.
    if (not callable(getattr(statistics, 'evaluate', None)) or statistics.REPETITIONS != 5000
            or statistics.SEED != 20261028 or statistics.COHORTS != ('DeepShield', 'OpenFake')
            or statistics.MATCHED_FRACTIONS != {'DeepShield': (9807, 10000), 'OpenFake': (3639, 4000)}
            or statistics.MIN_DISCORDANT_GROUPS != 10):
        raise ValueError('Štatistický evaluátor nezodpovedá vopred určenému pravidlu.')
    input_files = [models / name for name in ('manifest.json', 'manifest.sha256', 'complete.json')]
    input_files += [data / name for name in ('protocol.json', 'protocol.sha256', 'metadata_complete.json')]
    input_files += [ROOT / name for name in ('freeze_models.py', 'prepare_candidate_plan.py', 'statistics.py', 'freeze_protocol.py')]
    anchors = {str(path): sha(path) for path in input_files}
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = output.with_name('.' + output.name + '.staging-' + uuid.uuid4().hex)
    stage.mkdir()
    (stage / 'code').mkdir()
    for name in ('statistics.py', 'freeze_protocol.py'):
        shutil.copyfile(ROOT / name, stage / 'code' / name)
    protocol = {
        'version': VERSION, 'created_utc': datetime.now(timezone.utc).isoformat(),
        'scope': 'Interný predtestový plán nového porovnania; nie vykonaný test ani verejná registrácia.',
        'statistical_protocol_locked': True, 'adoption_rule_locked': True,
        'ready_for_inference': False, 'images_downloaded': False, 'all_images_verified': False,
        'new_results_computed': False, 'frozen_models': str(models), 'metadata_plan': str(data),
        'counts': dict(counts), 'environment': model_info['environment'], 'anchors': anchors,
        'comparison': {'candidate': 'lr_resize_only', 'reference': 'frozen_v4',
                       'direction': 'candidate-minus-reference', 'model_training': 'none',
                       'predictions': 'Original deterministic EfficientNet prediction, identical errors for both selectors.',
                       'threshold_recalibration': 'none; comparison is by ranking at fixed coverage, not certified deployment.'},
        'primary_endpoints': {'cohorts': ['DeepShield', 'OpenFake'], 'metrics': ['aurc', 'matched_fixed_coverage_risk'],
                              'matched_coverages': MATCHED, 'accepted_rule': 'floor(q * N), identical k for both models',
                              'planned_accepted': {'DeepShield': 1471, 'OpenFake': 5458},
                              'bootstrap_accepted_rule': 'floor(q * N_replica), recomputed inside every paired group replicate',
                              'historical_absolute_k_reused': False, 'new_full_threshold_k_used': False},
        'secondary_endpoints': {'risk_at_coverage': [.8, .9, .95], 'decision_role': 'descriptive_only'},
        'inference': {'repetitions': 5000, 'seed': 20261028, 'resampling': 'whole paired origin/sampling groups',
                      'interval': 'two-sided pointwise percentile 95%, quantiles .025 and .975',
                      'aurc_ties': 'complete equal-score blocks',
                      'coverage_ties': 'predefined SHA256 sample identity without labels/errors',
                      'min_discordant_groups_per_cohort': 10,
                      'adequacy_note': 'Conservative support/variability checks are not a guarantee of exact bootstrap coverage.',
                      'fixed_realization': 'fixed previously trained detector and selectors; no retraining uncertainty'},
        'adoption': {'required_upper_ci_bound': 'strictly less than zero for all four primary contrasts',
                     'logic': 'intersection-union: all conditions required; intervals are not simultaneous',
                     'allowed_deterioration': 0, 'noninferiority_claim': False,
                     'pass': 'eligible_for_replacement', 'otherwise': 'retain_full',
                     'missing_inadequate_degenerate_or_zero_boundary': 'retain_full',
                     'automatic_model_or_thesis_edit': False},
        'stopping': {'fixed_metadata_sample_size': True, 'outcome_based_extension': False,
                     'outcome_based_metric_or_model_change': False,
                     'unfilled_content_audit_quota': 'incomplete; do not silently lower quotas or report full test'},
        'remaining_before_inference': ['Download according to fixed candidate/reserve order.',
                                       'Verify bytes, decoded RGB, pHash and origin/sampling closure against complete known history.',
                                       'Publish verified image manifest and final data receipt with this design/model digest.',
                                       'Validate execution adapter on historical technical inputs, then seal execution code before new inference.'],
        'limitations': ['N is based on available unseen metadata, not a demonstrated power calculation.',
                        'The small historical matched-risk advantage may remain unresolved with this sample size.',
                        'DeepShield quota mixture differs from the historical 10000-image mixture.',
                        'Incomplete historical perceptual hashes and unknown public pretraining remain documented limitations.'],
    }
    protocol['files'] = {path.relative_to(stage).as_posix(): sha(path) for path in stage.rglob('*') if path.is_file()}
    for name, checksum in anchors.items():
        if sha(name) != checksum:
            raise ValueError('Zdroj sa zmenil počas uzamykania návrhu.')
    write_new(stage / 'protocol.json', protocol)
    protocol_sha = sha(stage / 'protocol.json')
    (stage / 'protocol.sha256').write_text(protocol_sha + '\n', encoding='ascii')
    write_new(stage / 'design.complete.json', {'version': VERSION, 'protocol_sha256': protocol_sha,
              'design_frozen': True, 'statistical_protocol_locked': True, 'adoption_rule_locked': True,
              'ready_for_inference': False, 'new_results_computed': False})
    os.rename(stage, output)
    return verify(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--models', type=Path, default=DEFAULT_MODELS)
    parser.add_argument('--data-plan', type=Path, default=DEFAULT_DATA)
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    result = verify(args.output) if args.verify else freeze(args.output, args.models, args.data_plan)
    print(json.dumps({'design_frozen': True, 'ready_for_inference': result['ready_for_inference'],
                      'counts': result['counts'], 'adoption': result['adoption'], 'path': str(args.output)}, ensure_ascii=False))


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
    main()
