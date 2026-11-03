"""Ručný spúšťač pre PyCharm. Bez argumentov vykoná celý NOVÝ experiment.

Samotné vytvorenie tohto súboru nič nespúšťa. --describe vypíše plán;
--stage prepare iba pripraví a uzamkne vstupy. Výsledky patria výlučne
do article_detector_adaptation/runs/<názov>, nikdy do starých experimentov.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

sys.dont_write_bytecode = True
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from article_detector_adaptation.common import ROOT, PROJECT, lock, output_path, read, sha, write
from article_detector_adaptation.prepare import config_from, prepare_run, verify


class Tee:
    def __init__(self, terminal, logfile):
        self.terminal, self.logfile = terminal, logfile
    def write(self, value):
        self.terminal.write(value); self.logfile.write(value); self.logfile.flush()
    def flush(self):
        self.terminal.flush(); self.logfile.flush()
    def isatty(self):
        return False


def _reexec(cfg, current_python):
    """The release uses the interpreter selected by the caller."""
    return None

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--config', type=Path, default=ROOT/'config.json')
    parser.add_argument('--run', type=Path, help='Nový podpriečinok runs; relatívna cesta sa rieši voči priečinku experimentu.')
    parser.add_argument('--stage', choices=('all','prepare','train','selectors','evaluate','report'), default='all')
    parser.add_argument('--detector', choices=('both','clip','resnet50'), default='both')
    parser.add_argument('--current-python', action='store_true', help='Použi zvolený interpreter namiesto predvoleného výpočtového venv.')
    parser.add_argument('--describe', action='store_true', help='Vypíš konfiguráciu a plán bez tvorby behu či importu numerických knižníc.')
    args = parser.parse_args(argv)
    cfg = config_from(args.config.resolve())
    run = args.run or Path('runs')/cfg['run_name']
    run = output_path(run if run.is_absolute() else ROOT/run)
    if args.describe:
        print(json.dumps({'run':str(run),'stage':args.stage,'config':cfg,
            'order':['freeze_input_plan','train_clip','train_resnet50',
                     'fit_and_assess_clip_selector','fit_and_assess_resnet50_selector',
                     'retrospective_clip_test','retrospective_resnet50_test','report'],
            'status':'Description only. No run created; no numerical module imported.'},ensure_ascii=False,indent=2))
        return 0
    if args.stage in ('all','evaluate') and args.detector != 'both':
        parser.error('Záverečná fáza vyhodnotí oba detektory až po zmrazení oboch vývojových vetiev. Na jednotlivý tréning použite --stage train --detector clip/resnet50.')
    dispatched = _reexec(cfg, args.current_python)
    if dispatched is not None:
        return dispatched
    selected = cfg['detectors'] if args.detector == 'both' else [args.detector]
    with lock(run/'.run.lock'):
        protocol = prepare_run(run,cfg)
        if args.stage == 'prepare':
            print('Príprava hotová. Tréning ani inferencia v tejto fáze neprebehli.', flush=True)
            return 0
        logs = run/'logs'; logs.mkdir(exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        with (logs/(stamp+'.log')).open('x',encoding='utf-8') as stream:
            previous_out, previous_err = sys.stdout,sys.stderr
            sys.stdout,sys.stderr = Tee(previous_out,stream),Tee(previous_err,stream)
            try:
                print('Samostatný experiment:',run,flush=True)
                print('Fáza:',args.stage,'Detektory:',', '.join(selected),flush=True)
                if args.stage in ('all','train'):
                    from article_detector_adaptation.training import train_detector
                    for detector in selected:
                        train_detector(run,detector)
                if args.stage in ('all','selectors'):
                    from article_detector_adaptation.selector import fit_selector
                    for detector in selected:
                        fit_selector(run,detector)
                if args.stage in ('all','evaluate'):
                    from article_detector_adaptation.selector import evaluate_selector
                    # No test outcomes before both development branches are frozen.
                    for detector in cfg['detectors']:
                        if not (run/detector/'selector'/'complete.json').is_file():
                            raise RuntimeError('Najprv dokončite vývoj selektora pre oba detektory: '+detector)
                    freeze = run/'evaluation_freeze.json'
                    frozen = {'protocol_sha256':sha(run/'protocol.json'),
                              'development':{d:sha(run/d/'selector'/'complete.json') for d in cfg['detectors']}}
                    if freeze.exists() and read(freeze) != frozen:
                        raise ValueError('Po prvom povolení testu sa vývojové artefakty zmenili.')
                    if not freeze.exists(): write(freeze,frozen)
                    for detector in cfg['detectors']:
                        evaluate_selector(run,detector)
                if args.stage in ('all','report'):
                    from article_detector_adaptation.selector import write_summary
                    write_summary(run)
                verify(run,cfg)
                print('Fáza dokončená:',args.stage,flush=True)
            finally:
                sys.stdout,sys.stderr = previous_out,previous_err
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
        raise SystemExit(main())
    except KeyboardInterrupt:
        print('\nPrerušené. Rovnaký príkaz obnoví dokončenú epochu a platné profily.',file=sys.stderr)
        raise SystemExit(130)
    except Exception as error:
        import traceback
        traceback.print_exc()
        print('\nExperiment sa zastavil; pôvodné výsledky sa nemenia. Dôvod:',error,file=sys.stderr)
        raise SystemExit(2)
