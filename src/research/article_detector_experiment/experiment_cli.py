"""Spustenie v PyCharme: technický pilot CLIP, potom ResNet-50, v novom priečinku."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
if __package__ in (None, ''):
    sys.path.insert(0, str(HERE.parent))

from article_detector_experiment.common import ROOT, lock, output_path
from article_detector_experiment.data import prepare, verify


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--detector', choices=('clip', 'resnet50', 'all'), default='all')
    parser.add_argument('--mode', choices=('pilot', 'development'), default='pilot')
    parser.add_argument('--run', type=Path, default=ROOT / 'runs/pilot_20261027')
    parser.add_argument('--stage', choices=('all', 'prepare', 'extract', 'analyze', 'verify'), default='all')
    parser.add_argument('--assets', type=Path, default=ROOT / 'assets')
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    parser.add_argument('--batch-size', type=int, default=6)
    parser.add_argument('--max-new', type=int)
    parser.add_argument('--bootstrap', type=int, default=1000)
    args = parser.parse_args()
    run = output_path(args.run)
    if args.bootstrap < 0:
        parser.error('--bootstrap musí byť nezáporný.')
    if args.mode == 'development' and run.name == 'pilot_20261027':
        parser.error('Celý vývojový experiment potrebuje nový --run.')
    with lock(run.parent / ('.' + run.name + '.lock')):
        if args.stage in ('prepare', 'all'):
            protocol = prepare(run, mode=args.mode)
            print('Pripravené: ' + str(run), flush=True)
            print(protocol['counts'], flush=True)
        else:
            protocol = verify(run)
        if args.stage in ('prepare', 'verify'):
            print('Kontrola integrity úspešná. Pôvodné vstupy sa nezmenili.', flush=True)
            return
        names = ('clip', 'resnet50') if args.detector == 'all' else (args.detector,)
        for detector in names:
            if args.stage in ('extract', 'all'):
                from article_detector_experiment.prepare_assets import prepare as prepare_assets
                from article_detector_experiment.profiles import extract
                prepare_assets(detector, args.assets)
                result = extract(run, detector, args.assets, device=args.device,
                                 batch_size=args.batch_size, max_new=args.max_new)
                if not result['complete']:
                    print('Čiastočný beh uložený; rovnaký príkaz pokračuje.', flush=True)
                    return
            if args.stage in ('analyze', 'all'):
                from article_detector_experiment.report import analyze, build_report
                result = analyze(run, detector, bootstrap=args.bootstrap)
                build_report(run)
                print(f'{detector}: pilotné/vývojové hodnotenie hotové; {result["profiles"]} výberových profilov.', flush=True)
        verify(run)
        print('Report: ' + str(run / 'report.html'), flush=True)


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
