"""Rebuild the article's numerical tables and figures in a separate work copy."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'work/manuscript',
                        help='New output directory. It must not contain existing files.')
    args=parser.parse_args()
    destination=args.output.resolve()
    protected=(ROOT/'analysis').resolve(),(ROOT/'evidence').resolve(),(ROOT/'models').resolve(),(ROOT/'src').resolve()
    if destination==ROOT or any(destination==p or destination.is_relative_to(p) for p in protected):
        parser.error('Choose a work directory outside the packaged source, model, and evidence directories.')
    if destination.exists():
        parser.error('Output directory already exists; choose a new directory.')
    shutil.copytree(ROOT/'analysis/manuscript',destination,
                    ignore=shutil.ignore_patterns('__pycache__','.mplconfig','*.pyc'))
    commands=[['build_tables.py'],['build_figures.py','--from-aggregate'],
              ['build_adaptation_assets.py'],['build_detector_history.py'],
              ['build_historical_transfer.py'],['build_selector_coverage_figures.py'],
              ['build_selector_history_tables.py'],['build_contribution_figure.py'],
              ['build_retrospective_table.py'],['class_analysis/build_class_analysis.py'],
              ['cluster_sensitivity/analyze.py']]
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1')
    for command in commands:
        print('Running: '+ ' '.join(command),flush=True)
        subprocess.run([sys.executable,*command],cwd=destination,env=env,check=True)
    print('Generated figures and tables: '+str(destination))


if __name__=='__main__':
    main()
