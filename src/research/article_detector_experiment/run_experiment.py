"""Spúšťač pre PyCharm: automaticky použije overené prostredie experimentu."""
from pathlib import Path
import sys

sys.dont_write_bytecode = True
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from article_detector_experiment.launcher import main


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
    raise SystemExit(main())
