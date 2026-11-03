"""Výber Pythonu a overeného snapshotu bez importu numerických knižníc.

Existujúce protokoly, snapshoty ani výsledky sa pri oprave spúšťača nemenia.
Výskumný kód zostáva kontrolovaný pôvodnými SHA256 a verziami prostredia.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import runpy
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
PACKAGE = 'article_detector_experiment'
PACKAGES = ('torch', 'torchvision', 'Pillow', 'numpy', 'scipy', 'scikit-learn')
DEFAULT_ENVIRONMENT = {
    'python': '3.12.14',
    'packages': {'torch': '2.8.0+cu126', 'torchvision': '0.23.0', 'Pillow': '12.3.0',
                 'numpy': '2.5.3', 'scipy': '1.18.1', 'scikit-learn': '1.8.0'},
}
REEXEC_MARKER = 'ARTICLE_EXPERIMENT_RUNTIME_REEXEC'


class LauncherError(RuntimeError):
    pass


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def current_environment():
    versions = {}
    for name in PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return {'python': platform.python_version(), 'packages': versions}


def run_path(argv):
    # Zvyšné voľby bez zmeny spracuje pôvodné experimentálne CLI.
    # Rovnako ako pôvodné CLI prijíma aj jednoznačné skrátenie --ru/--r.
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=True)
    parser.add_argument('--run', type=Path, default=ROOT / 'runs/pilot_20261027')
    args, _ = parser.parse_known_args(argv)
    run = args.run.resolve()
    base = (ROOT / 'runs').resolve()
    if run == base or not run.is_relative_to(base):
        raise LauncherError('Výstup musí byť podpriečinok article_detector_experiment/runs.')
    return run


def frozen_protocol(run):
    """Pred importom overí protokol aj každý vykonateľný súbor snapshotu."""
    path = run / 'protocol.json'
    if not path.exists():
        return None
    try:
        if sha(path) != (run / 'protocol.sha256').read_text(encoding='ascii').strip():
            raise LauncherError('Zmenený protokol; existujúci beh sa nespustí.')
        p = json.loads(path.read_text(encoding='utf-8'))
        if p['version'] != 'article-detectors-v1':
            raise LauncherError('Nepodporovaná verzia protokolu.')
        code = p['code']
        if not isinstance(code, dict) or not code:
            raise LauncherError('Protokol neobsahuje uzamknutý kód.')
        for name, checksum in code.items():
            if Path(name).name != name or not name.endswith('.py') or '/' in name or '\\' in name:
                raise LauncherError('Neplatná cesta v uzamknutom kóde.')
            relative = 'snapshot/code/' + name
            if p['files'].get(relative) != checksum or sha(run / relative) != checksum:
                raise LauncherError('Poškodený snapshot kódu: ' + name)
        required = {'__init__.py', 'common.py', 'data.py', 'profiles.py', 'report.py',
                    'adapters.py', 'evaluation.py', 'prepare_assets.py'}
        if not required.issubset(code) or not {'experiment_cli.py', 'run_experiment.py'}.intersection(code):
            raise LauncherError('Snapshot neobsahuje úplný experimentálny kód.')
        actual = {f.name for f in (run / 'snapshot/code').iterdir()}
        if actual != set(code):
            raise LauncherError('Snapshot obsahuje chýbajúce alebo neuzamknuté súbory.')
        expected = p['environment']
        if not isinstance(expected['python'], str) or set(expected['packages']) != set(PACKAGES):
            raise LauncherError('Neplatná definícia numerického prostredia.')
        return p
    except LauncherError:
        raise
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise LauncherError('Protokol alebo snapshot nie je možné bezpečne overiť: ' + str(exc)) from exc


def probe_environment(python):
    code = (
        'import json,platform; from importlib import metadata; '
        f'names={PACKAGES!r}; '
        "print(json.dumps({'python':platform.python_version(),'packages':{n:metadata.version(n) for n in names}}))"
    )
    try:
        result = subprocess.run([str(python), '-B', '-c', code], capture_output=True,
                                text=True, encoding='utf-8', timeout=30, check=True)
        return json.loads(result.stdout)
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def select_interpreter(expected, current, candidates, probe=probe_environment):
    if current == expected:
        return None
    for candidate in candidates:
        if Path(candidate).is_file() and probe(candidate) == expected:
            return Path(candidate).resolve()
    raise LauncherError(
        f"Aktuálny Python {current['python']} nezodpovedá požadovanému {expected['python']} "
        'a overený kompatibilný interpret nie je dostupný. Obnovte prostredie '
        'the recorded package versions podľa README_SK.md. Protokol ani výsledky sa nemenili.'
    )


def dispatch(argv, run, protocol):
    """Vykoná aktuálne CLI pri novom behu alebo presne uložený kód pri obnovení."""
    old_argv = sys.argv
    try:
        if protocol is None:
            sys.argv = [str(ROOT / 'experiment_cli.py'), *argv]
            runpy.run_module(PACKAGE + '.experiment_cli', run_name='__main__')
            return
        snapshot = run / 'snapshot/code'
        for name in list(sys.modules):
            if name == PACKAGE or name.startswith(PACKAGE + '.'):
                del sys.modules[name]
        spec = importlib.util.spec_from_file_location(
            PACKAGE, snapshot / '__init__.py', submodule_search_locations=[str(snapshot)])
        package = importlib.util.module_from_spec(spec)
        sys.modules[PACKAGE] = package
        spec.loader.exec_module(package)
        common = importlib.import_module(PACKAGE + '.common')
        # Dáta zostávajú na pôvodnom mieste, vykonávaný kód pochádza zo snapshotu.
        common.ROOT, common.PROJECT = ROOT, ROOT.parent
        data = importlib.import_module(PACKAGE + '.data')
        data.ROOT = snapshot
        module = 'experiment_cli' if 'experiment_cli.py' in protocol['code'] else 'run_experiment'
        sys.argv = [str(snapshot / (module + '.py')), *argv]
        print('Používam overený uložený kód behu: ' + str(run), flush=True)
        runpy.run_module(PACKAGE + '.' + module, run_name='__main__')
    finally:
        sys.argv = old_argv


def main(argv=None):
    sys.dont_write_bytecode = True
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    argv = list(sys.argv[1:] if argv is None else argv)
    try:
        run = run_path(argv)
        if '-h' in argv or '--help' in argv:
            dispatch(argv, run, None)
            return 0
        protocol = frozen_protocol(run)
        expected = protocol['environment'] if protocol is not None else DEFAULT_ENVIRONMENT
        current = current_environment()
        candidate = Path(sys.executable)
        python = select_interpreter(expected, current, [candidate])
        if python is not None:
            if os.environ.get(REEXEC_MARKER):
                raise LauncherError('Ani overený podproces nemá správne prostredie; opakované spúšťanie bolo zastavené.')
            print(f"PyCharm spustil Python {current['python']}; experiment vyžaduje {expected['python']}.", flush=True)
            print('Automaticky používam overený interpret: ' + str(python), flush=True)
            env = os.environ.copy()
            env.update({REEXEC_MARKER: '1', 'PYTHONUTF8': '1', 'PYTHONDONTWRITEBYTECODE': '1'})
            return subprocess.call([str(python), '-B', '-u', str(ROOT / 'run_experiment.py'), *argv], env=env)
        dispatch(argv, run, protocol)
        return 0
    except LauncherError as exc:
        print('Experiment sa nespustil: ' + str(exc), file=sys.stderr, flush=True)
        return 2
