"""Spusti cez Run: overí dáta V4 a začne/obnoví samostatný tréning V4.

Pre dokončený DeepShield použi prepare_training_v4.py podľa návodu.
Predvolene sa pokračuje v detector_experiments/detector_v4/training_main.
Nový porovnávací pokus potrebuje vlastný --output-dir. Existujúci V3 sa nemení.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def compute_python(project):
    return sys.executable

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--python")
    parser.add_argument("--help-training", action="store_true")
    args, extra = parser.parse_known_args(argv)
    project = args.project_root.resolve()
    data = (args.data_dir or project / "detector_experiments" / "detector_v4" / "data").resolve()
    run = (args.resume or args.output_dir or data.parent / "training_main").resolve()
    if args.resume and args.output_dir and args.resume.resolve() != args.output_dir.resolve():
        parser.error("--resume a --output-dir sa nezhodujú.")
    engine = project / "detector_training_v4.py"
    python = args.python or compute_python(project)
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
    if args.help_training:
        return subprocess.run([python, "-B", str(engine), "--help"], env=env, cwd=project).returncode
    if not (data / "data_protocol.json").is_file():
        raise ValueError(f"Dáta V4 ešte nie sú uzamknuté: {data}. Spusti prepare_training_v4.py.")
    saved_args = []
    if (run / "config.json").is_file() and "--check" not in extra and "--smoke" not in extra:
        config = json.loads((run / "config.json").read_text(encoding="utf-8"))
        if config.get("version") != 4:
            raise ValueError("Cieľový priečinok nepatrí tréningu V4.")
        engine = run / "snapshot" / "detector_training_v4.py"
        import hashlib
        expected = config["source_sha256"]["detector_training_v4.py"]
        if not engine.is_file() or hashlib.sha256(engine.read_bytes()).hexdigest() != expected:
            raise ValueError("Zmrazený tréningový skript sa zmenil.")
        if Path(config["training"]["data_dir"]).resolve() != data:
            raise ValueError("Obnovenie musí používať pôvodný dátový priečinok.")
        python = args.python or config["environment"]["executable"]
        for key, value in config["training"].items():
            if key == "data_dir" or value is None:
                continue
            if isinstance(value, bool):
                if value:
                    saved_args.append("--" + key.replace("_", "-"))
            else:
                saved_args.extend(["--" + key.replace("_", "-"), str(value)])
    command = [python, "-B", "-u", str(engine), "--data-dir", str(data), "--output-dir", str(run), *saved_args, *extra]
    if (run / "last_training_state.pt").is_file() and "--check" not in extra and "--smoke" not in extra:
        command.extend(["--resume", str(run)])
    print(f"Python: {python}\nDáta: {data}\nBeh V4: {run}", flush=True)
    return subprocess.run(command, env=env, cwd=project).returncode


# Release entry point: require an explicit research-job request.
if __name__ == '__main__':
    import sys as _release_sys
    if not any(arg in _release_sys.argv for arg in ('--help', '-h', '--execute-research')):
        print('Research source. Read src/RESEARCH_CODE.txt before running a job. '
              'Use --help to inspect options or --execute-research to run explicitly.')
        raise SystemExit(0)
    if '--execute-research' in _release_sys.argv:
        _release_sys.argv.remove('--execute-research')

if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Prerušené. Rovnaký Run obnoví poslednú dokončenú epochu.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(f"Chyba: {exc}", file=sys.stderr)
        raise SystemExit(2)
