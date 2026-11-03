"""Spoločný vstup; automaticky zvolí kompatibilný interpreter ešte pred importom ML."""
import argparse
import os
from pathlib import Path
import subprocess
import sys


def main(default_action):
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream,"reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser=argparse.ArgumentParser(description="Porovnanie selektorov na uložených profiloch pri zmrazenom detektore.")
    parser.add_argument("--project-root",type=Path,default=Path(__file__).resolve().parents[1],help="Priečinok ImageDetection (alebo jeho rodič AIModels).")
    parser.add_argument("--run",type=Path,help="Nový oddelený experiment; predvolené selective_experiments/selector_comparison_bahat.")
    parser.add_argument("--suite",choices=("cached","full"),default="full" if default_action=="full" else "cached")
    parser.add_argument("--max-new",type=int,help="Iba extract: najviac toľko nových úplných profilov; potom bezpečne skončí.")
    parser.add_argument("--check",action="store_true",help="Iba overí integritu pripraveného porovnania.")
    args=parser.parse_args()
    project=args.project_root.resolve()
    if (project/"ImageDetection").is_dir():
        project=project/"ImageDetection"
    pinned=Path(sys.executable)
    if not pinned.exists():
        parser.error("Chýba kompatibilný Python: " + str(pinned))
    if Path(sys.executable).resolve() != pinned.resolve():
        print("Python pre porovnanie: " + str(pinned),flush=True)
        raise SystemExit(subprocess.call([str(pinned),"-B","-u",sys.argv[0],*sys.argv[1:]],
                                         env=dict(os.environ,PYTHONDONTWRITEBYTECODE="1",PYTHONIOENCODING="utf-8")))
    if args.max_new is not None and (args.max_new < 1 or default_action != "extract"):
        parser.error("--max-new je kladné číslo dostupné iba v extrakčnom skripte.")
    run=(args.run or project/"selective_experiments/selector_comparison_bahat").resolve()
    from .common import lock,verify
    try:
        with lock(run.parent/("."+run.name+".lock")):
            if args.check:
                verify(run); print("Protokol, prostredie, dáta a kód: OK.",flush=True); return
            from .data import prepare
            protocol=prepare(project,run)
            if default_action=="prepare":
                return
            if default_action=="extract":
                from .extract import extract
                extract(run,protocol,args.max_new); return
            if default_action in ("train","cached","full"):
                from .models import train
                train(run,protocol,args.suite)
            if default_action in ("test","cached","full"):
                from .report import evaluate
                evaluate(run,protocol,args.suite)
    except KeyboardInterrupt:
        print("Zastavené. Dokončené modely/profily zostávajú uložené; rovnaký príkaz nadviaže.",flush=True)
        raise SystemExit(130)
    except (ValueError,RuntimeError,FileNotFoundError) as exc:
        print("Chyba: " + str(exc),file=sys.stderr,flush=True)
        raise SystemExit(2)
