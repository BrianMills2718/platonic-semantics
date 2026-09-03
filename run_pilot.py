#!/usr/bin/env python3
import argparse, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def call(args):
    print("\n$", " ".join(args))
    subprocess.run(args, check=True, cwd=ROOT)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=None)
    ap.add_argument("--prompt-mode", default="bare", choices=["bare","neutral"])
    ap.add_argument("--permutations", type=int, default=250)
    ap.add_argument("--bootstrap", type=int, default=500)
    ap.add_argument("--seed", type=int, default=20260903)
    args = ap.parse_args()

    py = sys.executable
    cmd = [py, "src/extract_representations.py", "--prompt-mode", args.prompt_mode]
    if args.device:
        cmd += ["--device", args.device]
    call(cmd)
    call([
        py, "src/analyze_semantic_geometry.py",
        "--prompt-mode", args.prompt_mode,
        "--permutations", str(args.permutations),
        "--bootstrap", str(args.bootstrap),
        "--seed", str(args.seed),
    ])
    call([py, "src/make_report.py"])

if __name__ == "__main__":
    main()
