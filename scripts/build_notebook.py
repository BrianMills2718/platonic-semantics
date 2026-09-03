#!/usr/bin/env python3
"""Regenerate the Colab notebook's embedded experiment payload from the tree.

The notebook ships a base64 ZIP of the experiment so it runs in Colab with no
credentials, which matters because this repository is private. The cost of that
is a second copy of the source that can silently drift from `src/`. This script
is the only supported way to update it, and `tests/test_notebook_in_sync.py`
fails CI whenever the notebook and the tree disagree.

    python3 scripts/build_notebook.py          # rewrite the notebook
    python3 scripts/build_notebook.py --check  # exit 1 if out of sync
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import pathlib
import sys
import zipfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "semantic_relational_atlas_real_run.ipynb"
PAYLOAD_MARKER = "# 3) Unpack the complete hardened experiment embedded in this notebook"
PROJECT_DIR = "/content/semantic_relational_atlas"

PAYLOAD_FILES = [
    "run_pilot.py",
    "experiment_config.json",
    "requirements.txt",
    "src/extract_representations.py",
    "src/analyze_semantic_geometry.py",
    "src/make_report.py",
    "src/validate_with_conceptnet.py",
    "benchmark/concepts.csv",
    "benchmark/relations.csv",
]


def build_payload() -> str:
    """Byte-for-byte reproducible so --check is a real equality test."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for rel in sorted(PAYLOAD_FILES):
            path = ROOT / rel
            if not path.exists():
                raise SystemExit(f"missing payload file: {rel}")
            info = zipfile.ZipInfo(rel, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            z.writestr(info, path.read_bytes())
    return base64.b64encode(buf.getvalue()).decode("ascii")


def unpack_cell(payload: str) -> list[str]:
    return [
        PAYLOAD_MARKER + "\n",
        "# Regenerate with scripts/build_notebook.py -- do not hand-edit.\n",
        "import base64, io, zipfile, os\n",
        f'payload = """{payload}"""\n',
        f'project_dir = "{PROJECT_DIR}"\n',
        "os.makedirs(project_dir, exist_ok=True)\n",
        "with zipfile.ZipFile(io.BytesIO(base64.b64decode(payload))) as z:\n",
        "    z.extractall(project_dir)\n",
        'print("Project unpacked to:", project_dir)\n',
        'print("Files:", sorted(os.listdir(project_dir)))\n',
    ]


def embedded_payload(nb: dict) -> str | None:
    for cell in nb["cells"]:
        src = "".join(cell["source"])
        if PAYLOAD_MARKER in src:
            start = src.index('payload = """') + len('payload = """')
            return src[start : src.index('"""', start)]
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    want = build_payload()
    have = embedded_payload(nb)
    if have is None:
        raise SystemExit("no payload cell found in the notebook")

    if want == have:
        print("notebook payload is in sync with the tree")
        return 0
    if args.check:
        print(
            "notebook payload is STALE: it does not match src/ and benchmark/.\n"
            "Run: python3 scripts/build_notebook.py",
            file=sys.stderr,
        )
        return 1

    for cell in nb["cells"]:
        if PAYLOAD_MARKER in "".join(cell["source"]):
            cell["source"] = unpack_cell(want)
    NOTEBOOK.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"rewrote {NOTEBOOK.relative_to(ROOT)} ({len(want)} base64 chars)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
