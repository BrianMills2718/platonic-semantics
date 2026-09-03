"""The Colab notebook embeds a copy of the experiment. Keep it honest.

Before this check, the notebook carried a frozen base64 ZIP plus a cell that
patched the extractor with `str.replace` and no assertion -- so once the
embedded copy changed, the patch would silently no-op and the notebook would
run different code from the repository without saying so.
"""
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_notebook_payload_matches_tree():
    r = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "build_notebook.py"), "--check"],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stdout + r.stderr
