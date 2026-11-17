"""Repository paths. Not part of the original analysis."""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def ensure_src_on_path():
    import sys
    src = str(REPO_ROOT / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
