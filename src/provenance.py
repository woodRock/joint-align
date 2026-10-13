"""Which gcmsalign produced a result: version, git commit and whether its source tree was dirty.

gcmsalign is installed editable from ~/Desktop/alignment, so its code (and the data under its
data/fish_oil) can change between runs. Every result file records this line.
"""
from __future__ import annotations

import subprocess

import gcmsalign
from gcmsalign.config import REPO_ROOT


def _git(*args: str) -> str:
    try:
        return subprocess.run(['git', '-C', str(REPO_ROOT), *args], capture_output=True,
                              text=True, check=True, timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ''


def gcmsalign_revision() -> str:
    """e.g. 'gcmsalign 2.0.0 @ 5b75419 (src clean)'; 'unknown commit' outside a git checkout."""
    from importlib.metadata import PackageNotFoundError, version
    try:
        ver = version('gcmsalign')
    except PackageNotFoundError:
        ver = '?'
    sha = _git('rev-parse', '--short', 'HEAD') or 'unknown commit'
    dirty = _git('status', '--porcelain', '--', 'src')
    state = 'src dirty' if dirty else 'src clean'
    return f'gcmsalign {ver} @ {sha} ({state}) from {gcmsalign.__file__.rsplit("/src/", 1)[0]}'
