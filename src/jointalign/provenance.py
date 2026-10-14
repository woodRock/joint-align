"""Which code produced a result: version, git commit and whether its source tree was dirty.

gcmsalign_revision(): gcmsalign is installed editable from ~/Desktop/alignment, so its code (and
the data under its data/fish_oil) can change between runs. Every paper-one result file records
this line.

jointalign_revision(): the same for this package (many-batch-align), for downstream projects that
must record which version of the joint model produced their numbers.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import gcmsalign
from gcmsalign.config import REPO_ROOT


def _git(*args: str, cwd: Path | str = REPO_ROOT) -> str:
    try:
        return subprocess.run(['git', '-C', str(cwd), *args], capture_output=True,
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
    return f'gcmsalign {ver} @ {sha} ({state}) from {display_path(gcmsalign.__file__.rsplit("/src/", 1)[0])}'


def display_path(p) -> str:
    """A path for reports that are committed: home-relative (~/...), never the user's absolute path."""
    p, home = Path(p).expanduser().resolve(), Path.home().resolve()
    try:
        return f'~/{p.relative_to(home)}'
    except ValueError:
        return str(p)


def jointalign_revision() -> str:
    """e.g. 'many-batch-align 1.0.0 @ 1a2b3c4 (src clean)'. Installed non-editable (no git checkout
    around the package): 'many-batch-align 1.0.0 (installed, no git)'."""
    from importlib.metadata import PackageNotFoundError, version

    import jointalign
    try:
        ver = version('many-batch-align')
    except PackageNotFoundError:
        ver = '?'
    root = Path(jointalign.__file__).resolve().parents[2]    # <root>/src/jointalign/__init__.py
    top = _git('rev-parse', '--show-toplevel', cwd=root)
    if not top or Path(top).resolve() != root:    # site-packages, possibly inside some other repo
        return f'many-batch-align {ver} (installed, no git)'
    sha = _git('rev-parse', '--short', 'HEAD', cwd=root) or 'unknown commit'
    state = 'src dirty' if _git('status', '--porcelain', '--', 'src', cwd=root) else 'src clean'
    return f'many-batch-align {ver} @ {sha} ({state})'
