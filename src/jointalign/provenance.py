"""Which code produced a result: version, git commit and whether its source tree was dirty.

Since joint-align 2.0 the lines come from alignbench.provenance (one implementation for every
package): header() lists alignbench, every alignbench plugin (this package among them) and the
numeric environment. jointalign_revision() and display_path() keep their names for downstream
projects; warpalign_revision() reports warpalign if it is installed (only script 31 needs it).
"""
from __future__ import annotations

from alignbench.provenance import display_path, revision
from alignbench.provenance import header as _header

__all__ = ['display_path', 'header', 'jointalign_revision', 'warpalign_revision']


def jointalign_revision() -> str:
    """e.g. 'joint-align 2.0.0 @ 1a2b3c4 (src clean) from ~/Desktop/joint-align', or
    'joint-align 2.0.0 @ 1a2b3c4 (pinned https://…)' / '(installed, no git)'."""
    return revision('joint-align')


def warpalign_revision() -> str:
    """warpalign's provenance line (script 31's WarpTransformer), or 'warpalign (not installed)'."""
    return revision('warpalign')


def header() -> list[str]:
    """The provenance lines for the top of a result file: alignbench, joint-align, environment."""
    return _header(extra=('joint-align',))
