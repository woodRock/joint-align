"""Tests for the installable package: public API and provenance lines."""
from __future__ import annotations

import re

import jointalign
from jointalign import multibatch as mb
from jointalign import provenance


def test_public_api_reexports_multibatch():
    for name in ('Anchors', 'JointWarp', 'match_anchors', 'harvest_anchors', 'fit_joint', 'fit_pairwise',
                 'pchip_pairwise'):
        assert getattr(jointalign, name) is getattr(mb, name)
    assert set(jointalign.__all__) >= {'fit_joint', '__version__'}
    assert re.fullmatch(r'\d+\.\d+\.\d+', jointalign.__version__)


def test_jointalign_revision_in_checkout():
    # Editable install of this checkout (as in development and CI): commit, src state, home-relative path.
    rev = provenance.jointalign_revision()
    ver = re.escape(jointalign.__version__)
    assert re.fullmatch(rf'joint-align {ver} @ \w+ \(src (clean|dirty)\) from .+', rev)


def test_provenance_is_alignbench_s(monkeypatch):
    # One implementation for every package (alignbench.provenance, which tests pinned, editable and
    # plain installs); these names stay for downstream projects.
    from alignbench import provenance as ab

    assert provenance.jointalign_revision() == ab.revision('joint-align')
    assert provenance.warpalign_revision() == ab.revision('warpalign')
    lines = provenance.header()
    assert lines[0].startswith('alignbench ') and any(line.startswith('joint-align ') for line in lines)
    assert lines[-1].startswith('environment: ')


def test_display_path_is_home_relative(tmp_path):
    from pathlib import Path
    assert provenance.display_path(Path.home() / 'x' / 'y') == '~/x/y'
    assert provenance.display_path('/') == '/'
