"""Tests for the installable package: public API and provenance lines."""
from __future__ import annotations

import re
import shutil

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
    rev = provenance.jointalign_revision()
    assert re.fullmatch(rf'many-batch-align {re.escape(jointalign.__version__)} @ \w+ \(src (clean|dirty)\)', rev)


def test_jointalign_revision_installed_without_git(tmp_path, monkeypatch):
    # A copy of the package outside any git checkout, as in site-packages.
    fake = tmp_path / 'lib' / 'site-packages' / 'jointalign' / '__init__.py'
    fake.parent.mkdir(parents=True)
    shutil.copy(jointalign.__file__, fake)
    monkeypatch.setattr(jointalign, '__file__', str(fake))
    assert provenance.jointalign_revision() == f'many-batch-align {jointalign.__version__} (installed, no git)'


def test_jointalign_revision_inside_another_repo(tmp_path, monkeypatch):
    # site-packages of a venv inside some other git repository must not report that repository.
    other = tmp_path / 'paper-two'
    fake = other / '.venv' / 'lib' / 'jointalign' / '__init__.py'
    fake.parent.mkdir(parents=True)
    fake.touch()
    provenance._git('init', '-q', cwd=other)
    monkeypatch.setattr(jointalign, '__file__', str(fake))
    assert provenance.jointalign_revision().endswith('(installed, no git)')


def test_display_path_is_home_relative(tmp_path):
    from pathlib import Path
    assert provenance.display_path(Path.home() / 'x' / 'y') == '~/x/y'
    assert provenance.display_path('/') == '/'
