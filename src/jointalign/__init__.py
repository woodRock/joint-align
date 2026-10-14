"""
jointalign: joint retention-time alignment of GC-MS chromatograms across many batches.

One monotone warp per batch into a shared retention-time frame, fitted jointly to spectral anchors
from every cross-batch sample pair (the joint model). Distribution name: joint-align.

Submodules: multibatch (the joint model and its baselines, re-exported here), synthetic (public
stand-in data with known drifts), provenance (revision lines for results) and neural_joint (the
neural refinement; needs torch, so it is not imported here).
"""
from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from jointalign.multibatch import (
    Anchors,
    JointWarp,
    fit_joint,
    fit_pairwise,
    harvest_anchors,
    match_anchors,
    pchip_pairwise,
)

try:
    __version__ = version('joint-align')
except PackageNotFoundError:  # pragma: no cover - imported from a source tree that was never installed
    __version__ = 'unknown'

__all__ = [
    'Anchors',
    'JointWarp',
    '__version__',
    'fit_joint',
    'fit_pairwise',
    'harvest_anchors',
    'match_anchors',
    'pchip_pairwise',
]
