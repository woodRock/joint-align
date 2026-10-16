"""
Synthetic GC-MS batches with known retention-time drift, in the warpalign data layout.

Moved to alignbench in joint-align 2.0 (alignbench.datasets.synthetic, dataset id
'synthetic:fishlike'); this module re-exports it under the old names, with the same arrays and
truth for the same arguments and seed. See alignbench.datasets.synthetic for the generative model.
"""
from __future__ import annotations

from alignbench.axis import FISH_OIL_AXIS
from alignbench.datasets.synthetic import (
    DEFAULT_BATCHES,
    BatchSpec,
    _generate,
    _library,
    _sample,
    _truth,
    make_dataset,
    make_world,
    true_map,
)

N_BINS = FISH_OIL_AXIS.n_bins
RUN_MIN = FISH_OIL_AXIS.run_min
BIN_MIN = FISH_OIL_AXIS.bin_min
TIME_AXIS = FISH_OIL_AXIS.times

__all__ = ['BIN_MIN', 'DEFAULT_BATCHES', 'N_BINS', 'RUN_MIN', 'TIME_AXIS', 'BatchSpec', '_generate', '_library',
           '_sample', '_truth', 'make_dataset', 'make_world', 'true_map']
