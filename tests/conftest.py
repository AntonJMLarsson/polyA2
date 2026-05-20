"""Pytest setup for polyA2.

`support_functions` performs heavy module-level work on import: it loads
`hist_map.pickle` (saved with an older numpy and not loadable on current numpy),
then builds a sklearn KDTree from it. None of that is needed for the tests
here, so we stub the pickle load before the first import and let the KDTree
build over a single dummy vector.
"""
import os
import sys
import pickle
import numpy as np


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


class _FakeHist:
    def __init__(self):
        self._vec = np.ones(60, dtype=float)
        self._ms_len = 10


_orig_pickle_load = pickle.load


def _patched_load(f, *args, **kwargs):
    name = getattr(f, "name", "")
    if "hist_map" in str(name):
        return [_FakeHist()]
    return _orig_pickle_load(f, *args, **kwargs)


pickle.load = _patched_load
os.chdir(ROOT)

import support_functions  # noqa: F401, E402  -- import for side-effect

pickle.load = _orig_pickle_load
