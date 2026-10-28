"""Hierarchical block bootstrap (docs/5 section 5).

Levels are resampled with replacement from the outside in (recording, event, arc, ...); within the innermost
level, moving blocks of length block_len are drawn. Observations must be in time order inside each innermost
group. `stat` receives the resampled index array, so a paired comparison applies the same indices to both models.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class Hierarchy:
    levels: Sequence[NDArray[np.int64]]  # one group label per observation and level, outermost first
    block_len: int  # moving-block length in observations, ceil(3 tau_int / T_s)


@dataclass(frozen=True)
class Interval:
    estimate: float
    lo: float
    hi: float


def _moving_blocks(idx: IntArray, block_len: int, rng: np.random.Generator) -> IntArray:
    n = idx.size
    b = min(block_len, n)
    starts = rng.integers(0, n - b + 1, -(-n // b))
    return np.concatenate([idx[s : s + b] for s in starts])[:n]


def resample(h: Hierarchy, rng: np.random.Generator) -> IntArray:
    def draw(idx: IntArray, depth: int) -> list[IntArray]:
        if depth == len(h.levels):
            return [_moving_blocks(idx, h.block_len, rng)]
        labels = h.levels[depth][idx]
        _, first = np.unique(labels, return_index=True)
        groups = [idx[labels == labels[f]] for f in np.sort(first)]
        picks = rng.integers(0, len(groups), len(groups)) if len(groups) > 1 else [0]
        return [part for p in picks for part in draw(groups[p], depth + 1)]

    n = h.levels[0].size if h.levels else 0
    return np.concatenate(draw(np.arange(n), 0))


def bootstrap_ci(
    stat: Callable[[IntArray], float], hierarchy: Hierarchy, n_boot: int = 2000, seed: int = 0, level: float = 0.95
) -> Interval:
    """Percentile interval of stat over n_boot hierarchical resamples."""
    rng = np.random.default_rng(seed)
    n = hierarchy.levels[0].size
    reps = np.array([stat(resample(hierarchy, rng)) for _ in range(n_boot)])
    alpha = (1 - level) / 2
    return Interval(stat(np.arange(n)), float(np.quantile(reps, alpha)), float(np.quantile(reps, 1 - alpha)))
