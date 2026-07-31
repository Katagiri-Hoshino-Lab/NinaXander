"""Shared paired-test utilities for item-aligned evaluation outcomes."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np


def mcnemar_exact(
    x: Sequence[int],
    y: Sequence[int],
) -> tuple[int, int, float]:
    """Return discordant counts and a finite exact two-sided McNemar p-value."""

    if len(x) != len(y):
        raise ValueError(f"paired outcomes differ in length: {len(x)} != {len(y)}")
    b = sum(1 for x_i, y_i in zip(x, y) if x_i == 1 and y_i == 0)
    c = sum(1 for x_i, y_i in zip(x, y) if x_i == 0 and y_i == 1)
    n = b + c
    if n == 0:
        return b, c, 1.0

    # The exact lower binomial tail is evaluated in log space. Computing
    # sum(C(n, i)) * 2**-n directly underflows for full-set QA comparisons
    # with more than 1,074 discordant pairs and incorrectly emits p=0.
    k = min(b, c)
    log_terms = [
        math.lgamma(n + 1)
        - math.lgamma(i + 1)
        - math.lgamma(n - i + 1)
        - n * math.log(2.0)
        for i in range(k + 1)
    ]
    maximum = max(log_terms)
    log_tail = maximum + math.log(sum(math.exp(term - maximum) for term in log_terms))
    return b, c, min(1.0, math.exp(math.log(2.0) + log_tail))


def paired_bootstrap_ci(
    x: Sequence[int],
    y: Sequence[int],
    iterations: int = 10_000,
    seed: int = 0,
) -> tuple[float, float]:
    """Return a paired-bootstrap 95% interval for mean(x)-mean(y)."""

    if len(x) != len(y):
        raise ValueError(f"paired outcomes differ in length: {len(x)} != {len(y)}")
    rng = np.random.default_rng(seed)
    differences = np.asarray(x, dtype=np.int8) - np.asarray(y, dtype=np.int8)
    n = differences.size
    samples = np.empty(iterations, dtype=np.float64)
    batch_size = min(1000, iterations)
    for start in range(0, iterations, batch_size):
        stop = min(start + batch_size, iterations)
        indices = rng.integers(0, n, size=(stop - start, n))
        samples[start:stop] = differences[indices].mean(axis=1)
    samples.sort()
    return (
        float(samples[int(0.025 * iterations)]),
        float(samples[int(0.975 * iterations)]),
    )
