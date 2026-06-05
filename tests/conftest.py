"""Shared pytest fixtures.

The simulation threads a single ``numpy.random.Generator`` through every
randomness-bearing call (``Habitat.simulate_week``, ``Creature.reproduce``,
``SpeciesRegistry``, the ``_mate_*`` helpers, ``_gale_shapley``,
``try_spontaneous_isolation``) rather than reading the global ``np.random``
singleton.  Tests obtain that generator from the ``rng`` fixture below.
"""

import numpy as np
import pytest


@pytest.fixture
def rng() -> np.random.Generator:
    """A fresh, fixed-seed generator for a single test.

    Function-scoped, so every test gets its own identically-seeded stream and
    stays isolated and deterministic.  Thread it into any simulation call that
    now requires an explicit generator.
    """
    return np.random.default_rng(1234567)
