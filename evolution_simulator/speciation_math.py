"""
Pure geometric/clustering helpers for speciation detection.

These functions have no dependency on ``SpeciesRegistry`` state — they operate
purely on the arrays handed to them — so they live here, importable and testable
on their own.  ``SpeciesRegistry`` keeps the helpers that genuinely read
instance state (``_closest_centroid``, ``_closest_candidate``,
``_rebuild_centroid_matrix``) or config knobs (``_select_clusters``); those call
into ``_spherical_kmeans`` below.

  - ``cos``               cosine similarity of two vectors
  - ``unit_rows``         L2-normalize each row (turns k-means into *spherical*)
  - ``kmeanspp_init``     k-means++ (cosine variant) seeding
  - ``spherical_kmeans``  Lloyd's spherical k-means with k-means++ restarts

# ------------------------------------------------------------------
# Spherical k-means (for sub-cluster split detection)
# ------------------------------------------------------------------
# We need to partition a species' members in COSINE geometry, because cosine is
# the metric that governs mating compatibility.  "Spherical" k-means is ordinary
# k-means run on L2-normalized vectors: once every point lies on the unit
# sphere, squared Euclidean distance and cosine are equivalent objectives —
#     ||x - c||^2 = 2 - 2 (x . c)     for unit x, c
# so minimizing Euclidean distortion is the same as maximizing cosine similarity
# to the assigned centre.  Implemented in numpy rather than scikit-learn: at this
# scale (small per-species n, dim 245, K <= split_max_k, run only every
# respeciate_every weeks) the optimised library buys nothing and would add a
# heavy sklearn+scipy dependency.
"""

import numpy as np


def cos(a: np.ndarray, b: np.ndarray) -> float:
    """
    Cosine similarity of two vectors.

    Parameters
    ----------
    a : np.ndarray
        First vector.
    b : np.ndarray
        Second vector; must be the same shape as *a*.

    Returns
    -------
    float
        Cosine similarity in [-1, 1].  Returns 0.0 if either vector has
        near-zero norm (< 1e-12) to avoid division by zero.
    """
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))


def unit_rows(mat: np.ndarray) -> np.ndarray:
    """
    L2-normalise each row of *mat* onto the unit sphere.

    This converts ordinary k-means into *spherical* k-means: on unit vectors,
    ``X @ Cᵀ`` equals cosine similarity, so all downstream assignment and
    centroid arithmetic operates in cosine geometry without explicit
    normalisation.  Rows with near-zero norm (< 1e-12) are left unscaled.

    Parameters
    ----------
    mat : np.ndarray, shape (N, D)
        Matrix whose rows should be normalised.

    Returns
    -------
    np.ndarray, shape (N, D)
        Row-normalised copy of *mat*.
    """
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    return mat / norms


def kmeanspp_init(X: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """
    Choose k initial cluster centres using the k-means++ cosine variant.

    Spreads seeds so that Lloyd's iterations are far less likely to land in a
    poor local optimum than uniform-random seeding:

      1. Pick the first centre uniformly at random from the rows of *X*.
      2. For every point, compute its cosine distance (1 − cosine) to the
         nearest centre chosen so far.
      3. Pick the next centre with probability proportional to that distance —
         points far from all current centres are most likely to be chosen.
      4. Repeat steps 2–3 until *k* centres are selected.

    Parameters
    ----------
    X : np.ndarray, shape (N, D)
        Row-normalised data matrix (unit rows assumed; use ``unit_rows`` first).
        ``X @ centresᵀ`` gives cosine similarity.
    k : int
        Number of cluster centres to select.
    rng : np.random.Generator
        Random generator for reproducible seeding.

    Returns
    -------
    np.ndarray, shape (k, D)
        The k selected initial centres (rows from *X*).
    """
    n = X.shape[0]
    centroids = [X[rng.integers(n)]]
    for _ in range(1, k):
        sims = X @ np.stack(centroids).T  # (n, chosen) cosine
        dist = np.clip(1.0 - sims.max(axis=1), 0.0, None)  # dist to nearest centre
        total = float(dist.sum())
        if total <= 1e-12:  # all points coincide
            centroids.append(X[rng.integers(n)])
        else:
            centroids.append(X[rng.choice(n, p=dist / total)])
    return np.stack(centroids)


def spherical_kmeans(
    X: np.ndarray, k: int, seed: int, n_init: int = 3, max_iter: int = 50
) -> tuple[np.ndarray, np.ndarray]:
    """
    Cluster unit-row matrix X into k clusters with spherical k-means.

    Runs Lloyd's algorithm ``n_init`` times from independent k-means++ seedings
    and keeps the best result, where "best" maximizes the total cosine
    similarity of points to their assigned centre (the spherical analogue of
    minimizing k-means inertia).

    Each Lloyd iteration:
      - **Assign**: label every point by its most-similar centre (``argmax`` of
        the cosine matrix ``X @ centresᵀ``).
      - **Update**: recompute each centre as the *spherical mean* of its members
        — sum the member unit vectors and re-normalize, which is the point on
        the sphere maximizing summed cosine to the cluster.
      - An emptied cluster is reseeded to a random point so k clusters are always
        returned.
      - Iteration stops once labels stop changing (converged) or after
        ``max_iter`` sweeps.

    ``seed`` makes the result deterministic (the whole simulation is seeded), so
    repeated runs reproduce the same partition.

    Returns
    -------
    (labels, centroids) : labels is (n,) int; centroids is (k, d) with UNIT
    rows, so callers can take dot products as cosines directly.
    """
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    best_labels: np.ndarray | None = None
    best_centroids: np.ndarray | None = None
    best_inertia = -np.inf  # total cosine-to-centre; maximize
    for _ in range(n_init):
        centroids = kmeanspp_init(X, k, rng)
        labels = np.full(n, -1, dtype=int)
        for _ in range(max_iter):
            new_labels = np.argmax(X @ centroids.T, axis=1)  # assign
            for j in range(k):  # update
                pts = X[new_labels == j]
                if len(pts) == 0:
                    centroids[j] = X[rng.integers(n)]  # reseed empty cluster
                else:
                    c = pts.sum(axis=0)
                    nrm = float(np.linalg.norm(c))
                    centroids[j] = c / nrm if nrm > 1e-12 else pts[0]  # spherical mean
            if np.array_equal(new_labels, labels):
                labels = new_labels
                break  # converged
            labels = new_labels
        inertia = float((X @ centroids.T)[np.arange(n), labels].sum())
        if inertia > best_inertia:
            best_inertia = inertia
            best_labels = labels.copy()
            best_centroids = centroids.copy()
    return best_labels, best_centroids
