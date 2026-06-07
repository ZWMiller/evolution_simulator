<a id="evolution_simulator.speciation_math"></a>

# evolution\_simulator.speciation\_math

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

__------------------------------------------------------------------__

__Spherical k-means (for sub-cluster split detection)__

__------------------------------------------------------------------__

__We need to partition a species' members in COSINE geometry, because cosine is__

__the metric that governs mating compatibility.  "Spherical" k-means is ordinary__

__k-means run on L2-normalized vectors: once every point lies on the unit__

__sphere, squared Euclidean distance and cosine are equivalent objectives —__

__    ||x - c||^2 = 2 - 2 (x . c)     for unit x, c__

__so minimizing Euclidean distortion is the same as maximizing cosine similarity__

__to the assigned centre.  Implemented in numpy rather than scikit-learn: at this__

__scale (small per-species n, dim 245, K <= split_max_k, run only every__

__respeciate_every weeks) the optimised library buys nothing and would add a__

__heavy sklearn+scipy dependency.__


<a id="evolution_simulator.speciation_math.cos"></a>

#### cos

```python
def cos(a: np.ndarray, b: np.ndarray) -> float
```

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

<a id="evolution_simulator.speciation_math.unit_rows"></a>

#### unit\_rows

```python
def unit_rows(mat: np.ndarray) -> np.ndarray
```

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

<a id="evolution_simulator.speciation_math.kmeanspp_init"></a>

#### kmeanspp\_init

```python
def kmeanspp_init(X: np.ndarray, k: int,
                  rng: np.random.Generator) -> np.ndarray
```

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

<a id="evolution_simulator.speciation_math.spherical_kmeans"></a>

#### spherical\_kmeans

```python
def spherical_kmeans(X: np.ndarray,
                     k: int,
                     seed: int,
                     n_init: int = 3,
                     max_iter: int = 50) -> tuple[np.ndarray, np.ndarray]
```

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
