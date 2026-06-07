<a id="evolution_simulator.traits"></a>

# evolution\_simulator.traits

Trait catalogs and bulk (population-level) trait measurement.

This module owns *what traits exist and how to measure them across many
creatures*, kept separate from the ``Creature`` class (which only computes a
single trait from its own genes via ``Creature._compute_trait``):

  - ``LOGGED_TRAITS`` / ``_TRAIT_SCALING`` / ``_TRAIT_INTERNAL_KEY`` and
    ``_batch_compute_traits`` — the scaled biological-range traits recorded in
    per-habitat / per-species statistics, computed in one vectorized pass.
  - ``PHENOTYPE_TRAITS`` / ``compute_phenotype_matrix`` / ``compute_phenotype``
    — the raw [0, 1] phenotype axis used for anagenesis / morphological-
    divergence detection.

Layering: this module imports the gene loci and the OWA kernel from
``genetics`` only — it does *not* import ``Creature`` (the OWA decay for a batch
is read from the creatures' own class at call time in ``_batch_compute_traits``).
The DAG is ``genetics <- traits`` and ``genetics <- creature``, with no edge
between ``creature`` and ``traits``.

<a id="evolution_simulator.traits.compute_phenotype_matrix"></a>

#### compute\_phenotype\_matrix

```python
def compute_phenotype_matrix(
        gene_matrix: np.ndarray,
        owa_alpha: float = DEFAULT_OWA_ALPHA,
        trait_gene_indices: dict | None = None,
        traits: tuple[str, ...] = PHENOTYPE_TRAITS) -> np.ndarray
```

Vectorized OWA phenotype values for a batch of genomes.

Returns the RAW sigmoid value in [0, 1] for each trait (not scaled to
biological ranges) so that cosine comparisons treat every trait on the same
footing rather than letting wide-range traits like max_lifespan dominate.

Parameters
----------
gene_matrix : (N, GENE_DIMS) float array
owa_alpha   : OWA decay rate (defaults to DEFAULT_OWA_ALPHA)
trait_gene_indices : trait→loci map (defaults to DEFAULT_TRAIT_GENE_INDICES)
traits      : ordered trait names to include (defaults to PHENOTYPE_TRAITS)

Returns (N, len(traits)) float64 array of raw [0, 1] values.

<a id="evolution_simulator.traits.compute_phenotype"></a>

#### compute\_phenotype

```python
def compute_phenotype(genes: np.ndarray,
                      owa_alpha: float = DEFAULT_OWA_ALPHA) -> np.ndarray
```

Phenotype vector for a single genome.

Convenience wrapper around ``compute_phenotype_matrix`` for the
single-genome case.

Parameters
----------
genes : np.ndarray, shape (GENE_DIMS,)
    Full 500-dimensional gene vector for one creature.
owa_alpha : float, optional
    OWA decay rate; see ``owa_aggregate``.  Defaults to
    ``DEFAULT_OWA_ALPHA``.

Returns
-------
np.ndarray, shape (len(PHENOTYPE_TRAITS),)
    Raw sigmoid values in [0, 1] for each trait in ``PHENOTYPE_TRAITS``.
