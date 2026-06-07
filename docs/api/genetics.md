<a id="evolution_simulator.genetics"></a>

# evolution\_simulator.genetics

Genome layout: the single source of truth for how the 500-dimensional gene
vector is partitioned into trait / resource / compatibility loci.

Everything that needs a gene-index set imports it from here:
  - ``GENE_DIMS``                  total dimensionality of every gene vector
  - ``DEFAULT_TRAIT_GENE_INDICES`` trait name -> contributing loci (polygenic)
  - ``DEFAULT_FOOD_GENE_INDICES``  loci used for food-finding cosine geometry
  - ``DEFAULT_WATER_GENE_INDICES`` loci used for water-finding cosine geometry

It also owns the genotype -> phenotype map itself — the Ordered Weighted
Averaging (OWA) kernel and its default decay constant:
  - ``DEFAULT_OWA_ALPHA``          OWA decay rate (subclasses override via the
                                   ``Creature.OWA_ALPHA`` class attribute)
  - ``owa_aggregate``             vectorized OWA over an (N, k) block of loci
  - ``owa_value``                 scalar OWA for a single creature's loci

``Creature.TRAIT_GENE_INDICES`` and ``Habitat.FOOD_GENE_INDICES`` /
``WATER_GENE_INDICES`` default to the module-level index constants but remain
class attributes so species/habitat subclasses can override them; likewise
``Creature.OWA_ALPHA`` defaults to ``DEFAULT_OWA_ALPHA``.

<a id="evolution_simulator.genetics.owa_aggregate"></a>

#### owa\_aggregate

```python
def owa_aggregate(vals: np.ndarray, alpha: float) -> np.ndarray
```

Vectorised Ordered Weighted Averaging over a block of gene values.

For each row of *vals*: sort the k gene values descending, combine them
with normalised exponentially-decaying weights
``w_i = alpha * (1 - alpha)^i``, dot-product to get a raw signal, then
pass through sigmoid to produce a value in [0, 1].

This is the single OWA kernel shared by ``Creature._compute_trait``
(single genome), ``traits._batch_compute_traits``, and
``traits.compute_phenotype_matrix`` (batches), so the
genotype → phenotype map is defined in exactly one place.

Parameters
----------
vals : np.ndarray, shape (N, k)
    Gene values at one trait's loci for N creatures.
alpha : float
    OWA decay rate in (0, 1).  Higher values concentrate more weight on
    the top-ranked locus (faster decay).  The highest locus receives
    weight ≈ alpha, the next ≈ alpha*(1-alpha), etc., normalised to
    sum to 1.

Returns
-------
np.ndarray, shape (N,)
    Sigmoid-normalised trait values in [0, 1] for each creature.

<a id="evolution_simulator.genetics.owa_value"></a>

#### owa\_value

```python
def owa_value(vals: np.ndarray, alpha: float) -> float
```

Scalar OWA for a single genome's loci.

Convenience wrapper around ``owa_aggregate`` for the common single-creature
case.  See ``owa_aggregate`` for the full algorithm description.

Parameters
----------
vals : np.ndarray, shape (k,)
    Gene values at one trait's loci for a single creature.
alpha : float
    OWA decay rate; see ``owa_aggregate``.

Returns
-------
float
    Sigmoid-normalised trait value in [0, 1].

<a id="evolution_simulator.genetics.DEFAULT_FOOD_GENE_INDICES"></a>

#### DEFAULT\_FOOD\_GENE\_INDICES

158 total indices

<a id="evolution_simulator.genetics.DEFAULT_WATER_GENE_INDICES"></a>

#### DEFAULT\_WATER\_GENE\_INDICES

175 total indices
