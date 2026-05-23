# Future Features & Improvements

Ideas for making the simulator more biologically grounded, more publishable, and
more capable of producing interesting dynamics. Roughly ordered from "natural
extensions of current design" to "more ambitious redesigns."

---

## 1. Rougher Fitness Surface (Cosine + Perturbation)

**What**: Keep `(cos θ + 1) / 2` as the base fitness signal, but add a fixed
per-habitat perturbation function that creates local ruggedness. Options:

- **Epistatic mask**: A per-habitat binary or float mask that amplifies or
  suppresses the contribution of specific loci to the dot product. Two creatures
  with the same cosine similarity to the habitat vector but different locus
  profiles could have very different fitness. This models environment-specific
  gene expression.
- **Fourier perturbation**: Add a sum of low-frequency sinusoidal components to
  the fitness landscape — a technique from NK landscape theory for controlling
  the degree of ruggedness analytically.
- **Local optima injection**: Seed a small number of secondary "niche vectors"
  per habitat that offer a local fitness peak for specialists. The main habitat
  vector is the generalist optimum; niches reward specialization at the cost of
  generalism.

**Why**: Directly addresses the "too smooth" critique. Neutral drift, local
optima, and evolutionary stasis become possible. Even small perturbations in
high dimensions can have large effects on dynamics.

**Biological analog**: Gene-environment interactions (GxE effects), where the
same genotype has different fitness in different habitats not just because of
the main habitat signal but because specific allele combinations interact with
specific environmental features.

---

## 2. Genotype-to-Phenotype Projection Layer

**What**: Insert a learned or fixed transformation M between the gene vector g
and the vector used in fitness computation: `phenotype = M @ g` (or a nonlinear
version). Fitness is then computed as cosine similarity between `phenotype` and
the habitat's "phenotype vector" (a transformed version of the current habitat
vector).

Options for M:
- **Fixed random projection**: M is a random Gaussian matrix, fixed at
  initialization. This decouples genome space from phenotype space without
  adding any learned parameters. Creates degeneracy (multiple genomes map to
  the same phenotype) and context-dependence for free.
- **Species-specific M**: Each species lineage gets its own M, initialized from
  the founding genome at speciation. This models divergent development — the
  same gene at locus 47 means something different in a rabbit than a bird.
- **Sparse M with pleiotropy structure**: M is sparse, encoding an explicit
  gene-to-trait map where some genes affect many traits (pleiotropic) and some
  are trait-specific. The sparsity pattern could itself evolve.

**Why**: The most substantive biological critique of the current design is that
genome space and fitness space are the same. A projection layer, even a fixed
random one, creates the abstraction that real development provides. This alone
would address Critique 3 in `publication_critiques.md` and bring the model
closer to Aevol's architecture while keeping cosine similarity as the fitness
geometry.

**Biological analog**: Gene regulatory networks, protein folding, developmental
cascades — all of which mediate between DNA sequence and organismal trait.

---

## 3. Multi-Vector Habitats ("Habitat Phenotype")

**What**: Instead of a single habitat vector, give each habitat a set of
orthogonal (or near-orthogonal) environment vectors, one per "resource type" or
"selection pressure axis." Examples:
- `food_vector`: alignment determines food discovery probability
- `water_vector`: alignment determines water discovery probability
- `predator_vector`: misalignment (1 - cosine similarity) determines predator
  evasion probability — rewarding creatures that are DIFFERENT from the predator
  signature
- `pathogen_vector`: similar to predator but for disease load

Each creature's phenotype is compared to each axis independently. This creates
genuinely orthogonal selection pressures and a multi-peaked fitness landscape
even within a single habitat.

**Why**: Directly addresses Critique 4. Also creates much richer ecological
dynamics — a creature could be food-adapted but pathogen-susceptible, driving
tradeoffs. Multi-axis habitats also make the "environmental vector" concept
biologically defensible (each axis represents a measurable ecological variable).

**Biological analog**: Hutchinson's n-dimensional ecological niche concept —
each environmental axis is one dimension of the niche space.

---

## 4. Evolving Habitat Vectors (Environmental Change)

**What**: Allow habitat vectors to drift slowly over simulation time, either:
- Directionally (e.g., glacial advance — systematic rotation of the habitat
  vector over thousands of days)
- Stochastically (random walk in habitat vector space, modeling climate
  variability)
- Event-driven (catastrophic shift of the habitat vector, modeling mass
  extinction scenarios)

**Why**: Tests adaptation lag and extinction under environmental change — one of
the most important questions in current conservation biology. The cosine
geometry makes the "distance from optimum" directly measurable throughout a
run, enabling clean time-series analysis of adaptation lag.

**Biological analog**: Climate change, habitat fragmentation, island colonization.

---

## 5. Diploid Genomes with Dominance

**What**: Give each creature two gene vectors (haploid pair) instead of one.
Phenotypic trait expression uses the OWA aggregation across both vectors
combined, with the dominance parameter controlling how much the higher-valued
locus "wins" over the lower-valued one at each position. Recombination during
mating: offspring gets one strand from each parent via a crossover operator
(random crossover points in the 500-dimensional space).

**Why**: Diploidy enables heterozygosity, which buffers populations against
deleterious mutations and creates Mendelian inheritance dynamics. It also makes
hybridization between species more interesting — F1 hybrids get one strand from
each species, creating a phenotype that is a mixture of both parents' niches.
Currently hybridization between diverged species always produces an intermediate
phenotype; diploidy would enable overdominance (hybrid vigor).

**Biological analog**: Standard eukaryotic genetics. Would also enable modeling
of inbreeding depression, heterosis, and Haldane's rule.

---

## 6. Developmental Noise / Canalization

**What**: Add a per-creature developmental noise parameter (itself heritable)
that controls how much random perturbation is added to the phenotype vector
after G→P projection. High canalization (low noise) means the phenotype
faithfully reflects the genotype. Low canalization means the same genotype
produces variable phenotypes.

Selection will favor canalization in stable environments (noise is costly) but
evolvability in variable ones (noise allows exploration). This is a known
theoretical result and having it emerge from the simulator would be notable.

**Why**: Canalization (Waddington 1942) is a major concept in evo-devo.
Demonstrating its evolution from first principles in a simulation is
publishable on its own.

---

## 7. Explicit Ecology: Predator-Prey and Competition Dynamics

**What**: Allow creatures to "consume" other creatures as a food source, with
predation probability governed by cosine similarity between predator and prey
gene vectors (predators are specialized for specific prey niches). Currently
all competition is mediated through resource scarcity; direct predation is
absent.

**Why**: The current resource geometry handles producer-consumer dynamics
implicitly. Explicit predation creates Red Queen dynamics (prey evolve to
diverge from predator genomes; predators evolve to track prey) which the cosine
geometry can handle elegantly — anti-alignment from a predator's genome becomes
a survival advantage.

---

## 8. Horizontal Gene Transfer / Hybridization Mechanism

**What**: Allow low-probability gene transfer between unrelated individuals
(modeling HGT in microbes, or introgression in plants). Currently mating only
happens between males and females of compatible species. HGT would allow
occasional transfer of gene segments between lineages, creating reticulate
evolution.

**Why**: Relevant to the microbial evolution case. Also creates interesting
dynamics where a well-adapted gene cluster from one species can "invade" another
lineage. The cosine similarity speciation mechanism makes this tractable —
post-HGT genome similarity can be rechecked immediately.

---

## 9. Spatial Habitat Structure (Within-Habitat Geography)

**What**: Currently habitats are well-mixed (all creatures in a habitat compete
equally and can mate with any compatible individual). Adding a 2D spatial grid
within each habitat would enable:
- Isolation by distance (creatures more likely to mate with spatial neighbors)
- Clinal variation (gradient in habitat vector across the spatial extent)
- Parapatric speciation (speciation along a gradient without geographic barrier)

**Why**: Isolation by distance is one of the most important drivers of
speciation in nature. Parapatric speciation along environmental gradients is
well-documented (Rosenzweig's "zone of interaction" model). Having this emerge
from a spatially structured version of the cosine model would be strong
validation.

---

## 10. Adaptive Radiation Experiment Suite

**What**: Build a standardized experimental framework (separate from the main
simulator) for running controlled experiments:
- Island colonization: Single habitat, single founding individual, measure
  radiation rate
- Competitive exclusion: Two species, one habitat, measure time to exclusion vs.
  cosine distance between their gene vectors
- Mass extinction recovery: Crash a population, measure recovery rate vs.
  genetic diversity of survivors
- Allopatric speciation: Two isolated habitats, seed from same founder, measure
  speciation time vs. habitat vector divergence

These would generate the quantitative results needed for a paper and would
make the simulator a reusable research tool others could cite.

---

## Priority Order (Personal Recommendation)

For publication readiness, in roughly this order:

1. **Multi-vector habitats** (#3) — addresses the strongest biological critique
   and creates richer dynamics with minimal code change
2. **Genotype-to-phenotype projection** (#2) — addresses the genome/fitness
   conflation critique; a fixed random projection is maybe 10 lines of code
3. **Rougher fitness surface** (#1) — NK-style epistatic mask is moderate effort,
   high payoff for biological realism
4. **Adaptive radiation experiments** (#10) — needed for any paper regardless
   of other features
5. **Diploid genomes** (#5) — most ambitious but most impactful for modeling
   sexual reproduction accurately
