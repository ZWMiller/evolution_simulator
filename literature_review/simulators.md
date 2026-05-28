# Evolutionary Simulators: Literature Survey

Compiled from lit review conducted May 2026. Focus: how each system represents
genomes and computes fitness/traits — the axes most relevant to this project.

---

## Tierra (Thomas Ray, 1991)

**Genome representation**: Sequences of machine instructions (assembly-like opcodes). Each organism is a self-replicating program running on a virtual CPU.
**Fitness/trait computation**: Replication speed is the implicit fitness proxy. There is no numeric gene-to-trait map — the program IS the organism.
**Speciation**: None built in. Ecological diversity emerges from parasites and hyperparasites competing in shared memory.
**Dimensionality**: Variable-length instruction strings (~80–800 instructions typical).
**Notable**: The original "digital evolution" platform. Demonstrated that parasitism and ecological complexity emerge without explicit design.
**Links**: Ray, T.S. (1991). "An approach to the synthesis of life." *Artificial Life II*, SFI Studies. [tierra.slhs.udel.edu](http://tierra.slhs.udel.edu)

---

## Avida (Ofria, Adami, Pennock et al., 2004)

**Genome representation**: Sequences of ~26 simple computer instructions (discrete, symbolic). Genome length variable, typically 50–200 instructions.
**Fitness/trait computation**: Organisms gain "merit" (metabolic rate boost) by executing Boolean logic tasks (AND, OR, NAND, EQU, etc.) using environmental inputs. Phenotype = which logic tasks the instruction sequence happens to compute. No numeric projection — fitness is entirely mediated by instruction execution.
**Speciation**: None built in. Applied post-hoc by researchers as an analysis step.
**Dimensionality**: ~50–200 discrete instruction loci.
**Notable**: The dominant academic digital evolution platform; ~1,000+ citations. Used to study major transitions in evolution, the evolution of complexity, and epistasis. Ofria lab at MSU actively maintains it.
**Links**: [avida.devosoft.org](https://avida.devosoft.org) | Ofria & Wilke (2004) *Artificial Life* 10(2).

---

## Polyworld (Larry Yaeger, 1994/2009)

**Genome representation**: Fixed-length binary string (~26–200 genes depending on version) encoding neural network architecture parameters: neuron counts per group, connection densities, color, metabolism, mutation rate. Genes are binary but decoded to continuous architectural values.
**Fitness/trait computation**: Brain architecture (number of neurons, connection topology) is derived from genome. Synaptic weights are NOT in the genome — learned via Hebbian plasticity during lifetime. Food acquisition via movement and visual perception.
**Speciation**: Not a built-in feature.
**Dimensionality**: ~26–200 binary loci decoded to continuous parameters.
**Notable**: One of the first simulators to encode neural architecture (not weights) in the genome. Demonstrated Zipf's law in agent behavior distributions. Mutation rate is one of the encoded loci — a direct precedent for heritable mutation rates.
**Links**: [polyworld.sourceforge.net](http://polyworld.sourceforge.net) | Yaeger, L. (1994). "Computational genetics, physiology, metabolism, neural systems, learning, vision, and behavior." *Artificial Life III*.

---

## Framsticks (Komosinski & Rotaru-Varga, 2001–present)

**Genome representation**: Specialized "genetic language" (f0, f1, f8 encodings) describing 3D stick-figure morphology and a neural network brain. Parameters within this language can be continuous floats.
**Fitness/trait computation**: User-defined explicit fitness function (e.g., maximize speed, maximize distance traveled). Phenotype = physical morphology + neural controller.
**Speciation**: Not emergent. Phenotypic dissimilarity heuristics available as a post-hoc analysis tool.
**Dimensionality**: Variable; grammar/language based, not a fixed-length vector.
**Notable**: One of the most flexible morphology-evolution platforms. Widely used in evolutionary robotics. Good example of a genotype-to-phenotype decoding pipeline.
**Links**: [framsticks.com](https://framsticks.com) | Komosinski, M. & Rotaru-Varga, A. (2001). *Artificial Life* 7(4).

---

## Aevol (Knibbe, Faure, Beslon et al., 2004–present)

**Genome representation**: Binary string (or 4-base ATGC in some versions) that undergoes a biologically-inspired transcription/translation into a set of "protein" Gaussian functions.
**Fitness/trait computation**: The decoded phenotype is the sum of Gaussian functions; fitness = how closely this sum matches a predefined environmental target curve. This is the closest analog in the literature to "organism vs. environment comparison," but uses L2 distance on decoded phenotypes rather than cosine similarity on raw gene vectors.
**Speciation**: Single-population model; none built in.
**Dimensionality**: Variable-length binary/nucleotide string.
**Notable**: Has a genuine genotype-to-phenotype decoding step (genome → protein Gaussians → phenotype curve → fitness), which is the multi-layer abstraction that biologists argue real evolution has. Actively developed; v9 released 2025.
**Links**: [aevol.fr](https://aevol.fr) | Knibbe et al. (2007) *PLOS Computational Biology*. bioRxiv preprint for v9: [2025.04.10.648095](https://www.biorxiv.org/content/10.1101/2025.04.10.648095).

---

## EcoSim (Phelps et al., ~2015)

**Genome representation**: Haploid real-valued vector of up to ~390 loci (alleles in range −12 to +12), encoding edges of a Fuzzy Cognitive Map (FCM). This is structurally the closest to this project's approach — a continuous float genome vector per individual.
**Fitness/trait computation**: Behavior emerges from FCM activation dynamics fed by sensory inputs. Fitness is implicit: survival and reproduction. No dot-product or cosine-similarity fitness geometry.
**Speciation**: Detected via hierarchical clustering (2-means) on genetic distance, with a similarity threshold. Functionally parallel to the progenitor-registry speciation here, though EcoSim clusters the full live population rather than comparing to stored progenitors.
**Dimensionality**: ~390 real-valued loci.
**Notable**: The most structurally similar major simulator to this project — real-valued vector genome with emergent speciation from genetic distance. Key difference: each locus is semantically mapped to a specific FCM edge (no pleiotropy via shared loci), and fitness is purely behavioral.
**Links**: Phelps, S. et al. (2015). "Speciation without pre-defined fitness functions." *PLOS ONE*. [PMC4570812](https://pmc.ncbi.nlm.nih.gov/articles/PMC4570812/)

---

## Biosim4 (David Miller, 2021)

**Genome representation**: 32-bit integers displayed in hex, encoding neural network connections (source neuron, target neuron, weight). Bit-string under the hood; genome → neural net brain.
**Fitness/trait computation**: Selection by survival in a spatial grid environment. Fitness = reaching a "survival zone" on each generation. Purely behavioral/spatial.
**Speciation**: Not implemented.
**Dimensionality**: Variable-length sequence of 32-bit "gene" integers.
**Notable**: Very popular open-source project (~12k GitHub stars). Good visual demonstration of natural selection and gene flow. Not academically published but widely referenced in public discourse about evolution.
**Links**: [github.com/davidrmiller/biosim4](https://github.com/davidrmiller/biosim4)

---

## SLiM (Haller & Messer, 2017–present)

**Genome representation**: Diploid chromosomes with individual mutations (SNPs) tracked explicitly. Each mutation has a selection coefficient and dominance. Quantitative trait phenotype = additive sum of QTL effect sizes.
**Fitness/trait computation**: Can be spatially explicit. Fitness is defined by user-written Eidos scripts; trait phenotype is additive across loci. No cosine-similarity geometry.
**Speciation**: Not emergent; researchers impose reproductive isolation rules.
**Dimensionality**: Realistically-scaled chromosomes (millions of sites). Individual creatures tracked.
**Notable**: Gold standard in academic population genetics simulation. ~1,000+ citations. Most biologically realistic in terms of mutation models, recombination, demography. Best choice if the goal is to match real genomic data. Very different design philosophy from A-Life simulators.
**Links**: [messerlab.org/slim](https://messerlab.org/slim) | Haller & Messer (2019) *Molecular Biology and Evolution* 36(3). [PMC10793872](https://pmc.ncbi.nlm.nih.gov/articles/PMC10793872/)

---

## Derrida-Higgs Speciation Model (theoretical)

**Genome representation**: Bit strings (±1 entries) of length B. Not a full simulator — a mathematical model.
**Fitness/trait computation**: No explicit fitness function. Mating is restricted by genome similarity threshold q_min. Speciation emerges from reproductive isolation driven by genetic drift.
**Speciation**: The central object of study. Speciation threshold on pairwise genome similarity is the core mechanism.
**Notable**: Well-cited theoretical framework for sympatric speciation from genetic drift alone. The concept of "speciation threshold on genome similarity" directly parallels the cosine threshold in the progenitor-registry approach here, but uses Hamming/overlap metric on bit strings and pairwise population comparison rather than a registry.
**Links**: Higgs, P.G. & Derrida, B. (1992). *Journal of Molecular Evolution* 35. Ongoing theoretical extensions: [arXiv:2603.01701](https://arxiv.org/html/2603.01701v1)

---

## ALIEN — Artificial Life Environment (Heinemann, 2019–present)

**Genome representation**: Cell clusters described by a genetic program specifying morphology, connection topology, and behavior. Continuous physics simulation.
**Fitness/trait computation**: Survival in a continuous 2D particle-physics world. Energy acquisition via metabolism. Fully emergent — no explicit fitness.
**Speciation**: Not a formal mechanism, but clusters emerge.
**Notable**: Visually impressive; simulates chemical-style energy flow and physical body structure evolution. Very different goals from this project.
**Links**: [alien-project.org](https://www.alien-project.org)

---

## Meta-Analysis & Review Papers

### Digital Evolution for Ecology Research: A Review (Dolson & Ofria, 2021)

**Citation**: Dolson, E. & Ofria, C. (2021). "Digital Evolution for Ecology Research: A Review." *Frontiers in Ecology and Evolution*, Vol. 9. DOI: [10.3389/fevo.2021.750779](https://doi.org/10.3389/fevo.2021.750779)
**Link**: [frontiersin.org/articles/10.3389/fevo.2021.750779/full](https://www.frontiersin.org/journals/ecology-and-evolution/articles/10.3389/fevo.2021.750779/full)

**What it covers**: A comprehensive survey of digital evolution platforms suitable for ecological research, authored by researchers from the Avida lab (Ofria is the creator of Avida). Reviews 13+ systems and evaluates them across a common set of axes.

**Platforms reviewed**: Avida, EcoSim, Symbulation, MABE, Aevol, Chromaria, DISHTINY, Urdar, Polyworld, Geb, Echo, Tierra, and several evolutionary computation systems.

**Comparison axes used**:
- Genome representation (computer programs vs. neural networks vs. numerical values)
- Fitness mechanism (implicit vs. explicit)
- Ecological interaction types supported (competition, predation, facilitation, parasitism, symbiosis, cooperation)
- Speciation support
- Position on a "complexity vs. tractability" spectrum

**Main conclusions**:
- Digital evolution uniquely bridges field research and mathematical models by enabling individual-based instantiation with emergent complexity and no pre-defined outcomes
- Results replicated in both digital and biological systems carry stronger evidential weight than either alone
- Most platforms cluster toward either high-complexity/low-tractability (Avida, DISHTINY) or low-complexity/high-tractability (theoretical population genetics); few occupy the middle

**Notable gaps identified**:
- Very little work on facilitative ecological interactions (commensalism); Chromaria is the only platform targeting this
- Few studies on how complex ecological interaction networks evolve their topology
- Perturbation ecology and ecosystem fragmentation are underexplored
- Geb and Polyworld have ecological foundations but have generated little published ecology research
- Practitioners face computational skill barriers; more accessible standalone implementations are needed

**Relevance to this project**: This is the primary meta-analysis paper for the field. If submitting to *Frontiers in Ecology and Evolution* or any ecology-adjacent venue, this paper must be cited prominently. The comparison axes it uses (genome type, fitness mechanism, ecological interaction support) are the natural axes for positioning this project in related-work sections. Notably, the review does not cover any simulator using cosine/dot-product genome-environment fitness geometry — the gap this project occupies.

---

## Key Comparison Table

| Simulator | Genome type | Genome dimensionality | Fitness mechanism | Emergent speciation |
|---|---|---|---|---|
| **This project** | Continuous float vector | 500 | Cosine similarity to habitat vector | Yes (progenitor registry, cosine threshold) |
| Tierra | Instruction sequence | 80–800 | Replication speed | No |
| Avida | Instruction sequence | 50–200 | Logic task execution | No (post-hoc only) |
| Polyworld | Binary → neural arch | 26–200 | Behavioral/perceptual | No |
| Framsticks | Genetic language | Variable | User-defined function | No |
| Aevol | Binary/nucleotide | Variable | L2 distance to env. target curve | No |
| EcoSim | Real-valued float vector | ~390 | Behavioral (FCM dynamics) | Yes (k-means clustering) |
| Biosim4 | Bit-string → neural net | Variable | Spatial survival | No |
| SLiM | Diploid chromosome | Millions | Additive QTL model | No (user-imposed) |
| Derrida-Higgs | Bit string | Variable | None (mating threshold only) | Yes (the whole point) |
