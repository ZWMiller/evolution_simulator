# TODO

Cross-machine task tracking for active development items.

---

## In Progress

## Backlog

### Intelligence as predation offset
Use the `intelligence` trait as an offset to predation rate — smarter creatures
should be better at avoiding predators. Needs to be wired into the predation
calculation in `habitat.py` alongside the existing `PREDATION_ALPHA` /
`base_predation_rate` logic.

### Fix runaway speciation
3,253 species from a single founding genome over 5,000 weeks is far too many.
Two likely fixes to research and implement:

1. **Minimum population before speciation is declared** — a single-birth
   divergence should not count as a new species. Require the candidate lineage
   to sustain some minimum headcount (e.g. N individuals) before the registry
   promotes it.

2. **Stability window** — require the diverged lineage to persist for some
   minimum number of weeks without going extinct before it is recorded as a
   new species. Prevents ephemeral genetic outliers from inflating the count.

Relevant code: `SpeciesRegistry.assign_species()` in `species.py`.

---

## Done
