"""
Experiment: how far does phenotype drift from the founding type over many
generations, and does it plateau?

Motivation
----------
The anagenesis detector fires when a species' living PHENOTYPE centroid drifts
below ``anagenesis_threshold`` (centred cosine) from its frozen founding type.
A short (1500-week) run floored at ~0.93, which made a 0.80 default inert.  But
phenotype here is selection-driven: a population adapts toward its FIXED habitat
optimum and then plateaus, so drift is bounded by the founder→optimum distance.
This script measures the drift trajectory over many generations to find where it
plateaus, so the anagenesis threshold can be set to a biologically meaningful
"substantially transformed" level rather than to an artifact of run length.

Method
------
Run the multi-habitat long config with anagenesis DISABLED (threshold 0) so each
species' type stays frozen and we observe raw cumulative drift; cladogenesis
stays on (real dynamics).  Every SAMPLE weeks, print the min/median centred
phenotype-centroid-to-type cosine across living species, plus each surviving
founder's own cosine, alongside an approximate generation count.

Run
---
    poetry run python experiments/drift_trajectory.py
"""
import sys
from pathlib import Path

import numpy as np

# Make the project importable regardless of the invoking working directory.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evolution_simulator.simulation import SimulationRunner
from evolution_simulator.creature import compute_phenotype_matrix

WEEKS = 12000
SAMPLE = 500
CONFIG = Path(__file__).resolve().parent.parent / "configs" / "multi_habitat_long.toml"


def centred_cos(a: np.ndarray, b: np.ndarray) -> float:
    a = a - 0.5
    b = b - 0.5
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(np.dot(a, b) / (na * nb)) if na and nb else 1.0


def main() -> None:
    r = SimulationRunner(CONFIG)
    r.config["simulation"]["weeks"] = WEEKS
    r.setup()
    reg = r.species_registry
    reg.anagenesis_threshold = 0.0  # disable anagenesis so types stay frozen
    founders = list(reg.all_species)

    gen = np.mean([
        c.weeks_to_sexual_viability
        for h in r.habitats.values()
        for c in h.alive_creatures
    ])
    print(f"approx generation length: {gen:.1f} weeks  ->  "
          f"{WEEKS} weeks ~= {WEEKS / gen:.0f} generations")
    print(f"{'week':>6} {'gens':>5} {'living':>6} {'min_cos':>8} {'med_cos':>8}  founders_alive")

    for w in range(1, WEEKS + 1):
        r.step()
        if sum(h.population_size for h in r.habitats.values()) == 0:
            print(f"extinct at week {w}")
            break
        if w % SAMPLE != 0:
            continue

        alive = [c for h in r.habitats.values() for c in h.alive_creatures]
        bysp: dict[str, list] = {}
        for c in alive:
            bysp.setdefault(c.species, []).append(c)

        vals = []
        for sp, ms in bysp.items():
            if sp not in reg._type_phenotype:
                continue
            cen = compute_phenotype_matrix(np.stack([m.genes for m in ms])).mean(0)
            vals.append(centred_cos(cen, reg._type_phenotype[sp]))

        fnd = []
        for f in founders:
            ms = bysp.get(f)
            if ms:
                cen = compute_phenotype_matrix(np.stack([m.genes for m in ms])).mean(0)
                fnd.append(f"{centred_cos(cen, reg._type_phenotype[f]):.3f}")

        print(f"{w:>6} {w / gen:>5.0f} {len(alive):>6} "
              f"{min(vals):>8.4f} {np.median(vals):>8.4f}  {fnd}")


if __name__ == "__main__":
    main()
