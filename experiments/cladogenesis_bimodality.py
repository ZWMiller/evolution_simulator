"""
Experiment: visualise the bimodality that the k-means cladogenesis split
detector acts on as a single founding species diverges across two isolated
habitats.

Method
------
Two isolated habitats (Forest + Plains), one shared founding genome biased
toward the average of both habitat vectors so the population survives in both.
Every SAMPLE_EVERY weeks the compatibility gene vectors (245-dim) of all
living creatures are extracted, unit-normalised, and two projections are
plotted as histograms coloured by habitat:

  Panel A — Stable separation-axis projection
    axis = normalise(Plains_centroid − Forest_centroid)
    where centroids are the per-habitat mean unit vectors (NOT k-means labels).
    This removes label-switching: the axis always points from Forest toward
    Plains, so the histograms are consistent across every frame.

  Panel B — Cosine similarity to Forest centroid
    Simpler: each creature's cosine similarity to the Forest habitat mean.
    Forest creatures cluster high; Plains creatures cluster lower.
    Both panels show the same bimodal signal; Panel A separates the peaks
    further by cancelling shared adaptations.

The k-means K=2 centroid cosine is still computed (it's what the detector
measures) and tracked in the time-series alongside the per-habitat centroid
cosine, so you can see how noisy the k-means signal is vs. the ground truth.

Output
------
experiments/bimodality_output/<timestamp>/week_NNNNN.png  (one per sample)
experiments/bimodality_output/<timestamp>/centroid_cosine.png

Run
---
    poetry run python experiments/cladogenesis_bimodality.py
"""

import sys
from datetime import datetime
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")  # headless — no display needed

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evolution_simulator.creature import Creature
from evolution_simulator.genetics import GENE_DIMS
from evolution_simulator.simulation import SimulationRunner
from evolution_simulator.speciation_math import spherical_kmeans

CONFIG = Path(__file__).resolve().parent / "simulation_configs" / "bimodality_experiment.toml"
CREATURES_PER_HABITAT = 500  # 250 female + 250 male per habitat
GENOME_NOISE = 0.05
SAMPLE_EVERY = 10  # how often (weeks) to snapshot and plot
WEEKS = 3000


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def unit_rows(X: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    return X / norms


def unit_vec(v: np.ndarray) -> np.ndarray:
    nrm = float(np.linalg.norm(v))
    return v / nrm if nrm > 1e-12 else v


def separation_axis(c0: np.ndarray, c1: np.ndarray) -> np.ndarray:
    """Unit vector pointing from c0 to c1 (the direction of cluster difference)."""
    return unit_vec(c1 - c0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    r = SimulationRunner(CONFIG)
    r.config["simulation"]["weeks"] = WEEKS
    r.setup()

    reg = r.species_registry
    hab_ids = list(r.habitats.keys())  # ["forest_1", "plains_1"]
    hab_labels = {hid: r.habitats[hid].name for hid in hab_ids}

    # --- Manual seeding with a shared founding genome ---
    # Build one genome biased toward the average of both habitat vectors so
    # the population is viable in both environments from the start. Using the
    # average rather than a habitat-specific vector is the only way to honour
    # the identical-ancestor requirement while surviving the raised food/water
    # selection pressure (mirror mode cannot apply habitat bias).
    rng = np.random.default_rng(r.config["simulation"].get("seed", 42))
    habs = list(r.habitats.values())
    avg_vec = sum(np.asarray(h.vector, dtype=float) for h in habs)
    avg_norm = float(np.linalg.norm(avg_vec))
    avg_dir = avg_vec / avg_norm if avg_norm > 1e-12 else avg_vec
    hab_scaled = avg_dir * np.sqrt(GENE_DIMS)

    bias = 0.4  # 40% habitat direction, 60% random
    founding_genes = (1.0 - bias) * rng.standard_normal(GENE_DIMS) + bias * hab_scaled
    species_name = reg.register_founding_species(founding_genes)

    for hab in habs:
        for i in range(CREATURES_PER_HABITAT):
            genes = founding_genes + rng.standard_normal(GENE_DIMS) * GENOME_NOISE
            c = Creature(genes=genes)
            c.sex = "female" if i % 2 == 0 else "male"
            c.species = species_name
            c.age = c.weeks_to_sexual_viability + 1
            hab.add_creature(c)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = Path(__file__).resolve().parent / "bimodality_output" / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    # Approximate generation length for display
    gen = float(
        np.mean([c.weeks_to_sexual_viability for h in r.habitats.values() for c in h.alive_creatures])
    )
    print(f"Generation length ≈ {gen:.1f} weeks  ({WEEKS} weeks ≈ {WEEKS / gen:.0f} generations)")
    print(f"Output → {out_dir}\n")
    print(
        f"{'Week':>6} {'Gens':>5} {'N_' + hab_ids[0]:>12} {'N_' + hab_ids[1]:>12} {'Hab cos':>9} {'KM cos':>9} {'Split?':>7}"
    )

    split_isolation_threshold = reg.split_isolation_threshold

    # Time-series tracking
    ts_weeks: list[int] = []
    ts_km_cos: list[float] = []  # k-means centroid cosine (noisy detector signal)
    ts_hab_cos: list[float] = []  # per-habitat centroid cosine (stable ground truth)
    split_week: int | None = None
    split_fired_species: set[str] = set()

    colors = {hab_ids[0]: "#2196F3", hab_ids[1]: "#FF5722"}

    for w in range(1, WEEKS + 1):
        r.step()

        # Detect if a new cladogenesis_kmeans_subcluster event fired this step
        for ev in reg.speciation_events:
            if (
                ev.get("event_type") == "cladogenesis_kmeans_subcluster"
                and ev.get("new_species") not in split_fired_species
            ):
                split_fired_species.add(ev["new_species"])
                if split_week is None:
                    split_week = w

        if w % SAMPLE_EVERY != 0:
            continue

        # Collect living creatures per habitat
        by_hab: dict[str, list] = {hid: list(h.alive_creatures) for hid, h in r.habitats.items()}
        n_per_hab = {hid: len(cs) for hid, cs in by_hab.items()}

        all_creatures = [c for cs in by_hab.values() for c in cs]
        N = len(all_creatures)
        if N < 4:
            print(f"{w:>6} {w / gen:>5.1f}  (population too small to analyse)")
            continue

        # Build unit compatibility matrix and per-creature habitat label
        compat = np.stack([c.genes[reg._compat_indices] for c in all_creatures])
        units = unit_rows(compat)

        hab_color: list[str] = []
        for hid, cs in by_hab.items():
            hab_color.extend([hid] * len(cs))
        hab_arr = np.array(hab_color)

        # Per-habitat unit centroids — stable, label-switch-free axis.
        per_hab_c: dict[str, np.ndarray] = {}
        for hid in hab_ids:
            mask = hab_arr == hid
            if mask.any():
                per_hab_c[hid] = unit_vec(units[mask].mean(0))

        if len(per_hab_c) == 2:
            c0 = per_hab_c[hab_ids[0]]
            c1 = per_hab_c[hab_ids[1]]
            hab_centroid_cos = float(c0 @ c1)
            axis = separation_axis(c0, c1)  # always Forest→Plains
        else:
            hab_centroid_cos = 1.0
            axis = np.zeros(units.shape[1])
            axis[0] = 1.0

        proj_sep = units @ axis  # Forest → negative, Plains → positive
        proj_cos = units @ per_hab_c.get(hab_ids[0], axis)  # cosine to Forest centroid

        # K=2 spherical k-means — for the detector signal comparison only
        _, km_centroids = spherical_kmeans(units, k=2, seed=w)
        km_cos = float(km_centroids[0] @ km_centroids[1])

        ts_weeks.append(w)
        ts_km_cos.append(km_cos)
        ts_hab_cos.append(hab_centroid_cos)

        # ---- plot ----
        split_note = f"split fired wk {split_week}" if split_week else "no split yet"
        h0_lbl = hab_labels[hab_ids[0]]
        h1_lbl = hab_labels[hab_ids[1]]

        fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(12, 4.5))
        fig.suptitle(
            f"Week {w} ({w / gen:.1f} gens) | "
            f"hab cos = {hab_centroid_cos:.4f} | km cos = {km_cos:.4f} | "
            f"threshold = {split_isolation_threshold:.2f} | {split_note}",
            fontsize=9,
        )

        bins_a = np.linspace(-0.7, 0.7, 43)  # fixed range; 42 bins of width ~0.033
        bins_b = np.linspace(max(proj_cos.min() - 0.01, -1), min(proj_cos.max() + 0.01, 1), 31)

        for hid in hab_ids:
            mask = hab_arr == hid
            if not mask.any():
                continue
            ax_a.hist(proj_sep[mask], bins=bins_a, alpha=0.6, color=colors[hid], label=hab_labels[hid])
            ax_b.hist(proj_cos[mask], bins=bins_b, alpha=0.6, color=colors[hid], label=hab_labels[hid])

        ax_a.set_title(f"Panel A: Separation-axis projection\naxis = normalise({h1_lbl} − {h0_lbl})")
        ax_a.set_xlabel(f"← {h0_lbl}           {h1_lbl} →")
        ax_a.set_xlim(-0.7, 0.7)
        ax_a.set_ylabel("Count")
        ax_a.legend(loc="upper right")

        ax_b.set_title(f"Panel B: Cosine similarity to {h0_lbl} centroid\n(shared signal still present)")
        ax_b.set_xlabel("Cosine similarity")
        ax_b.set_ylabel("Count")
        ax_b.legend(loc="upper left")

        plt.tight_layout()
        out_path = out_dir / f"week_{w:05d}.png"
        fig.savefig(out_path, dpi=120)
        plt.close(fig)

        print(
            f"{w:>6} {w / gen:>5.1f}  "
            f"{n_per_hab.get(hab_ids[0], 0):>12} {n_per_hab.get(hab_ids[1], 0):>12}  "
            f"{hab_centroid_cos:>8.4f}  {km_cos:>8.4f}  "
            f"{'YES' if split_week and split_week <= w else '—':>7}"
        )

        if sum(n_per_hab.values()) == 0:
            print("All habitats extinct — stopping early.")
            break

    # ------------------------------------------------------------------
    # Summary time-series: centroid-to-centroid cosine over time
    # ------------------------------------------------------------------
    if ts_weeks:
        fig_ts, ax_ts = plt.subplots(figsize=(11, 4.5))
        ax_ts.plot(
            ts_weeks,
            ts_hab_cos,
            color="#1565C0",
            linewidth=1.5,
            label="per-habitat centroid cos (ground truth)",
        )
        ax_ts.plot(
            ts_weeks,
            ts_km_cos,
            color="#999999",
            linewidth=0.8,
            alpha=0.7,
            label="k-means centroid cos (detector signal)",
        )
        ax_ts.axhline(
            split_isolation_threshold,
            color="red",
            linestyle="--",
            linewidth=1,
            label=f"split_isolation_threshold = {split_isolation_threshold}",
        )
        if split_week:
            ax_ts.axvline(
                split_week,
                color="orange",
                linestyle=":",
                linewidth=1.5,
                label=f"split fired (week {split_week})",
            )
        ax_ts.set_xlabel("Week")
        ax_ts.set_ylabel("Cosine similarity (unit centroids)")
        ax_ts.set_title("Centroid-to-centroid cosine similarity over time")
        ax_ts.legend()
        plt.tight_layout()
        ts_path = out_dir / "centroid_cosine.png"
        fig_ts.savefig(ts_path, dpi=120)
        plt.close(fig_ts)
        print(f"\nTime-series plot → {ts_path}")

    if split_week:
        print(f"\nSplit first fired at week {split_week} ({split_week / gen:.1f} gens).")
    else:
        print(f"\nNo split detected within {WEEKS} weeks — try a longer run.")

    print(f"Frame PNGs → {out_dir}/week_NNNNN.png")


if __name__ == "__main__":
    main()
