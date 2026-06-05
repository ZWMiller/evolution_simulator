"""
Calibrate a member-level reproductive-isolation gate for the cladogenesis split
detector.

Why
---
`detect_subcluster_splits` decides two sub-clusters are reproductively isolated
by comparing their **centroids** (`centroid_a · centroid_b < split_isolation_threshold`,
0.75).  The speciation-churn diagnosis
(`reports/general_findings/speciation_churn_diagnosis.md`) showed that this is the
wrong granularity: ~47% of accepted splits keep interbreeding with their parent,
at cross-cluster compatibility ~0.88 — far above the 0.70 mating floor — because
the angle between two cluster *means* understates how much the member
distributions overlap.

The proposed replacement gate is a **member-level interbreed fraction**: of all
cross-cluster opposite-sex pairs, what fraction clears the actual mating gate
(`is_compatible`: cosine >= COMPATIBILITY_FLOOR + 0.15 * mean(selectivity)).  A
split should fire only when that fraction is low (the two groups genuinely cannot
interbreed).  This script does NOT change the detector — it runs the real
(deterministic) simulation and, at every respeciate cadence, *observes* each
species' k-means partition and records both metrics side by side, plus the
detector's own accept/reject decision, so we can read the right threshold off the
data before wiring the gate in.

The same primitive (`interbreed_fraction`) is what a future merge check would use,
read the other way (merge when the fraction is high).

What it writes
--------------
experiments/calib_output/<timestamp>/split_isolation_calibration.csv — one row per
measured cluster pair (the detector-selected partition's non-primary clusters vs
the primary, and a forced-K=2 probe for every eligible species), with the
centroid cosine, the member-level interbreed fraction, sizes, and whether the
current centroid rule accepted the split.  A summary is printed to stdout.

Usage
-----
    poetry run python experiments/calibrate_split_isolation.py \
        reports/general_findings/speciation_churn_diagnosis_diag_config.toml --weeks 2500
"""

import argparse
import csv
import statistics as st
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np

from evolution_simulator.creature import Creature
from evolution_simulator.genetics import DEFAULT_TRAIT_GENE_INDICES
from evolution_simulator.simulation import SimulationRunner
from evolution_simulator.speciation_math import spherical_kmeans, unit_rows

_COMPAT_IDX = np.array(DEFAULT_TRAIT_GENE_INDICES["compatibility_genes"], dtype=int)
_FLOOR = Creature.COMPATIBILITY_FLOOR  # 0.70


# ---------------------------------------------------------------------------
# The candidate metric: member-level interbreed fraction
# ---------------------------------------------------------------------------


def _compat_units(creatures: list) -> np.ndarray:
    """L2-normalized 245-dim compatibility sub-vectors for a list of creatures."""
    g = np.stack([c.genes[_COMPAT_IDX] for c in creatures])
    n = np.linalg.norm(g, axis=1, keepdims=True)
    return g / np.where(n > 1e-12, n, 1.0)


def _dir_fraction(males: list, females: list) -> tuple[int, int]:
    """(#compatible, #pairs) over all male x female pairs, via the is_compatible gate."""
    if not males or not females:
        return 0, 0
    cos = _compat_units(males) @ _compat_units(females).T  # (M, F)
    sel_m = np.array([c.selectivity for c in males])
    sel_f = np.array([c.selectivity for c in females])
    # is_compatible's per-pair threshold: FLOOR + 0.15 * mean(selectivity)
    thresh = _FLOOR + 0.15 * (sel_m[:, None] + sel_f[None, :]) / 2.0
    compatible = cos >= thresh
    return int(compatible.sum()), int(compatible.size)


def interbreed_fraction(group_a: list, group_b: list) -> tuple[float, int]:
    """
    Exhaustive cross-group interbreed fraction.

    Considers every cross-group OPPOSITE-SEX pair in both directions
    (A-male x B-female and B-male x A-female) and returns the fraction that
    clears the mating gate, plus the number of pairs the estimate is based on.
    Measures genetic+selectivity capability to interbreed; transient state
    (pregnancy, age viability) is intentionally ignored — isolation is about
    whether two populations *can* breed, not whether a given encounter conceives.
    """
    am = [c for c in group_a if c.sex == "male"]
    af = [c for c in group_a if c.sex == "female"]
    bm = [c for c in group_b if c.sex == "male"]
    bf = [c for c in group_b if c.sex == "female"]
    c1, n1 = _dir_fraction(am, bf)
    c2, n2 = _dir_fraction(bm, af)
    n = n1 + n2
    return (float("nan") if n == 0 else (c1 + c2) / n), n


# ---------------------------------------------------------------------------
# Observation at one respeciate cadence
# ---------------------------------------------------------------------------


def _observe(runner: SimulationRunner, week: int, excluded_ids: set, rows: list) -> None:
    """
    Reconstruct what detect_subcluster_splits saw this cadence and record metrics.

    `excluded_ids` is the set of candidate-member ids captured BEFORE this week's
    step, i.e. the candidates the detector excluded when it ran.  Using the
    pre-step set (not the post-step set) keeps just-seeded split-off members in
    the population so we can re-measure the very splits the detector just made.
    """
    reg = runner.species_registry
    # Mirror detect_subcluster_splits' member selection + iteration order so the
    # k-means seeding (k-means++ samples by index) reproduces the same partition.
    by_species: dict[str, list] = {}
    for hab in runner.habitats.values():
        for c in hab.alive_creatures:
            if c.creature_id in excluded_ids:
                continue
            if c.species in reg._registry:
                by_species.setdefault(c.species, []).append(c)

    for sp, members in by_species.items():
        if len(members) < 2 * reg.min_species_population:
            continue
        compat = np.stack([reg._compat(c.genes) for c in members])
        units = unit_rows(compat)

        # (a) The detector's actual decision for this species this cadence.
        labels, centroids = reg._select_clusters(units, seed=week)
        selected_k = int(centroids.shape[0])
        if selected_k >= 2:
            type_unit = unit_rows(reg._compat(reg._registry[sp])[np.newaxis, :])[0]
            primary = int(np.argmax(centroids @ type_unit))
            primary_members = [members[i] for i, lab in enumerate(labels) if lab == primary]
            for j in range(selected_k):
                if j == primary:
                    continue
                child = [members[i] for i, lab in enumerate(labels) if lab == j]
                if len(child) < reg.min_species_population:
                    continue
                rep = max(
                    child,
                    key=lambda m: float(unit_rows(reg._compat(m.genes)[np.newaxis, :])[0] @ centroids[j]),
                )
                frac, npairs = interbreed_fraction(child, primary_members)
                rows.append(
                    {
                        "week": week,
                        "species": sp,
                        "row_type": "selected_child_vs_primary",
                        "n_members": len(members),
                        "selected_k": selected_k,
                        "primary_size": len(primary_members),
                        "child_size": len(child),
                        "centroid_cos": round(float(centroids[j] @ centroids[primary]), 4),
                        "interbreed_frac": None if np.isnan(frac) else round(frac, 4),
                        "n_cross_pairs": npairs,
                        "rep_creature_id": rep.creature_id,
                    }
                )

        # (b) A uniform forced-K=2 probe for every eligible species, so the
        #     scatter includes species the centroid rule left at K=1.
        k2_labels, k2_cent = spherical_kmeans(units, 2, week + 2)
        ca = [members[i] for i, lab in enumerate(k2_labels) if lab == 0]
        cb = [members[i] for i, lab in enumerate(k2_labels) if lab == 1]
        if ca and cb:
            frac, npairs = interbreed_fraction(ca, cb)
            rows.append(
                {
                    "week": week,
                    "species": sp,
                    "row_type": "k2_probe",
                    "n_members": len(members),
                    "selected_k": selected_k,
                    "primary_size": max(len(ca), len(cb)),
                    "child_size": min(len(ca), len(cb)),
                    "centroid_cos": round(float(k2_cent[0] @ k2_cent[1]), 4),
                    "interbreed_frac": None if np.isnan(frac) else round(frac, 4),
                    "n_cross_pairs": npairs,
                    "rep_creature_id": "",
                }
            )


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _candidate_member_ids(reg) -> set:
    ids: set = set()
    for cand in reg._candidates.values():
        ids |= cand["members"]
    return ids


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("config", type=Path, help="TOML config (use the diag config for a comparable trajectory)")
    ap.add_argument("--weeks", type=int, default=None, help="override config weeks")
    args = ap.parse_args()

    runner = SimulationRunner(args.config)
    sim = runner.config["simulation"]
    # We observe live objects, not the JSON logs — silence per-week logging.
    sim["events_every"] = 0
    sim["stats_every"] = 10**9
    sim["output_dir"] = tempfile.mkdtemp(prefix="calib_split_")
    if args.weeks is not None:
        sim["weeks"] = args.weeks
    weeks = sim["weeks"]
    respeciate_every = sim.get("respeciate_every", 10)

    runner.setup()
    print(
        f"calibrating: config={args.config.name} weeks={weeks} seed={sim.get('seed')} "
        f"respeciate_every={respeciate_every}"
    )

    rows: list[dict] = []
    for _ in range(weeks):
        # Snapshot the candidates the detector will exclude this week BEFORE it runs.
        pre_candidates = _candidate_member_ids(runner.species_registry)
        runner.step()
        w = runner.week
        if w == 1 or (respeciate_every >= 1 and w % respeciate_every == 0):
            _observe(runner, w, pre_candidates, rows)
        if w % 500 == 0:
            print(f"  week {w}/{weeks}: rows so far {len(rows)}")

    # --- Write CSV ---
    out_dir = Path("experiments/calib_output") / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "split_isolation_calibration.csv"
    fields = [
        "week",
        "species",
        "row_type",
        "n_members",
        "selected_k",
        "primary_size",
        "child_size",
        "centroid_cos",
        "interbreed_frac",
        "n_cross_pairs",
        "rep_creature_id",
    ]
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {len(rows)} rows -> {csv_path}")

    # --- Summary ---
    selected = [
        r for r in rows if r["row_type"] == "selected_child_vs_primary" and r["interbreed_frac"] is not None
    ]
    probes = [r for r in rows if r["row_type"] == "k2_probe" and r["interbreed_frac"] is not None]

    def describe(label, rs):
        if not rs:
            print(f"\n{label}: (none)")
            return
        fr = sorted(r["interbreed_frac"] for r in rs)
        cc = [r["centroid_cos"] for r in rs]
        print(f"\n{label}: n={len(rs)}")
        print(
            f"  interbreed_frac : min {fr[0]:.3f}  p25 {fr[len(fr) // 4]:.3f}  median {st.median(fr):.3f}  "
            f"p75 {fr[3 * len(fr) // 4]:.3f}  max {fr[-1]:.3f}"
        )
        print(f"  centroid_cos    : min {min(cc):.3f}  median {st.median(cc):.3f}  max {max(cc):.3f}")

    describe("ACCEPTED splits (detector said split; child vs parent)", selected)
    describe("Forced-K=2 probe (all eligible species)", probes)

    if selected:
        print("\n=== of the splits the detector ACCEPTS, how many a member-level gate would still allow ===")
        print("    (these are real splits today; lower retained % = more spurious splits the gate removes)")
        n = len(selected)
        for thr in (0.02, 0.05, 0.10, 0.15, 0.20, 0.30):
            keep = sum(1 for r in selected if r["interbreed_frac"] <= thr)
            print(
                f"    interbreed_frac <= {thr:.2f}:  retained {keep}/{n} ({keep / n:.0%})  "
                f"=> removed {n - keep} ({1 - keep / n:.0%})"
            )
        leaky = sum(1 for r in selected if r["interbreed_frac"] > 0.20)
        print(
            f"\n  accepted splits with interbreed_frac > 0.20 (clearly still interbreeding): "
            f"{leaky}/{n} ({leaky / n:.0%})"
        )
        # selected splits all have centroid_cos < split_isolation_threshold by construction,
        # yet interbreed_frac spans a wide range — that gap is the whole point.
        print(
            f"  (all accepted splits have centroid_cos < {runner.species_registry.split_isolation_threshold} "
            f"by the current rule, yet interbreed_frac ranges {min(r['interbreed_frac'] for r in selected):.2f}"
            f"-{max(r['interbreed_frac'] for r in selected):.2f})"
        )


if __name__ == "__main__":
    main()
