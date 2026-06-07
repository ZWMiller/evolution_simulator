#!/usr/bin/env bash
# Regenerate docs/api/*.md from source docstrings.
# Run from the repo root:  bash scripts/generate_api_docs.sh
# Or via poetry:           poetry run bash scripts/generate_api_docs.sh

set -euo pipefail

OUTDIR="docs/api"
mkdir -p "$OUTDIR"

modules=(
    "evolution_simulator.creature"
    "evolution_simulator.habitat"
    "evolution_simulator.habitats.types"
    "evolution_simulator.species"
    "evolution_simulator.simulation"
    "evolution_simulator.genetics"
    "evolution_simulator.traits"
    "evolution_simulator.mating"
    "evolution_simulator.speciation_math"
    "evolution_simulator.log_builder"
)

filenames=(
    "creature"
    "habitat"
    "habitat_types"
    "species"
    "simulation"
    "genetics"
    "traits"
    "mating"
    "speciation_math"
    "log_builder"
)

for i in "${!modules[@]}"; do
    mod="${modules[$i]}"
    fname="${filenames[$i]}"
    outfile="$OUTDIR/${fname}.md"
    echo "Generating $outfile ..."
    pydoc-markdown -m "$mod" > "$outfile"
done

echo "Done. Files written to $OUTDIR/"
