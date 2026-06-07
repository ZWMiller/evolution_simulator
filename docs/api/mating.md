<a id="evolution_simulator.mating"></a>

# evolution\_simulator.mating

Mating strategies: who gets paired with whom.

All four pairing algorithms plus their helpers live here as standalone
functions.  ``Habitat.simulate_week`` dispatches to one of the ``_mate_*``
functions by the configured ``mating_strategy`` name.  The strategies have no
state dependency on ``Habitat`` beyond the viable male/female lists and a couple
of constants:

  - ``Creature.COMPATIBILITY_FLOOR`` — read off the creature class.
  - ``MATING_SHARPNESS_K`` — a ``Habitat`` (subclass-overridable) attribute, so
    ``_mate_weighted_matrix`` takes it as an explicit ``mating_sharpness_k``
    argument rather than reaching back into the habitat.

The strategy only determines *who pairs with whom*; conception probability and
litter size are always downstream, inside ``_attempt_mating`` ->
``is_compatible`` / ``reproduce``.
