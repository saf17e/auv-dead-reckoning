# dead_reckoning/

Physics-based dead-reckoning position estimator for the AUV — fills the gaps
between sparse sonar fixes by integrating the vehicle's own equations of
motion forward in time, using the real per-thruster commanded output as the
forcing input. See `../dead_reckoning_estimator_handoff.md` for the original
design brief and the plan this subpackage implements.

Everything the estimator needs lives in this one self-contained subpackage
rather than scattered across the flat `robofest_6/` module directory, so it
can be reviewed, tested, and removed as one unit. It has one intentional
external dependency on the rest of the package: it imports
`/debug/final_pwm` and `/bno` values the way `thruster_node.py` already
publishes/reads them, and `thruster_node.py` in turn imports its thruster
geometry and PWM-table loading from here (see "Relationship to
thruster_node.py" below) so the two never define the same numbers twice.

## Files

### `cad_dynamics.py` — STEP file ingestion and dynamic-state calculation

The single entry point for turning a CAD export into every physical
constant the rest of this package needs. Run manually whenever the CAD
changes:

```
python -m robofest_6.dead_reckoning.cad_dynamics --step path/to/hull.step [--density 1000] [--mass 12.4]
```

- Parses the STEP file via `cadquery`/OpenCASCADE (a real B-rep kernel, not
  a pre-tessellated mesh format) and reads mass, center of mass, and the
  yaw moment of inertia directly off the solid geometry — exact, not
  fitted. STEP geometry alone only encodes *relative* mass distribution
  unless the file carries material assignments, so you must also supply
  either `--density` (kg/m^3) or `--mass` (a bench-measured total mass, from
  which density is back-solved) to get real mass/inertia numbers rather
  than ones assuming unit density.
- Tessellates that same B-rep once into a triangle mesh (only place this
  happens — the resolution is chosen deliberately here rather than
  inherited from an ad-hoc STL export) and uses it to compute:
  - `A(theta)` — projected silhouette (frontal) area vs. horizontal
    direction of travel, via convex-hull projection at each sampled angle.
  - `D_quad(theta) = 0.5 * rho_water * Cd * A(theta)` — quadratic form drag,
    `Cd` a single literature constant applied uniformly until online
    fine-tuning (Node A) corrects the scale.
  - `M_a(theta)` — added mass, via a slender-body/strip-theory
    equivalent-ellipse approximation of the cross-section perpendicular to
    the direction of travel.
- Writes `config/hydrodynamic_tables.npz` (the three tables) and updates the
  `vehicle_params`/`cad_source` blocks of `config/dynamics_estimator.yaml`
  in place, leaving `coefficients`/`runtime` untouched.
- The mesh-only geometry functions (`frontal_area_table`,
  `added_mass_table`) take plain numpy vertex arrays and have no cadquery
  dependency, so `test/test_cad_dynamics.py` exercises them directly
  against a synthetic box mesh without needing cadquery/OCP installed.

### `dynamics_core.py` — shared physics engine

The equations of motion (plan Part 1), imported by both nodes so the actual
physics is defined exactly once:

- `VehicleParams` / `DynamicsCoefficients` — dataclasses for, respectively,
  the CAD-exact mass/inertia/water-density constants and the larger
  online-tunable coefficient vector (added-mass scales, drag scale, the
  flat linear-drag 3-vector, speed-regime blend scalars, and the
  bi-directional PWM correction scale/bias pairs).
- `load_hydrodynamic_tables()` / `interp_table()` — loads
  `hydrodynamic_tables.npz` (or a flat literature-placeholder table if
  `cad_dynamics.py` hasn't been run yet) and does circular linear
  interpolation into the 360-entry per-degree tables.
- `speed_regime_blend()` — the smooth low-speed/high-speed sigmoid blend
  used to interpolate each table's `k_low`/`k_high` correction scalars.
- `compute_state_derivative()` / `rk4_step()` — the actual ODE right-hand
  side and its 4-stage RK4 step. Internally rotates into a
  velocity-aligned (along-track / cross-track) frame so a single
  1-D-in-angle table can stand in for a full direction-dependent tensor
  without a much larger fitted table; yaw uses its own flat coefficients
  since "direction of travel" isn't meaningful for a pure rotation.
- `DeadReckoningIntegrator` — owns one node's running velocity/position
  state; `step()` integrates velocity via RK4 and always rotates it into
  the world frame using the *measured* yaw passed in (from `/bno`), never
  a yaw integrated from this model, per the handoff doc's fusion strategy.
- Batched variants (`compute_state_derivative_batch`, `rk4_step_batch`,
  `_batch_backend()`) — the exact same equations, vectorized across an
  ensemble dimension, used only by Node A's online fine-tuning. Runs on
  `torch` CUDA tensors when torch + a CUDA device are available, and
  transparently falls back to plain numpy on the CPU otherwise.

### `vehicle_geometry.py` — thruster mounts and the thrust allocation matrix

`THRUSTER_POSITIONS`/`THRUSTER_UNIT_DIRECTIONS`/`build_TAM_6DOF()`, extracted
from `thruster_node.py`'s original geometry block. If
`config/dynamics_estimator.yaml` carries a `thruster_geometry` block (not
present yet — to be added once CAD-exact mount positions/orientations are
available), those values are used; otherwise this falls back to the same
box-corner placeholder geometry `thruster_node.py` originally hardcoded,
which is an acknowledged rough approximation, not the real hull.
`GEOMETRY_SOURCE` reports which one is active.

### `pwm_thrust_table.py` — PWM↔thrust lookup and correction

Wraps the external `pwm_new.json` bench-calibration table (loaded once,
same path `thruster_node.py` uses). `thrust_to_pwm()` is
`thruster_node.py`'s original direction (force → PWM, for commanding
thrusters). `pwm_to_thrust()` is the new inverse direction the estimator
needs (measured/commanded PWM → estimated thrust), wrapped with the
bi-directional `scale`/`bias` correction layer described in the handoff
plan — separate forward/reverse pairs, since real propellers don't produce
equal thrust per PWM unit in both rotation directions. Both scale/bias
pairs default to the identity correction `(1, 0)` until Node A's online
fine-tuning has produced fitted values.

### `dynamics_estimator_node.py` — Node B (production, no sonar)

Subscribes only to `/debug/final_pwm` and `/bno`. Converts PWM back to
per-thruster thrust, maps it through the shared TAM into body-frame
force/moment, and integrates position with `dynamics_core`. Loads
coefficients from `config/dynamics_estimator_fitted.yaml` if Node A has
produced one, else the static prior in `config/dynamics_estimator.yaml`.
Publishes estimated pose on `/dead_reckoning/pose`. This is the node meant
to run standalone once the sonar rig is unattached, or whenever an
absolute-fix-free estimate is needed.

### `dynamics_estimator_calibration_node.py` — Node A (sonar-in-the-loop fine-tuning)

Subscribes to `/debug/final_pwm`, `/bno`, `/sonar/x`, `/sonar/y`. Buffers
raw PWM + measured yaw between trusted sonar fixes (confidence > 90, the
same threshold `thruster_node.py` already gates on). On each new trusted
fix, replays the buffered window through a small ensemble of candidate
coefficient vectors — batched across the ensemble with `dynamics_core`'s
batch functions — scores each candidate against how far its predicted
displacement landed from the sonar-observed displacement, and folds the
best candidates into a running estimate with an exponential forgetting
factor (`FORGETTING_LAMBDA`). When no sonar fix has landed for a while, the
running estimate is gently pulled back toward the CAD/literature prior
(`REGULARIZATION_PULL`) so it can't wander unbounded on buffer noise alone.
Periodically (`WRITE_INTERVAL_SEC`) writes the running estimate to
`config/dynamics_estimator_fitted.yaml`. Publishes no pose of its own — it
exists purely to keep Node B's coefficients current.

### `config/`

- `dynamics_estimator.yaml` — static prior: CAD-derived (or, until
  `cad_dynamics.py` has run, literature-placeholder) vehicle params, the
  default coefficient vector, and runtime settings (topic names, loop
  rate, sonar confidence threshold).
- `dynamics_estimator_fitted.yaml` — Node A's live output; absent until
  Node A has run at least once, at which point Node B prefers it over the
  static prior.
- `hydrodynamic_tables.npz` — `A`/`D_quad`/`M_a` tables written by
  `cad_dynamics.py`; falls back to a flat literature-guess table (see
  `dynamics_core._flat_placeholder_tables()`) if absent.

### `test/`

- `test_dynamics_core.py` — table interpolation/wrapping, speed-regime
  blend limits, zero-thrust equilibrium, forward-thrust sign sanity, the
  TAM forward map, and the PWM correction round-trip.
- `test_cad_dynamics.py` — `frontal_area_table()`/`added_mass_table()`
  against a synthetic box mesh with hand-checkable expected areas/added
  masses, with no cadquery dependency.

## Relationship to `thruster_node.py`

`thruster_node.py` owns the hardware (ESC/servo output, PID stabilization,
sonar-lock gating for the live control loop) and is left in place at the
package root. It imports its thruster geometry and PWM-table loading from
`vehicle_geometry.py`/`pwm_thrust_table.py` here instead of keeping its own
copies, so the two packages can't silently drift apart on what the
thruster layout or calibration table actually is. Nothing in this
subpackage writes to hardware or affects the live control loop; Node A and
Node B are purely observers of `/debug/final_pwm` and `/bno`.

## Known open questions (implementation-time, not design)

- **STEP-parsing library**: `cadquery` is preferred (pip-installable
  wheels, clean mass-property/tessellation API); FreeCAD's Python API is
  the documented fallback if OCP wheels aren't available on the Orin's
  ARM64 Python.
- **GPU backend**: `dynamics_core`'s batch functions use `torch` if
  importable and `torch.cuda.is_available()`; otherwise they run
  identically on numpy/CPU. Whether CuPy would be a better fit than torch
  on this specific JetPack/L4T install is worth checking against whichever
  wheels are already present on the Orin before assuming torch.
- **CAD-exact thruster mount geometry**: `vehicle_geometry.py` is ready to
  consume a `thruster_geometry` block in `dynamics_estimator.yaml`, but
  nothing populates it yet — that needs the thruster mount
  positions/orientations pulled from the CAD assembly (not just the hull
  used for `cad_dynamics.py`'s hydrodynamic tables) and written into that
  config block.
