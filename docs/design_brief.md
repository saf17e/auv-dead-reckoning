# Dead-Reckoning Estimator — Handoff for Claude Code

## Context

Agastya AUV / SAUVC27 project (ROS 2, Python + C++). Goal: eventually replace
sonar-based position fixes with a software-based estimator. Visual odometry
was tried and often fails due to lack of features in the pool environment.

This doc captures a design discussion to be continued in Claude Code, which
has access to the actual codebase — including the existing `thruster_node.py`
in this directory. **Read `thruster_node.py` first** to pull real values for
the open items below (thrust allocation matrix, message field names, control
loop rate, kill-switch/staleness handling) instead of asking the user to
retype them.

## Goal

Build a physics-based dead-reckoning position estimator: log thruster
commands + PID outputs, model the vehicle's hydrodynamics (mass, added mass,
drag), and integrate forward in time (RK4) to estimate position as a
drift-prone bridge between sparse absolute fixes (sonar now, possibly vision
later). Target end state is EKF fusion of this estimator with sonar.

**Explicitly not a sonar replacement on its own** — open-loop integration
from thrust will drift within tens of seconds in practice. It's meant to fill
gaps between sonar locks and eventually serve as the EKF predict step.

## Proposed architecture

### 1. Dynamics model — 3-DOF planar (surge/sway/yaw)

```
(M + M_a) * v̇ + D(v) * v = τ
```

- `M` = mass/inertia diag(m, m, I_z)
- `M_a` = added mass (water accelerates with vehicle; often 10-40% of vehicle
  mass depending on hull shape) — not derivable from datasheet, needs fitting
- `D(v)` = drag = `D_lin * v + D_quad * v * |v|` (linear dominates at low
  speed, quadratic at higher speed)
- `τ` = net force/moment from thrusters, via the thrust allocation matrix
  already used in `thruster_node.py`

Integrate body-frame velocity via RK4 → rotate into world frame by current
yaw → integrate position. Should run at the existing control loop rate.

### 2. Thrust model (command ≠ actual thrust)

- Need PWM/command → thrust (N) curve, ideally from datasheet or bench test
  with a load cell
- First-order lag for prop spin-up/spin-down (don't assume instantaneous
  thrust)
- If ESC current/RPM telemetry is available, prefer that over commanded PWM
  as the thrust proxy — closer to ground truth, absorbs battery voltage sag

### 3. System identification (the part that determines accuracy)

Can't get `M_a` / `D_lin` / `D_quad` from a datasheet — must fit from logged
pool data:

- Log timestamped thrust commands + timestamped sonar/overhead-camera
  position for: pure surge pulse, pure sway pulse, pure yaw spin, then a
  mixed path
- Fit coefficients via least-squares (scipy `curve_fit` / `optimize`)
  minimizing simulated-vs-ground-truth trajectory error over the window
- Validate on a held-out run (different thrust profile than fitting run) —
  check open-loop prediction stays within tolerance over a typical mission
  leg length, not just instantaneously

If no logged pool data exists yet: build the pipeline with
literature-typical placeholder coefficients for a vehicle this size, flagged
clearly as uncalibrated, to be replaced once pool data is captured.

### 4. Fusion strategy

- Use IMU yaw/yaw-rate directly rather than integrating yaw from the
  dynamics model — IMU is much more reliable for orientation
- Dynamics model supplies surge/sway velocity between sonar fixes
- Sonar (when locked) resets accumulated position error — structurally
  similar to the existing `ping_x`/`ping_y` correction logic, just becomes a
  lower-confidence higher-rate fill-in
- Target: EKF with dynamics model as predict step, absolute fixes (sonar,
  later vision) as correct step. A simple complementary filter
  (dynamics + periodic sonar snap) is a reasonable intermediate step before
  full EKF.

### 5. Suggested node structure

New node, e.g. `dynamics_estimator_node.py`, separate from `thruster_node.py`
to keep hardware ownership and estimation logic decoupled (consistent with
existing package separation pattern: hardware in C++, logic in Python).

- Subscribes: thruster command topic (the vector `thruster_node.py` already
  consumes), IMU topic
- Publishes: estimated pose/velocity, ideally in the same format the sonar
  position output uses today, so `PathTracker` / task nodes can consume it
  as a drop-in

## Open items to resolve against the real codebase

Pull these from `thruster_node.py` and related files rather than
re-deriving:

1. **Vehicle physical parameters** — mass, displaced volume/buoyancy trim,
   yaw moment of inertia (I_z), center of mass vs. center of buoyancy offset,
   hull frontal area estimates (surge/sway/yaw) for initial drag ballpark
2. **Thruster specs** — thrust curve (PWM→N), thruster count/position/
   orientation (thrust allocation matrix — likely already implemented in
   `thruster_node.py`), spin-up/spin-down time constant, ESC telemetry
   availability
3. **Message/topic contracts** — exact fields of the 6-element
   direction/vector message (units, i/j/yaw meaning, frame convention), IMU
   topic message type and fields, sonar topic fields (ping_x, ping_y,
   lock/confidence status)
4. **Pool frame convention** — origin bottom-left, x increasing east, y
   increasing north, yaw likely clockwise-positive (confirm against current
   code, this may have been revised)
5. **Ground truth data** — does logged pool data (thrust commands paired
   with sonar/camera position over time) already exist, or does this need a
   fresh capture session first?
6. **Runtime details** — control loop rate (Hz), ROS 2 distro, and whether
   the estimator should publish to the same topic/format sonar currently
   uses for a clean drop-in swap into `PathTracker`

## Working style notes for continuation

- Prefer surgical, minimal diffs with clear annotation of what changed and
  why; leave unrelated code untouched
- Config-driven over hardcoded (topic names, DOF names, file paths →
  config file)
- Don't remove commented-out fallback/intent-documenting code unless
  explicitly asked
- Present design tradeoffs/options before implementing when there's a real
  choice to make, rather than picking silently
