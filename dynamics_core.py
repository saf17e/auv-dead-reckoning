#!/usr/bin/env python3
"""dynamics_core.py — shared 3-DOF planar dynamics engine for the
dead-reckoning estimator (plan Part 1). Imported by both
dynamics_estimator_node.py (Node B) and dynamics_estimator_calibration_node.py
(Node A) so the physics is defined exactly once and the two nodes can't
silently diverge.

Equation of motion (body frame, v = [u, v_sway, r]):

    (M + M_a(theta)) v_dot + D(v, theta) v = tau

`theta` is the current direction of travel in the horizontal body plane
(atan2(v_sway, u)). Direction-dependent added mass/drag are handled by
working in a velocity-aligned (l = along-track, n = cross-track) frame,
where at any instant v_l = speed and v_n = 0 by construction — this lets a
single 1-D-in-angle table (A/D_quad/M_a vs theta) stand in for a full
per-direction 2x2 tensor without needing a much larger fitted table. Yaw
uses its own flat (non-angle-dependent) coefficients, since "direction of
travel" isn't a meaningful concept for a pure rotation.

Yaw itself is never integrated here — callers pass in measured yaw
(from /bno) for the body->world rotation each step, per the handoff doc's
fusion strategy.
"""
from dataclasses import dataclass, field, asdict
import math
import os
import numpy as np
import yaml

CONFIG_DIR = os.path.join(os.path.dirname(__file__), 'config')
HYDRO_TABLES_PATH = os.path.join(CONFIG_DIR, 'hydrodynamic_tables.npz')
PRIOR_CONFIG_PATH = os.path.join(CONFIG_DIR, 'dynamics_estimator.yaml')
FITTED_CONFIG_PATH = os.path.join(CONFIG_DIR, 'dynamics_estimator_fitted.yaml')

N_TABLE_SAMPLES = 360  # one entry per degree, 0..359


# ---------------------------------------------------------------------------
# Vehicle parameters (mass/inertia — exact from CAD, not fitted)
# ---------------------------------------------------------------------------
@dataclass
class VehicleParams:
    mass_kg: float = 12.0          # literature-placeholder until cad_dynamics.py runs
    inertia_z_kgm2: float = 0.35   # literature-placeholder
    rho_water_kgm3: float = 997.0

    @classmethod
    def from_dict(cls, d):
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ---------------------------------------------------------------------------
# Online-tunable coefficient vector (plan §2.1's theta vector)
# ---------------------------------------------------------------------------
@dataclass
class DynamicsCoefficients:
    # Added-mass scale corrections on top of the CAD-derived M_a(theta) table,
    # one per axis family (surge/along-track, sway/cross-track, yaw).
    xu_dot_scale: float = 1.0
    yv_dot_scale: float = 1.0
    nr_dot_scale: float = 1.0
    inertia_yaw_added_kgm2: float = 0.05  # flat added yaw inertia (no angle table for yaw)

    # Quadratic (form) drag overall scale correction on top of D_quad(theta).
    cd_scale: float = 1.0
    # Linear (viscous) drag, kept as a flat 3-axis constant (plan §1.2's D_lin).
    d_lin: list = field(default_factory=lambda: [2.0, 3.0, 0.5])
    # Flat quadratic yaw drag (yaw has no direction-of-travel table either).
    d_quad_yaw: float = 0.3

    # Speed-regime blending (plan §1.5), shared threshold/width, independent
    # low/high scalars per direction-dependent table.
    v_regime_mps: float = 0.3
    blend_width_mps: float = 0.2
    k_low_dquad: float = 1.0
    k_high_dquad: float = 1.0
    k_low_ma: float = 1.0
    k_high_ma: float = 1.0

    # Bi-directional PWM->thrust correction (plan §1.4).
    pwm_scale_fwd: float = 1.0
    pwm_bias_fwd: float = 0.0
    pwm_scale_rev: float = 1.0
    pwm_bias_rev: float = 0.0

    @classmethod
    def from_dict(cls, d):
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def to_dict(self):
        return asdict(self)


def load_coefficients(prefer_fitted=True) -> DynamicsCoefficients:
    """Node B's loading rule: use Node A's fitted output if it exists,
    else fall back to the static CAD/literature prior."""
    if prefer_fitted and os.path.exists(FITTED_CONFIG_PATH):
        with open(FITTED_CONFIG_PATH, 'r') as f:
            cfg = yaml.safe_load(f) or {}
        if 'coefficients' in cfg:
            return DynamicsCoefficients.from_dict(cfg['coefficients'])

        
    if os.path.exists(PRIOR_CONFIG_PATH):
        with open(PRIOR_CONFIG_PATH, 'r') as f:
            cfg = yaml.safe_load(f) or {}
        if 'coefficients' in cfg:
            return DynamicsCoefficients.from_dict(cfg['coefficients'])
    return DynamicsCoefficients()


def load_vehicle_params() -> VehicleParams:
    if os.path.exists(PRIOR_CONFIG_PATH):
        with open(PRIOR_CONFIG_PATH, 'r') as f:
            cfg = yaml.safe_load(f) or {}
        if 'vehicle_params' in cfg:
            return VehicleParams.from_dict(cfg['vehicle_params'])
    return VehicleParams()


# ---------------------------------------------------------------------------
# Hydrodynamic lookup tables — A(theta), D_quad(theta), M_a(theta)
# ---------------------------------------------------------------------------
def _make_fallback_hydro_tables():
    """Returns flat literature-guess hydro tables used as a fallback until
    cad_dynamics.py has produced a real table from the CAD mesh."""
    return {
        'A': np.full(N_TABLE_SAMPLES, 0.05),       # m^2, rough frontal-area guess
        'D_quad': np.full(N_TABLE_SAMPLES, 20.0),  # N/(m/s)^2, rough guess
        'M_a': np.full(N_TABLE_SAMPLES, 3.0),      # kg, rough guess
    }


def load_hydro_tables_from_disk(path=HYDRO_TABLES_PATH):
    """Loads A/D_quad/M_a angle-indexed tables from the .npz produced by
    cad_dynamics.py, falling back to flat placeholder values if the file
    doesn't exist yet."""
    if not os.path.exists(path):
        return _make_fallback_hydro_tables()
    data = np.load(path)
    return {'A': data['A'], 'D_quad': data['D_quad'], 'M_a': data['M_a']}


def angle_table_lookup(table: np.ndarray, theta_rad: float) -> float:
    """Circular linear interpolation into a 360-entry (1-deg-resolution)
    angle-keyed table — wraps seamlessly at 360°/0°."""
    angle_deg = math.degrees(theta_rad) % 360.0
    idx_lo = int(math.floor(angle_deg))
    idx_hi = (idx_lo + 1) % N_TABLE_SAMPLES
    frac = angle_deg - idx_lo
    return (1 - frac) * table[idx_lo] + frac * table[idx_hi]


def blend_low_high_speed_scalar(k_low, k_high, speed, v_regime, width):
    """Returns a smoothly blended scalar between k_low (low-speed) and
    k_high (high-speed) using a sigmoid centred at v_regime with transition
    width `width` (plan §1.5).  Avoids the RK4-hostile discontinuity that a
    hard threshold switch would produce."""
    if width <= 0:
        return k_high if speed >= v_regime else k_low
    sigmoid = 1.0 / (1.0 + math.exp(-(speed - v_regime) / width))
    return k_low + (k_high - k_low) * sigmoid


# ---------------------------------------------------------------------------
# Equation of motion
# ---------------------------------------------------------------------------
def calc_body_accels(body_vel, body_wrench, coeffs: DynamicsCoefficients,
                     params: VehicleParams, tables: dict) -> np.ndarray:
    """Given current body velocity [vel_surge, vel_sway, vel_yaw] and the
    applied wrench [force_x, force_y, moment_z], applies added-mass,
    quadratic drag, and linear drag from the hydro tables to return the
    body-frame acceleration [accel_surge, accel_sway, accel_yaw].
    Used by Node B's RK4 integrator at every 100 Hz tick."""
    vel_surge, vel_sway, vel_yaw = body_vel
    speed = math.hypot(vel_surge, vel_sway)
    theta = math.atan2(vel_sway, vel_surge) if speed > 1e-6 else 0.0

    frontal_area_along   = angle_table_lookup(tables['A'], theta)
    frontal_area_cross   = angle_table_lookup(tables['A'], theta + math.pi / 2)
    drag_quad_along_raw  = angle_table_lookup(tables['D_quad'], theta)
    drag_quad_cross_raw  = angle_table_lookup(tables['D_quad'], theta + math.pi / 2)
    added_mass_along_raw = angle_table_lookup(tables['M_a'], theta)
    added_mass_cross_raw = angle_table_lookup(tables['M_a'], theta + math.pi / 2)

    blend_drag_quad  = blend_low_high_speed_scalar(coeffs.k_low_dquad, coeffs.k_high_dquad,
                                                   speed, coeffs.v_regime_mps, coeffs.blend_width_mps)
    blend_added_mass = blend_low_high_speed_scalar(coeffs.k_low_ma, coeffs.k_high_ma,
                                                   speed, coeffs.v_regime_mps, coeffs.blend_width_mps)

    drag_quad_along  = drag_quad_along_raw * coeffs.cd_scale * blend_drag_quad
    added_mass_along = added_mass_along_raw * blend_added_mass * coeffs.xu_dot_scale
    added_mass_cross = added_mass_cross_raw * blend_added_mass * coeffs.yv_dot_scale

    mass_eff_along = params.mass_kg + added_mass_along
    mass_eff_cross = params.mass_kg + added_mass_cross

    # Linear drag applied in body frame first, folded into a net force,
    # then rotated into the velocity-aligned (along, cross) frame for the
    # direction-dependent quadratic-drag/added-mass solve.
    drag_lin = coeffs.d_lin
    force_net_x = body_wrench[0] - drag_lin[0] * vel_surge
    force_net_y = body_wrench[1] - drag_lin[1] * vel_sway

    cos_theta, sin_theta = math.cos(theta), math.sin(theta)
    tau_along = cos_theta * force_net_x + sin_theta * force_net_y
    tau_cross  = -sin_theta * force_net_x + cos_theta * force_net_y

    drag_quad_along_force = drag_quad_along * speed * speed  # vel_cross == 0 at this instant by construction
    accel_along = (tau_along - drag_quad_along_force) / mass_eff_along
    accel_cross  = tau_cross / mass_eff_cross

    accel_surge = cos_theta * accel_along - sin_theta * accel_cross
    accel_sway  = sin_theta * accel_along + cos_theta * accel_cross

    mass_eff_yaw = params.inertia_z_kgm2 + coeffs.inertia_yaw_added_kgm2 * coeffs.nr_dot_scale
    accel_yaw    = (body_wrench[2] - drag_lin[2] * vel_yaw - coeffs.d_quad_yaw * vel_yaw * abs(vel_yaw)) / mass_eff_yaw

    return np.array([accel_surge, accel_sway, accel_yaw])


def integrate_one_step_rk4(body_vel, body_wrench, dt, coeffs, params, tables) -> np.ndarray:
    """Advances body_vel by one timestep dt using a standard 4-stage RK4
    scheme, holding body_wrench constant across the step.
    Called once per 100 Hz tick by Node B's dead-reckoning integrator."""
    k1 = calc_body_accels(body_vel,                   body_wrench, coeffs, params, tables)
    k2 = calc_body_accels(body_vel + 0.5 * dt * k1,  body_wrench, coeffs, params, tables)
    k3 = calc_body_accels(body_vel + 0.5 * dt * k2,  body_wrench, coeffs, params, tables)
    k4 = calc_body_accels(body_vel + dt * k3,         body_wrench, coeffs, params, tables)
    return body_vel + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


# ---------------------------------------------------------------------------
# Batched (ensemble) variant of the exact same equations, for Node A's
# GPU-accelerated parameter-fitting rollout (plan §2.1). Runs on torch
# tensors (GPU, if CUDA is available on the Orin) when torch is importable,
# and transparently falls back to plain numpy on the CPU otherwise — the
# math is identical either way, only the array backend changes.
# ---------------------------------------------------------------------------
try:
    import torch
    HAS_TORCH = torch.cuda.is_available()
except ImportError:
    torch = None
    HAS_TORCH = False


def _pick_array_backend():
    """Returns (xp, device) — xp is the array module used for Node A's
    ensemble rollout, device is where tensors live ('cuda' or None for numpy)."""
    if HAS_TORCH:
        return torch, 'cuda'
    return np, None


def _cast_to_backend(xp, device, arr):
    """Casts a numpy array to the active backend (torch tensor on CUDA, or
    numpy array on CPU) so the batch engine can operate on it."""
    if xp is torch:
        return torch.as_tensor(arr, dtype=torch.float64, device=device)
    return np.asarray(arr, dtype=np.float64)


def angle_table_lookup_batch(xp, table_1d_np, theta: 'array (N,)'):
    """Batch (ensemble-N) circular linear interpolation into an angle-keyed
    1-D table — same logic as angle_table_lookup but vectorised over the
    entire ensemble in one shot for Node A's GPU rollout."""
    angle_deg = xp.remainder(theta * (180.0 / math.pi), 360.0)
    idx_lo = xp.floor(angle_deg).long() if xp is torch else angle_deg.astype(np.int64)
    idx_hi = (idx_lo + 1) % N_TABLE_SAMPLES
    frac = angle_deg - (idx_lo.double() if xp is torch else idx_lo)
    table = _cast_to_backend(xp, theta.device if xp is torch else None, table_1d_np)
    return (1 - frac) * table[idx_lo] + frac * table[idx_hi]


def calc_body_accels_batch(xp, body_vel, body_wrench, coeffs_b: dict, tables_np: dict,
                           mass_kg: float, inertia_z_kgm2: float):
    """Batch (ensemble-N) version of calc_body_accels, used by Node A's
    GPU-accelerated parameter-fitting rollout.  body_vel: (N,3), body_wrench:
    (N,3) or (3,) broadcastable, coeffs_b: dict of (N,) tensors/arrays.
    Returns ensemble accelerations [accel_surge, accel_sway, accel_yaw] (N,3)."""
    vel_surge, vel_sway, vel_yaw = body_vel[:, 0], body_vel[:, 1], body_vel[:, 2]
    speed = xp.sqrt(vel_surge * vel_surge + vel_sway * vel_sway) + 1e-12
    theta = xp.atan2(vel_sway, vel_surge) if xp is torch else np.arctan2(vel_sway, vel_surge)

    frontal_area_along   = angle_table_lookup_batch(xp, tables_np['A'], theta)
    drag_quad_along_raw  = angle_table_lookup_batch(xp, tables_np['D_quad'], theta)
    added_mass_along_raw = angle_table_lookup_batch(xp, tables_np['M_a'], theta)
    theta_cross = theta + math.pi / 2
    added_mass_cross_raw = angle_table_lookup_batch(xp, tables_np['M_a'], theta_cross)

    blend_drag_quad  = coeffs_b['k_low_dquad'] + (coeffs_b['k_high_dquad'] - coeffs_b['k_low_dquad']) * \
        (1.0 / (1.0 + xp.exp(-(speed - coeffs_b['v_regime_mps']) / coeffs_b['blend_width_mps'])))
    blend_added_mass = coeffs_b['k_low_ma'] + (coeffs_b['k_high_ma'] - coeffs_b['k_low_ma']) * \
        (1.0 / (1.0 + xp.exp(-(speed - coeffs_b['v_regime_mps']) / coeffs_b['blend_width_mps'])))

    drag_quad_along  = drag_quad_along_raw  * coeffs_b['cd_scale'] * blend_drag_quad
    added_mass_along = added_mass_along_raw * blend_added_mass * coeffs_b['xu_dot_scale']
    added_mass_cross = added_mass_cross_raw * blend_added_mass * coeffs_b['yv_dot_scale']

    mass_eff_along = mass_kg + added_mass_along
    mass_eff_cross = mass_kg + added_mass_cross

    force_x  = body_wrench[:, 0] if body_wrench.ndim == 2 else body_wrench[0]
    force_y  = body_wrench[:, 1] if body_wrench.ndim == 2 else body_wrench[1]
    moment_z = body_wrench[:, 2] if body_wrench.ndim == 2 else body_wrench[2]

    drag_lin_x, drag_lin_y, drag_lin_yaw = coeffs_b['d_lin0'], coeffs_b['d_lin1'], coeffs_b['d_lin2']
    force_net_x = force_x  - drag_lin_x * vel_surge
    force_net_y = force_y  - drag_lin_y * vel_sway

    cos_theta, sin_theta = xp.cos(theta), xp.sin(theta)
    tau_along = cos_theta * force_net_x + sin_theta * force_net_y
    tau_cross  = -sin_theta * force_net_x + cos_theta * force_net_y

    drag_quad_along_force = drag_quad_along * speed * speed
    accel_along = (tau_along - drag_quad_along_force) / mass_eff_along
    accel_cross  = tau_cross / mass_eff_cross

    accel_surge = cos_theta * accel_along - sin_theta * accel_cross
    accel_sway  = sin_theta * accel_along + cos_theta * accel_cross

    mass_eff_yaw = inertia_z_kgm2 + coeffs_b['inertia_yaw_added_kgm2'] * coeffs_b['nr_dot_scale']
    accel_yaw    = (moment_z - drag_lin_yaw * vel_yaw - coeffs_b['d_quad_yaw'] * vel_yaw * xp.abs(vel_yaw)) / mass_eff_yaw

    return xp.stack([accel_surge, accel_sway, accel_yaw], dim=1) if xp is torch else np.stack([accel_surge, accel_sway, accel_yaw], axis=1)


def integrate_one_step_rk4_batch(xp, body_vel, body_wrench, dt, coeffs_b, tables_np, mass_kg, inertia_z_kgm2):
    """Batch 4-stage RK4 integrator used exclusively by Node A's ensemble
    rollout — advances all N candidate body velocities simultaneously by dt."""
    k1 = calc_body_accels_batch(xp, body_vel,                    body_wrench, coeffs_b, tables_np, mass_kg, inertia_z_kgm2)
    k2 = calc_body_accels_batch(xp, body_vel + 0.5 * dt * k1,   body_wrench, coeffs_b, tables_np, mass_kg, inertia_z_kgm2)
    k3 = calc_body_accels_batch(xp, body_vel + 0.5 * dt * k2,   body_wrench, coeffs_b, tables_np, mass_kg, inertia_z_kgm2)
    k4 = calc_body_accels_batch(xp, body_vel + dt * k3,          body_wrench, coeffs_b, tables_np, mass_kg, inertia_z_kgm2)
    return body_vel + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


class DeadReckoningIntegrator:
    """Owns the running velocity/position state for one node instance.
    Position integration always uses measured yaw (never the model's own
    integrated yaw) to rotate body-frame velocity into the world frame."""

    def __init__(self, coeffs: DynamicsCoefficients, params: VehicleParams, tables: dict):
        self.coeffs = coeffs
        self.params = params
        self.tables = tables
        self.body_vel = np.zeros(3)   # [vel_surge, vel_sway, vel_yaw] body frame
        self.pos_x = 0.0
        self.pos_y = 0.0

    def advance_and_integrate(self, body_wrench: np.ndarray, measured_yaw_rad: float, dt: float):
        """Advances body velocity one RK4 step with body_wrench, then rotates
        body-frame velocity into the world frame using measured_yaw_rad and
        accumulates world-frame position.  Returns (pos_x, pos_y, body_vel)."""
        self.body_vel = integrate_one_step_rk4(self.body_vel, body_wrench, dt, self.coeffs, self.params, self.tables)
        cos_yaw, sin_yaw = math.cos(measured_yaw_rad), math.sin(measured_yaw_rad)
        vel_surge, vel_sway, _ = self.body_vel
        vel_world_x = cos_yaw * vel_surge - sin_yaw * vel_sway
        vel_world_y = sin_yaw * vel_surge + cos_yaw * vel_sway
        self.pos_x += vel_world_x * dt
        self.pos_y += vel_world_y * dt
        return self.pos_x, self.pos_y, self.body_vel.copy()
