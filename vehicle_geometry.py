#!/usr/bin/env python3
"""vehicle_geometry.py — thruster mount geometry and the 6-DOF thrust
allocation matrix, shared by thruster_node.py and the dead-reckoning
estimator nodes.

Geometry precedence: if config/dynamics_estimator.yaml carries a
`thruster_geometry` block (written by cad_dynamics.py from the real CAD
assembly), those exact positions/orientations are used. Otherwise this
module falls back to the box-corner placeholder geometry that
thruster_node.py originally hardcoded — acknowledged to be a rough
approximation, not the real hull.
"""
import os
import numpy as np
import yaml

THRUSTER_NAMES = ["T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8"]

CONFIG_DIR = os.path.join(os.path.dirname(__file__), 'config')
DYNAMICS_CONFIG_PATH = os.path.join(CONFIG_DIR, 'dynamics_estimator.yaml')

THRUSTER_FORCE_LIMIT_KGF = 2.9

# ---------------------------------------------------------------------------
# Placeholder geometry (fallback only — same values thruster_node.py used).
# ---------------------------------------------------------------------------
_AUV_LENGTH = 0.6
_AUV_WIDTH = 0.4
_AUV_HEIGHT = 0.2

_angle_x = np.radians(45)
_angle_y = np.radians(60)
_angle_z = np.radians(60)

_fx_base = np.cos(_angle_x)
_fy_base = np.cos(_angle_y)
_fz_base = np.cos(_angle_z)

_PLACEHOLDER_POSITIONS = {
    'T1': [+_AUV_LENGTH/2, +_AUV_WIDTH/2, +_AUV_HEIGHT/2],
    'T2': [+_AUV_LENGTH/2, -_AUV_WIDTH/2, +_AUV_HEIGHT/2],
    'T3': [-_AUV_LENGTH/2, -_AUV_WIDTH/2, +_AUV_HEIGHT/2],
    'T4': [-_AUV_LENGTH/2, +_AUV_WIDTH/2, +_AUV_HEIGHT/2],
    'T5': [+_AUV_LENGTH/2, +_AUV_WIDTH/2, -_AUV_HEIGHT/2],
    'T6': [+_AUV_LENGTH/2, -_AUV_WIDTH/2, -_AUV_HEIGHT/2],
    'T7': [-_AUV_LENGTH/2, -_AUV_WIDTH/2, -_AUV_HEIGHT/2],
    'T8': [-_AUV_LENGTH/2, +_AUV_WIDTH/2, -_AUV_HEIGHT/2],
}

_PLACEHOLDER_SIGNS = {
    'T1': [1, -1, -1], 'T2': [-1, -1, 1],
    'T3': [-1, 1, -1], 'T4': [1, 1, 1],
    'T5': [-1, 1, -1], 'T6': [1, 1, 1],
    'T7': [1, -1, -1], 'T8': [-1, -1, 1],
}


def _load_cad_geometry():
    """Return (positions, signs, force_scale) dicts from the CAD-derived config
    if present, else (None, None, None)."""
    if not os.path.exists(DYNAMICS_CONFIG_PATH):
        return None, None, None
    with open(DYNAMICS_CONFIG_PATH, 'r') as f:
        cfg = yaml.safe_load(f) or {}
    geom = cfg.get('thruster_geometry')
    if not geom:
        return None, None, None
    positions = {name: geom['positions'][name] for name in THRUSTER_NAMES}
    # signs come from CAD as unit thrust-direction vectors, not +-1 flags,
    # since CAD-exact mount angles are rarely axis-aligned.
    signs = {name: geom['directions'][name] for name in THRUSTER_NAMES}
    return positions, signs, geom.get('source', 'cad')


_cad_positions, _cad_signs, GEOMETRY_SOURCE = _load_cad_geometry()

if _cad_positions is not None:
    THRUSTER_POSITIONS = _cad_positions
    THRUSTER_UNIT_DIRECTIONS = _cad_signs
else:
    GEOMETRY_SOURCE = 'placeholder'
    THRUSTER_POSITIONS = _PLACEHOLDER_POSITIONS
    THRUSTER_UNIT_DIRECTIONS = {
        name: [
            _PLACEHOLDER_SIGNS[name][0] * _fx_base,
            _PLACEHOLDER_SIGNS[name][1] * _fy_base,
            _PLACEHOLDER_SIGNS[name][2] * _fz_base,
        ]
        for name in THRUSTER_NAMES
    }


def build_TAM_6DOF():
    """Build the 6x8 Thrust Allocation Matrix mapping per-thruster unit
    thrust -> [Fx, Fy, Fz, Mx, My, Mz], from whichever geometry source
    (CAD or placeholder) is active."""
    TAM6 = np.zeros((6, 8))
    for idx, name in enumerate(THRUSTER_NAMES):
        fx, fy, fz = THRUSTER_UNIT_DIRECTIONS[name]
        pos = THRUSTER_POSITIONS[name]
        TAM6[0, idx] = fx
        TAM6[1, idx] = fy
        TAM6[2, idx] = fz
        TAM6[3, idx] = pos[1] * fz - pos[2] * fy   # Mx
        TAM6[4, idx] = pos[2] * fx - pos[0] * fz   # My
        TAM6[5, idx] = pos[0] * fy - pos[1] * fx   # Mz
    return TAM6


TAM_6DOF = build_TAM_6DOF()
TAM_6DOF_INV = np.linalg.pinv(TAM_6DOF)  # 8x6 pseudo-inverse


def _norm_col(col_idx):
    col = TAM_6DOF_INV[:, col_idx].copy()
    m = np.max(np.abs(col))
    return col / m if m > 1e-9 else col


CMD_FORWARD_ALLOC = _norm_col(0)  # Surge  (Fx) <- i
CMD_RIGHT_ALLOC   = _norm_col(1)  # Sway   (Fy) <- j
CMD_UP_ALLOC      = _norm_col(2)  # Heave  (Fz) <- k
CMD_YAW_ALLOC     = _norm_col(5)  # Yaw    (Mz) <- yaw_dir
