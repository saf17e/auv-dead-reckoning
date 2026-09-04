#!/usr/bin/env python3
"""cad_dynamics.py — STEP file ingestion and full dynamic-state extraction
(plan §2.3.1).

The single entry point for turning the CAD export into every physical
constant the rest of dead_reckoning/ needs: mass, center of mass, the yaw
moment of inertia, displaced volume, and the direction-dependent
hydrodynamic tables A(theta)/D_quad(theta)/M_a(theta) (plan §1.2). Run
manually whenever the CAD changes — this is a preprocessing step, not a ROS
node:

    python -m robofest_6.dead_reckoning.cad_dynamics --step path/to/hull.step

Mass/CoM/inertia come straight from the STEP B-rep via `cadquery`
(OpenCASCADE bindings) — exact geometry, no fitting needed, modulo the
density assumption noted below. The hydrodynamic tables need a triangle
mesh (tessellation of the same B-rep, done once here) since projecting a
silhouette or approximating a cross-section as an ellipse is a mesh
operation, not a B-rep one.

The mesh-only geometry functions (`frontal_area_table`,
`added_mass_table`) take plain numpy vertex arrays and have no cadquery
dependency, so they're independently unit-testable against a synthetic
mesh (see test_cad_dynamics.py) even on a machine without cadquery/OCP
installed.
"""
import argparse
import math
import os

import numpy as np
import yaml
from scipy.spatial import ConvexHull

try:
    import cadquery as cq
    HAS_CADQUERY = True
except ImportError:
    cq = None
    HAS_CADQUERY = False

CONFIG_DIR = os.path.join(os.path.dirname(__file__), 'config')
HYDRO_TABLES_PATH = os.path.join(CONFIG_DIR, 'hydrodynamic_tables.npz')
PRIOR_CONFIG_PATH = os.path.join(CONFIG_DIR, 'dynamics_estimator.yaml')

N_ANGLES = 360


# ---------------------------------------------------------------------------
# STEP -> (mass properties, triangle mesh). The only cadquery-dependent code
# in this module.
# ---------------------------------------------------------------------------
def load_step(step_path, tessellation_tol=0.5, angular_tol=0.3):
    if not HAS_CADQUERY:
        raise RuntimeError(
            "cadquery is not installed. Install it (`pip install cadquery`) "
            "to parse STEP files — FreeCAD's Python API is the documented "
            "fallback if cadquery/OCP wheels aren't available on this "
            "platform (see dead_reckoning/README.md)."
        )
    result = cq.importers.importStep(step_path)
    shape = result.val()

    vertices, faces = shape.tessellate(tessellation_tol, angular_tol)
    vertices_np = np.array([[v.x, v.y, v.z] for v in vertices])
    faces_np = np.array(faces)
    return shape, vertices_np, faces_np


def compute_mass_properties(shape, density_kgm3=None, known_mass_kg=None):
    """Volume, center of mass, and yaw (Izz) inertia about the CoM, straight
    from the OpenCASCADE B-rep. STEP geometry alone only gives *relative*
    mass distribution (volume moments assuming uniform unit density) unless
    the source file carries material assignments — so an explicit density
    (or a bench-measured total mass, from which density is back-solved) is
    required to turn those into real mass/inertia numbers."""
    # cadquery exposes volume/CoM/inertia via the underlying OCP GProp
    # object; the exact call shape can shift slightly between cadquery
    # versions, so this is intentionally isolated in one place to patch if
    # the installed version differs from what's assumed here.
    from OCP.GProp import GProp_GProps
    from OCP.BRepGProp import BRepGProp

    gprops = GProp_GProps()
    BRepGProp.VolumeProperties_s(shape.wrapped, gprops)

    volume_m3 = gprops.Mass()  # "mass" at implicit unit density == geometric volume
    com = gprops.CentreOfMass()
    center_of_mass = np.array([com.X(), com.Y(), com.Z()])

    inertia_mat = gprops.MatrixOfInertia()  # about the CoM, unit density
    izz_unit_density = inertia_mat.Value(3, 3)

    if known_mass_kg is not None:
        density_kgm3 = known_mass_kg / volume_m3
    elif density_kgm3 is None:
        density_kgm3 = 1000.0  # sealed-enclosure placeholder; override at the CLI

    mass_kg = density_kgm3 * volume_m3
    inertia_z_kgm2 = density_kgm3 * izz_unit_density

    return {
        'volume_m3': float(volume_m3),
        'center_of_mass_m': center_of_mass.tolist(),
        'density_kgm3': float(density_kgm3),
        'mass_kg': float(mass_kg),
        'inertia_z_kgm2': float(inertia_z_kgm2),
    }


# ---------------------------------------------------------------------------
# Pure-mesh geometry (no cadquery dependency) — direction-dependent tables.
# ---------------------------------------------------------------------------
def frontal_area_table(vertices: np.ndarray, n_angles=N_ANGLES) -> np.ndarray:
    """A(theta): projected silhouette area vs. horizontal direction of
    travel theta, sampled every 360/n_angles degrees.

    For each theta, project every mesh vertex onto the plane perpendicular
    to the travel direction (spanned by the cross-track axis and the
    vertical axis) and take the convex hull area of that 2D point set.
    Using the convex hull rather than a full concave silhouette rasterization
    is a deliberate simplification — reasonable for a mostly-convex AUV hull,
    and cheap enough to sample at 1-degree resolution; flagged as a future
    refinement (proper rasterized silhouette) if residuals ever show hull
    concavity matters for this vehicle's shape.

    The per-angle rotation/projection is vectorized across all angles at
    once (a single batched matrix multiply), which is the actual
    embarrassingly-parallel, GPU-friendly part of this computation — the
    per-angle convex hull call itself has no off-the-shelf GPU
    implementation and is run as a cheap O(n_angles) loop.
    """
    thetas = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)
    cross_track = np.stack([-np.sin(thetas), np.cos(thetas), np.zeros(n_angles)], axis=1)  # (n_angles, 3)

    u_all = vertices @ cross_track.T   # (n_vertices, n_angles) — batched projection
    w_all = np.broadcast_to(vertices[:, 2:3], (vertices.shape[0], n_angles))

    areas = np.zeros(n_angles)
    for i in range(n_angles):
        pts = np.stack([u_all[:, i], w_all[:, i]], axis=1)
        try:
            areas[i] = ConvexHull(pts).volume  # 2D hull's "volume" is its area
        except Exception:
            areas[i] = 0.0
    return areas


def added_mass_table(vertices: np.ndarray, n_angles=N_ANGLES, rho_water_kgm3=997.0) -> np.ndarray:
    """M_a(theta): naval-architecture strip-theory approximation — the
    cross-section perpendicular to the direction of travel is treated as an
    equivalent ellipse, using the standard 2D added-mass formula for an
    ellipse accelerating along its minor axis (rho*pi*b^2 per unit length,
    b = half the cross-track extent), then scaled by the hull's extent
    along the travel direction as a first-order stand-in for integrating
    varying cross-sections along the body length (plan §1.2). This is a
    physically-motivated starting table, not a precise CFD result — refined
    online by Node A exactly like the drag coefficient.
    """
    thetas = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)
    along = np.stack([np.cos(thetas), np.sin(thetas), np.zeros(n_angles)], axis=1)
    cross_track = np.stack([-np.sin(thetas), np.cos(thetas), np.zeros(n_angles)], axis=1)

    l_all = vertices @ along.T          # (n_vertices, n_angles)
    u_all = vertices @ cross_track.T    # (n_vertices, n_angles)

    length_along = l_all.max(axis=0) - l_all.min(axis=0)     # (n_angles,)
    beam_cross_track = u_all.max(axis=0) - u_all.min(axis=0)  # (n_angles,)

    b = beam_cross_track / 2.0
    m_a_per_length = rho_water_kgm3 * np.pi * b ** 2
    return m_a_per_length * length_along


def compute_hydrodynamic_tables(vertices, n_angles=N_ANGLES, cd_literature=0.8, rho_water_kgm3=997.0):
    A = frontal_area_table(vertices, n_angles)
    D_quad = 0.5 * rho_water_kgm3 * cd_literature * A
    M_a = added_mass_table(vertices, n_angles, rho_water_kgm3)
    return {'A': A, 'D_quad': D_quad, 'M_a': M_a}


# ---------------------------------------------------------------------------
# Output — writes both files both Node A and Node B read at startup.
# ---------------------------------------------------------------------------
def write_outputs(mass_props: dict, tables: dict, output_dir=CONFIG_DIR,
                   rho_water_kgm3=997.0, step_path=''):
    os.makedirs(output_dir, exist_ok=True)

    npz_path = os.path.join(output_dir, 'hydrodynamic_tables.npz')
    np.savez(npz_path, A=tables['A'], D_quad=tables['D_quad'], M_a=tables['M_a'])

    yaml_path = os.path.join(output_dir, 'dynamics_estimator.yaml')
    cfg = {}
    if os.path.exists(yaml_path):
        with open(yaml_path, 'r') as f:
            cfg = yaml.safe_load(f) or {}

    cfg['vehicle_params'] = {
        'mass_kg': mass_props['mass_kg'],
        'inertia_z_kgm2': mass_props['inertia_z_kgm2'],
        'rho_water_kgm3': rho_water_kgm3,
    }
    cfg['cad_source'] = {
        'step_path': step_path,
        'volume_m3': mass_props['volume_m3'],
        'center_of_mass_m': mass_props['center_of_mass_m'],
        'density_kgm3': mass_props['density_kgm3'],
    }
    cfg.setdefault('coefficients', {})

    with open(yaml_path, 'w') as f:
        yaml.safe_dump(cfg, f)

    return npz_path, yaml_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--step', required=True, help='Path to the STEP file (.step/.stp)')
    parser.add_argument('--density', type=float, default=None,
                         help='Uniform density override (kg/m^3), used if --mass is not given')
    parser.add_argument('--mass', type=float, default=None,
                         help='Known total mass (kg, e.g. from a bench scale) — overrides --density')
    parser.add_argument('--angles', type=int, default=N_ANGLES,
                         help='Number of direction samples for the hydrodynamic tables (default 360)')
    parser.add_argument('--tessellation-tol', type=float, default=0.5,
                         help='STEP tessellation linear tolerance (mm-equivalent CAD units)')
    parser.add_argument('--cd', type=float, default=0.8,
                         help='Literature quadratic-drag coefficient applied uniformly across all angles')
    parser.add_argument('--rho-water', type=float, default=997.0)
    parser.add_argument('--output-dir', default=CONFIG_DIR)
    args = parser.parse_args()

    shape, vertices, faces = load_step(args.step, tessellation_tol=args.tessellation_tol)
    mass_props = compute_mass_properties(shape, density_kgm3=args.density, known_mass_kg=args.mass)
    tables = compute_hydrodynamic_tables(vertices, n_angles=args.angles,
                                          cd_literature=args.cd, rho_water_kgm3=args.rho_water)
    npz_path, yaml_path = write_outputs(mass_props, tables, args.output_dir,
                                         rho_water_kgm3=args.rho_water, step_path=args.step)

    print(f"mass_kg            = {mass_props['mass_kg']:.4f}")
    print(f"inertia_z_kgm2     = {mass_props['inertia_z_kgm2']:.6f}")
    print(f"volume_m3          = {mass_props['volume_m3']:.6f}")
    print(f"center_of_mass_m   = {mass_props['center_of_mass_m']}")
    print(f"A(0deg)  [bow-on]  = {tables['A'][0]:.5f} m^2")
    print(f"A(90deg) [beam-on] = {tables['A'][args.angles // 4]:.5f} m^2")
    print(f"wrote {npz_path}")
    print(f"wrote {yaml_path}")


if __name__ == '__main__':
    main()
