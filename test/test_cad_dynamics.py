#!/usr/bin/env python3
"""Unit tests for cad_dynamics.py's pure-mesh geometry functions (plan
Part 3.2). These do not touch cadquery/STEP parsing at all — they exercise
frontal_area_table()/added_mass_table() against a synthetic box mesh (just
the 8 corner vertices; these functions only need vertices, not faces),
where the expected answer is easy to check by hand."""
import math

import numpy as np
import pytest

from robofest_6.dead_reckoning import cad_dynamics as cd


def _box_vertices(lx, ly, lz):
    xs = [-lx / 2, lx / 2]
    ys = [-ly / 2, ly / 2]
    zs = [-lz / 2, lz / 2]
    return np.array([[x, y, z] for x in xs for y in ys for z in zs])


def test_frontal_area_bow_on_and_beam_on():
    lx, ly, lz = 0.6, 0.4, 0.2
    verts = _box_vertices(lx, ly, lz)
    areas = cd.frontal_area_table(verts, n_angles=360)

    # theta=0 (moving along +x): silhouette seen face-on is the y-z face.
    assert areas[0] == pytest.approx(ly * lz, rel=1e-6)
    # theta=90deg (moving along +y): silhouette is the x-z face.
    assert areas[90] == pytest.approx(lx * lz, rel=1e-6)


def test_added_mass_table_matches_ellipse_strip_formula():
    lx, ly, lz = 0.6, 0.4, 0.2
    rho = 997.0
    verts = _box_vertices(lx, ly, lz)
    table = cd.added_mass_table(verts, n_angles=360, rho_water_kgm3=rho)

    expected_theta0 = rho * math.pi * (ly / 2.0) ** 2 * lx
    assert table[0] == pytest.approx(expected_theta0, rel=1e-6)

    expected_theta90 = rho * math.pi * (lx / 2.0) ** 2 * ly
    assert table[90] == pytest.approx(expected_theta90, rel=1e-6)


def test_compute_hydrodynamic_tables_shapes_and_positivity():
    verts = _box_vertices(0.6, 0.4, 0.2)
    tables = cd.compute_hydrodynamic_tables(verts, n_angles=36)
    for key in ('A', 'D_quad', 'M_a'):
        assert tables[key].shape == (36,)
        assert np.all(tables[key] >= 0)
