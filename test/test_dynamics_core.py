#!/usr/bin/env python3
"""Unit tests for dynamics_core.py (plan Part 3.1)."""
import json
import math
import os
import tempfile

import numpy as np
import pytest

from robofest_6.dead_reckoning import dynamics_core as core
from robofest_6.dead_reckoning import vehicle_geometry as geom
from robofest_6.dead_reckoning import pwm_thrust_table as pwm_table


def test_interp_table_wraps_at_360():
    table = np.arange(360, dtype=float)
    assert core.interp_table(table, math.radians(0)) == pytest.approx(0.0)
    assert core.interp_table(table, math.radians(359.5)) == pytest.approx(179.5)  # midpoint of 359 and wrapped 0
    assert core.interp_table(table, math.radians(10.5)) == pytest.approx(10.5)


def test_speed_regime_blend_limits():
    lo, hi = 1.0, 3.0
    v_regime, width = 0.5, 0.05
    assert core.speed_regime_blend(lo, hi, 0.0, v_regime, width) == pytest.approx(lo, abs=1e-3)
    assert core.speed_regime_blend(lo, hi, 5.0, v_regime, width) == pytest.approx(hi, abs=1e-3)
    mid = core.speed_regime_blend(lo, hi, v_regime, v_regime, width)
    assert mid == pytest.approx((lo + hi) / 2.0, abs=1e-6)


def test_zero_thrust_zero_velocity_stays_at_rest():
    coeffs = core.DynamicsCoefficients()
    params = core.VehicleParams()
    tables = core._flat_placeholder_tables()
    v = np.zeros(3)
    v_next = core.rk4_step(v, np.zeros(3), 0.01, coeffs, params, tables)
    assert np.allclose(v_next, 0.0)


def test_forward_thrust_produces_positive_surge_acceleration():
    coeffs = core.DynamicsCoefficients()
    params = core.VehicleParams()
    tables = core._flat_placeholder_tables()
    v = np.zeros(3)
    tau = np.array([10.0, 0.0, 0.0])
    v_dot = core.compute_state_derivative(v, tau, coeffs, params, tables)
    assert v_dot[0] > 0
    assert v_dot[1] == pytest.approx(0.0)
    assert v_dot[2] == pytest.approx(0.0)


def test_integrator_stays_at_origin_with_no_thrust():
    coeffs = core.DynamicsCoefficients()
    params = core.VehicleParams()
    tables = core._flat_placeholder_tables()
    integ = core.DeadReckoningIntegrator(coeffs, params, tables)
    for _ in range(50):
        x, y, v = integ.step(np.zeros(3), measured_yaw_rad=0.3, dt=0.01)
    assert x == pytest.approx(0.0)
    assert y == pytest.approx(0.0)


def test_tam_6dof_forward_command_yields_positive_fx():
    thrust = np.array(geom.CMD_FORWARD_ALLOC)
    tau = geom.TAM_6DOF @ thrust
    assert tau[0] > 0  # net Fx should be positive for a "forward" allocation


def test_pwm_thrust_table_round_trip():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, 'pwm.json')
        # linear-ish synthetic calibration table, both directions
        data = {str(f): f * 50.0 for f in np.linspace(-2.9, 2.9, 59)}
        with open(path, 'w') as f:
            json.dump(data, f)

        table = pwm_table.PwmThrustTable(path)
        raw = table.raw_pwm_to_thrust(100.0)
        assert raw == pytest.approx(2.0, abs=0.15)

        corrected = table.pwm_to_thrust(100.0, scale_fwd=2.0, bias_fwd=1.0)
        assert corrected == pytest.approx(2.0 * raw + 1.0)

        corrected_rev = table.pwm_to_thrust(-100.0, scale_rev=0.5, bias_rev=-0.1)
        raw_rev = table.raw_pwm_to_thrust(-100.0)
        assert corrected_rev == pytest.approx(0.5 * raw_rev - 0.1)
