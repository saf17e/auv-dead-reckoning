#!/usr/bin/env python3
"""pwm_thrust_table.py — shared PWM<->thrust calibration lookup.

`thrust_to_pwm()` is the same direction thruster_node.py already uses
(force -> PWM, for commanding thrusters) and is exposed here so
thruster_node.py can import it instead of keeping its own copy.

`pwm_to_thrust()` is the new inverse direction the dead-reckoning
estimator needs (measured/commanded PWM -> estimated thrust), wrapped with
a bi-directional scale/bias correction layer (see plan §1.4) since the raw
pwm_new.json table is known to be somewhat unreliable and reverse-thrust
efficiency differs from forward.
"""
import bisect
import json
import os

PWM_JSON_PATH = '/home/agastya/amurov26/pwm_new.json'


class PwmThrustTable:
    """Loads pwm_new.json once and exposes both lookup directions."""

    def __init__(self, path=PWM_JSON_PATH):
        self.path = path
        with open(path, 'r') as f:
            raw = json.load(f)
        # sorted by thrust (kgf), for thrust_to_pwm (thruster_node.py's direction)
        by_thrust = sorted([(float(k), v) for k, v in raw.items()])
        self._thrusts = [item[0] for item in by_thrust]
        self._pwms_by_thrust = [item[1] for item in by_thrust]
        # sorted by pwm, for pwm_to_thrust (the estimator's direction)
        by_pwm = sorted([(v, float(k)) for k, v in raw.items()])
        self._pwms = [item[0] for item in by_pwm]
        self._thrusts_by_pwm = [item[1] for item in by_pwm]

    @staticmethod
    def _nearest(sorted_keys, values, query):
        idx = bisect.bisect_left(sorted_keys, query)
        if idx == 0:
            return values[0]
        if idx == len(sorted_keys):
            return values[-1]
        before, after = sorted_keys[idx - 1], sorted_keys[idx]
        if after - query < query - before:
            return values[idx]
        return values[idx - 1]

    def thrust_to_pwm(self, thrust_kgf: float) -> float:
        """Bench-calibration table lookup, force (kgf) -> raw PWM units.
        Unchanged behavior from thruster_node.py's original force_to_pwm
        table-lookup half (caller still applies its own +-400 clamp)."""
        return self._nearest(self._thrusts, self._pwms_by_thrust, thrust_kgf)

    def raw_pwm_to_thrust(self, pwm: float) -> float:
        """Uncorrected inverse lookup: PWM units -> table thrust (kgf)."""
        return self._nearest(self._pwms, self._thrusts_by_pwm, pwm)

    def pwm_to_thrust(self, pwm: float, scale_fwd=1.0, bias_fwd=0.0,
                       scale_rev=1.0, bias_rev=0.0) -> float:
        """Corrected inverse lookup used by the dead-reckoning estimator:

            f = scale_dir * table_lookup(pwm) + bias_dir,  dir = sign(pwm)

        Forward (pwm >= 0) and reverse (pwm < 0) get independent
        scale/bias pairs since propeller thrust is typically not symmetric
        between rotation directions (plan §1.4). `scale`/`bias` default to
        the identity correction (1, 0) until Node A's online fine-tuning
        has produced fitted values.
        """
        raw = self.raw_pwm_to_thrust(pwm)
        if pwm >= 0:
            return scale_fwd * raw + bias_fwd
        return scale_rev * raw + bias_rev


_default_table = None


def get_default_table(path=PWM_JSON_PATH) -> PwmThrustTable:
    """Process-wide singleton, mirroring how thruster_node.py loads the
    table once at node startup rather than per-call."""
    global _default_table
    if _default_table is None or _default_table.path != path:
        _default_table = PwmThrustTable(path)
    return _default_table
