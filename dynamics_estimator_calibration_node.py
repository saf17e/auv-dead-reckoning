#!/usr/bin/env python3
"""dynamics_estimator_calibration_node.py — Node A: sonar-in-the-loop online
fine-tuning of the dead-reckoning coefficient vector (plan §2.1).

Subscribes to the same `/debug/final_pwm` + `/bno` stream Node B uses, plus
`/sonar/x`/`/sonar/y`. Buffers raw per-thruster PWM + measured yaw between
trusted sonar fixes (confidence > 90, same threshold thruster_node.py already
gates on). Each time a new trusted fix arrives, it replays that buffered
window through a small ensemble of candidate coefficient vectors — batched
across the ensemble dimension with dynamics_core's batch functions (torch on
CUDA if available, numpy otherwise) — scores each candidate against how far
its predicted position landed from the sonar-observed position, and updates
a running best estimate with an exponential forgetting factor plus mild
regularization back toward the CAD/literature prior. Periodically writes the
running estimate to config/dynamics_estimator_fitted.yaml for Node B to pick
up.

This node never publishes a pose of its own — it exists purely to keep
Node B's coefficients current.
"""
import math
import time
import numpy as np
import yaml
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray, Float32MultiArray

from . import vehicle_geometry as geom
from . import pwm_thrust_table as pwm_table
from . import dynamics_core as core

ENSEMBLE_SIZE = 64
JITTER_STD_FRACTION = 0.05   # ensemble spread as a fraction of each param's prior value
FORGETTING_LAMBDA = 0.3      # weight given to each new correction vs. the running estimate
REGULARIZATION_PULL = 0.02   # per-tick pull toward the prior when no sonar fix has landed
SONAR_CONFIDENCE_THRESHOLD = 90.0
WRITE_INTERVAL_SEC = 30.0
MAX_BUFFER_SECONDS = 20.0


class DynamicsEstimatorCalibrationNode(Node):
    def __init__(self):
        super().__init__('dynamics_estimator_calibration_node')

        self.declare_parameter('loop_rate_hz', 100.0)
        self.declare_parameter('final_pwm_topic', '/debug/final_pwm')
        self.declare_parameter('bno_topic', '/bno')
        self.declare_parameter('sonar_x_topic', '/sonar/x')
        self.declare_parameter('sonar_y_topic', '/sonar/y')

        self.dt = 1.0 / self.get_parameter('loop_rate_hz').value

        self.params = core.load_vehicle_params()
        self.tables = core.load_hydro_tables_from_disk()
        self.pwm_table = pwm_table.get_default_table()
        self.tam3 = geom.TAM_6DOF[:3, :]

        # `d_lin` is a 3-vector field on DynamicsCoefficients, not a scalar —
        # excluded from the flat ensemble vector and held fixed at the
        # prior's value for every candidate (still correctable indirectly
        # via cd_scale/k_low/k_high, which scale the quadratic term it sits
        # alongside).
        prior = core.load_coefficients(prefer_fitted=False)
        prior_dict = prior.to_dict()
        self.d_lin_fixed = list(prior_dict.pop('d_lin'))
        self.field_names = list(prior_dict.keys())
        self.mean = np.array([prior_dict[k] for k in self.field_names], dtype=np.float64)
        self.prior_vec = self.mean.copy()
        self._seed_ensemble_from_prior()

        self._buffer = []           # list of (pwm_8vec, yaw_rad)
        self._latest_pwm = [0.0] * 8
        self._latest_yaw = 0.0
        self._have_pwm = False
        self._have_yaw = False
        self._last_fix_xy = None
        self._raw_x = self._raw_y = 0.0
        self._conf_x = self._conf_y = 0.0
        self._last_write_time = time.monotonic()

        self.create_subscription(Float64MultiArray, self.get_parameter('final_pwm_topic').value,
                                  self._receive_pwm_command, 10)
        self.create_subscription(Float64MultiArray, self.get_parameter('bno_topic').value,
                                  self._receive_imu_yaw, 10)
        self.create_subscription(Float32MultiArray, self.get_parameter('sonar_x_topic').value,
                                  self._receive_sonar_x, 1)
        self.create_subscription(Float32MultiArray, self.get_parameter('sonar_y_topic').value,
                                  self._receive_sonar_y, 1)

        self.timer = self.create_timer(self.dt, self._tick_buffer_and_regularize)

        xp, device = core._pick_array_backend()
        self.get_logger().info(
            f"dynamics_estimator_calibration_node started "
            f"(ensemble={ENSEMBLE_SIZE}, backend={'torch/cuda' if device else 'numpy/cpu'})")

    # -- ensemble management -------------------------------------------------
    def _seed_ensemble_from_prior(self):
        """Seeds all N ensemble candidates with Gaussian jitter around the
        prior mean — called once at startup."""
        spread = np.abs(self.mean) * JITTER_STD_FRACTION + 1e-6
        self.ensemble = self.mean[None, :] + np.random.randn(ENSEMBLE_SIZE, len(self.mean)) * spread[None, :]

    def _reseed_ensemble_around_mean(self):
        """Reseeds all N ensemble candidates with Gaussian jitter around the
        current running mean — called after every sonar correction."""
        spread = np.abs(self.mean) * JITTER_STD_FRACTION + 1e-6
        self.ensemble = self.mean[None, :] + np.random.randn(ENSEMBLE_SIZE, len(self.mean)) * spread[None, :]

    def _build_ensemble_coeffs_dict(self):
        """Slices the (N, num_fields) ensemble array into a dict of (N,)
        arrays keyed the way dynamics_core's batch functions expect, and
        broadcasts the fixed d_lin values across all candidates."""
        idx = {name: i for i, name in enumerate(self.field_names)}
        out = {name: self.ensemble[:, idx[name]] for name in self.field_names}
        out['d_lin0'] = np.full(ENSEMBLE_SIZE, self.d_lin_fixed[0])
        out['d_lin1'] = np.full(ENSEMBLE_SIZE, self.d_lin_fixed[1])
        out['d_lin2'] = np.full(ENSEMBLE_SIZE, self.d_lin_fixed[2])
        return out, idx

    # -- topic callbacks -------------------------------------------------------
    def _receive_pwm_command(self, msg):
        """Stores the latest 8-thruster PWM command vector from /debug/final_pwm."""
        self._latest_pwm = list(msg.data[:8])
        self._have_pwm = True

    def _receive_imu_yaw(self, msg):
        """Stores the latest body yaw (radians) from the BNO IMU /bno topic."""
        if len(msg.data) >= 3:
            self._latest_yaw = math.radians(msg.data[2])
            self._have_yaw = True

    def _receive_sonar_x(self, msg):
        """Stores the latest sonar X position and confidence; may trigger an
        ensemble update if both axes have high-confidence fixes."""
        self._raw_x, self._conf_x = float(msg.data[0]), float(msg.data[1])
        self._try_trigger_ensemble_update()

    def _receive_sonar_y(self, msg):
        """Stores the latest sonar Y position and confidence; may trigger an
        ensemble update if both axes have high-confidence fixes."""
        self._raw_y, self._conf_y = float(msg.data[0]), float(msg.data[1])
        self._try_trigger_ensemble_update()

    def _try_trigger_ensemble_update(self):
        """Checks that both sonar axes are above the confidence threshold.
        If so, uses the current fix to close out the buffered window and
        trigger a scored ensemble update."""
        if self._conf_x < SONAR_CONFIDENCE_THRESHOLD or self._conf_y < SONAR_CONFIDENCE_THRESHOLD:
            return
        fix = (self._raw_x, self._raw_y)
        if self._last_fix_xy is None:
            self._last_fix_xy = fix
            self._buffer.clear()
            return
        if len(self._buffer) < 2:
            self._last_fix_xy = fix
            self._buffer.clear()
            return
        self._score_and_update_ensemble(self._last_fix_xy, fix)
        self._last_fix_xy = fix
        self._buffer.clear()




    def _tick_buffer_and_regularize(self):
        """100 Hz timer tick: appends the current sensor snapshot to the
        rolling PWM/yaw buffer, applies prior regularization to the running
        mean, and periodically writes fitted coefficients to disk."""
        if self._have_pwm and self._have_yaw:
            self._buffer.append((list(self._latest_pwm), self._latest_yaw))
            max_len = int(MAX_BUFFER_SECONDS / self.dt)
            if len(self._buffer) > max_len:
                self._buffer.pop(0)
        else:
            return

        # Prior regularization: gently pull the running mean back toward the
        # CAD/literature prior whenever sonar corrections aren't landing, so
        # a long gap doesn't let the estimate wander on buffer noise alone.
        self.mean += REGULARIZATION_PULL * self.dt * (self.prior_vec - self.mean)

        if time.monotonic() - self._last_write_time > WRITE_INTERVAL_SEC:
            self._save_fitted_coeffs_to_yaml()
            self._last_write_time = time.monotonic()






    # -- GPU-batched ensemble rollout -----------------------------------------
    def _score_and_update_ensemble(self, start_xy, end_xy):
        """Replays the buffered PWM/yaw window through all N ensemble candidates
        in parallel (GPU if available), scores each by position error against
        the sonar-observed displacement, then updates the running mean with an
        exponential forgetting blend of the top-k candidates."""
        pwm_seq = np.array([b[0] for b in self._buffer])   # (T, 8)
        yaw_seq = np.array([b[1] for b in self._buffer])   # (T,)
        T = pwm_seq.shape[0]

        raw_thrust_seq = np.vectorize(self.pwm_table.raw_pwm_to_thrust)(pwm_seq)  # (T, 8)
        fwd_mask = pwm_seq >= 0  # (T, 8)

        coeffs_b, idx = self._build_ensemble_coeffs_dict()

        scale_fwd = coeffs_b['pwm_scale_fwd'][:, None, None]
        bias_fwd  = coeffs_b['pwm_bias_fwd'][:, None, None]
        scale_rev = coeffs_b['pwm_scale_rev'][:, None, None]
        bias_rev  = coeffs_b['pwm_bias_rev'][:, None, None]

        raw_thrust = raw_thrust_seq[None, :, :]          # (1, T, 8)
        fwd_mask_b  = fwd_mask[None, :, :]               # (1, T, 8)
        corrected_thrust = np.where(fwd_mask_b,
                                    scale_fwd * raw_thrust + bias_fwd,
                                    scale_rev * raw_thrust + bias_rev)  # (N, T, 8)

        body_wrench_seq = np.einsum('ij,ntj->nti', self.tam3, corrected_thrust)  # (N, T, 3)

        xp, device = core._pick_array_backend()
        body_vel = np.zeros((ENSEMBLE_SIZE, 3))
        pos_x = np.zeros(ENSEMBLE_SIZE)
        pos_y = np.zeros(ENSEMBLE_SIZE)

        for t in range(T):
            tau_t = body_wrench_seq[:, t, :]
            if xp is not np:
                body_vel_xp = core._cast_to_backend(xp, device, body_vel)
                tau_xp      = core._cast_to_backend(xp, device, tau_t)
                body_vel_xp = core.integrate_one_step_rk4_batch(xp, body_vel_xp, tau_xp, self.dt, coeffs_b,
                                                                 self.tables, self.params.mass_kg, self.params.inertia_z_kgm2)
                body_vel = body_vel_xp.cpu().numpy()
            else:
                body_vel = core.integrate_one_step_rk4_batch(np, body_vel, tau_t, self.dt, coeffs_b,
                                                              self.tables, self.params.mass_kg, self.params.inertia_z_kgm2)
            cos_yaw, sin_yaw = math.cos(yaw_seq[t]), math.sin(yaw_seq[t])
            pos_x += (cos_yaw * body_vel[:, 0] - sin_yaw * body_vel[:, 1]) * self.dt
            pos_y += (sin_yaw * body_vel[:, 0] + cos_yaw * body_vel[:, 1]) * self.dt

        pred_disp_x, pred_disp_y = pos_x, pos_y
        obs_disp_x = end_xy[0] - start_xy[0]
        obs_disp_y = end_xy[1] - start_xy[1]
        position_error = np.sqrt((pred_disp_x - obs_disp_x) ** 2 + (pred_disp_y - obs_disp_y) ** 2)

        k = max(1, ENSEMBLE_SIZE // 8)
        best_idx = np.argsort(position_error)[:k]
        weights = 1.0 / (position_error[best_idx] + 1e-6)
        weights /= weights.sum()
        updated = (self.ensemble[best_idx] * weights[:, None]).sum(axis=0)

        self.mean = FORGETTING_LAMBDA * updated + (1 - FORGETTING_LAMBDA) * self.mean
        self._reseed_ensemble_around_mean()
        self.get_logger().info(f"sonar correction applied, best-candidate position error: {position_error[best_idx[0]]:.3f} m")

    # -- persistence -----------------------------------------------------------
    def _save_fitted_coeffs_to_yaml(self):
        """Writes the current running mean coefficient vector to the fitted
        YAML file so Node B picks it up on its next coefficient reload."""
        values = dict(zip(self.field_names, self.mean.tolist()))
        values['d_lin'] = list(self.d_lin_fixed)
        out = {'coefficients': values}
        with open(core.FITTED_CONFIG_PATH, 'w') as f:
            yaml.safe_dump(out, f)
        self.get_logger().info(f"wrote fitted coefficients to {core.FITTED_CONFIG_PATH}")


def main(args=None):
    rclpy.init(args=args)
    node = DynamicsEstimatorCalibrationNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
