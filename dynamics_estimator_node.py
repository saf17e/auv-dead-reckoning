#!/usr/bin/env python3
"""dynamics_estimator_node.py — Node B: pure open-loop dead reckoning.

No sonar, ever (plan §2.2). Subscribes to the same real per-thruster
command stream thruster_node.py drives the hardware with
(`/debug/final_pwm`) plus measured yaw (`/bno`), converts PWM back to
per-thruster thrust via the corrected pwm_thrust_table, maps that into
body-frame force/moment via the shared TAM, and integrates position
forward with dynamics_core's RK4 engine. Loads whatever coefficients
dynamics_estimator_calibration_node.py (Node A) has most recently fitted,
falling back to the static CAD/literature prior if Node A hasn't produced
fitted output yet.
"""
import math
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from geometry_msgs.msg import PoseStamped

from . import vehicle_geometry as geom
from . import pwm_thrust_table as pwm_table
from . import dynamics_core as core


class DynamicsEstimatorNode(Node):
    def __init__(self):
        super().__init__('dynamics_estimator_node')

        self.declare_parameter('loop_rate_hz', 100.0)
        self.declare_parameter('final_pwm_topic', '/debug/final_pwm')
        self.declare_parameter('bno_topic', '/bno')
        self.declare_parameter('pose_topic', '/dead_reckoning/pose')

        loop_rate_hz = self.get_parameter('loop_rate_hz').value

        self.coeffs = core.load_coefficients(prefer_fitted=True)
        self.params = core.load_vehicle_params()
        self.tables = core.load_hydro_tables_from_disk()
        self.integrator = core.DeadReckoningIntegrator(self.coeffs, self.params, self.tables)

        self.pwm_table = pwm_table.get_default_table()
        self.tam = geom.TAM_6DOF[:3, :]  # only Fx, Fy, Mz rows needed for 3-DOF planar model

        self._latest_pwm = [0.0] * 8
        self._latest_yaw_rad = 0.0
        self._have_pwm = False
        self._have_yaw = False

        self.create_subscription(
            Float64MultiArray, self.get_parameter('final_pwm_topic').value,
            self._receive_pwm_command, 10)
        self.create_subscription(
            Float64MultiArray, self.get_parameter('bno_topic').value,
            self._receive_imu_yaw, 10)

        self.pose_pub = self.create_publisher(
            PoseStamped, self.get_parameter('pose_topic').value, 10)

        self._dt = 1.0 / loop_rate_hz
        self.timer = self.create_timer(self._dt, self._tick_integrate_and_publish)

        self.get_logger().info(
            f"dynamics_estimator_node started (geometry source: {geom.GEOMETRY_SOURCE}, "
            f"coefficients: {'fitted' if self._fitted_coeffs_available() else 'prior'})")

    def _fitted_coeffs_available(self):
        """Returns True when Node A has already written at least one set of
        fitted coefficients to disk, so they can be loaded instead of the prior."""
        import os
        return os.path.exists(core.FITTED_CONFIG_PATH)

    def _receive_pwm_command(self, msg: Float64MultiArray):
        """Stores the latest 8-thruster PWM command vector from /debug/final_pwm."""
        self._latest_pwm = list(msg.data[:8])
        self._have_pwm = True

    def _receive_imu_yaw(self, msg: Float64MultiArray):
        # Same field convention thruster_node.py already reads yaw from
        # (bno[2] == yaw, degrees) — kept in radians internally here.
        if len(msg.data) >= 3:
            self._latest_yaw_rad = math.radians(msg.data[2])
            self._have_yaw = True

    def _tick_integrate_and_publish(self):
        """100 Hz timer tick: converts the latest PWM snapshot to per-thruster
        forces, maps through the TAM to a body wrench, advances the RK4
        integrator one step, and publishes the updated dead-reckoning pose."""
        if not (self._have_pwm and self._have_yaw):
            return

        thruster_forces = [
            self.pwm_table.pwm_to_thrust(
                pwm,
                self.coeffs.pwm_scale_fwd, self.coeffs.pwm_bias_fwd,
                self.coeffs.pwm_scale_rev, self.coeffs.pwm_bias_rev,
            )
            for pwm in self._latest_pwm
        ]
        body_wrench = self.tam @ thruster_forces  # [force_x, force_y, moment_z]

        pos_x, pos_y, body_vel = self.integrator.advance_and_integrate(body_wrench, self._latest_yaw_rad, self._dt)

        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'pool'
        msg.pose.position.x = float(pos_x)
        msg.pose.position.y = float(pos_y)
        msg.pose.orientation.z = math.sin(self._latest_yaw_rad / 2.0)
        msg.pose.orientation.w = math.cos(self._latest_yaw_rad / 2.0)
        self.pose_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = DynamicsEstimatorNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
