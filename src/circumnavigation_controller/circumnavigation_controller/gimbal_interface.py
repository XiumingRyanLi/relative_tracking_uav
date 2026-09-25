#!/usr/bin/env python3
"""MAVROS gimbal plumbing: claim the gimbal manager, send pitch/yaw commands,
keep the attitude feedback (plus a short history for capture-time lookup).

What to point the gimbal at is decided by the node's control loop
(GimbalController); this class only talks to MAVROS.
"""
import math

import numpy as np
import tf_transformations
from mavros_msgs.srv import GimbalManagerConfigure, GimbalManagerPitchyaw

try:
    from .camera_frames import TimedHistory
except ImportError:
    from camera_frames import TimedHistory


class GimbalInterface:
    def __init__(
        self,
        node,
        device_id: int,
        command_period_sec: float,
        initial_pitch_rad: float,
        history_sec: float,
    ):
        self._log = node.get_logger()
        self.device_id = device_id
        self.command_period_sec = command_period_sec

        self.config_client = node.create_client(
            GimbalManagerConfigure, "/mavros/gimbal_control/manager/configure"
        )
        self.pitchyaw_client = node.create_client(
            GimbalManagerPitchyaw, "/mavros/gimbal_control/manager/pitchyaw"
        )
        self._configured = False
        self._config_in_progress = False
        self._last_cmd_time = 0.0

        # Last commanded angles (rad). The image PD builds on these. The
        # gimbal boots at initial_pitch_rad (ArduPilot Neutral mode, see
        # config/gimbal_startup.parm).
        self.current_pitch = initial_pitch_rad
        self.current_yaw = 0.0

        # Attitude feedback: base_link FRD -> gimbal, as reported by the FCU.
        self.have_attitude = False
        self.T_frd_gimbal = np.eye(4)
        self.flags = 0
        self.history = TimedHistory(history_sec)

    # ---- commands ----
    def send(self, pitch_rad: float, yaw_rad: float, now: float):
        """Send a pitch/yaw command, at most once per command_period_sec.
        The first calls claim the gimbal manager instead."""
        if not self._configured:
            self._configure()
            return

        # Limit service calls to avoid spamming COMMAND_LONG.
        if now - self._last_cmd_time < self.command_period_sec:
            return

        if not self.pitchyaw_client.wait_for_service(timeout_sec=0.1):
            self._log.warning("Gimbal pitchyaw service not ready.", throttle_duration_sec=2.0)
            return

        req = GimbalManagerPitchyaw.Request()
        req.pitch = float(math.degrees(pitch_rad))
        req.yaw = float(math.degrees(yaw_rad))
        req.pitch_rate = 0.0
        req.yaw_rate = 0.0
        req.flags = 0
        req.gimbal_device_id = self.device_id

        self.current_pitch = pitch_rad
        self.current_yaw = yaw_rad
        self._last_cmd_time = now
        self.pitchyaw_client.call_async(req)

    def _configure(self):
        if self._configured or self._config_in_progress:
            return
        if not self.config_client.wait_for_service(timeout_sec=0.1):
            self._log.warning("Gimbal configure service not ready.", throttle_duration_sec=2.0)
            return

        req = GimbalManagerConfigure.Request()
        req.sysid_primary = -2      # take primary control
        req.compid_primary = -2
        req.sysid_secondary = 0
        req.compid_secondary = 0
        req.gimbal_device_id = self.device_id

        self._config_in_progress = True
        self.config_client.call_async(req).add_done_callback(self._on_config_done)

    def _on_config_done(self, fut):
        self._config_in_progress = False
        try:
            res = fut.result()
        except Exception as e:
            self._log.error(f"Gimbal configure failed: {e}")
            return
        if res.success:
            self._configured = True
            self._log.info("Gimbal manager configured.")
        else:
            self._log.warning(f"Gimbal configure rejected, result={res.result}")

    # ---- feedback ----
    def on_attitude_status(self, msg, t_sec: float):
        q = msg.q
        self.T_frd_gimbal = tf_transformations.quaternion_matrix([q.x, q.y, q.z, q.w])
        self.flags = msg.flags
        self.have_attitude = True
        self.history.append(t_sec, self.T_frd_gimbal.copy())

    def attitude_rpy_deg(self):
        """Reported base_link FRD -> gimbal attitude as roll/pitch/yaw deg."""
        if not self.have_attitude:
            return None, None, None
        return tuple(
            math.degrees(a)
            for a in tf_transformations.euler_from_matrix(self.T_frd_gimbal, axes="sxyz")
        )
