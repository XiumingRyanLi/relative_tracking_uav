#!/usr/bin/env python3
"""Flight logistics: GUIDED -> arm -> takeoff -> hover -> tracking, plus safety RTL.

Owns the MAVROS mode/arming/takeoff service clients. The node feeds it FCU
state and altitude and calls tick() / check_safety() from its timers; the
control loops only look at `tracking_enabled` and `rtl_initiated`.
"""
from typing import Callable, Optional

from mavros_msgs.msg import State
from mavros_msgs.srv import CommandBool, CommandTOL, SetMode


class FlightSequencer:
    ARM_TO_TAKEOFF_DELAY_SEC = 5.0
    MAX_FLIGHT_TIME_SEC = 1200.0

    def __init__(
        self,
        node,
        takeoff_altitude: float,
        hover_delay_sec: float,
        boundary_limit: float,
        on_tracking_enabled: Optional[Callable[[], None]] = None,
    ):
        self._node = node
        self._log = node.get_logger()
        self.takeoff_altitude = takeoff_altitude
        self.hover_delay_sec = hover_delay_sec
        self.boundary_limit = boundary_limit
        self._on_tracking_enabled = on_tracking_enabled

        self.set_mode_client = node.create_client(SetMode, "/mavros/set_mode")
        self.arming_client = node.create_client(CommandBool, "/mavros/cmd/arming")
        self.takeoff_client = node.create_client(CommandTOL, "/mavros/cmd/takeoff")

        self.fcu_state = State()
        self._guided_requested = False
        self._guided_confirmed = False
        self._arm_requested = False
        self._armed_confirmed = False
        self._tko_requested = False
        self._tko_reached = False
        self.tracking_enabled = False
        self.rtl_initiated = False
        self._armed_time = None
        self._takeoff_complete_time = None
        self._armed_wait_logged = False

    # ---- inputs ----
    def on_state(self, msg: State, now: float):
        self.fcu_state = msg
        if msg.mode == "GUIDED" and not self._guided_confirmed:
            self._guided_confirmed = True
            self._log.info("GUIDED confirmed.")
        if msg.armed and not self._armed_confirmed:
            self._armed_confirmed = True
            self._armed_time = now
            self._log.info("Armed confirmed.")

    def on_altitude(self, alt: float, now: float):
        if self._armed_confirmed and not self._tko_reached and alt >= self.takeoff_altitude - 0.5:
            self._tko_reached = True
            self._takeoff_complete_time = now
            self._log.info(f"Takeoff complete at {alt:.2f} m.")

    # ---- sequencing (orchestrator timer) ----
    def tick(self, now: float, have_drone_pose: bool):
        if not self.fcu_state.connected:
            return

        if not self._guided_confirmed:
            if not self._guided_requested:
                self._request_guided()
            return

        if not have_drone_pose:
            return

        if not self._armed_confirmed:
            if not self._arm_requested:
                self._request_arm()
            return

        if self._armed_time is None:
            return

        if not self._tko_requested:
            if now - self._armed_time < self.ARM_TO_TAKEOFF_DELAY_SEC:
                if not self._armed_wait_logged:
                    self._log.info(
                        f"Armed — waiting {self.ARM_TO_TAKEOFF_DELAY_SEC:.0f} s before takeoff..."
                    )
                    self._armed_wait_logged = True
                return
            self._request_takeoff()
            return

        if self._tko_reached and not self.tracking_enabled and self._takeoff_complete_time is not None:
            if now - self._takeoff_complete_time >= self.hover_delay_sec:
                if self._on_tracking_enabled is not None:
                    self._on_tracking_enabled()
                self.tracking_enabled = True
                self._log.info("Tracking enabled.")

    def _request_guided(self):
        if not self.set_mode_client.wait_for_service(timeout_sec=0.5):
            self._log.warning("SetMode service not ready.")
            return
        self._guided_requested = True
        req = SetMode.Request()
        req.custom_mode = "GUIDED"
        self.set_mode_client.call_async(req).add_done_callback(self._on_set_mode_done)
        self._log.info("Requesting GUIDED...")

    def _on_set_mode_done(self, fut):
        try:
            res = fut.result()
        except Exception as e:
            self._log.error(f"SetMode error: {e}")
            self._guided_requested = False
            return
        if not getattr(res, "mode_sent", False):
            self._log.error("GUIDED rejected by FCU.")
            self._guided_requested = False

    def _request_arm(self):
        if not self.arming_client.wait_for_service(timeout_sec=0.5):
            self._log.warning("Arming service not ready.")
            return
        self._arm_requested = True
        req = CommandBool.Request()
        req.value = True
        self.arming_client.call_async(req).add_done_callback(self._on_arm_done)
        self._log.info("Requesting ARM...")

    def _on_arm_done(self, fut):
        try:
            res = fut.result()
        except Exception as e:
            self._log.error(f"Arm error: {e}")
            self._arm_requested = False
            return
        if not getattr(res, "success", False):
            self._log.error("Arm rejected by FCU.")
            self._arm_requested = False

    def _request_takeoff(self):
        if not self.takeoff_client.wait_for_service(timeout_sec=0.5):
            self._log.warning("Takeoff service not ready.")
            return
        self._tko_requested = True
        req = CommandTOL.Request()
        req.altitude = float(self.takeoff_altitude)
        self.takeoff_client.call_async(req).add_done_callback(self._on_takeoff_done)
        self._log.info(f"Requesting takeoff to {self.takeoff_altitude:.1f} m...")

    def _on_takeoff_done(self, fut):
        try:
            res = fut.result()
        except Exception as e:
            self._log.error(f"Takeoff error: {e}")
            self._tko_requested = False
            return
        if not getattr(res, "success", False):
            self._log.error("Takeoff rejected by FCU.")
            self._tko_requested = False

    # ---- safety (safety timer) ----
    def check_safety(self, now: float, x: float, y: float):
        if self.rtl_initiated or not self._tko_reached or self._takeoff_complete_time is None:
            return
        if now - self._takeoff_complete_time >= self.MAX_FLIGHT_TIME_SEC:
            self.initiate_rtl(f"{self.MAX_FLIGHT_TIME_SEC:.0f} s timer expired")
            return
        if abs(x) > self.boundary_limit or abs(y) > self.boundary_limit:
            self.initiate_rtl(f"boundary violation ({x:.1f}, {y:.1f})")

    def initiate_rtl(self, reason: str):
        if self.rtl_initiated:
            return
        self.rtl_initiated = True
        self.tracking_enabled = False
        self._log.warning(f"SAFETY RTL: {reason}")

        if not self.set_mode_client.wait_for_service(timeout_sec=1.0):
            self._log.error("SetMode service not ready for RTL.")
            return
        req = SetMode.Request()
        req.custom_mode = "RTL"
        self.set_mode_client.call_async(req).add_done_callback(
            lambda f: self._on_rtl_done(f, reason)
        )

    def _on_rtl_done(self, fut, reason):
        try:
            res = fut.result()
        except Exception as e:
            self._log.error(f"RTL error: {e}")
            return
        if getattr(res, "mode_sent", False):
            self._log.info(f"RTL accepted ({reason})")
        else:
            self._log.error(f"RTL rejected ({reason})")
