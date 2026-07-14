#!/usr/bin/env python3
import math
from dataclasses import dataclass
from collections import deque


@dataclass
class CinematicOffset:
    x: float
    y: float
    z: float
    finished: bool = False


class CinematicPlanner:
    """
    Generates target-relative offsets for cinematic UAV shots.

    Target-relative convention:
        +x = front of target
        -x = back of target
        +y = left of target
        -y = right of target
        +z = above target
    """

    LOCATIONS = {
        "front":       ( 1.0,  0.0),
        "back":        (-1.0,  0.0),
        "left":        ( 0.0,  1.0),
        "right":       ( 0.0, -1.0),
        "front_left":  ( 0.707,  0.707),
        "front_right": ( 0.707, -0.707),
        "back_left":   (-0.707,  0.707),
        "back_right":  (-0.707, -0.707),
    }

    def __init__(self):
        self.queue = deque()
        self.current_action = None
        self.action_start_time = None

        # Default fallback shot.
        self.default_radius = 3.0
        self.default_height = 3.0
        self.default_location = "back"

        # Store the last commanded cinematic offset.
        # When the action queue finishes, hold this position.
        self.previous_offset = CinematicOffset(
            x=-self.default_radius,
            y=0.0,
            z=self.default_height,
            finished=True,
        )
                

    def clear(self):
        self.queue.clear()
        self.current_action = None
        self.action_start_time = None

    def add_action(self, action: dict):
        self.queue.append(action)

    def set_sequence(self, actions):
        self.clear()
        for action in actions:
            self.add_action(action)

    @staticmethod
    def clamp01(u: float) -> float:
        return max(0.0, min(1.0, u))

    @staticmethod
    def smoothstep(u: float) -> float:
        u = max(0.0, min(1.0, u))
        return u * u * (3.0 - 2.0 * u)

    @staticmethod
    def lerp(a: float, b: float, u: float) -> float:
        return a + (b - a) * u

    def _location_to_offset(self, location: str, radius: float, height: float):
        ux, uy = self.LOCATIONS.get(location, self.LOCATIONS["back"])
        return ux * radius, uy * radius, height

    def _start_next_action(self, now: float):
        if not self.queue:
            self.current_action = None
            self.action_start_time = None
            return

        self.current_action = self.queue.popleft()
        self.action_start_time = now

    def update(self, now: float) -> CinematicOffset:
        if self.current_action is None:
            self._start_next_action(now)

        # If there is no current action and no queued action,
        # hold the previous cinematic position.
        if self.current_action is None:
            return CinematicOffset(
                x=self.previous_offset.x,
                y=self.previous_offset.y,
                z=self.previous_offset.z,
                finished=True,
            )

        action = self.current_action
        duration = max(float(action.get("duration", 1.0)), 1e-3)
        elapsed = now - self.action_start_time

        u = elapsed / duration
        u = max(0.0, min(1.0, u))

        # Optional smoothing for cinematic motion
        s = u * u * (3.0 - 2.0 * u)

        action_type = action.get("type", "hold_location")

        if action_type == "hold_location":
            offset = self._hold_location(action)

        elif action_type == "move_location":
            offset = self._move_location(action, s)

        elif action_type == "orbit":
            offset = self._orbit(action, s)

        elif action_type == "overpass":
            offset = self._overpass(action, s)

        elif action_type == "push_in":
            offset = self._push_pull(action, s, push=True)

        elif action_type == "pull_out":
            offset = self._push_pull(action, s, push=False)

        else:
            offset = self._hold_location(action)

        # If the current action has finished, mark this offset finished
        # and start the next action next cycle.
        if u >= 1.0:
            offset.finished = True

            # Store the final position before moving to the next action.
            self.previous_offset = CinematicOffset(
                x=offset.x,
                y=offset.y,
                z=offset.z,
                finished=True,
            )

            self._start_next_action(now)

            return offset

        # Store current offset while action is running.
        self.previous_offset = CinematicOffset(
            x=offset.x,
            y=offset.y,
            z=offset.z,
            finished=False,
        )

        return offset
    


    def _hold_location(self, action: dict) -> CinematicOffset:
        location = action.get("location", "back")
        radius = float(action.get("radius", self.default_radius))
        height = float(action.get("height", self.default_height))

        x, y, z = self._location_to_offset(location, radius, height)
        return CinematicOffset(x, y, z)

    def _move_location(self, action: dict, s: float) -> CinematicOffset:
        start_location = action.get("from", "back")
        end_location = action.get("to", "front")
        via = action.get("via", "left")

        radius = float(action.get("radius", self.default_radius))
        height = float(action.get("height", self.default_height))

        if start_location == "back" and end_location == "front":
            side = 1.0 if via == "left" else -1.0

            theta = math.pi * (1.0 - s)

            x = radius * math.cos(theta)
            y = side * radius * math.sin(theta)
            z = height

            return CinematicOffset(x, y, z)

        if start_location == "front" and end_location == "back":
            side = 1.0 if via == "left" else -1.0

            theta = math.pi * s

            x = radius * math.cos(theta)
            y = side * radius * math.sin(theta)
            z = height

            return CinematicOffset(x, y, z)

        x0, y0, z0 = self._location_to_offset(start_location, radius, height)
        x1, y1, z1 = self._location_to_offset(end_location, radius, height)

        x = self.lerp(x0, x1, s)
        y = self.lerp(y0, y1, s)
        z = self.lerp(z0, z1, s)

        return CinematicOffset(x, y, z)

    def _orbit(self, action: dict, s: float) -> CinematicOffset:
        radius = float(action.get("radius", self.default_radius))
        height = float(action.get("height", self.default_height))

        start_location = action.get("start", "back")
        angle_deg = float(action.get("angle_deg", 180.0))
        direction = action.get("direction", "clockwise")

        ux, uy = self.LOCATIONS.get(start_location, self.LOCATIONS["back"])
        start_theta = math.atan2(uy, ux)

        sign = -1.0 if direction == "clockwise" else 1.0
        theta = start_theta + sign * math.radians(angle_deg) * s

        x = radius * math.cos(theta)
        y = radius * math.sin(theta)
        z = height

        return CinematicOffset(x, y, z)

    def _overpass(self, action: dict, s: float) -> CinematicOffset:
        start_location = action.get("from", "back")
        end_location = action.get("to", "front")

        radius = float(action.get("radius", self.default_radius))
        start_height = float(action.get("start_height", self.default_height))
        peak_height = float(action.get("peak_height", self.default_height + 3.0))
        end_height = float(action.get("end_height", self.default_height))

        x0, y0, _ = self._location_to_offset(start_location, radius, start_height)
        x1, y1, _ = self._location_to_offset(end_location, radius, end_height)

        x = self.lerp(x0, x1, s)
        y = self.lerp(y0, y1, s)

        # Parabolic height profile: start -> peak -> end
        base_z = self.lerp(start_height, end_height, s)
        arc = 4.0 * s * (1.0 - s)
        z = base_z + arc * (peak_height - max(start_height, end_height))

        return CinematicOffset(x, y, z)

    def _push_pull(self, action: dict, s: float, push: bool) -> CinematicOffset:
        location = action.get("location", "back")

        height = float(action.get("height", self.default_height))
        near_radius = float(action.get("near_radius", 3.0))
        far_radius = float(action.get("far_radius", 8.0))

        if push:
            radius = self.lerp(far_radius, near_radius, s)
        else:
            radius = self.lerp(near_radius, far_radius, s)

        x, y, z = self._location_to_offset(location, radius, height)
        return CinematicOffset(x, y, z)