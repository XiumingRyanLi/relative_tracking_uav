#!/usr/bin/env python3
import math
from dataclasses import dataclass
from collections import deque

# Camera-to-car distance DOPE needs to see the whole car in the 46 x 26 deg
# sim camera (straight down, the 4.4 m car spans the 26 deg direction: >= 9.4 m).
MIN_TARGET_RANGE = 10.0
CAR_CENTRE_Z = 0.6          # DOPE box centre above the ground (m)

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

    def __init__(self, transition_speed: float = 5.0, transition_min_sec: float = 2.0):
        # A new sequence first flies from the current shot point to its start
        # point (see set_sequence) at this speed relative to the car.
        self.transition_speed = transition_speed
        self.transition_min_sec = transition_min_sec
        self._has_flown = False
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
        """Replace the queue. Once a shot has been flown, a transition to the
        new sequence's start point goes first: jumping there made the drone
        cut straight past (or at) the car, closer than DOPE can see it."""
        actions = list(actions)
        start = self._offset(actions[0], 0.0) if actions and self._has_flown else None
        self.clear()
        if start is not None:
            transition = self._make_transition(self.previous_offset, start)
            if transition is not None:
                self.add_action(transition)
        for action in actions:
            self.add_action(action)

    def _make_transition(self, frm: CinematicOffset, to: CinematicOffset):
        """Action flying around the car from `frm` to `to`, None if already there."""
        r0, r1 = math.hypot(frm.x, frm.y), math.hypot(to.x, to.y)
        th0, th1 = math.atan2(frm.y, frm.x), math.atan2(to.y, to.x)
        dth = abs(math.atan2(math.sin(th1 - th0), math.cos(th1 - th0)))
        length = 0.5 * (r0 + r1) * dth + abs(r1 - r0) + abs(to.z - frm.z)
        if length < 0.5:
            return None
        return {
            "type": "transition",
            "from": (frm.x, frm.y, frm.z),
            "to": (to.x, to.y, to.z),
            "duration": max(self.transition_min_sec, length / max(self.transition_speed, 0.1)),
        }

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

        offset = self._offset(action, s)
        self._has_flown = True

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
    


    def _offset(self, action: dict, s: float) -> CinematicOffset:
        """Shot offset of `action` at smoothed progress s (0..1)."""
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
        elif action_type == "transition":
            offset = self._transition(action, s)
        else:
            offset = self._hold_location(action)

        # --------------------------- change offset depending on the camera model view
        # (Overpass and transition climb rather than step outwards, which
        # keeps their paths continuous.)
        return self._enforce_min_range(
            offset, overhead_ok=action_type in ("overpass", "transition")
        )

    def _transition(self, action: dict, s: float) -> CinematicOffset:
        """Around the car, the short way: bearing, radius and height blend
        from `from` to `to`, so the drone keeps its distance instead of
        crossing the car."""
        x0, y0, z0 = action["from"]
        x1, y1, z1 = action["to"]
        r0, r1 = math.hypot(x0, y0), math.hypot(x1, y1)
        th1 = math.atan2(y1, x1)
        th0 = math.atan2(y0, x0) if r0 > 0.5 else th1   # overhead: no bearing yet
        if r1 <= 0.5:
            th1 = th0
        dth = math.atan2(math.sin(th1 - th0), math.cos(th1 - th0))
        theta = th0 + dth * s
        r = self.lerp(r0, r1, s)
        return CinematicOffset(r * math.cos(theta), r * math.sin(theta), self.lerp(z0, z1, s))

    def _hold_location(self, action: dict) -> CinematicOffset:
        location = action.get("location", "back")
        radius = float(action.get("radius", self.default_radius))
        height = float(action.get("height", self.default_height))

        x, y, z = self._location_to_offset(location, radius, height)
        return CinematicOffset(x, y, z)

    def _move_location(self, action: dict, s: float) -> CinematicOffset:
        """Arc round the car at a constant radius and height from `from` to
        `to`, on the side that passes `via` (a straight line between e.g.
        back and left cut the corner to 0.7 x radius from the car)."""
        radius = float(action.get("radius", self.default_radius))
        height = float(action.get("height", self.default_height))

        th0 = self._bearing(action.get("from", "back"))
        th1 = self._bearing(action.get("to", "front"))
        thv = self._bearing(action.get("via", "left"))

        two_pi = 2.0 * math.pi
        ccw = (th1 - th0) % two_pi          # counter-clockwise sweep, 0..2pi
        via_ccw = (thv - th0) % two_pi
        eps = 1e-6
        if ccw < eps:
            sweep = 0.0                                  # from == to
        elif eps < via_ccw < ccw - eps:
            sweep = ccw                                  # via on the ccw side
        elif via_ccw > ccw + eps:
            sweep = ccw - two_pi                         # via on the cw side
        else:
            sweep = ccw if ccw <= math.pi else ccw - two_pi   # via is an end: short way

        theta = th0 + sweep * s
        return CinematicOffset(radius * math.cos(theta), radius * math.sin(theta), height)

    def _bearing(self, location: str) -> float:
        ux, uy = self.LOCATIONS.get(location, self.LOCATIONS["back"])
        return math.atan2(uy, ux)

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

    

    def _enforce_min_range(self, offset: CinematicOffset, overhead_ok: bool) -> CinematicOffset:
        dz = offset.z - CAR_CENTRE_Z
        horiz = math.hypot(offset.x, offset.y)
        if math.hypot(horiz, dz) >= MIN_TARGET_RANGE:
            return offset
        if overhead_ok or horiz < 0.5:
            # Overpass (or directly above): only climbing keeps the path continuous.
            offset.z = CAR_CENTRE_Z + math.sqrt(MIN_TARGET_RANGE ** 2 - horiz ** 2)
        else:
            # Everything else: move straight out along the same bearing, keep the height.
            need = math.sqrt(max(MIN_TARGET_RANGE ** 2 - dz ** 2, 0.0))
            offset.x *= need / horiz
            offset.y *= need / horiz
        return offset