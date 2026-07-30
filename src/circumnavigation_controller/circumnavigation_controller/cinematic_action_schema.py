#!/usr/bin/env python3
"""
cinematic_action_schema.py

Single source of truth for what a cinematic action looks like: which
fields each action type takes, what kind each field is (a named
location / an enum / a bounded float), and what its default and valid
range are.

Both relative_position_controller.py (validation before it ever reaches
CinematicPlanner) and cinematic_gui.py (building the input form) import
from here, so the schema only has to be edited in one place.
"""
try:
    from .cinematic_planner import CinematicPlanner
except ImportError:
    from cinematic_planner import CinematicPlanner

LOCATIONS = set(CinematicPlanner.LOCATIONS.keys())

# Per-field definition. "kind" drives both validation (controller) and
# widget choice (GUI):
#   location -> must be one of LOCATIONS
#   enum     -> must be one of "options"
#   float    -> cast to float and clamped to "bounds"
# "default" is used whenever a field is missing/invalid; the controller
# may override it per-message with a planner-derived default (see
# dynamic_defaults in validate_action).
FIELD_SPECS = {
    "location":     {"kind": "location", "default": "back"},
    "from":         {"kind": "location", "default": "back"},
    "to":           {"kind": "location", "default": "front"},
    "start":        {"kind": "location", "default": "back"},
    "via":          {"kind": "enum", "options": ("left", "right"), "default": "left"},
    "direction":    {"kind": "enum", "options": ("clockwise", "counterclockwise"), "default": "clockwise"},
    "radius":       {"kind": "float", "bounds": (0.5, 15.0), "default": 3.0, "unit": " m"},
    "height":       {"kind": "float", "bounds": (0.5, 20.0), "default": 3.0, "unit": " m"},
    "start_height": {"kind": "float", "bounds": (0.5, 20.0), "default": 3.0, "unit": " m"},
    "peak_height":  {"kind": "float", "bounds": (0.5, 20.0), "default": 6.0, "unit": " m"},
    "end_height":   {"kind": "float", "bounds": (0.5, 20.0), "default": 3.0, "unit": " m"},
    "near_radius":  {"kind": "float", "bounds": (0.5, 15.0), "default": 3.0, "unit": " m"},
    "far_radius":   {"kind": "float", "bounds": (0.5, 15.0), "default": 8.0, "unit": " m"},
    "angle_deg":    {"kind": "float", "bounds": (0.0, 360.0), "default": 180.0, "unit": " deg"},
    "duration":     {"kind": "float", "bounds": (0.1, 300.0), "default": 8.0, "unit": " s"},
}

# Which fields each action type takes, in display order, plus a
# type-specific default duration (a 5s hold and a 15s orbit are both
# reasonable defaults; one universal default duration wouldn't be).
ACTION_SCHEMA = {
    "hold_location": {
        "fields": ["location", "radius", "height", "duration"],
        "duration_default": 5.0,
    },
    "move_location": {
        "fields": ["from", "to", "via", "radius", "height", "duration"],
        "duration_default": 8.0,
    },
    "orbit": {
        "fields": ["start", "radius", "height", "angle_deg", "direction", "duration"],
        "duration_default": 10.0,
    },
    "overpass": {
        "fields": ["from", "to", "radius", "start_height", "peak_height", "end_height", "duration"],
        "duration_default": 12.0,
    },
    "push_in": {
        "fields": ["location", "height", "near_radius", "far_radius", "duration"],
        "duration_default": 6.0,
    },
    "pull_out": {
        "fields": ["location", "height", "near_radius", "far_radius", "duration"],
        "duration_default": 6.0,
    },
}

ACTION_TYPES = list(ACTION_SCHEMA.keys())

FIELD_LABELS = {
    "location": "Location", "from": "From", "to": "To", "via": "Via", "start": "Start",
    "radius": "Radius", "height": "Height", "start_height": "Start height",
    "peak_height": "Peak height", "end_height": "End height",
    "near_radius": "Near radius", "far_radius": "Far radius",
    "angle_deg": "Angle", "direction": "Direction", "duration": "Duration",
}

# Canonical row order for the GUI form (superset of every action's fields).
FIELD_ORDER = [
    "location", "from", "to", "via", "start",
    "radius", "height", "start_height", "peak_height", "end_height",
    "near_radius", "far_radius", "angle_deg", "direction", "duration",
]


def validate_action(raw, index, dynamic_defaults=None, warn=None):
    """Validate/clamp one action dict against ACTION_SCHEMA.

    raw: the untrusted dict as received (e.g. from JSON off the GUI topic)
    index: position in the incoming list, for logging only
    dynamic_defaults: optional {field: value} overrides, e.g. the
        controller passes the CinematicPlanner's current default_radius/
        default_height so fields fall back to flight-relevant values
        instead of this module's generic defaults
    warn: optional callable(str) for logging; if None, warnings are dropped

    Returns a cleaned action dict, or None if the action must be skipped
    entirely (unrecognizable type / not a dict). Never raises -- a
    malformed message degrades to "skip this field/action", not a crash.
    """
    dynamic_defaults = dynamic_defaults or {}

    def _warn(message):
        if warn is not None:
            warn(message)

    if not isinstance(raw, dict):
        _warn(f"Cinematic action[{index}]: not an object; skipping.")
        return None

    action_type = raw.get("type")
    schema = ACTION_SCHEMA.get(action_type)
    if schema is None:
        _warn(f"Cinematic action[{index}]: unknown type {action_type!r}; skipping.")
        return None

    action = {"type": action_type}

    for field in schema["fields"]:
        spec = FIELD_SPECS[field]
        default = dynamic_defaults.get(field, spec["default"])
        if field == "duration":
            default = schema.get("duration_default", default)

        value = raw.get(field, default)

        if spec["kind"] in ("location", "enum"):
            valid_set = LOCATIONS if spec["kind"] == "location" else spec["options"]
            if value not in valid_set:
                _warn(f"Cinematic action[{index}]: invalid {field}={value!r}; using {default!r}.")
                value = default
            action[field] = value

        else:  # float
            try:
                value = float(value)
            except (TypeError, ValueError):
                _warn(f"Cinematic action[{index}]: invalid {field}={value!r}; using {default}.")
                value = float(default)
            lo, hi = spec["bounds"]
            clamped = max(lo, min(hi, value))
            if clamped != value:
                _warn(
                    f"Cinematic action[{index}]: {field}={value} out of range "
                    f"[{lo}, {hi}]; clamped to {clamped}."
                )
            action[field] = clamped

    return action
