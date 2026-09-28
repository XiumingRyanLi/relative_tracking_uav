import math
import os

import numpy as np
import pytest

from circumnavigation_controller.race_line import RaceDriver, RaceLine

RACE_LINES = os.path.join(os.path.dirname(__file__), '..', 'config', 'race_lines')


def stadium(straight=100.0, radius=20.0, spacing=1.0):
    """Closed loop: two straights joined by semicircles, counter-clockwise."""
    pts = []
    for y in np.arange(0.0, straight, spacing):
        pts.append((radius, y))
    for a in np.arange(0.0, math.pi, spacing / radius):
        pts.append((radius * math.cos(a), straight + radius * math.sin(a)))
    for y in np.arange(straight, 0.0, -spacing):
        pts.append((-radius, y))
    for a in np.arange(math.pi, 2 * math.pi, spacing / radius):
        pts.append((radius * math.cos(a), radius * math.sin(a)))
    return np.array(pts)


def test_profile_fast_on_straights_slow_in_corners():
    line = RaceLine(stadium(), max_speed=15.0, lat_accel=2.5)
    straight = line.speed[np.abs(line.curvature) < 1e-3]
    corner = line.speed[np.abs(line.curvature) > 0.049]  # inside the arcs
    assert straight.max() == pytest.approx(15.0)
    # Corner speed limit sqrt(lat_accel * R) = sqrt(2.5 * 20).
    assert corner.max() <= math.sqrt(2.5 * 20.0) + 0.1


def test_profile_respects_accel_and_brake():
    accel, brake = 2.0, 4.0
    line = RaceLine(stadium(), accel=accel, brake=brake)
    v, v_next = line.speed, np.roll(line.speed, -1)
    dv2 = (v_next ** 2 - v ** 2) / (2.0 * line.ds)
    assert dv2.max() <= accel + 1e-6
    assert dv2.min() >= -brake - 1e-6


def test_driver_steers_back_to_line():
    line = RaceLine(stadium())
    driver = RaceDriver(line)
    # On the first straight (x = 20, heading +y), 2 m to the right of it:
    # steer left (positive yaw rate).
    cmd = driver.step(22.0, 30.0, math.pi / 2, 10.0)
    assert cmd.yaw_rate > 0.0
    assert cmd.cross_track == pytest.approx(-2.0, abs=0.1)
    assert cmd.speed > 0.0


@pytest.mark.parametrize('world', ['iris_monza', 'iris_silverstone', 'iris_oschersleben'])
def test_shipped_race_lines_start_at_spawn(world):
    line = RaceLine.from_csv(os.path.join(RACE_LINES, world + '.csv'))
    # The car spawns at (0, 20) facing +y in every track world.
    assert np.hypot(*(line.pts[0] - (0.0, 20.0))) < 2.0
    heading = line.pts[3] - line.pts[0]
    assert heading[1] > 0.9 * np.linalg.norm(heading)
    assert 1000.0 < line.length < 2500.0


def test_speed_zone_caps_and_brakes_into_it():
    free = RaceLine(stadium(), brake=4.0)
    zoned = RaceLine(stadium(), brake=4.0, speed_zones=[(30.0, 60.0, 5.0)])
    inside = (zoned.s >= 30.0) & (zoned.s <= 60.0)
    assert zoned.speed[inside].max() <= 5.0 + 1e-9
    # Braking starts before the zone, and the rest of the lap is unchanged.
    assert zoned.speed[(zoned.s > 20.0) & (zoned.s < 30.0)].max() < free.speed.max()
    assert np.allclose(zoned.speed[zoned.s > 150.0], free.speed[free.s > 150.0])


def test_driver_ignores_fishtail():
    # On the line, travelling straight along it (+y), but the nose is swung
    # 5 deg left (fishtail): the slip makes the course straight, so no
    # correction; steering from the nose would turn right.
    line = RaceLine(stadium())
    nose = math.pi / 2 + math.radians(5.0)
    v = 15.0
    v_fwd, v_lat = v * math.cos(math.radians(5.0)), -v * math.sin(math.radians(5.0))
    straight_on = RaceDriver(line).step(20.0, 30.0, nose, v_fwd, v_lat)
    from_nose = RaceDriver(line).step(20.0, 30.0, nose, v_fwd)
    assert abs(straight_on.yaw_rate) < 0.01
    assert from_nose.yaw_rate < -0.05
