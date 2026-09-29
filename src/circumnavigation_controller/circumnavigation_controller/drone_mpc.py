#!/usr/bin/env python3
"""Drone MPC: acados OCP (model and cost written in CasADi), pure Python, no ROS.

Model (world ENU), from command -> response fits on the race logs of
2026-09-28 (horizontal: ~0.8 s delay + ~0.45 s lag; yaw ~0.25 s + ~0.3 s):
    state   x = [px py pz  vx vy vz  psi r  ux uy uz ur]      (NX = 12)
            u* = the velocity / yaw-rate command currently applied
    input   du = [dux duy duz dur]  rate of change of the command (NU = 4)
    p' = v,  v_xy' = (u_xy - v_xy)/tau_xy,  v_z' = (u_z - v_z)/tau_z,
    psi' = r,  r' = (u_r - r)/tau_r,  u' = du
The transport delay is not in the OCP: the caller predicts the drone ahead
by the delay with the commands already sent (propagate_delay) and plans from
there, so the first planned command is the one to send now.

Cost per stage (nonlinear least squares, all references are parameters):
    distance to the car vs the shot radius, and the angle round the car (as a
        chord, r_ref * |unit(drone - car) - unit(shot - car)|), each divided by
        its band half-width (5 m; r_ref * 10 deg around the car, >= 2 m),
        so the cost contours are the "rainbow band" slice around the car,
        scaled by w_pos (drops as the car prediction gets uncertain);
    height error;
    facing the car: sin and 1-cos of (yaw - bearing to the car), no wrap,
        scaled by w_yaw (0 when the shot point is nearly over the car);
    command changes du (smooth commands).
Constraints: |u_xy| <= v_max (hard), |u_z|, |u_r| boxes, |du| boxes
(acceleration limits), and range to the car >= r_min (soft: DOPE can't see
the whole car closer than ~10 m) -- against the predicted car, and over the
first part of the plan also against a car that brakes hard / accelerates
from now (P_BRK_*, P_ACC_*; inactive where P_R_MIN_SCEN = 0). The filter
only sees braking ~1 s late (0.5 m of deviation in 0.5 s is inside DOPE's
noise), so the plan has to be safe for it before it is seen: run
20260929_094556 overran a car braking into a hairpin to 4.4 m and lost it.

Parameters per stage p (NP = 21): see P_* indices. Weights and bounds are
set at runtime, so only N / horizon / structure changes regenerate code.
"""
import ctypes
import hashlib
import json
import math
import os
import sys
import time

import numpy as np

# ---- state / input / parameter layout ----
NX, NU = 12, 4
PX, PY, PZ, VX, VY, VZ, PSI, R, UX, UY, UZ, UR = range(NX)
(P_REF_X, P_REF_Y, P_REF_Z, P_CAR_X, P_CAR_Y, P_CAR_Z, P_DIR_X, P_DIR_Y,
 P_S_RAD, P_S_TAN, P_W_POS, P_W_YAW, P_R_MIN, P_TAU_XY, P_TAU_Z, P_TAU_R,
 P_BRK_X, P_BRK_Y, P_ACC_X, P_ACC_Y, P_R_MIN_SCEN) = range(21)
NP = 21
NY, NY_E = 10, 6
FORMULATION_VERSION = 5      # bump when the OCP structure below changes


def _acados_env():
    """Make acados importable/loadable without installing into system Python:
    ACADOS_SOURCE_DIR (default ~/acados), its Python interface on sys.path,
    and its shared libraries preloaded (LD_LIBRARY_PATH is read only at
    process start, so setting it here would be too late)."""
    root = os.environ.setdefault("ACADOS_SOURCE_DIR", os.path.expanduser("~/acados"))
    iface = os.path.join(root, "interfaces", "acados_template")
    if iface not in sys.path:
        sys.path.insert(0, iface)
    for lib in ("libblasfeo.so", "libhpipm.so", "libqpOASES_e.so", "libacados.so"):
        path = os.path.join(root, "lib", lib)
        if os.path.exists(path):
            ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
    return root


class DroneMPC:
    def __init__(
        self,
        horizon_sec: float = 2.0,
        n_steps: int = 20,
        build_dir: str = os.path.expanduser("~/.ros/drone_mpc"),
        w_radial: float = 90.0,
        w_tangential: float = 10.0,
        w_height: float = 5.0,
        w_yaw: float = 10.0,
        yaw_band_deg: float = 10.0,
        r_du_xy: float = 0.1,
        r_du_z: float = 1.0,
        r_du_yaw: float = 2.0,
        terminal_factor: float = 2.0,
        v_max_xy: float = 15.0,
        v_max_z: float = 3.0,
        yaw_rate_max: float = 1.5,
        accel_max_xy: float = 5.0,
        accel_max_z: float = 2.0,
        yaw_accel_max: float = 3.0,
        range_slack_linear: float = 50.0,
        range_slack_quadratic: float = 50.0,
        max_range: float = 45.0,
        min_altitude: float = 2.0,
        force_rebuild: bool = False,
        verbose: bool = False,
    ):
        _acados_env()
        from acados_template import AcadosOcp, AcadosOcpSolver  # noqa: E402 (needs _acados_env)

        self.N = int(n_steps)
        self.T = float(horizon_sec)
        self.dt = self.T / self.N
        ocp = self._build_ocp(AcadosOcp)

        # Generate + compile only when the structure changed (or first run).
        os.makedirs(build_dir, exist_ok=True)
        key = hashlib.sha1(json.dumps([FORMULATION_VERSION, self.N, self.T]).encode()).hexdigest()[:12]
        work = os.path.join(build_dir, key)
        os.makedirs(work, exist_ok=True)
        ocp.code_export_directory = os.path.join(work, "c_generated_code")
        stamp = os.path.join(work, "built.ok")
        fresh = force_rebuild or not os.path.exists(stamp)
        cwd = os.getcwd()
        os.chdir(work)                      # acados writes some files relative to cwd
        try:
            t0 = time.monotonic()
            self.solver = AcadosOcpSolver(ocp, json_file="drone_mpc.json", generate=fresh, build=fresh,
                                          verbose=verbose)
            self.build_seconds = time.monotonic() - t0 if fresh else 0.0
        finally:
            os.chdir(cwd)
        if fresh:
            open(stamp, "w").write(key)
        self.work_dir = work

        # ---- runtime weights and bounds (no regeneration needed) ----
        self.set_weights(w_radial, w_tangential, w_height, w_yaw, yaw_band_deg,
                         r_du_xy, r_du_z, r_du_yaw, terminal_factor)
        self.set_limits(v_max_xy, v_max_z, yaw_rate_max, accel_max_xy, accel_max_z, yaw_accel_max,
                        range_slack_linear, range_slack_quadratic)
        self.set_max_range(max_range)
        self.set_min_altitude(min_altitude)
        self._initialized = False

    # ------------------------------------------------------------------
    def _build_ocp(self, AcadosOcp):
        import casadi as ca

        x = ca.SX.sym("x", NX)
        du = ca.SX.sym("du", NU)
        p = ca.SX.sym("p", NP)
        xdot = ca.SX.sym("xdot", NX)

        v = x[VX:VZ + 1]
        f = ca.vertcat(
            v,
            (x[UX] - x[VX]) / p[P_TAU_XY],
            (x[UY] - x[VY]) / p[P_TAU_XY],
            (x[UZ] - x[VZ]) / p[P_TAU_Z],
            x[R],
            (x[UR] - x[R]) / p[P_TAU_R],
            du,
        )

        # Position error split into DISTANCE to the car and ANGLE round it,
        # exactly (not projected on the shot's radial direction, which only
        # measures distance while the angle error is small: with a large
        # angle error a high radial weight then did not hold the distance).
        #   e_rad = |drone - car| - r_ref          (horizontal distance error)
        #   e_tan = r_ref * (unit(drone - car) - dir_ref), a 2-vector: its
        #           length is the chord 2 r_ref sin(dtheta / 2) -- ~ the arc
        #           for small angles, growing all the way to 180 deg, fading
        #           with r_ref for overhead shots.
        dcx, dcy = x[PX] - p[P_CAR_X], x[PY] - p[P_CAR_Y]
        dist_c = ca.sqrt(dcx * dcx + dcy * dcy + 1.0)
        r_ref = ca.sqrt((p[P_REF_X] - p[P_CAR_X]) ** 2 + (p[P_REF_Y] - p[P_CAR_Y]) ** 2)
        e_rad = dist_c - ca.sqrt(r_ref * r_ref + 1.0)
        e_tan_x = r_ref * (dcx / dist_c - p[P_DIR_X])
        e_tan_y = r_ref * (dcy / dist_c - p[P_DIR_Y])
        sw = ca.sqrt(p[P_W_POS])
        # Facing the car: sin / (1 - cos) of yaw - bearing, via the unit
        # vector to the car (no atan2, no wrap; +1 m^2 keeps it finite overhead).
        dx, dy = p[P_CAR_X] - x[PX], p[P_CAR_Y] - x[PY]
        dist = ca.sqrt(dx * dx + dy * dy + 1.0)
        s_e = (ca.sin(x[PSI]) * dx - ca.cos(x[PSI]) * dy) / dist
        c_e = (ca.cos(x[PSI]) * dx + ca.sin(x[PSI]) * dy) / dist
        swy = ca.sqrt(p[P_W_YAW])
        y_track = ca.vertcat(
            sw * e_rad / p[P_S_RAD],
            sw * e_tan_x / p[P_S_TAN],
            sw * e_tan_y / p[P_S_TAN],
            sw * (x[PZ] - p[P_REF_Z]),
            swy * s_e,
            swy * (1.0 - c_e),
        )

        ocp = AcadosOcp()
        m = ocp.model
        m.name = "drone_mpc"
        m.x, m.u, m.p, m.xdot = x, du, p, xdot
        m.f_expl_expr = f
        m.f_impl_expr = xdot - f
        m.cost_y_expr = ca.vertcat(y_track, du)
        m.cost_y_expr_e = y_track

        ocp.cost.cost_type = "NONLINEAR_LS"
        ocp.cost.cost_type_e = "NONLINEAR_LS"
        ocp.cost.W = np.eye(NY)
        ocp.cost.W_e = np.eye(NY_E)
        ocp.cost.yref = np.zeros(NY)
        ocp.cost.yref_e = np.zeros(NY_E)

        # Hard: horizontal command speed (circle). Soft: range to the car.
        speed2 = x[UX] ** 2 + x[UY] ** 2

        def range_to(cx, cy):
            return ca.sqrt((x[PX] - cx) ** 2 + (x[PY] - cy) ** 2 + (x[PZ] - p[P_CAR_Z]) ** 2 + 1e-6)

        rng = range_to(p[P_CAR_X], p[P_CAR_Y]) - p[P_R_MIN]
        rng_brk = range_to(p[P_BRK_X], p[P_BRK_Y]) - p[P_R_MIN_SCEN]
        rng_acc = range_to(p[P_ACC_X], p[P_ACC_Y]) - p[P_R_MIN_SCEN]
        # Maximum range is HORIZONTAL: with the 3D range, flying lower was a
        # way to "get closer" and the MPC dived toward the ground once the
        # car ran away (run 20260929_141000: 4 m -> 0.5 m). The minimum
        # range stays 3D (climbing over the car is legitimate).
        rng_h = ca.sqrt((x[PX] - p[P_CAR_X]) ** 2 + (x[PY] - p[P_CAR_Y]) ** 2 + 1e-6)
        # h rows: 0 command speed^2 (hard) | 1 range - r_min | 2, 3 scenario
        # ranges | 4 horizontal range (<= r_max) | 5 altitude (>= floor);
        # rows 1-5 soft. Bounds / penalties are set at runtime.
        m.con_h_expr = ca.vertcat(speed2, rng, rng_brk, rng_acc, rng_h, x[PZ])
        m.con_h_expr_e = ca.vertcat(rng, rng_h, x[PZ])
        c = ocp.constraints
        c.lh = np.array([0.0, 0.0, 0.0, 0.0, 0.0, -1e6])
        c.uh = np.array([15.0 ** 2, 1e6, 1e6, 1e6, 1e6, 1e6])
        c.lh_e = np.array([0.0, 0.0, -1e6]); c.uh_e = np.array([1e6, 1e6, 1e6])
        c.idxsh = np.array([1, 2, 3, 4, 5]); c.idxsh_e = np.array([0, 1, 2])
        ocp.cost.zl = np.zeros(5); ocp.cost.zu = np.zeros(5)
        ocp.cost.Zl = np.zeros(5); ocp.cost.Zu = np.zeros(5)
        ocp.cost.zl_e = np.zeros(3); ocp.cost.zu_e = np.zeros(3)
        ocp.cost.Zl_e = np.zeros(3); ocp.cost.Zu_e = np.zeros(3)
        # Boxes on the applied command (states) and its rate (inputs).
        c.idxbx = np.array([UX, UY, UZ, UR])
        c.lbx = -np.array([15.0, 15.0, 3.0, 1.5]); c.ubx = -c.lbx
        c.idxbx_e = c.idxbx.copy(); c.lbx_e = c.lbx.copy(); c.ubx_e = c.ubx.copy()
        c.idxbu = np.arange(NU)
        c.lbu = -np.array([5.0, 5.0, 2.0, 3.0]); c.ubu = -c.lbu
        c.x0 = np.zeros(NX)

        ocp.parameter_values = self.default_parameters()
        so = ocp.solver_options
        so.N_horizon = self.N
        so.tf = self.T
        so.nlp_solver_type = "SQP_RTI"
        so.qp_solver = "PARTIAL_CONDENSING_HPIPM"
        so.hessian_approx = "GAUSS_NEWTON"
        so.integrator_type = "ERK"
        so.sim_method_num_stages = 4
        so.sim_method_num_steps = 1
        so.qp_solver_warm_start = 1
        so.print_level = 0
        return ocp

    @staticmethod
    def default_parameters():
        p = np.zeros(NP)
        p[P_DIR_X] = -1.0
        p[P_S_RAD], p[P_S_TAN] = 5.0, 3.0
        p[P_W_POS], p[P_W_YAW] = 1.0, 1.0
        p[P_R_MIN] = 10.0
        p[P_TAU_XY], p[P_TAU_Z], p[P_TAU_R] = 0.45, 0.3, 0.3
        p[P_R_MIN_SCEN] = 0.0          # scenario constraints off unless set
        return p

    # ------------------------------------------------------------------
    def set_weights(self, w_radial, w_tangential, w_height, w_yaw, yaw_band_deg,
                    r_du_xy, r_du_z, r_du_yaw, terminal_factor):
        """Weights are relative to 'one band half-width' of position error.
        The yaw residual is sin(error), so it is divided by sin(yaw band): a
        yaw error equal to the band costs w_yaw."""
        yb = math.sin(math.radians(yaw_band_deg)) ** 2
        track = [w_radial, w_tangential, w_tangential, w_height, w_yaw / yb, w_yaw / yb]
        W = np.diag(track + [r_du_xy, r_du_xy, r_du_z, r_du_yaw])
        W_e = np.diag([terminal_factor * w for w in track])
        for k in range(self.N):
            self.solver.cost_set(k, "W", W)
        self.solver.cost_set(self.N, "W", W_e)

    def set_limits(self, v_max_xy, v_max_z, yaw_rate_max, accel_max_xy, accel_max_z, yaw_accel_max,
                   range_slack_linear, range_slack_quadratic):
        ubx = np.array([v_max_xy, v_max_xy, v_max_z, yaw_rate_max])
        ubu = np.array([accel_max_xy, accel_max_xy, accel_max_z, yaw_accel_max])
        for k in range(1, self.N + 1):
            self.solver.constraints_set(k, "lbx", -ubx)
            self.solver.constraints_set(k, "ubx", ubx)
        for k in range(self.N):
            self.solver.constraints_set(k, "lbu", -ubu)
            self.solver.constraints_set(k, "ubu", ubu)
        self._ubx = ubx
        self._slack = (range_slack_linear, range_slack_quadratic)
        self._r_max = 0.0
        self._z_min = None
        self._apply_h_bounds()

    def set_max_range(self, r_max, r_min=10.0, slack_linear=None, slack_quadratic=None):
        """Soft MAXIMUM horizontal range to the (predicted) car, r_max <= 0
        switches it off. When the shot point runs away faster than the
        drone can fly (e.g. 18 m to the right of a car in a long left turn:
        the point needs ~2x the car's speed, the drone tops out at 15 m/s)
        the cost alone just chases it round the outside while the car gets
        away (run 20260929_124410 lost it at 80 m). Past r_max the MPC gives
        up shot accuracy to keep the car within detection range: it cuts
        inside. Horizontal, so altitude is never a way to comply."""
        self._r_max = r_max
        self._apply_h_bounds()

    def set_min_altitude(self, z_min):
        """Soft altitude floor (local z, m); None switches it off. The MPC
        had no lower altitude limit at all."""
        self._z_min = z_min
        self._apply_h_bounds()

    def _apply_h_bounds(self):
        zl, Zl = self._slack
        r_on = self._r_max > 0.0
        z_on = self._z_min is not None
        lh = np.array([0.0, 0.0, 0.0, 0.0, 0.0, self._z_min if z_on else -1e6])
        uh = np.array([self._ubx[0] ** 2, 1e6, 1e6, 1e6, self._r_max if r_on else 1e6, 1e6])
        #       slack rows:  range_min, brake, accel, range_max, altitude
        zl_v = np.array([zl, zl, zl, 0.0, 4.0 * zl if z_on else 0.0])
        Zl_v = np.array([Zl, Zl, Zl, 0.0, 4.0 * Zl if z_on else 0.0])
        zu_v = np.array([0.0, 0.0, 0.0, zl if r_on else 0.0, 0.0])
        Zu_v = np.array([0.0, 0.0, 0.0, Zl if r_on else 0.0, 0.0])
        for k in range(1, self.N):      # stage 0 is the fixed x0: no path constraint / slack there
            self.solver.constraints_set(k, "lh", lh)
            self.solver.constraints_set(k, "uh", uh)
            self.solver.cost_set(k, "zl", zl_v)
            self.solver.cost_set(k, "Zl", Zl_v)
            self.solver.cost_set(k, "zu", zu_v)
            self.solver.cost_set(k, "Zu", Zu_v)
        self.solver.constraints_set(self.N, "lh", np.array([0.0, 0.0, lh[5]]))
        self.solver.constraints_set(self.N, "uh", np.array([1e6, uh[4], 1e6]))
        self.solver.cost_set(self.N, "zl", np.array([zl, 0.0, zl_v[4]]))
        self.solver.cost_set(self.N, "Zl", np.array([Zl, 0.0, Zl_v[4]]))
        self.solver.cost_set(self.N, "zu", np.array([0.0, zu_v[3], 0.0]))
        self.solver.cost_set(self.N, "Zu", np.array([0.0, Zu_v[3], 0.0]))

    # ------------------------------------------------------------------
    def solve(self, x0, params):
        """x0: state at the plan origin (already delay-compensated).
        params: (N+1, NP) stage parameters. Returns (command [ux uy uz ur]
        to send now, info dict with status, solve time and the plan)."""
        x0 = np.asarray(x0, dtype=float).copy()
        x0[UX:UR + 1] = np.clip(x0[UX:UR + 1], -self._ubx, self._ubx)   # keep x0 feasible
        s = self.solver
        if not self._initialized:
            # Cold start: hold the current state/command over the horizon.
            for k in range(self.N + 1):
                s.set(k, "x", x0)
            for k in range(self.N):
                s.set(k, "u", np.zeros(NU))
            self._initialized = True
        else:
            # Unwrap yaw of x0 next to the previous plan (the cost is periodic,
            # but the warm start should not be 2*pi away).
            prev_psi = s.get(0, "x")[PSI]
            x0[PSI] = prev_psi + math.remainder(x0[PSI] - prev_psi, 2.0 * math.pi)
        s.set(0, "lbx", x0)
        s.set(0, "ubx", x0)
        for k in range(self.N + 1):
            s.set(k, "p", np.asarray(params[k], dtype=float))
        t0 = time.monotonic()
        status = s.solve()
        solve_ms = 1000.0 * (time.monotonic() - t0)
        x1 = s.get(1, "x")
        plan = np.array([s.get(k, "x") for k in range(self.N + 1)])
        return x1[UX:UR + 1].copy(), {"status": status, "solve_ms": solve_ms, "plan": plan}

    def reset(self):
        """Forget the warm start (e.g. after a mode switch)."""
        self._initialized = False


def propagate_delay(x_now, sent, now, delays, taus, step=0.02):
    """Predict the drone state `max(delays)` ahead of `now`.

    x_now: [px py pz vx vy vz psi r] measured now.
    sent: list of (t_sent, [ux uy uz ur]) commands sent so far, ascending.
    delays: (d_xy, d_z, d_r) transport delay per channel (s).
    taus: (tau_xy, tau_z, tau_r) first-order lags.
    Channel c at time tau is driven by the command sent at tau - d_c; where
    that is later than `now` (a shorter-delay channel) the last command sent
    is held. Returns the 12-state x0 for the MPC, with u = last command sent.
    """
    x = np.array(x_now, dtype=float)
    last = np.array(sent[-1][1], dtype=float) if sent else np.zeros(4)
    horizon = max(delays)
    times = [t for t, _ in sent]
    tau_xy, tau_z, tau_r = taus

    def cmd_at(t_plant, d):
        t_cmd = t_plant - d
        if not sent or t_cmd >= now:
            return last
        i = np.searchsorted(times, t_cmd, side="right") - 1
        return np.array(sent[max(i, 0)][1], dtype=float) if i >= 0 else np.zeros(4)

    n = max(1, int(math.ceil(horizon / step)))
    d = horizon / n
    for k in range(n):
        t = now + k * d
        uxy = cmd_at(t, delays[0]); uz = cmd_at(t, delays[1]); ur = cmd_at(t, delays[2])
        x[0:3] += x[3:6] * d
        x[3] += (uxy[0] - x[3]) / tau_xy * d
        x[4] += (uxy[1] - x[4]) / tau_xy * d
        x[5] += (uz[2] - x[5]) / tau_z * d
        x[6] += x[7] * d
        x[7] += (ur[3] - x[7]) / tau_r * d
    return np.concatenate([x, last]), horizon
