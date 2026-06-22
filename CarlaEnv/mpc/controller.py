from dataclasses import asdict
import time

import numpy as np

from .common import MPCConfig, MPCDebug, wrap_angle_numpy
from .reference import build_reference_trajectory
from .solver import KinematicMPCSolver


class KinematicMPCController:
    """
    Modular CasADi-based MPC for route tracking.
    """

    def __init__(self, target_speed_kmh=20.0, config=None):
        self.target_speed_mps = float(target_speed_kmh) / 3.6
        self.config = config or MPCConfig()
        self.solver = KinematicMPCSolver(self.config)
        self.debug = MPCDebug(reference_speed_mps=self.target_speed_mps)
        self.reset()

    def reset(self):
        self._warm_start_x = np.zeros((self.solver.n_states, self.config.horizon + 1), dtype=np.float64)
        self._warm_start_u = np.zeros((self.solver.n_controls, self.config.horizon), dtype=np.float64)
        self.debug = MPCDebug(reference_speed_mps=self.target_speed_mps)

    def run_step(self, state, route_waypoints, current_waypoint_index):
        if not route_waypoints:
            return 0.0, 0.0, self.debug

        start_index, reference = build_reference_trajectory(
            route_waypoints,
            current_waypoint_index,
            self.config.horizon,
            self.config.reference_step,
            self.solver.n_states,
            self.target_speed_mps,
        )
        self.debug.nearest_waypoint_index = start_index
        solution_x, solution_u = self._solve(state, reference)

        steer_norm = float(np.clip(solution_u[0, 0], -1.0, 1.0))
        accel = float(np.clip(solution_u[1, 0], -self.config.max_decel, self.config.max_accel))
        self._shift_warm_start(solution_x, solution_u)
        return steer_norm, accel, self.debug

    def get_debug_snapshot(self):
        return asdict(self.debug)

    def _solve(self, state, reference):
        initial_state = np.array([state.x, state.y, state.yaw, state.speed], dtype=np.float64)
        parameters = np.concatenate([initial_state, reference.reshape(-1)])
        initial_guess = np.concatenate(
            [
                self._warm_start_x.reshape(-1, order="F"),
                self._warm_start_u.reshape(-1, order="F"),
            ]
        )

        start_t = time.perf_counter()
        result = self.solver.solver(
            x0=initial_guess,
            lbx=self.solver.lbx,
            ubx=self.solver.ubx,
            lbg=self.solver.lbg,
            ubg=self.solver.ubg,
            p=parameters,
        )
        self.debug.solve_time_ms = float((time.perf_counter() - start_t) * 1000.0)

        solution = np.array(result["x"]).reshape(-1)
        x_entries = self.solver.n_states * (self.config.horizon + 1)
        solution_x = solution[:x_entries].reshape(self.solver.x_shape, order="F")
        solution_u = solution[x_entries:].reshape(self.solver.u_shape, order="F")

        stats = self.solver.solver.stats()
        self.debug.objective_value = float(result["f"])
        self.debug.solver_success = bool(stats.get("success", False))
        self.debug.solver_message = str(stats.get("return_status", "unknown"))
        self.debug.reference_speed_mps = self.target_speed_mps
        self.debug.iterations = int(stats.get("iter_count", 0) or 0)
        self._update_cost_breakdown(solution_x, solution_u, reference)
        return solution_x, solution_u

    def _update_cost_breakdown(self, solution_x, solution_u, reference):
        q_matrix = self.config.q_matrix()
        r_matrix = self.config.r_matrix()
        s_matrix = self.config.s_matrix()
        qf_matrix = self.config.qf_matrix()
        position_cost = 0.0
        heading_cost = 0.0
        speed_cost = 0.0
        control_cost = 0.0
        smoothness_cost = 0.0

        for step_idx in range(self.config.horizon):
            state_error = solution_x[:, step_idx] - reference[step_idx]
            heading_error = wrap_angle_numpy(state_error[2])
            tracking_error = np.array(
                [
                    state_error[0],
                    state_error[1],
                    heading_error,
                    state_error[3],
                ],
                dtype=np.float64,
            )

            # Keep the debug breakdown grouped by interpretation, even though the
            # solver itself now uses matrix-form Q/R/S/Qf costs.
            position_cost += float(tracking_error[:2].T @ q_matrix[:2, :2] @ tracking_error[:2])
            heading_cost += float(tracking_error[2] * q_matrix[2, 2] * tracking_error[2])
            speed_cost += float(tracking_error[3] * q_matrix[3, 3] * tracking_error[3])
            control_cost += float(solution_u[:, step_idx].T @ r_matrix @ solution_u[:, step_idx])

            if step_idx > 0:
                delta_u = solution_u[:, step_idx] - solution_u[:, step_idx - 1]
                smoothness_cost += float(delta_u.T @ s_matrix @ delta_u)

        terminal_error = solution_x[:, self.config.horizon] - reference[self.config.horizon]
        terminal_heading_error = wrap_angle_numpy(terminal_error[2])
        terminal_tracking_error = np.array(
            [
                terminal_error[0],
                terminal_error[1],
                terminal_heading_error,
                terminal_error[3],
            ],
            dtype=np.float64,
        )
        terminal_cost = float(terminal_tracking_error.T @ qf_matrix @ terminal_tracking_error)

        self.debug.position_cost = float(position_cost)
        self.debug.heading_cost = float(heading_cost)
        self.debug.speed_cost = float(speed_cost)
        self.debug.control_cost = float(control_cost)
        self.debug.smoothness_cost = float(smoothness_cost)
        self.debug.terminal_cost = float(terminal_cost)

    def _shift_warm_start(self, solution_x, solution_u):
        self._warm_start_x[:, :-1] = solution_x[:, 1:]
        self._warm_start_x[:, -1] = solution_x[:, -1]
        self._warm_start_u[:, :-1] = solution_u[:, 1:]
        self._warm_start_u[:, -1] = solution_u[:, -1]
