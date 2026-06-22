import numpy as np

try:
    import casadi as ca
except ImportError as exc:
    raise ImportError(
        "CasADi is required for CarlaEnv.mpc. Use the Python environment where `import casadi` works."
    ) from exc


class KinematicMPCSolver:
    def __init__(self, config):
        self.config = config
        self.n_states = 4
        self.n_controls = 2
        self._build_solver()

    def _build_solver(self):
        horizon = self.config.horizon
        q_matrix = ca.DM(self.config.q_matrix())
        r_matrix = ca.DM(self.config.r_matrix())
        s_matrix = ca.DM(self.config.s_matrix())
        qf_matrix = ca.DM(self.config.qf_matrix())

        x_var, u_var, p_var = self._build_symbolic_variables(horizon)
        dynamics = self._build_dynamics_model()
        constraints = self._build_dynamics_constraints(x_var, u_var, p_var, dynamics, horizon)
        cost = self._build_running_cost(x_var, u_var, p_var, q_matrix, r_matrix, s_matrix, horizon)
        cost += self._build_terminal_cost(x_var, p_var, qf_matrix, horizon)

        # CasADi/Ipopt expects a single optimization vector, so flatten X and U.
        opt_variables = ca.vertcat(ca.reshape(x_var, -1, 1), ca.reshape(u_var, -1, 1))
        nlp = {"f": cost, "x": opt_variables, "g": ca.vertcat(*constraints), "p": p_var}
        options = {"ipopt.print_level": 0, "ipopt.sb": "yes", "ipopt.max_iter": 80, "print_time": 0}
        self.solver = ca.nlpsol("mpc_solver", "ipopt", nlp, options)

        x_entries = self.n_states * (horizon + 1)
        u_entries = self.n_controls * horizon
        self.x_shape = (self.n_states, horizon + 1)
        self.u_shape = (self.n_controls, horizon)

        self._build_variable_bounds(x_entries, u_entries, horizon)
        self._build_constraint_bounds(horizon)

    def _build_symbolic_variables(self, horizon):
        # X contains the predicted state trajectory over the horizon:
        # columns are time steps, rows are [x, y, yaw, speed].
        x_var = ca.SX.sym("X", self.n_states, horizon + 1)
        # U contains the control trajectory:
        # rows are [steer_normalized, accel].
        u_var = ca.SX.sym("U", self.n_controls, horizon)
        # P packs the current measured state followed by the reference states.
        p_var = ca.SX.sym("P", self.n_states + (horizon + 1) * self.n_states)
        return x_var, u_var, p_var

    def _build_dynamics_model(self):
        wheelbase = self.config.wheelbase

        def dynamics(state_vec, control_vec):
            steer_norm = control_vec[0]
            accel = control_vec[1]
            steer_rad = steer_norm * self.config.max_steer_rad
            # Simple kinematic bicycle model with scalar speed.
            return ca.vertcat(
                state_vec[3] * ca.cos(state_vec[2]),
                state_vec[3] * ca.sin(state_vec[2]),
                (state_vec[3] / wheelbase) * ca.tan(steer_rad),
                accel,
            )

        return dynamics

    def _build_running_cost(self, x_var, u_var, p_var, q_matrix, r_matrix, s_matrix, horizon):
        cost = 0
        for step_idx in range(horizon):
            ref_offset = self.n_states + step_idx * self.n_states
            ref_state = p_var[ref_offset : ref_offset + self.n_states]
            state_error = x_var[:, step_idx] - ref_state
            heading_error = ca.atan2(ca.sin(state_error[2]), ca.cos(state_error[2]))
            tracking_error = ca.vertcat(
                state_error[0],
                state_error[1],
                heading_error,
                state_error[3],
            )

            # e_k^T Q e_k + u_k^T R u_k
            cost += ca.mtimes([tracking_error.T, q_matrix, tracking_error])
            cost += ca.mtimes([u_var[:, step_idx].T, r_matrix, u_var[:, step_idx]])

            if step_idx > 0:
                delta_u = u_var[:, step_idx] - u_var[:, step_idx - 1]
                # (u_k - u_{k-1})^T S (u_k - u_{k-1})
                cost += ca.mtimes([delta_u.T, s_matrix, delta_u])
        return cost

    def _build_terminal_cost(self, x_var, p_var, qf_matrix, horizon):
        terminal_ref = p_var[
            self.n_states + horizon * self.n_states : self.n_states + (horizon + 1) * self.n_states
        ]
        terminal_error = x_var[:, horizon] - terminal_ref
        terminal_heading_error = ca.atan2(ca.sin(terminal_error[2]), ca.cos(terminal_error[2]))
        terminal_tracking_error = ca.vertcat(
            terminal_error[0],
            terminal_error[1],
            terminal_heading_error,
            terminal_error[3],
        )
        return ca.mtimes([terminal_tracking_error.T, qf_matrix, terminal_tracking_error])

    def _build_dynamics_constraints(self, x_var, u_var, p_var, dynamics, horizon):
        constraints = [x_var[:, 0] - p_var[: self.n_states]]
        for step_idx in range(horizon):
            next_state = x_var[:, step_idx] + self.config.dt * dynamics(x_var[:, step_idx], u_var[:, step_idx])
            constraints.append(x_var[:, step_idx + 1] - next_state)
        return constraints

    def _build_variable_bounds(self, x_entries, u_entries, horizon):
        # By default, states are unbounded until we apply specific limits below.
        self.lbx = np.full(x_entries + u_entries, -np.inf, dtype=np.float64)
        self.ubx = np.full(x_entries + u_entries, np.inf, dtype=np.float64)

        for step_idx in range(horizon + 1):
            speed_idx = 3 + step_idx * self.n_states
            # Hard state bounds on predicted speed.
            self.lbx[speed_idx] = self.config.min_speed
            self.ubx[speed_idx] = self.config.max_speed

        control_start = x_entries
        for step_idx in range(horizon):
            steer_idx = control_start + step_idx * self.n_controls
            accel_idx = steer_idx + 1
            # Hard input bounds on steering and longitudinal acceleration.
            self.lbx[steer_idx] = -1.0
            self.ubx[steer_idx] = 1.0
            self.lbx[accel_idx] = -self.config.max_decel
            self.ubx[accel_idx] = self.config.max_accel

    def _build_constraint_bounds(self, horizon):
        # All dynamics constraints are equalities, so both lower and upper are zero.
        self.lbg = np.zeros(self.n_states * (horizon + 1), dtype=np.float64)
        self.ubg = np.zeros(self.n_states * (horizon + 1), dtype=np.float64)
