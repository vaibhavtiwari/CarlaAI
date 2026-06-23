from dataclasses import dataclass
from math import radians, sqrt

import numpy as np


def wrap_angle_numpy(angle):
    return np.arctan2(np.sin(angle), np.cos(angle))


@dataclass
class MPCConfig:
    horizon: int = 20
    dt: float = 0.2
    wheelbase: float = 2.875
    max_steer_rad: float = radians(35.0)
    max_accel: float = 3.0
    max_decel: float = 5.0
    min_speed: float = 0.0
    max_speed: float = 20.0
    reference_step: int = 3
    # Tracking weights
    w_position: float = 30.0
    w_heading: float = 20.0
    w_speed: float = 2.0
    # Control effort weights
    w_steer: float = 3.0
    w_accel: float = 0.2
    # Smoothness weights
    w_steer_rate: float = 3.0
    w_accel_rate: float = 0.8
    # Terminal state weights (Horizon-end guidance)
    w_terminal_position: float = 7.0
    w_terminal_heading: float = 20.0
    w_terminal_speed: float = 3.0

    def q_matrix(self):
        return np.diag(
            [
                self.w_position,
                self.w_position,
                self.w_heading,
                self.w_speed,
            ]
        ).astype(np.float64)

    def r_matrix(self):
        return np.diag(
            [
                self.w_steer,
                self.w_accel,
            ]
        ).astype(np.float64)

    def s_matrix(self):
        return np.diag(
            [
                self.w_steer_rate,
                self.w_accel_rate,
            ]
        ).astype(np.float64)

    def qf_matrix(self):
        return np.diag(
            [
                self.w_terminal_position,
                self.w_terminal_position,
                self.w_terminal_heading,
                self.w_terminal_speed,
            ]
        ).astype(np.float64)


@dataclass
class MPCDebug:
    objective_value: float = 0.0
    solver_success: bool = False
    solver_message: str = ""
    nearest_waypoint_index: int = 0
    reference_speed_mps: float = 0.0
    iterations: int = 0
    solve_time_ms: float = 0.0
    position_cost: float = 0.0
    heading_cost: float = 0.0
    speed_cost: float = 0.0
    control_cost: float = 0.0
    smoothness_cost: float = 0.0
    terminal_cost: float = 0.0


@dataclass
class VehicleState:
    x: float
    y: float
    yaw: float
    speed: float


def build_vehicle_state(vehicle):
    transform = vehicle.get_transform()
    velocity = vehicle.get_velocity()
    speed = sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)
    return VehicleState(
        x=float(transform.location.x),
        y=float(transform.location.y),
        yaw=float(wrap_angle_numpy(radians(float(transform.rotation.yaw)))),
        speed=float(speed),
    )
