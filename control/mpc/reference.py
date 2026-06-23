from math import radians

import numpy as np


def build_reference_trajectory(route_waypoints, current_waypoint_index, horizon, reference_step, n_states, target_speed_mps):
    route_count = len(route_waypoints)
    start_index = max(0, int(current_waypoint_index))
    reference = np.zeros((horizon + 1, n_states), dtype=np.float64)

    for step_idx in range(horizon + 1):
        idx = min(start_index + step_idx * reference_step, route_count - 1)
        waypoint = route_waypoints[idx][0]
        location = waypoint.transform.location
        reference[step_idx] = np.array(
            [
                float(location.x),
                float(location.y),
                float(radians(waypoint.transform.rotation.yaw)),
                target_speed_mps,
            ],
            dtype=np.float64,
        )
    return start_index, reference
