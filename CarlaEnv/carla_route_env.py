import time

import numpy as np
import pygame
from pygame.locals import K_DOWN, K_LEFT, K_RIGHT, K_UP, K_a, K_d, K_s, K_w

try:
    from .base_env import CarlaBaseEnv
    from .planner import compute_route_waypoints
    from .wrappers import vector
except ImportError:
    from base_env import CarlaBaseEnv
    from planner import compute_route_waypoints
    from wrappers import vector


class CarlaRouteEnv(CarlaBaseEnv):
    """
    CARLA point-to-point route-following environment with shared simulator lifecycle.
    """

    def __init__(self, *args, **kwargs):
        self.max_distance = 3000
        self.route_completion_distance_threshold = 2.0
        self.route_completion_stationary_speed_mps = 0.25
        self._route_transition_status = "not_checked"
        super().__init__(*args, **kwargs)

    def _get_vehicle_spawn_point(self):
        return self.world.map.get_spawn_points()[0]

    def _reset_task(self, is_training):
        self._debug(f"Route reset task start (is_training={is_training})")
        self.num_routes_completed = -1
        self.routes_completed = 0.0
        self._route_transition_status = "reset"
        self._create_new_route()
        self._debug("Route reset task complete")

    def _before_step(self):
        if self._should_rollover_route():
            self._route_transition_status = "triggered"
            self._create_new_route()
        else:
            self._route_transition_status = "waiting"

    def _should_rollover_route(self):
        if self.current_waypoint_index >= len(self.route_waypoints) - 1:
            return True

        final_waypoint, _ = self.route_waypoints[-1]
        final_distance = self.vehicle.get_transform().location.distance(final_waypoint.transform.location)
        speed_mps = self.vehicle.get_speed()
        if (
            final_distance <= self.route_completion_distance_threshold
            and speed_mps <= self.route_completion_stationary_speed_mps
        ):
            self._route_transition_status = "threshold_reached"
            return True
        return False

    def _create_new_route(self):
        self._debug("Route creation: soft reset vehicle")
        self._soft_reset_vehicle()
        self._debug("Route creation: sampling start/end spawn points")
        self.start_wp, self.end_wp = [
            self.world.map.get_waypoint(spawn.location)
            for spawn in np.random.choice(self.world.map.get_spawn_points(), 2, replace=False)
        ]
        self._debug(
            "Route creation: computing waypoints "
            f"from ({self.start_wp.transform.location.x:.1f}, {self.start_wp.transform.location.y:.1f}) "
            f"to ({self.end_wp.transform.location.x:.1f}, {self.end_wp.transform.location.y:.1f})"
        )
        self.route_waypoints = compute_route_waypoints(self.world.map, self.start_wp, self.end_wp, resolution=1.0)
        self._debug(f"Route creation: computed {len(self.route_waypoints)} waypoints")
        self.current_waypoint_index = 0
        self.num_routes_completed += 1
        self._route_transition_status = "new_route_loaded"
        self._debug("Route creation: teleporting vehicle to route start")
        self.vehicle.set_transform(self.start_wp.transform)
        self.vehicle.set_simulate_physics(False)
        self.vehicle.set_simulate_physics(True)
        self._debug("Route creation: waiting for reset stabilization")
        self._wait_for_reset()
        self._debug("Route creation: refresh route visuals")
        self._refresh_route_visuals()

    def _after_waypoint_tracking(self, transform):
        if self.current_waypoint_index < len(self.route_waypoints) - 1:
            self.next_waypoint, self.next_road_maneuver = self.route_waypoints[
                (self.current_waypoint_index + 1) % len(self.route_waypoints)
            ]
        self.current_waypoint, self.current_road_maneuver = self.route_waypoints[
            self.current_waypoint_index % len(self.route_waypoints)
        ]
        self.routes_completed = self.num_routes_completed + (
            self.current_waypoint_index + 1
        ) / len(self.route_waypoints)

    def _after_step_metrics(self):
        if self.distance_traveled >= self.max_distance:
            self.terminal_reason = "Max route distance reached"
            self.terminal_state = True

    def _get_scenario_name(self):
        return "Route"

    def _get_render_metrics(self):
        route_count = len(getattr(self, "route_waypoints", []))
        return [
            "Routes completed:    % 7.2f" % self.routes_completed,
            "Route progress:    % 4d/%-4d" % (self.current_waypoint_index, route_count),
            "Route rollover: % 12s" % self._route_transition_status,
            "Route end tol:   % 7.2f m" % self.route_completion_distance_threshold,
            "Distance traveled: % 7d m" % self.distance_traveled,
            "Center deviance:   % 7.2f m" % self.distance_from_center,
            "Avg center dev:    % 7.2f m" % (self.center_lane_deviation / self.step_count),
            "Avg speed:      % 7.2f km/h" % (3.6 * self.speed_accum / self.step_count),
        ]

    def _draw_path(self, life_time=60.0, skip=0):
        return

    def _get_task_rollout_metrics(self):
        return {
            "routes_completed": self.routes_completed,
            "current_waypoint_index": self.current_waypoint_index,
            "route_num_waypoints": len(self.route_waypoints),
        }


def reward_fn(env):
    early_termination = False
    if early_termination:
        if time.time() - env.start_t > 5.0 and env.vehicle.get_speed() < 1.0 / 3.6:
            env.terminal_state = True
        if env.distance_from_center > 3.0:
            env.terminal_state = True

    fwd = vector(env.vehicle.get_velocity())
    wp_fwd = vector(env.current_waypoint.transform.rotation.get_forward_vector())
    if np.dot(fwd[:2], wp_fwd[:2]) > 0:
        return env.vehicle.get_speed()
    return 0


if __name__ == "__main__":
    env = CarlaRouteEnv(obs_res=(160, 80), reward_fn=reward_fn)
    action = np.zeros(env.action_space.shape[0])
    while True:
        env.reset()
        while True:
            pygame.event.pump()
            keys = pygame.key.get_pressed()
            if keys[K_LEFT] or keys[K_a]:
                action[0] = -0.5
            elif keys[K_RIGHT] or keys[K_d]:
                action[0] = 0.5
            else:
                action[0] = 0.0
            action[0] = np.clip(action[0], -1, 1)
            action[1] = 1.0 if keys[K_UP] or keys[K_w] else 0.0
            if keys[K_DOWN] or keys[K_s]:
                action[1] = 0.0

            obs, _, done, info = env.step(action)
            if info["closed"]:
                exit(0)
            env.render()
            if done:
                break
    env.close()
