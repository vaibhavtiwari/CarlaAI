import time

import carla
import numpy as np
import pygame
from pygame.locals import K_DOWN, K_LEFT, K_RIGHT, K_UP, K_a, K_d, K_s, K_w

try:
    from .base_env import CarlaBaseEnv
    from .planner import RoadOption, compute_route_waypoints
    from .wrappers import vector
except ImportError:
    from base_env import CarlaBaseEnv
    from planner import RoadOption, compute_route_waypoints
    from wrappers import vector


class CarlaLapEnv(CarlaBaseEnv):
    """
    CARLA lap-following environment with shared simulator lifecycle managed by CarlaBaseEnv.
    """

    def _get_vehicle_spawn_point(self):
        lap_start_wp = self.world.map.get_waypoint(self.world.map.get_spawn_points()[1].location)
        spawn_transform = lap_start_wp.transform
        spawn_transform.location += carla.Location(z=1.0)
        self.lap_start_wp = lap_start_wp
        return spawn_transform

    def _post_world_init(self):
        self.route_waypoints = compute_route_waypoints(
            self.world.map,
            self.lap_start_wp,
            self.lap_start_wp,
            resolution=1.0,
            plan=[RoadOption.STRAIGHT] + [RoadOption.RIGHT] * 2 + [RoadOption.STRAIGHT] * 5,
        )
        self.current_waypoint_index = 0
        self.checkpoint_waypoint_index = 0

    def _reset_task(self, is_training):
        self._soft_reset_vehicle()
        if is_training:
            waypoint, _ = self.route_waypoints[self.checkpoint_waypoint_index % len(self.route_waypoints)]
            self.current_waypoint_index = self.checkpoint_waypoint_index
        else:
            waypoint, _ = self.route_waypoints[0]
            self.current_waypoint_index = 0

        transform = waypoint.transform
        transform.location += carla.Location(z=1.0)
        self.vehicle.set_transform(transform)
        self.vehicle.set_simulate_physics(False)
        self.vehicle.set_simulate_physics(True)
        self._wait_for_reset()
        self.start_waypoint_index = self.current_waypoint_index
        self.laps_completed = 0.0

    def _after_waypoint_tracking(self, transform):
        self.current_waypoint, self.current_road_maneuver = self.route_waypoints[
            self.current_waypoint_index % len(self.route_waypoints)
        ]
        self.next_waypoint, self.next_road_maneuver = self.route_waypoints[
            (self.current_waypoint_index + 1) % len(self.route_waypoints)
        ]

    def _after_step_metrics(self):
        self.laps_completed = (self.current_waypoint_index - self.start_waypoint_index) / len(self.route_waypoints)
        if self.laps_completed >= 3:
            self.terminal_reason = "Lap target reached"
            self.terminal_state = True
        if self.is_training:
            checkpoint_frequency = 50
            self.checkpoint_waypoint_index = (self.current_waypoint_index // checkpoint_frequency) * checkpoint_frequency

    def _get_scenario_name(self):
        return "Lap"

    def _get_render_metrics(self):
        return [
            "Laps completed:    % 7.2f %%" % (self.laps_completed * 100.0),
            "Distance traveled: % 7d m" % self.distance_traveled,
            "Center deviance:   % 7.2f m" % self.distance_from_center,
            "Avg center dev:    % 7.2f m" % (self.center_lane_deviation / self.step_count),
            "Avg speed:      % 7.2f km/h" % (3.6 * self.speed_accum / self.step_count),
        ]

    def _draw_path(self, life_time=60.0, skip=0):
        return

    def _get_task_rollout_metrics(self):
        return {
            "laps_completed": self.laps_completed,
            "current_waypoint_index": self.current_waypoint_index,
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
    env = CarlaLapEnv(obs_res=(160, 80), reward_fn=reward_fn)
    action = np.zeros(env.action_space.shape[0])
    while True:
        env.reset(is_training=True)
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
