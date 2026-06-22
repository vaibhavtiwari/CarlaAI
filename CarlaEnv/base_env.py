import os
import subprocess
import time

import carla
import numpy as np
import pygame
from pygame.locals import K_ESCAPE

try:
    import gym
    from gym.utils import seeding
except ImportError:
    import gymnasium as gym
    from gymnasium.utils import seeding

try:
    from .hud import HUD
    from .planner import RoadOption
    from .rollout import EpisodeTrace
    from .wrappers import (
        Camera,
        Vehicle,
        World,
        build_road_overlay_segments,
        camera_transforms,
        distance_to_line,
        draw_route_overlay,
        get_actor_display_name,
        vector,
    )
except ImportError:
    from hud import HUD
    from planner import RoadOption
    from rollout import EpisodeTrace
    from wrappers import (
        Camera,
        Vehicle,
        World,
        build_road_overlay_segments,
        camera_transforms,
        distance_to_line,
        draw_route_overlay,
        get_actor_display_name,
        vector,
    )


class CarlaBaseEnv(gym.Env):
    metadata = {
        "render.modes": ["human", "rgb_array", "rgb_array_no_hud", "state_pixels"]
    }

    def __init__(
        self,
        host="127.0.0.1",
        port=2000,
        viewer_res=(1280, 720),
        obs_res=(1280, 720),
        reward_fn=None,
        encode_state_fn=None,
        synchronous=True,
        fps=30,
        action_smoothing=0.9,
        start_carla=True,
        show_waypoints=True,
    ):
        self.carla_process = None
        if start_carla:
            self.carla_process = self._start_carla(synchronous=synchronous, fps=fps)

        pygame.init()
        pygame.font.init()
        width, height = viewer_res
        if obs_res is None:
            out_width, out_height = width, height
        else:
            out_width, out_height = obs_res

        self.display = pygame.display.set_mode((width, height), pygame.HWSURFACE | pygame.DOUBLEBUF)
        self.clock = pygame.time.Clock()
        self.synchronous = synchronous
        self.viewer_res = viewer_res
        self.debug_enabled = os.environ.get("CARLA_ENV_DEBUG", "0") == "1"

        self.seed()
        self.action_space = gym.spaces.Box(np.array([-1, 0]), np.array([1, 1]), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(*obs_res, 3), dtype=np.float32)
        self.metadata["video.frames_per_second"] = self.fps = self.average_fps = fps
        self.action_smoothing = action_smoothing
        self.encode_state_fn = (lambda x: x) if not callable(encode_state_fn) else encode_state_fn
        self.reward_fn = (lambda x: 0) if not callable(reward_fn) else reward_fn
        self.show_waypoints = show_waypoints
        self.closed = False
        self.episode_trace = EpisodeTrace(self._get_scenario_name())
        self.collision_events = []
        self.lane_invasion_events = []

        self.world = None
        available_bev_height = max(220, height - out_height - 40)
        self.bev_width = min(max(320, width // 3), 480)
        self.bev_height = min(self.bev_width, available_bev_height)
        self.observation = self.observation_buffer = None
        self.viewer_image = self.viewer_image_buffer = None
        self.last_observation_frame = None
        self.last_viewer_frame = None
        self.observation_frames_received = 0
        self.viewer_frames_received = 0
        try:
            self._debug("Initializing CARLA base env")
            self.client = carla.Client(host, port)
            self.client.set_timeout(60.0)

            self.world = World(self.client)
            if self.synchronous:
                settings = self.world.get_settings()
                settings.synchronous_mode = True
                settings.fixed_delta_seconds = 1.0 / self.fps
                self.world.apply_settings(settings)
                self._debug(f"Applied synchronous settings with fixed_delta_seconds={1.0 / self.fps:.4f}")

            self.road_overlay_segments = build_road_overlay_segments(self.world.map)

            vehicle_spawn_point = self._get_vehicle_spawn_point()
            self.vehicle = Vehicle(
                self.world,
                vehicle_spawn_point,
                on_collision_fn=lambda e: self._on_collision(e),
                on_invasion_fn=lambda e: self._on_invasion(e),
            )

            self.hud = HUD(width, height)
            self.hud.set_vehicle(self.vehicle)
            self.world.on_tick(self.hud.on_world_tick)

            self.dashcam = Camera(
                self.world,
                out_width,
                out_height,
                transform=camera_transforms["dashboard"],
                attach_to=self.vehicle,
                on_recv_image=lambda e: self._set_observation_image(e),
                sensor_tick=0.0 if self.synchronous else 1.0 / self.fps,
            )
            self.camera = Camera(
                self.world,
                width,
                height,
                transform=camera_transforms["spectator"],
                attach_to=self.vehicle,
                on_recv_image=lambda e: self._set_viewer_image(e),
                sensor_tick=0.0 if self.synchronous else 1.0 / self.fps,
            )

            self._post_world_init()
            self._debug("Post world init complete")
            self._prime_sensor_buffers()
            self._debug(
                f"Initial sensor warmup complete: obs_frames={self.observation_frames_received}, "
                f"viewer_frames={self.viewer_frames_received}"
            )
        except Exception as e:
            self.close()
            raise e

        self.reset()

    def _start_carla(self, synchronous, fps):
        if "CARLA_ROOT" not in os.environ:
            raise Exception("${CARLA_ROOT} has not been set!")
        dist_dir = os.path.join(os.environ["CARLA_ROOT"], "Dist")
        if not os.path.isdir(dist_dir):
            raise Exception('Expected to find directory "Dist" under ${CARLA_ROOT}!')
        sub_dirs = [
            os.path.join(dist_dir, sub_dir)
            for sub_dir in os.listdir(dist_dir)
            if os.path.isdir(os.path.join(dist_dir, sub_dir))
        ]
        if len(sub_dirs) == 0:
            raise Exception(
                'Could not find a packaged distribution of CALRA! '
                '(try building CARLA with the "make package" command in ${CARLA_ROOT})'
            )
        sub_dir = sub_dirs[0]
        carla_path = os.path.join(sub_dir, "LinuxNoEditor", "CarlaUE4.sh")
        launch_command = [carla_path, "Town07"]
        if synchronous:
            launch_command.append("-benchmark")
        launch_command.append("-fps=%i" % fps)
        print("Running command:")
        print(" ".join(launch_command))
        process = subprocess.Popen(launch_command, stdout=subprocess.PIPE, universal_newlines=True)
        print("Waiting for CARLA to initialize")
        for line in process.stdout:
            if "LogCarla: Number Of Vehicles" in line:
                break
        time.sleep(2)
        return process

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self, is_training=False):
        self._debug(f"Reset start (is_training={is_training})")
        self._reset_task(is_training=is_training)
        self._reset_episode_state(is_training=is_training)
        self._refresh_route_visuals()
        self._prime_sensor_buffers()
        self._debug(
            f"Reset warmup complete: obs_frames={self.observation_frames_received}, "
            f"viewer_frames={self.viewer_frames_received}"
        )
        return self.step(None)[0]

    def _reset_episode_state(self, is_training):
        self.terminal_state = False
        self.closed = False
        self.extra_info = []
        self.observation = self.observation_buffer = None
        self.viewer_image = self.viewer_image_buffer = None
        self.start_t = time.time()
        self.step_count = 0
        self.is_training = is_training
        self.total_reward = 0.0
        self.previous_location = self.vehicle.get_transform().location
        self.distance_traveled = 0.0
        self.center_lane_deviation = 0.0
        self.speed_accum = 0.0
        self.last_reward = 0.0
        self.terminal_reason = "Running..."
        self.collision_events = []
        self.lane_invasion_events = []
        self._collision_this_step = False
        self._lane_invasion_this_step = False
        self.episode_trace = EpisodeTrace(self._get_scenario_name())

    def close(self):
        if self.carla_process:
            self.carla_process.terminate()
        pygame.quit()
        if self.world is not None:
            self.world.destroy()
        self.closed = True

    def render(self, mode="human"):
        self._debug(
            f"Render call: viewer={'set' if self.viewer_image is not None else 'none'}, "
            f"obs={'set' if self.observation is not None else 'none'}"
        )
        maneuver = self._get_maneuver_name()
        self.extra_info.extend(
            [
                "Scenario:        % 11s" % self._get_scenario_name(),
                "Reward: % 19.2f" % self.last_reward,
                "",
                "Maneuver:        % 11s" % maneuver,
            ]
        )
        self.extra_info.extend(self._get_render_metrics())

        self.display.blit(pygame.surfarray.make_surface(self.viewer_image.swapaxes(0, 1)), (0, 0))

        obs_h, obs_w = self.observation.shape[:2]
        view_h, view_w = self.viewer_image.shape[:2]
        pos = (view_w - obs_w - 10, 10)
        self.display.blit(pygame.surfarray.make_surface(self.observation.swapaxes(0, 1)), pos)
        self._render_bev_panel(view_w - self.bev_width - 10, pos[1] + obs_h + 10)

        self.hud.render(self.display, extra_info=self.extra_info)
        self.extra_info = []
        pygame.display.flip()

        if mode == "rgb_array_no_hud":
            return self.viewer_image
        if mode == "rgb_array":
            return np.array(pygame.surfarray.array3d(self.display), dtype=np.uint8).transpose([1, 0, 2])
        if mode == "state_pixels":
            return self.observation

    def step(self, action):
        if self.closed:
            raise Exception(
                'CarlaEnv.step() called after the environment was closed.'
                'Check for info["closed"] == True in the learning loop.'
            )

        self._debug(f"Step start #{self.step_count + 1} action={'none' if action is None else 'provided'}")
        self._before_step()
        self._update_clock_before_tick(action)
        self._apply_action(action)
        self._tick_world()
        self._debug(
            f"Post tick: obs_frames={self.observation_frames_received}, "
            f"viewer_frames={self.viewer_frames_received}"
        )

        self.observation = self._get_observation(timeout=2.0)
        self.viewer_image = self._get_viewer_image(timeout=2.0)
        if self.observation is None or self.viewer_image is None:
            self.terminal_reason = "Sensor frame timeout"
            self.terminal_state = True
            print("! Sensor frame timeout in CarlaBaseEnv.step()")
            self._debug("Step aborted due to sensor frame timeout")
            empty_state = np.zeros(self.observation_space.shape, dtype=np.float32)
            return empty_state, 0.0, self.terminal_state, {
                "closed": self.closed,
                "episode_summary": None,
                "step_metrics": None,
            }
        self._debug(
            f"Frames ready: obs_shape={getattr(self.observation, 'shape', None)}, "
            f"viewer_shape={getattr(self.viewer_image, 'shape', None)}"
        )
        encoded_state = self.encode_state_fn(self)

        transform = self.vehicle.get_transform()
        self._update_waypoint_progress(transform)
        self._update_center_deviation(transform)
        self._highlight_current_waypoint()
        self._update_motion_metrics(transform)
        self._after_step_metrics()

        self.last_reward = self.reward_fn(self)
        self.total_reward += self.last_reward
        self.step_count += 1
        self._record_rollout_step(transform)

        pygame.event.pump()
        if pygame.key.get_pressed()[K_ESCAPE]:
            self.terminal_reason = "User exit"
            self.close()
            self.terminal_state = True

        episode_summary = None
        if self.terminal_state:
            self.episode_trace.finalize(self.terminal_reason)
            episode_summary = self.episode_trace.summary()

        info = {
            "closed": self.closed,
            "episode_summary": episode_summary,
            "step_metrics": self.episode_trace.steps[-1] if self.episode_trace.steps else None,
        }
        self._collision_this_step = False
        self._lane_invasion_this_step = False
        return encoded_state, self.last_reward, self.terminal_state, info

    def _update_clock_before_tick(self, action):
        return

    def _apply_action(self, action):
        if action is None:
            return
        steer, throttle = [float(a) for a in action]
        self.vehicle.control.steer = self.vehicle.control.steer * self.action_smoothing + steer * (
            1.0 - self.action_smoothing
        )
        self.vehicle.control.throttle = self.vehicle.control.throttle * self.action_smoothing + throttle * (
            1.0 - self.action_smoothing
        )

    def _update_waypoint_progress(self, transform):
        waypoint_index = self.current_waypoint_index
        for _ in range(len(self.route_waypoints)):
            next_waypoint_index = waypoint_index + 1
            wp, _ = self.route_waypoints[next_waypoint_index % len(self.route_waypoints)]
            dot = np.dot(
                vector(wp.transform.get_forward_vector())[:2],
                vector(transform.location - wp.transform.location)[:2],
            )
            if dot > 0.0:
                waypoint_index += 1
            else:
                break
        self.current_waypoint_index = waypoint_index
        self._after_waypoint_tracking(transform)

    def _update_center_deviation(self, transform):
        self.distance_from_center = distance_to_line(
            vector(self.current_waypoint.transform.location),
            vector(self.next_waypoint.transform.location),
            vector(transform.location),
        )
        self.center_lane_deviation += self.distance_from_center

    def _update_motion_metrics(self, transform):
        self.distance_traveled += self.previous_location.distance(transform.location)
        self.previous_location = transform.location
        self.speed_accum += self.vehicle.get_speed()

    def _wait_for_reset(self):
        self._debug(f"Wait for reset start (synchronous={self.synchronous}, fps={self.fps})")
        if self.synchronous:
            for _ in range(max(3, min(self.fps // 2, 10))):
                self._tick_world()
        else:
            time.sleep(0.25)
        self._debug("Wait for reset complete")

    def _prime_sensor_buffers(self):
        warmup_ticks = 3 if self.synchronous else 1
        self._debug(f"Priming sensor buffers for {warmup_ticks} ticks")
        for _ in range(warmup_ticks):
            self._tick_world()
        self._get_observation(timeout=0.5)
        self._get_viewer_image(timeout=0.5)

    def _soft_reset_vehicle(self):
        self.vehicle.control.steer = float(0.0)
        self.vehicle.control.throttle = float(0.0)
        self.vehicle.tick()

    def _get_maneuver_name(self):
        if self.current_road_maneuver == RoadOption.LANEFOLLOW:
            return "Follow Lane"
        if self.current_road_maneuver == RoadOption.LEFT:
            return "Left"
        if self.current_road_maneuver == RoadOption.RIGHT:
            return "Right"
        if self.current_road_maneuver == RoadOption.STRAIGHT:
            return "Straight"
        if self.current_road_maneuver == RoadOption.VOID:
            return "VOID"
        fallback_name = getattr(self.current_road_maneuver, "name", None)
        if fallback_name:
            return str(fallback_name).replace("_", " ").title()
        return "INVALID(%s)" % str(self.current_road_maneuver)

    def _tick_world(self):
        if self.synchronous:
            self.clock.tick()
            self.world.tick()
            try:
                self.world.wait_for_tick(seconds=1.0 / self.fps + 0.2)
            except Exception:
                pass
        else:
            if self.fps <= 0:
                self.clock.tick()
            else:
                self.clock.tick_busy_loop(self.fps)
            self.world.tick()
            if self.fps > 0:
                time.sleep(1.0 / self.fps)
            self.average_fps = self.average_fps * 0.5 + self.clock.get_fps() * 0.5
        self.hud.tick(self.world, self.clock)

    def _get_observation(self, timeout=2.0):
        deadline = time.time() + timeout
        while self.observation_buffer is None:
            if time.time() >= deadline:
                return None if self.last_observation_frame is None else self.last_observation_frame.copy()
            time.sleep(0.001)
        obs = self.observation_buffer.copy()
        self.last_observation_frame = obs.copy()
        return obs

    def _get_viewer_image(self, timeout=2.0):
        deadline = time.time() + timeout
        while self.viewer_image_buffer is None:
            if time.time() >= deadline:
                return None if self.last_viewer_frame is None else self.last_viewer_frame.copy()
            time.sleep(0.001)
        image = self.viewer_image_buffer.copy()
        self.last_viewer_frame = image.copy()
        return image

    def _on_collision(self, event):
        self._collision_this_step = True
        self.collision_events.append(event)
        self.terminal_reason = "Collision"
        self.hud.notification("Collision with {}".format(get_actor_display_name(event.other_actor)))

    def _on_invasion(self, event):
        self._lane_invasion_this_step = True
        self.lane_invasion_events.append(event)
        lane_types = set(x.type for x in event.crossed_lane_markings)
        text = ["%r" % str(x).split()[-1] for x in lane_types]
        self.hud.notification("Crossed line %s" % " and ".join(text))

    def _set_observation_image(self, image):
        self.observation_buffer = image
        self.observation_frames_received += 1

    def _set_viewer_image(self, image):
        self.viewer_image_buffer = image
        self.viewer_frames_received += 1

    def _debug(self, message):
        if self.debug_enabled:
            print(f"[CarlaBaseEnv] {message}", flush=True)

    def _refresh_route_visuals(self):
        return

    def _highlight_current_waypoint(self):
        return

    def _render_bev_panel(self, x, y):
        panel = pygame.Surface((self.bev_width, self.bev_height))
        if self.show_waypoints:
            transform = self.vehicle.get_transform()
            draw_route_overlay(
                panel,
                self.route_waypoints,
                road_segments=getattr(self, "road_overlay_segments", None),
                current_waypoint_index=self.current_waypoint_index,
                step=3,
                vehicle_location=transform.location,
                vehicle_forward=vector(transform.get_forward_vector()),
            )
        else:
            panel.fill((18, 20, 24))
        self.display.blit(panel, (x, y))
        pygame.draw.rect(
            self.display,
            (255, 255, 255),
            pygame.Rect(x - 1, y - 1, self.bev_width + 2, self.bev_height + 2),
            1,
        )
        label = self.hud.font_mono.render("2D Route Map", True, (255, 255, 255))
        label_bg = pygame.Surface((label.get_width() + 10, label.get_height() + 6))
        label_bg.set_alpha(160)
        self.display.blit(label_bg, (x, y))
        self.display.blit(label, (x + 5, y + 3))

        route_count = len(getattr(self, "route_waypoints", []))
        if route_count > 0:
            progress_text = "WP %d/%d" % ((self.current_waypoint_index % route_count) + 1, route_count)
            speed_text = "Speed %.1f km/h" % (3.6 * self.vehicle.get_speed())
            footer = self.hud.font_mono.render("%s  |  %s" % (progress_text, speed_text), True, (210, 218, 228))
            footer_bg = pygame.Surface((self.bev_width, footer.get_height() + 8))
            footer_bg.set_alpha(140)
            self.display.blit(footer_bg, (x, y + self.bev_height - footer.get_height() - 8))
            self.display.blit(footer, (x + 6, y + self.bev_height - footer.get_height() - 4))

    def _record_rollout_step(self, transform):
        task_metrics = self._get_task_rollout_metrics()
        self.episode_trace.record_step(
            {
                "step": self.step_count,
                "reward": self.last_reward,
                "speed_kmh": 3.6 * self.vehicle.get_speed(),
                "distance_traveled": self.distance_traveled,
                "center_lane_deviation": self.center_lane_deviation,
                "distance_from_center": self.distance_from_center,
                "steer": float(self.vehicle.control.steer),
                "throttle": float(self.vehicle.control.throttle),
                "brake": float(self.vehicle.control.brake),
                "location_x": float(transform.location.x),
                "location_y": float(transform.location.y),
                "yaw": float(transform.rotation.yaw),
                "collision": self._collision_this_step,
                "lane_invasion": self._lane_invasion_this_step,
                "task_metrics": task_metrics,
            }
        )

    def get_episode_summary(self):
        self.episode_trace.finalize(self.terminal_reason)
        return self.episode_trace.summary()

    def _get_vehicle_spawn_point(self):
        raise NotImplementedError

    def _post_world_init(self):
        return

    def _reset_task(self, is_training):
        raise NotImplementedError

    def _before_step(self):
        return

    def _after_waypoint_tracking(self, transform):
        raise NotImplementedError

    def _after_step_metrics(self):
        return

    def _get_scenario_name(self):
        raise NotImplementedError

    def _get_render_metrics(self):
        raise NotImplementedError

    def _get_task_rollout_metrics(self):
        return {}
