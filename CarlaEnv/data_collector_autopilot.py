import os
import subprocess
import time
import random

import carla
import numpy as np
import pygame
from PIL import Image
from pygame.locals import *

from hud import HUD
from wrappers import *
from planner import compute_route_waypoints
from agents.navigation.local_planner import LocalPlanner
from agents.navigation.global_route_planner import GlobalRoutePlanner
from agents.navigation.global_route_planner_dao import GlobalRoutePlannerDAO


class AutopilotDataCollector:
    """
    Autonomous data collector for CARLA that uses route planning and local planning
    to automatically drive and collect camera/sensor data.
    
    Features:
    - Automatic route planning between random spawn points
    - Local planning for vehicle control
    - Traffic light and obstacle avoidance
    - Automatic destination switching when route completes
    - Collects exactly N images before stopping
    
    Note: Requires CARLA to be running. Either start it manually or pass start_carla=True
    """

    def __init__(self, host="127.0.0.1", port=2000,
                 num_images_to_save=10000, output_dir="images",
                 synchronous=True, fps=30, action_smoothing=0.9,
                 target_speed=20, frame_skip=2, min_save_distance=1.0,
                 min_route_distance=100.0, start_carla=False):
        """
        Initialize the autopilot data collector.
        
        Parameters:
        -----------
        host (str): IP address of CARLA server
        port (int): Port of CARLA server
        num_images_to_save (int): Total number of images to collect (default: 10000)
        output_dir (str): Directory to save images
        synchronous (bool): Run in synchronous mode (recommended for consistency)
        fps (int): FPS to run at
        action_smoothing (float): Action smoothing factor (0.0-1.0)
        target_speed (float): Target driving speed in km/h
        start_carla (bool): Automatically start CARLA
        """
        
        self.host = host
        self.port = port
        self.num_images_to_save = num_images_to_save
        self.output_dir = output_dir
        self.synchronous = synchronous
        self.fps = fps
        self.action_smoothing = action_smoothing
        self.target_speed = target_speed
        self.frame_skip = frame_skip
        self.min_save_distance = min_save_distance
        self.min_route_distance = min_route_distance
        self.frame_since_last_save = 0
        self.last_saved_location = None

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        self.rgb_dir = os.path.join(output_dir, "rgb")
        self.segmentation_dir = os.path.join(output_dir, "segmentation")
        os.makedirs(self.rgb_dir, exist_ok=True)
        os.makedirs(self.segmentation_dir, exist_ok=True)
        
        # Visualization state
        pygame.init()
        pygame.font.init()
        self.viewer_res = (1280, 720)
        self.display = pygame.display.set_mode(self.viewer_res, pygame.HWSURFACE | pygame.DOUBLEBUF)
        self.clock = pygame.time.Clock()
        self.hud = HUD(self.viewer_res[0], self.viewer_res[1])
        self.extra_info = []
        self.viewer_image = None
        self.viewer_image_buffer = None
        self.observation = {"rgb": None, "segmentation": None}
        self.observation_buffer = {"rgb": None, "segmentation": None}
        self.recording = True
        self.done = False

        # Initialize CARLA
        self.client = None
        self.world = None
        self.vehicle = None
        self.dashcam_rgb = None
        self.dashcam_seg = None
        self.viewer_camera = None
        self.local_planner = None
        self.global_planner = None
        self.images_collected = 0
        self.routes_completed = 0
        
        # Connect to CARLA
        self._connect_to_carla()
        self._setup_world()
        self._setup_vehicle()
        self._setup_cameras()
        self._setup_planners()
        
        print(f"\n{'='*60}")
        print("AutopilotDataCollector initialized successfully!")
        print(f"{'='*60}")
        print(f"Target images: {num_images_to_save}")
        print(f"Output directory: {output_dir}")
        print(f"Target speed: {target_speed} km/h")
        print(f"Synchronous mode: {synchronous}")
        print(f"FPS: {fps}")
        print(f"{'='*60}\n")

    def _connect_to_carla(self):
        """Connect to CARLA server"""
        print(f"Connecting to CARLA at {self.host}:{self.port}...")
        self.client = carla.Client(self.host, self.port)
        self.client.set_timeout(10.0)
        print("✓ Connected to CARLA")

    def _setup_world(self):
        """Setup CARLA world"""
        print("Setting up world...")
        self.world = World(self.client)
        
        if self.synchronous:
            settings = self.world.get_settings()
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = 1.0 / self.fps
            self.world.apply_settings(settings)
        
        # Keep the HUD updated on server ticks
        self.world.on_tick(self.hud.on_world_tick)
        
        print(f"✓ World ready (Map: {self.world.map.name})")

    def _setup_vehicle(self):
        """Spawn vehicle at a valid route start location"""
        print("Spawning vehicle...")
        spawn_points = self.world.map.get_spawn_points()
        self.spawn_point = random.choice(spawn_points)
        self.route_start = self.spawn_point.location

        self.vehicle = Vehicle(
            self.world,
            self.spawn_point,
            on_collision_fn=lambda e: self._on_collision(e),
            on_invasion_fn=lambda e: self._on_invasion(e)
        )
        print(f"✓ Vehicle spawned at route start {self.spawn_point.location}")
        self.vehicle_collision = False

    def _setup_cameras(self):
        """Setup RGB and segmentation cameras"""
        print("Setting up cameras...")
        
        self.dashcam_rgb = Camera(
            self.world, self.viewer_res[0], self.viewer_res[1],
            transform=camera_transforms["dashboard"],
            attach_to=self.vehicle,
            on_recv_image=lambda e: self._set_observation_image("rgb", e),
            sensor_tick=0.0 if self.synchronous else 1.0/self.fps,
            camera_type="sensor.camera.rgb",
            color_converter=carla.ColorConverter.Raw
        )
        
        self.dashcam_seg = Camera(
            self.world, self.viewer_res[0], self.viewer_res[1],
            transform=camera_transforms["dashboard"],
            attach_to=self.vehicle,
            on_recv_image=lambda e: self._set_observation_image("segmentation", e),
            sensor_tick=0.0 if self.synchronous else 1.0/self.fps,
            camera_type="sensor.camera.semantic_segmentation",
            color_converter=carla.ColorConverter.CityScapesPalette
        )
        
        self.viewer_camera = Camera(
            self.world, self.viewer_res[0], self.viewer_res[1],
            transform=camera_transforms["spectator"],
            attach_to=self.vehicle,
            on_recv_image=lambda e: self._set_viewer_image(e),
            sensor_tick=0.0 if self.synchronous else 1.0/self.fps,
            camera_type="sensor.camera.rgb",
            color_converter=carla.ColorConverter.Raw
        )
        
        print("✓ Cameras ready (RGB + Segmentation + Viewer)")
        
        # Small delay to ensure camera is ready
        time.sleep(1.0)

    def _setup_planners(self):
        """Setup global and local planners"""
        print("Setting up planners...")
        
        # Global Route Planner
        dao = GlobalRoutePlannerDAO(self.world.map, sampling_resolution=2.0)
        self.global_planner = GlobalRoutePlanner(dao)
        self.global_planner.setup()
        
        # Local Planner
        args_lateral = {'K_P': 1.0, 'K_D': 0.02, 'K_I': 0.0, 'dt': 1.0/20.0}
        args_longitudinal = {'K_P': 1.0, 'K_D': 0.1, 'K_I': 0.0, 'dt': 1.0/20.0}
        
        self.local_planner = LocalPlanner(
            self.vehicle,
            opt_dict={
                'target_speed': self.target_speed,
                'lateral_control_dict': args_lateral,
                'longitudinal_control_dict': args_longitudinal
            }
        )
        
        print("✓ Planners ready")

    def _plan_new_route(self):
        """Plan a new route using the vehicle's current location as start and a random destination."""
        spawn_points = self.world.map.get_spawn_points()
        
        # For the first route, start exactly at the selected spawn point.
        if self.routes_completed == 0 and hasattr(self, 'route_start'):
            start_loc = self.route_start
        else:
            start_loc = self.vehicle.get_location()

        # Project the start to the nearest road for robustness
        start_wp = self.world.map.get_waypoint(start_loc, project_to_road=True)
        
        # Choose a random destination that is not the current start
        end_spawn = random.choice(spawn_points)
        end_wp = self.world.map.get_waypoint(end_spawn.location, project_to_road=True)
        while end_wp.transform.location.distance(start_wp.transform.location) < self.min_route_distance:
            end_spawn = random.choice(spawn_points)
            end_wp = self.world.map.get_waypoint(end_spawn.location, project_to_road=True)
        
        # Compute route
        try:
            route = compute_route_waypoints(
                self.world.map, start_wp, end_wp, 
                resolution=2.0, plan=None
            )
            if not route or len(route) < 2:
                raise ValueError("Route is too short or empty")

            # Filter routes that are too short in distance
            start_loc = start_wp.transform.location
            end_loc = end_wp.transform.location
            distance = start_loc.distance(end_loc)
            if distance < self.min_route_distance:
                raise ValueError(f"Route distance too short: {distance:.1f} m")

            print(f"\n✓ Route planned: {len(route)} waypoints | distance={distance:.1f} m")
            print(f"  From: ({start_wp.transform.location.x:.1f}, {start_wp.transform.location.y:.1f})")
            print(f"  To:   ({end_wp.transform.location.x:.1f}, {end_wp.transform.location.y:.1f})")
            
            # Set route on local planner
            self.local_planner.set_global_plan(route)
            self.routes_completed += 1
            self.last_saved_location = None
            self.frame_since_last_save = 0
            
        except Exception as e:
            print(f"! Failed to plan route: {e}")
            print("  Will try again on next iteration...")
            return
    def _on_collision(self, event):
        """Callback for collision events"""
        self.vehicle_collision = True

    def _on_invasion(self, event):
        """Callback for lane invasion events"""
        pass

    def _set_observation_image(self, name, image):
        self.observation_buffer[name] = image

    def _set_viewer_image(self, image):
        self.viewer_image_buffer = image

    def _get_observation(self, name):
        while self.observation_buffer[name] is None:
            pass
        obs = self.observation_buffer[name].copy()
        self.observation_buffer[name] = None
        return obs

    def _get_viewer_image(self):
        while self.viewer_image_buffer is None:
            pass
        image = self.viewer_image_buffer.copy()
        self.viewer_image_buffer = None
        return image

    def _save_images(self):
        """Save current RGB and segmentation images"""
        if self.observation["rgb"] is None or self.observation["segmentation"] is None:
            return False
        
        try:
            rgb_path = os.path.join(self.rgb_dir, f"frame_{self.images_collected:06d}.png")
            Image.fromarray(self.observation["rgb"]).save(rgb_path)
            seg_path = os.path.join(self.segmentation_dir, f"frame_{self.images_collected:06d}.png")
            Image.fromarray(self.observation["segmentation"]).save(seg_path)
            self.images_collected += 1
            return True
        except Exception as e:
            print(f"! Error saving images: {e}")
            return False

    def _is_stuck(self):
        """Check if vehicle is stuck (very low speed for extended period)"""
        velocity = self.vehicle.get_velocity()
        speed = np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
        return speed < 0.1  # m/s

    def _tick_world(self):
        """Advance simulation by one tick"""
        if self.synchronous:
            self.clock.tick()
            self.world.tick()
            try:
                self.world.wait_for_tick(seconds=1.0/self.fps)
            except RuntimeError:
                pass
        else:
            self.clock.tick(self.fps)
            time.sleep(1.0/self.fps)

        # Update HUD state regularly
        self.hud.tick(self.world, self.clock)

    def collect_data(self):
        """Main data collection loop"""
        print("\n" + "="*60)
        print("Starting data collection...")
        print("="*60 + "\n")
        
        try:
            stuck_counter = 0
            waypoint_counter = 0
            
            # Initial route
            self._plan_new_route()
            
            while self.images_collected < self.num_images_to_save and not self.done:
                # Process pygame events so window remains responsive
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.done = True
                    elif event.type == KEYDOWN and event.key == K_ESCAPE:
                        self.done = True

                if self.done:
                    break

                # Check if current route is actually finished
                if self.local_planner.is_global_plan_complete():
                    print(f"\n[{self.images_collected}/{self.num_images_to_save}] "
                          f"Route completed. Planning new route...")
                    self._plan_new_route()
                
                # Get control from local planner
                try:
                    control = self.local_planner.run_step(debug=False)
                except Exception as e:
                    print(f"! Error in local planner: {e}")
                    print("  Planning new route...")
                    self._plan_new_route()
                    continue
                
                # Apply control to vehicle
                self.vehicle.apply_control(control)
                self.vehicle.control = control
                
                # Tick the world
                self._tick_world()
                
                # Update observations each step
                self.observation["rgb"] = self._get_observation("rgb")
                self.observation["segmentation"] = self._get_observation("segmentation")
                self.viewer_image = self._get_viewer_image()

                self.frame_since_last_save += 1
                save_due_to_frame = self.frame_since_last_save >= self.frame_skip
                save_due_to_distance = False
                current_location = self.vehicle.get_location()
                if self.last_saved_location is None:
                    save_due_to_distance = True
                else:
                    save_due_to_distance = current_location.distance(self.last_saved_location) >= self.min_save_distance

                if save_due_to_frame and save_due_to_distance:
                    if self._save_images():
                        waypoint_counter += 1
                        stuck_counter = 0
                        self.last_saved_location = current_location
                        self.frame_since_last_save = 0
                    else:
                        stuck_counter += 1
                else:
                    # still count as a frame but do not save images yet
                    pass

                if self.images_collected % 20 == 0:
                    print(f"DEBUG control: steer={control.steer:.3f} throttle={control.throttle:.3f} brake={control.brake:.3f} speed={self.vehicle.get_speed():.3f} m/s")
                
                # Check for stuck vehicle
                if self._is_stuck():
                    stuck_counter += 1
                    if stuck_counter > 100:  # ~3 seconds at 30 FPS
                        print(f"! Vehicle stuck. Planning new route...")
                        self._plan_new_route()
                        stuck_counter = 0
                else:
                    stuck_counter = max(0, stuck_counter - 1)
                
                # Render HUD and viewer
                self._render()

                # Print progress every 100 images
                if self.images_collected % 100 == 0:
                    print(f"Progress: {self.images_collected}/{self.num_images_to_save} images | "
                          f"Routes: {self.routes_completed} | Speed: {self.vehicle.get_speed():.2f} m/s")
                
                # Reset collision flag
                self.vehicle_collision = False
            
            print("\n" + "="*60)
            print(f"✓ Data collection complete!")
            print(f"  Total images saved: {self.images_collected}")
            print(f"  Total routes completed: {self.routes_completed}")
            print(f"  RGB dir: {self.rgb_dir}")
            print(f"  Segmentation dir: {self.segmentation_dir}")
            print("="*60 + "\n")
            
        except KeyboardInterrupt:
            print("\n\n! Data collection interrupted by user")
            print(f"  Images collected so far: {self.images_collected}")
        except Exception as e:
            print(f"\n! Error during data collection: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.cleanup()

    def _render(self):
        if self.viewer_image is None:
            return

        self.display.blit(pygame.surfarray.make_surface(self.viewer_image.swapaxes(0, 1)), (0, 0))

        if self.observation["rgb"] is not None:
            obs_h, obs_w = self.observation["rgb"].shape[:2]
            pos = (self.viewer_res[0] - obs_w - 10, 10)
            self.display.blit(pygame.surfarray.make_surface(self.observation["rgb"].swapaxes(0, 1)), pos)
        if self.observation["segmentation"] is not None:
            obs_h, obs_w = self.observation["segmentation"].shape[:2]
            pos = (self.viewer_res[0] - obs_w - 10, 20 + obs_h)
            self.display.blit(pygame.surfarray.make_surface(self.observation["segmentation"].swapaxes(0, 1)), pos)

        self.extra_info.extend([
            f"Images: {self.images_collected}/{self.num_images_to_save}",
            f"Speed: {self.vehicle.get_speed():.2f} m/s",
            f"Target speed: {self.target_speed:.1f} km/h",
            f"Routes: {self.routes_completed}",
            f"Throttle: {self.vehicle.control.throttle:.2f}",
            f"Steer: {self.vehicle.control.steer:.2f}"
        ])
        self.hud.render(self.display, extra_info=self.extra_info)
        self.extra_info = []
        pygame.display.flip()

        # Process pygame events and allow window close
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.done = True
            elif event.type == KEYDOWN and event.key == K_ESCAPE:
                self.done = True

    def cleanup(self):
        """Clean up and destroy all actors"""
        print("Cleaning up...")
        
        if hasattr(self, 'dashcam_rgb') and self.dashcam_rgb:
            self.dashcam_rgb.destroy()
        if hasattr(self, 'dashcam_seg') and self.dashcam_seg:
            self.dashcam_seg.destroy()
        if hasattr(self, 'viewer_camera') and self.viewer_camera:
            self.viewer_camera.destroy()
        if self.vehicle:
            self.vehicle.destroy()
        if self.world:
            self.world.destroy()
        pygame.quit()
        
        print("✓ Cleanup complete")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Autonomous CARLA data collector")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="CARLA host IP")
    parser.add_argument("--port", type=int, default=2000, help="CARLA port")
    parser.add_argument("--num_images", type=int, default=10000, help="Number of images to collect")
    parser.add_argument("--output_dir", type=str, default="autopilot_data", help="Output directory")
    parser.add_argument("--synchronous", type=int, default=1, help="Run in synchronous mode (0/1)")
    parser.add_argument("--fps", type=int, default=30, help="FPS to run at")
    parser.add_argument("--target_speed", type=float, default=20, help="Target speed in km/h")
    parser.add_argument("--frame_skip", type=int, default=2, help="Save one image every N frames")
    parser.add_argument("--min_save_distance", type=float, default=1.0, help="Minimum distance in meters between saved frames")
    parser.add_argument("--min_route_distance", type=float, default=100.0, help="Minimum route distance in meters")
    
    args = parser.parse_args()
    
    collector = AutopilotDataCollector(
        host=args.host,
        port=args.port,
        num_images_to_save=args.num_images,
        output_dir=args.output_dir,
        synchronous=bool(args.synchronous),
        fps=args.fps,
        target_speed=args.target_speed,
        frame_skip=args.frame_skip,
        min_save_distance=args.min_save_distance,
        min_route_distance=args.min_route_distance
    )
    
    collector.collect_data()


if __name__ == "__main__":
    main()
