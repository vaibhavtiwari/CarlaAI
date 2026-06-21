import carla
import random
import time
import collections
import math
import numpy as np
import weakref
import pygame

def print_transform(transform):
    print("Location(x={:.2f}, y={:.2f}, z={:.2f}) Rotation(pitch={:.2f}, yaw={:.2f}, roll={:.2f})".format(
            transform.location.x,
            transform.location.y,
            transform.location.z,
            transform.rotation.pitch,
            transform.rotation.yaw,
            transform.rotation.roll
        )
    )

def get_actor_display_name(actor, truncate=250):
    name = " ".join(actor.type_id.replace("_", ".").title().split(".")[1:])
    return (name[:truncate-1] + u"\u2026") if len(name) > truncate else name

def angle_diff(v0, v1):
    """ Calculates the signed angle difference (-pi, pi] between 2D vector v0 and v1 """
    angle = np.arctan2(v1[1], v1[0]) - np.arctan2(v0[1], v0[0])
    if angle > np.pi: angle -= 2 * np.pi
    elif angle <= -np.pi: angle += 2 * np.pi
    return angle

def distance_to_line(A, B, p):
    num   = np.linalg.norm(np.cross(B - A, A - p))
    denom = np.linalg.norm(B - A)
    if np.isclose(denom, 0):
        return np.linalg.norm(p - A)
    return num / denom

def vector(v):
    """ Turn carla Location/Vector3D/Rotation to np.array """
    if isinstance(v, carla.Location) or isinstance(v, carla.Vector3D):
        return np.array([v.x, v.y, v.z])
    elif isinstance(v, carla.Rotation):
        return np.array([v.pitch, v.yaw, v.roll])

camera_transforms = {
    "spectator": carla.Transform(carla.Location(x=-5.5, z=2.8), carla.Rotation(pitch=-15)),
    "dashboard": carla.Transform(carla.Location(x=1.6, z=1.7))
}


def build_road_overlay_segments(world_map, resolution=4.0):
    """Build lightweight road polylines for 2D map rendering."""
    segments = []
    seen = set()
    for waypoint in world_map.generate_waypoints(resolution):
        next_waypoints = waypoint.next(resolution)
        if not next_waypoints:
            continue
        for next_waypoint in next_waypoints:
            p0 = waypoint.transform.location
            p1 = next_waypoint.transform.location
            key = tuple(
                sorted(
                    (
                        (round(p0.x, 1), round(p0.y, 1), round(p0.z, 1)),
                        (round(p1.x, 1), round(p1.y, 1), round(p1.z, 1)),
                    )
                )
            )
            if key in seen:
                continue
            seen.add(key)
            segments.append((p0, p1, 0.5 * (waypoint.lane_width + next_waypoint.lane_width)))
    return segments

def draw_route_overlay(
    surface,
    route_waypoints,
    road_segments=None,
    current_waypoint_index=None,
    step=3,
    padding=18,
    vehicle_location=None,
    vehicle_forward=None,
    line_color=(120, 150, 210),
    point_color=(110, 130, 170),
    background_color=(18, 20, 24),
):
    if not route_waypoints:
        surface.fill(background_color)
        return

    width, height = surface.get_size()
    surface.fill(background_color)
    pygame.draw.rect(surface, (28, 32, 38), surface.get_rect(), border_radius=10)
    inner_rect = pygame.Rect(8, 8, width - 16, height - 16)
    pygame.draw.rect(surface, (20, 23, 28), inner_rect, border_radius=8)

    grid_color = (34, 39, 46)
    grid_spacing = 36
    for x in range(inner_rect.left, inner_rect.right, grid_spacing):
        pygame.draw.line(surface, grid_color, (x, inner_rect.top), (x, inner_rect.bottom), 1)
    for y in range(inner_rect.top, inner_rect.bottom, grid_spacing):
        pygame.draw.line(surface, grid_color, (inner_rect.left, y), (inner_rect.right, y), 1)

    locations = [wp.transform.location for wp, _ in route_waypoints]
    if road_segments:
        for p0, p1, _ in road_segments:
            locations.append(p0)
            locations.append(p1)
    xs = [location.x for location in locations]
    ys = [location.y for location in locations]
    if vehicle_location is not None:
        xs.append(vehicle_location.x)
        ys.append(vehicle_location.y)
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    span_x = max(1.0, max_x - min_x)
    span_y = max(1.0, max_y - min_y)
    usable_width = max(1.0, inner_rect.width - padding * 2)
    usable_height = max(1.0, inner_rect.height - padding * 2 - 24)
    scale = min(usable_width / span_x, usable_height / span_y)
    offset_x = inner_rect.left + (inner_rect.width - span_x * scale) * 0.5
    offset_y = inner_rect.top + (inner_rect.height - 24 - span_y * scale) * 0.5

    def project(location):
        px = offset_x + (location.x - min_x) * scale
        py = inner_rect.bottom - 24 - (offset_y - inner_rect.top + (location.y - min_y) * scale)
        return int(round(px)), int(round(py))

    if road_segments:
        for p0, p1, lane_width in road_segments:
            road_width = max(2, int(round(max(6.0, lane_width * scale * 0.55))))
            pygame.draw.line(
                surface,
                (92, 96, 102),
                project(p0),
                project(p1),
                width=road_width,
            )
            lane_mark_width = max(1, road_width // 5)
            pygame.draw.line(
                surface,
                (126, 129, 134),
                project(p0),
                project(p1),
                width=lane_mark_width,
            )

    last_index = len(route_waypoints) - 1
    draw_step = max(1, step)
    for index in range(0, last_index, draw_step):
        w0 = route_waypoints[index][0].transform.location
        w1 = route_waypoints[min(index + draw_step, last_index)][0].transform.location
        progress = 0.0 if last_index == 0 else index / last_index
        segment_color = (
            int(line_color[0] * (1.0 - progress) + 255 * progress * 0.35),
            int(line_color[1] * (1.0 - progress) + 205 * progress * 0.25),
            int(line_color[2] * (1.0 - progress) + 90 * progress * 0.15),
        )
        pygame.draw.line(
            surface,
            (52, 58, 68),
            project(w0),
            project(w1),
            width=5,
        )
        pygame.draw.line(
            surface,
            segment_color,
            project(w0),
            project(w1),
            width=2,
        )

    for index in range(0, len(route_waypoints), draw_step):
        waypoint = route_waypoints[index][0].transform.location
        color = point_color
        size = 2
        if index == 0:
            color = (80, 200, 120)
            size = 5
        elif index == last_index:
            color = (220, 90, 90)
            size = 5
        pygame.draw.circle(surface, color, project(waypoint), size)

    if current_waypoint_index is not None:
        waypoint = route_waypoints[current_waypoint_index % len(route_waypoints)][0].transform.location
        marker_pos = project(waypoint)
        pygame.draw.circle(surface, (90, 230, 230), marker_pos, 7, 2)
        pygame.draw.circle(surface, (90, 230, 230), marker_pos, 2)

    if vehicle_location is not None:
        vehicle_pos = project(vehicle_location)
        pygame.draw.circle(surface, (16, 18, 24), vehicle_pos, 9)
        pygame.draw.circle(surface, (245, 245, 245), vehicle_pos, 6)
        pygame.draw.circle(surface, (255, 196, 87), vehicle_pos, 3)
        if vehicle_forward is not None:
            norm = np.linalg.norm(vehicle_forward[:2])
            if norm > 1e-6:
                forward_world = carla.Location(
                    x=vehicle_location.x + float(vehicle_forward[0] / norm) * 6.0,
                    y=vehicle_location.y + float(vehicle_forward[1] / norm) * 6.0,
                    z=vehicle_location.z,
                )
                tip = project(forward_world)
                direction = np.array([tip[0] - vehicle_pos[0], tip[1] - vehicle_pos[1]], dtype=np.float32)
                direction_norm = np.linalg.norm(direction)
                if direction_norm > 1e-6:
                    direction /= direction_norm
                    perp = np.array([-direction[1], direction[0]], dtype=np.float32)
                    left = (
                        int(round(vehicle_pos[0] - direction[0] * 3 + perp[0] * 5)),
                        int(round(vehicle_pos[1] - direction[1] * 3 + perp[1] * 5)),
                    )
                    right = (
                        int(round(vehicle_pos[0] - direction[0] * 3 - perp[0] * 5)),
                        int(round(vehicle_pos[1] - direction[1] * 3 - perp[1] * 5)),
                    )
                    pygame.draw.polygon(surface, (255, 210, 90), [tip, left, right])

    start_pos = project(route_waypoints[0][0].transform.location)
    end_pos = project(route_waypoints[last_index][0].transform.location)
    pygame.draw.circle(surface, (80, 200, 120), start_pos, 8, 2)
    pygame.draw.circle(surface, (220, 90, 90), end_pos, 8, 2)

    legend_y = inner_rect.bottom - 16
    pygame.draw.line(surface, (80, 200, 120), (inner_rect.left + 12, legend_y), (inner_rect.left + 34, legend_y), 3)
    pygame.draw.line(surface, (90, 230, 230), (inner_rect.left + 74, legend_y), (inner_rect.left + 96, legend_y), 3)
    pygame.draw.line(surface, (245, 245, 245), (inner_rect.left + 140, legend_y), (inner_rect.left + 162, legend_y), 3)
    pygame.draw.line(surface, (220, 90, 90), (inner_rect.left + 192, legend_y), (inner_rect.left + 214, legend_y), 3)

#===============================================================================
# CarlaActorBase
#===============================================================================

class CarlaActorBase(object):
    def __init__(self, world, actor, managed=True):
        self.world = world
        self.actor = actor
        self.managed = managed
        self.world.actor_list.append(self)
        self.destroyed = False

    def destroy(self):
        if self.destroyed:
            raise Exception("Actor already destroyed.")
        else:
            print("Destroying ", self, "...")
            if self.managed:
                self.actor.destroy()
            self.world.actor_list.remove(self)
            self.destroyed = True

    def get_carla_actor(self):
        return self.actor

    def tick(self):
        pass

    def __getattr__(self, name):
        """Relay missing methods to underlying carla actor"""
        return getattr(self.actor, name)

#===============================================================================
# CollisionSensor
#===============================================================================

class CollisionSensor(CarlaActorBase):
    def __init__(self, world, vehicle, on_collision_fn):
        self.on_collision_fn = on_collision_fn

        # Collision history
        self.history = []

        # Setup sensor blueprint
        bp = world.get_blueprint_library().find("sensor.other.collision")

        # Create and setup sensor
        weak_self = weakref.ref(self)
        actor = world.spawn_actor(bp, carla.Transform(), attach_to=vehicle.get_carla_actor())
        actor.listen(lambda event: CollisionSensor.on_collision(weak_self, event))

        super().__init__(world, actor)

    @staticmethod
    def on_collision(weak_self, event):
        self = weak_self()
        if not self:
            return

        # Call on_collision_fn
        if callable(self.on_collision_fn):
            self.on_collision_fn(event)


#===============================================================================
# LaneInvasionSensor
#===============================================================================

class LaneInvasionSensor(CarlaActorBase):
    def __init__(self, world, vehicle, on_invasion_fn):
        self.on_invasion_fn = on_invasion_fn

        # Setup sensor blueprint
        bp = world.get_blueprint_library().find("sensor.other.lane_invasion")

        # Create sensor
        weak_self = weakref.ref(self)
        actor = world.spawn_actor(bp, carla.Transform(), attach_to=vehicle.get_carla_actor())
        actor.listen(lambda event: LaneInvasionSensor.on_invasion(weak_self, event))

        super().__init__(world, actor)

    @staticmethod
    def on_invasion(weak_self, event):
        self = weak_self()
        if not self:
            return

        # Call on_invasion_fn
        if callable(self.on_invasion_fn):
            self.on_invasion_fn(event)

#===============================================================================
# Camera
#===============================================================================

class Camera(CarlaActorBase):
    def __init__(self, world, width, height, transform=carla.Transform(),
                 sensor_tick=0.0, attach_to=None, on_recv_image=None,
                 camera_type="sensor.camera.rgb", color_converter=carla.ColorConverter.Raw):
        self.on_recv_image = on_recv_image
        self.color_converter = color_converter

        # Setup camera blueprint
        camera_bp = world.get_blueprint_library().find(camera_type)
        camera_bp.set_attribute("image_size_x", str(width))
        camera_bp.set_attribute("image_size_y", str(height))
        camera_bp.set_attribute("sensor_tick", str(sensor_tick))

        # Create and setup camera actor
        weak_self = weakref.ref(self)
        attach_actor = attach_to.get_carla_actor() if attach_to is not None else None
        actor = world.spawn_actor(camera_bp, transform, attach_to=attach_actor)
        actor.listen(lambda image: Camera.process_camera_input(weak_self, image))
        print("Spawned actor \"{}\"".format(actor.type_id))

        super().__init__(world, actor)
    
    @staticmethod
    def process_camera_input(weak_self, image):
        self = weak_self()
        if not self:
            return
        if callable(self.on_recv_image):
            image.convert(self.color_converter)
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            self.on_recv_image(array)

    def destroy(self):
        super().destroy()

#===============================================================================
# Vehicle
#===============================================================================

class Vehicle(CarlaActorBase):
    def __init__(self, world, transform=carla.Transform(),
                 on_collision_fn=None, on_invasion_fn=None,
                 vehicle_type="vehicle.lincoln.mkz_2020"):
        # Setup vehicle blueprint
        vehicle_bp = world.get_blueprint_library().find(vehicle_type)
        color = vehicle_bp.get_attribute("color").recommended_values[0]
        vehicle_bp.set_attribute("color", color)

        # Create vehicle actor. In busy worlds, the requested spawn point may
        # already be occupied, so fall back to alternative map spawn points.
        actor = world.try_spawn_actor(vehicle_bp, transform)
        if actor is None:
            spawn_points = list(world.map.get_spawn_points())
            random.shuffle(spawn_points)
            for fallback_transform in spawn_points:
                actor = world.try_spawn_actor(vehicle_bp, fallback_transform)
                if actor is not None:
                    print(
                        "Requested spawn was occupied; spawned vehicle at fallback "
                        f"location ({fallback_transform.location.x:.1f}, "
                        f"{fallback_transform.location.y:.1f}, "
                        f"{fallback_transform.location.z:.1f})"
                    )
                    break
        if actor is None:
            raise RuntimeError(
                "Failed to spawn vehicle: requested transform was occupied and "
                "no free fallback spawn points were available."
            )
        print("Spawned actor \"{}\"".format(actor.type_id))
            
        super().__init__(world, actor)

        # Maintain vehicle control
        self.control = carla.VehicleControl()

        if callable(on_collision_fn):
            self.collision_sensor = CollisionSensor(world, self, on_collision_fn=on_collision_fn)
        if callable(on_invasion_fn):
            self.lane_sensor = LaneInvasionSensor(world, self, on_invasion_fn=on_invasion_fn)

    @classmethod
    def from_existing_actor(cls, world, actor):
        self = cls.__new__(cls)
        super(Vehicle, self).__init__(world, actor, managed=False)
        self.control = actor.get_control()
        return self

    def tick(self):
        self.actor.apply_control(self.control)

    def get_speed(self):
        velocity = self.get_velocity()
        return np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)

    def get_closest_waypoint(self):
        return self.world.map.get_waypoint(self.get_transform().location, project_to_road=True)

#===============================================================================
# World
#===============================================================================

class World():
    def __init__(self, client):
        self.world = client.get_world()
        self.map = self.get_map()
        self.actor_list = []

    def tick(self):
        for actor in list(self.actor_list):
            actor.tick()
        self.world.tick()

    def destroy(self):
        print("Destroying all spawned actors")
        for actor in list(self.actor_list):
            actor.destroy()

    def get_carla_world(self):
        return self.world

    def __getattr__(self, name):
        """Relay missing methods to underlying carla object"""
        return getattr(self.world, name)
