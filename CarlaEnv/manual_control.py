import pygame
from pygame.locals import (
    KEYDOWN,
    QUIT,
    K_COMMA,
    K_DOWN,
    K_ESCAPE,
    K_LEFT,
    K_PERIOD,
    K_RIGHT,
    K_SPACE,
    K_UP,
    K_a,
    K_d,
    K_m,
    K_q,
    K_s,
    K_w,
)

import carla

try:
    from .hud import HUD
    from .wrappers import Camera, Vehicle, World, camera_transforms, get_actor_display_name
except ImportError:
    from hud import HUD
    from wrappers import Camera, Vehicle, World, camera_transforms, get_actor_display_name


def _handle_manual_control_event(event, control):
    if event.type != KEYDOWN:
        return False
    if event.key == K_ESCAPE:
        return True
    if event.key == K_q:
        control.reverse = not control.reverse
        control.gear = -1 if control.reverse else 1
    elif event.key == K_m:
        control.manual_gear_shift = not control.manual_gear_shift
        if control.manual_gear_shift:
            control.gear = -1 if control.reverse else max(1, control.gear)
    elif event.key == K_COMMA and control.manual_gear_shift:
        control.gear = max(-1, control.gear - 1)
        control.reverse = control.gear < 0
    elif event.key == K_PERIOD and control.manual_gear_shift:
        control.gear = min(5, control.gear + 1)
        control.reverse = control.gear < 0
    return False


def _apply_manual_key_state(control):
    keys = pygame.key.get_pressed()
    control.steer = -0.5 if (keys[K_LEFT] or keys[K_a]) else 0.5 if (keys[K_RIGHT] or keys[K_d]) else 0.0
    control.throttle = 1.0 if (keys[K_UP] or keys[K_w]) else 0.0
    control.brake = 1.0 if (keys[K_DOWN] or keys[K_s]) else 0.0
    control.hand_brake = bool(keys[K_SPACE])
    if not control.manual_gear_shift:
        control.gear = -1 if control.reverse else 1


def run_env_manual_drive(
    env_class,
    reward_fn,
    host,
    port,
    viewer_res,
    obs_res,
    synchronous,
    fps,
    start_carla,
    show_waypoints,
    is_training,
):
    env = env_class(
        host=host,
        port=port,
        viewer_res=viewer_res,
        obs_res=obs_res,
        reward_fn=reward_fn,
        synchronous=synchronous,
        fps=fps,
        start_carla=start_carla,
        show_waypoints=show_waypoints,
    )

    closed = False
    try:
        while not closed:
            env.reset(is_training=is_training)
            env.vehicle.control.manual_gear_shift = False
            env.vehicle.control.gear = 1
            env.vehicle.control.reverse = False
            while not closed:
                for event in pygame.event.get():
                    if event.type == QUIT or _handle_manual_control_event(event, env.vehicle.control):
                        closed = True
                        break

                _apply_manual_key_state(env.vehicle.control)
                _, _, done, info = env.step(None)
                if info["closed"]:
                    closed = True
                if not closed:
                    env.render()
                if done:
                    break
    finally:
        env.close()
        pygame.quit()


def control_existing_vehicle(
    host,
    port,
    viewer_res,
    obs_res,
    synchronous,
    fps,
    actor_id,
):
    pygame.init()
    pygame.font.init()
    display = pygame.display.set_mode(viewer_res, pygame.HWSURFACE | pygame.DOUBLEBUF)
    clock = pygame.time.Clock()

    client = carla.Client(host, port)
    client.set_timeout(10.0)
    world = World(client)
    actor = world.get_actor(actor_id)
    if actor is None:
        raise RuntimeError("Selected vehicle no longer exists.")

    vehicle = Vehicle.from_existing_actor(world, actor)
    hud = HUD(viewer_res[0], viewer_res[1])
    hud.set_vehicle(vehicle)
    world.on_tick(hud.on_world_tick)
    observation_buffer = {"obs": None, "view": None}

    def set_obs(image):
        observation_buffer["obs"] = image

    def set_view(image):
        observation_buffer["view"] = image

    dashcam = Camera(
        world,
        obs_res[0],
        obs_res[1],
        transform=camera_transforms["dashboard"],
        attach_to=vehicle,
        on_recv_image=set_obs,
        sensor_tick=0.0 if synchronous else 1.0 / fps,
    )
    spectator = Camera(
        world,
        viewer_res[0],
        viewer_res[1],
        transform=camera_transforms["spectator"],
        attach_to=vehicle,
        on_recv_image=set_view,
        sensor_tick=0.0 if synchronous else 1.0 / fps,
    )

    original_settings = world.get_settings()
    if synchronous:
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 1.0 / fps
        world.apply_settings(settings)

    vehicle.control = vehicle.get_control()
    closed = False
    try:
        while not closed:
            if not synchronous:
                clock.tick_busy_loop(fps)
            for event in pygame.event.get():
                if event.type == QUIT or _handle_manual_control_event(event, vehicle.control):
                    closed = True
                    break

            _apply_manual_key_state(vehicle.control)
            hud.tick(world, clock)
            world.tick()
            if synchronous:
                clock.tick()
                world.wait_for_tick(seconds=1.0 / fps + 0.1)
            if observation_buffer["view"] is not None:
                display.blit(pygame.surfarray.make_surface(observation_buffer["view"].swapaxes(0, 1)), (0, 0))
                if observation_buffer["obs"] is not None:
                    view_h, view_w = observation_buffer["view"].shape[:2]
                    obs_h, obs_w = observation_buffer["obs"].shape[:2]
                    display.blit(
                        pygame.surfarray.make_surface(observation_buffer["obs"].swapaxes(0, 1)),
                        (view_w - obs_w - 10, 10),
                    )
                hud.render(display, extra_info=[f"Controlling existing actor: {actor.id}", get_actor_display_name(actor)])
                pygame.display.flip()
    finally:
        try:
            dashcam.destroy()
        except Exception:
            pass
        try:
            spectator.destroy()
        except Exception:
            pass
        try:
            world.apply_settings(original_settings)
        except Exception:
            pass
        world.destroy()
        pygame.quit()
