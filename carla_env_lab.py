import os
import queue
import signal
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from tkinter import ttk

from CarlaEnv.config import load_json_config


REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_LAB_CONFIG_DIR = os.path.join(REPO_ROOT, "config")


class CarlaEnvLab:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("CARLA Env Lab")
        self.root.geometry("980x760")

        self.process = None
        self.log_queue = queue.Queue()
        self.stopping = False

        self.test_var = tk.StringVar(value="Manual Drive")
        self.manual_scenario_var = tk.StringVar(value="Lap")
        self.host_var = tk.StringVar(value="127.0.0.1")
        self.port_var = tk.StringVar(value="2000")
        self.fps_var = tk.StringVar(value="30")
        self.viewer_res_var = tk.StringVar(value="1280x720")
        self.obs_res_var = tk.StringVar(value="160x80")
        self.num_images_var = tk.StringVar(value="200")
        self.output_dir_var = tk.StringVar(value="images/lab_run")
        self.target_speed_var = tk.StringVar(value="20")
        self.start_carla_var = tk.BooleanVar(value=False)
        self.sync_var = tk.BooleanVar(value=True)
        self.show_waypoints_var = tk.BooleanVar(value=True)
        self.selected_vehicle_var = tk.StringVar(value="")
        self.vehicle_choices = []
        self.status_var = tk.StringVar(value="Ready")

        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.after(100, self._drain_logs)

    def _build_ui(self):
        frame = ttk.Frame(self.root, padding=12)
        frame.pack(fill=tk.BOTH, expand=True)

        title = ttk.Label(
            frame,
            text="CARLA Env Lab",
            font=("TkDefaultFont", 15, "bold"),
        )
        title.pack(anchor="w")

        subtitle = ttk.Label(
            frame,
            text="Run smoke tests for controls, sensors, routes, and observation/state space.",
        )
        subtitle.pack(anchor="w", pady=(0, 10))

        status_row = ttk.Frame(frame)
        status_row.pack(fill=tk.X, pady=(0, 8))
        ttk.Label(status_row, text="Status:").pack(side=tk.LEFT)
        ttk.Label(status_row, textvariable=self.status_var).pack(side=tk.LEFT, padx=(6, 0))

        config = ttk.LabelFrame(frame, text="Test Setup", padding=10)
        config.pack(fill=tk.X)

        self._add_labeled_entry(
            config,
            row=0,
            label="Test",
            widget=ttk.Combobox(
                config,
                textvariable=self.test_var,
                state="readonly",
                values=[
                    "Manual Drive",
                    "Manual Sensor Collector",
                    "Autopilot Route Collector",
                    "PID Controller Baseline",
                    "Lap Env Probe",
                    "Route Env Probe",
                ],
                width=28,
            ),
        )
        self._add_labeled_entry(
            config,
            0,
            "Scenario",
            ttk.Combobox(
                config,
                textvariable=self.manual_scenario_var,
                state="readonly",
                values=["Lap", "Route"],
                width=12,
            ),
            2,
        )
        self._add_labeled_entry(config, 0, "Host", ttk.Entry(config, textvariable=self.host_var, width=18), 4)
        self._add_labeled_entry(config, 0, "Port", ttk.Entry(config, textvariable=self.port_var, width=10), 6)

        self._add_labeled_entry(config, 1, "Viewer Res", ttk.Entry(config, textvariable=self.viewer_res_var, width=18))
        self._add_labeled_entry(config, 1, "Obs Res", ttk.Entry(config, textvariable=self.obs_res_var, width=18), 2)
        self._add_labeled_entry(config, 1, "FPS", ttk.Entry(config, textvariable=self.fps_var, width=10), 4)

        self._add_labeled_entry(config, 2, "Num Images", ttk.Entry(config, textvariable=self.num_images_var, width=18))
        self._add_labeled_entry(config, 2, "Output Dir", ttk.Entry(config, textvariable=self.output_dir_var, width=30), 2)
        self._add_labeled_entry(config, 2, "Target Speed", ttk.Entry(config, textvariable=self.target_speed_var, width=10), 4)

        ttk.Checkbutton(config, text="Start CARLA Automatically", variable=self.start_carla_var).grid(
            row=3, column=0, columnspan=2, sticky="w", pady=(8, 0)
        )
        ttk.Checkbutton(config, text="Synchronous Mode", variable=self.sync_var).grid(
            row=3, column=2, columnspan=2, sticky="w", pady=(8, 0)
        )
        ttk.Checkbutton(config, text="Show Waypoints", variable=self.show_waypoints_var).grid(
            row=3, column=4, columnspan=2, sticky="w", pady=(8, 0)
        )

        help_box = ttk.LabelFrame(frame, text="What Each Test Does", padding=10)
        help_box.pack(fill=tk.X, pady=(10, 0))

        help_text = (
            "Manual Drive: opens the pygame env window so you can drive with WASD/arrow keys in Lap or Route mode.\n"
            "Manual Sensor Collector: shows viewer + RGB + segmentation feeds and can save images.\n"
            "Autopilot Route Collector: runs route planning and automatic control, with live sensor previews.\n"
            "PID Controller Baseline: runs the classical local-planner PID controller on the route environment.\n"
            "Lap/Route Probe: creates the env, prints action/observation/state shapes, resets once, and steps once."
        )
        ttk.Label(help_box, text=help_text, justify=tk.LEFT).pack(anchor="w")

        controls = ttk.Frame(frame)
        controls.pack(fill=tk.X, pady=(10, 0))
        self.run_button = ttk.Button(controls, text="Run Selected Test", command=self._run_selected_test)
        self.run_button.pack(side=tk.LEFT)
        ttk.Button(controls, text="Load Setup", command=self._load_setup).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(controls, text="Save Setup", command=self._save_setup).pack(side=tk.LEFT, padx=(8, 0))
        self.stop_button = ttk.Button(controls, text="Stop Running Test", command=self._stop_process, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=(8, 0))
        self.force_stop_button = ttk.Button(
            controls,
            text="Force Kill",
            command=self._force_kill_process,
            state=tk.DISABLED,
        )
        self.force_stop_button.pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(controls, text="Clear Log", command=self._clear_log).pack(side=tk.LEFT, padx=(8, 0))

        vehicle_box = ttk.LabelFrame(frame, text="Existing Vehicles In World", padding=10)
        vehicle_box.pack(fill=tk.X, pady=(10, 0))

        vehicle_controls = ttk.Frame(vehicle_box)
        vehicle_controls.pack(fill=tk.X)

        ttk.Label(vehicle_controls, text="Vehicle").pack(side=tk.LEFT)
        self.vehicle_combo = ttk.Combobox(
            vehicle_controls,
            textvariable=self.selected_vehicle_var,
            state="readonly",
            values=[],
            width=70,
        )
        self.vehicle_combo.pack(side=tk.LEFT, padx=(8, 8), fill=tk.X, expand=True)
        ttk.Button(vehicle_controls, text="Refresh Vehicles", command=self._refresh_vehicles).pack(side=tk.LEFT)
        ttk.Button(vehicle_controls, text="Control Selected", command=self._control_selected_vehicle).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(vehicle_controls, text="Delete Selected", command=self._delete_selected_vehicle).pack(side=tk.LEFT, padx=(8, 0))

        ttk.Label(
            vehicle_box,
            text="Use this when a previous ego vehicle is still alive in the world. You can attach manual control to it or delete it.",
            justify=tk.LEFT,
        ).pack(anchor="w", pady=(8, 0))

        log_frame = ttk.LabelFrame(frame, text="Live Output", padding=10)
        log_frame.pack(fill=tk.BOTH, expand=True, pady=(10, 0))

        self.log_widget = tk.Text(log_frame, wrap=tk.WORD, state=tk.DISABLED)
        self.log_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar = ttk.Scrollbar(log_frame, command=self.log_widget.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_widget.configure(yscrollcommand=scrollbar.set)

        self._append_log("Use this launcher to validate the CARLA stack one piece at a time.\n")

    def _add_labeled_entry(self, parent, row, label, widget, column=0):
        ttk.Label(parent, text=label).grid(row=row, column=column, sticky="w", padx=(0, 8), pady=4)
        widget.grid(row=row, column=column + 1, sticky="ew", padx=(0, 16), pady=4)
        parent.grid_columnconfigure(column + 1, weight=1)

    def _build_command(self):
        python = sys.executable
        selected = self.test_var.get()
        if selected == "Manual Drive":
            env_kind = "lap" if self.manual_scenario_var.get() == "Lap" else "route"
            return self._build_manual_drive_command(env_kind)
        if selected == "Manual Sensor Collector":
            cmd = [
                python,
                "CarlaEnv/collect_data.py",
                "--host",
                self.host_var.get().strip(),
                "--port",
                self.port_var.get().strip(),
                "--fps",
                self.fps_var.get().strip(),
                "--synchronous",
                "1" if self.sync_var.get() else "0",
                "--viewer_res",
                self.viewer_res_var.get().strip(),
                "--obs_res",
                self.obs_res_var.get().strip(),
                "--num_images",
                self.num_images_var.get().strip(),
                "--output_dir",
                self.output_dir_var.get().strip(),
            ]
            if self.start_carla_var.get():
                cmd.append("-start_carla")
            return cmd
        if selected == "Autopilot Route Collector":
            cmd = [
                python,
                "CarlaEnv/data_collector_autopilot.py",
                "--host",
                self.host_var.get().strip(),
                "--port",
                self.port_var.get().strip(),
                "--viewer_res",
                self.viewer_res_var.get().strip(),
                "--obs_res",
                self.obs_res_var.get().strip(),
                "--fps",
                self.fps_var.get().strip(),
                "--synchronous",
                "1" if self.sync_var.get() else "0",
                "--num_images",
                self.num_images_var.get().strip(),
                "--output_dir",
                self.output_dir_var.get().strip(),
                "--target_speed",
                self.target_speed_var.get().strip(),
            ]
            if self.start_carla_var.get():
                cmd.append("-start_carla")
            return cmd
        if selected == "PID Controller Baseline":
            cmd = [
                python,
                "run_controller.py",
                "--host",
                self.host_var.get().strip(),
                "--port",
                self.port_var.get().strip(),
                "--viewer_res",
                self.viewer_res_var.get().strip(),
                "--obs_res",
                self.obs_res_var.get().strip(),
                "--fps",
                self.fps_var.get().strip(),
                "--synchronous",
                "1" if self.sync_var.get() else "0",
                "--show_waypoints",
                "1" if self.show_waypoints_var.get() else "0",
                "--target_speed",
                self.target_speed_var.get().strip(),
            ]
            if self.start_carla_var.get():
                cmd.append("-start_carla")
            return cmd
        if selected == "Lap Env Probe":
            return self._build_probe_command("lap")
        if selected == "Route Env Probe":
            return self._build_probe_command("route")
        raise ValueError(f"Unsupported test selection: {selected}")

    def _collect_setup_data(self):
        return {
            "test": self.test_var.get(),
            "manual_scenario": self.manual_scenario_var.get(),
            "host": self.host_var.get().strip(),
            "port": int(self.port_var.get().strip()),
            "fps": int(self.fps_var.get().strip()),
            "viewer_res": self.viewer_res_var.get().strip(),
            "obs_res": self.obs_res_var.get().strip(),
            "num_images": int(self.num_images_var.get().strip()),
            "output_dir": self.output_dir_var.get().strip(),
            "target_speed": float(self.target_speed_var.get().strip()),
            "start_carla": bool(self.start_carla_var.get()),
            "synchronous": bool(self.sync_var.get()),
            "show_waypoints": bool(self.show_waypoints_var.get()),
        }

    def _apply_setup_data(self, data):
        if "test" in data:
            self.test_var.set(str(data["test"]))
        if "manual_scenario" in data:
            self.manual_scenario_var.set(str(data["manual_scenario"]))
        if "host" in data:
            self.host_var.set(str(data["host"]))
        if "port" in data:
            self.port_var.set(str(data["port"]))
        if "fps" in data:
            self.fps_var.set(str(data["fps"]))
        if "viewer_res" in data:
            self.viewer_res_var.set(str(data["viewer_res"]))
        if "obs_res" in data:
            self.obs_res_var.set(str(data["obs_res"]))
        if "num_images" in data:
            self.num_images_var.set(str(data["num_images"]))
        if "output_dir" in data:
            self.output_dir_var.set(str(data["output_dir"]))
        if "target_speed" in data:
            self.target_speed_var.set(str(data["target_speed"]))
        if "start_carla" in data:
            self.start_carla_var.set(bool(data["start_carla"]))
        if "synchronous" in data:
            self.sync_var.set(bool(data["synchronous"]))
        if "show_waypoints" in data:
            self.show_waypoints_var.set(bool(data["show_waypoints"]))

    def _save_setup(self):
        try:
            setup = self._collect_setup_data()
        except Exception as exc:
            messagebox.showerror("Save Setup", f"Failed to collect setup values:\n{exc}")
            return

        os.makedirs(DEFAULT_LAB_CONFIG_DIR, exist_ok=True)
        path = filedialog.asksaveasfilename(
            parent=self.root,
            title="Save Lab Setup",
            initialdir=DEFAULT_LAB_CONFIG_DIR,
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        )
        if not path:
            return

        try:
            import json

            with open(path, "w", encoding="utf-8") as f:
                json.dump(setup, f, indent=2)
            self._append_log(f"\n[SETUP] Saved lab setup to {path}\n")
        except Exception as exc:
            messagebox.showerror("Save Setup", f"Failed to save setup:\n{exc}")

    def _load_setup(self):
        os.makedirs(DEFAULT_LAB_CONFIG_DIR, exist_ok=True)
        path = filedialog.askopenfilename(
            parent=self.root,
            title="Load Lab Setup",
            initialdir=DEFAULT_LAB_CONFIG_DIR,
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        )
        if not path:
            return

        try:
            data = load_json_config(path)
            self._apply_setup_data(data)
            self._append_log(f"\n[SETUP] Loaded lab setup from {path}\n")
        except Exception as exc:
            messagebox.showerror("Load Setup", f"Failed to load setup:\n{exc}")

    def _carla_python_snippet_prefix(self):
        return f"""
import carla
client = carla.Client({self.host_var.get().strip()!r}, int({self.port_var.get().strip()!r}))
client.set_timeout(5.0)
world = client.get_world()
"""

    def _run_short_python(self, code):
        command = [sys.executable, "-c", code]
        return subprocess.run(
            command,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )

    def _build_manual_drive_command(self, env_kind):
        python = sys.executable
        class_name = "CarlaLapEnv" if env_kind == "lap" else "CarlaRouteEnv"
        module_name = "CarlaEnv.carla_lap_env" if env_kind == "lap" else "CarlaEnv.carla_route_env"
        is_training = "True" if env_kind == "lap" else "False"
        drive_code = f"""
from {module_name} import {class_name}, reward_fn
from CarlaEnv.manual_control import run_env_manual_drive

run_env_manual_drive(
    env_class={class_name},
    reward_fn=reward_fn,
    host={self.host_var.get().strip()!r},
    port=int({self.port_var.get().strip()!r}),
    viewer_res=tuple(int(x) for x in {self.viewer_res_var.get().strip()!r}.split("x")),
    obs_res=tuple(int(x) for x in {self.obs_res_var.get().strip()!r}.split("x")),
    synchronous={self.sync_var.get()!r},
    fps=int({self.fps_var.get().strip()!r}),
    start_carla={self.start_carla_var.get()!r},
    show_waypoints={self.show_waypoints_var.get()!r},
    is_training={is_training},
)
"""
        return [python, "-c", drive_code]

    def _build_probe_command(self, env_kind):
        python = sys.executable
        probe_code = f"""
import numpy as np
from CarlaEnv.carla_{"lap" if env_kind == "lap" else "route"}_env import Carla{"Lap" if env_kind == "lap" else "Route"}Env

env = Carla{"Lap" if env_kind == "lap" else "Route"}Env(
    host={self.host_var.get().strip()!r},
    port=int({self.port_var.get().strip()!r}),
    obs_res=tuple(int(x) for x in {self.obs_res_var.get().strip()!r}.split("x")),
    viewer_res=tuple(int(x) for x in {self.viewer_res_var.get().strip()!r}.split("x")),
    fps=int({self.fps_var.get().strip()!r}),
    synchronous={self.sync_var.get()!r},
    start_carla={self.start_carla_var.get()!r},
    show_waypoints={self.show_waypoints_var.get()!r},
)
print("env_type=", type(env).__name__)
print("action_space=", env.action_space)
print("observation_space=", env.observation_space)
state = env.reset(is_training={env_kind == "lap"!r})
print("reset_state_type=", type(state).__name__)
print("reset_state_shape=", getattr(state, "shape", None))
zero_action = np.zeros(env.action_space.shape[0], dtype=np.float32)
state, reward, done, info = env.step(zero_action)
print("step_state_shape=", getattr(state, "shape", None))
print("reward=", reward)
print("done=", done)
print("info=", info)
rendered = env.render(mode="rgb_array")
print("render_rgb_array_shape=", getattr(rendered, "shape", None))
env.close()
"""
        return [python, "-c", probe_code]

    def _get_selected_vehicle_id(self):
        value = self.selected_vehicle_var.get().strip()
        if not value:
            return None
        try:
            return int(value.split("|", 1)[0].strip())
        except ValueError:
            return None

    def _run_selected_test(self):
        if self.process is not None:
            self._append_log("A test is already running. Stop it before starting another one.\n")
            return
        if self.stopping:
            self._append_log("The previous test is still stopping. Please wait for it to exit.\n")
            return

        if self.start_carla_var.get() and not os.environ.get("CARLA_ROOT"):
            self._append_log(
                "Start CARLA Automatically is enabled, but CARLA_ROOT is not set in this launcher process.\n"
                "Either unset that checkbox and start CARLA manually, or export CARLA_ROOT before launching the lab.\n"
            )
            return

        try:
            command = self._build_command()
        except Exception as exc:
            self._append_log(f"Failed to build command: {exc}\n")
            return

        self._append_log(f"\n[RUN] {' '.join(command)}\n")
        self.process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            preexec_fn=os.setsid if os.name != "nt" else None,
        )
        self.run_button.configure(state=tk.DISABLED)
        self.stop_button.configure(state=tk.NORMAL)
        self.force_stop_button.configure(state=tk.DISABLED)
        self.status_var.set("Running")
        self.stopping = False

        threading.Thread(target=self._stream_output, daemon=True).start()
        threading.Thread(target=self._wait_for_exit, daemon=True).start()

    def _refresh_vehicles(self):
        code = (
            self._carla_python_snippet_prefix()
            + """
vehicles = world.get_actors().filter("vehicle.*")
for actor in vehicles:
    transform = actor.get_transform()
    location = transform.location
    control = actor.get_control()
    print(
        f"{actor.id} | {actor.type_id} | "
        f"loc=({location.x:.1f}, {location.y:.1f}, {location.z:.1f}) | "
        f"throttle={control.throttle:.2f} steer={control.steer:.2f}"
    )
"""
        )
        result = self._run_short_python(code)
        if result.returncode != 0:
            self._append_log("\n[VEHICLES] Failed to query CARLA world.\n")
            if result.stdout:
                self._append_log(result.stdout)
            if result.stderr:
                self._append_log(result.stderr)
            return

        lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        self.vehicle_choices = lines
        self.vehicle_combo["values"] = lines
        if lines:
            self.selected_vehicle_var.set(lines[0])
            self._append_log(f"\n[VEHICLES] Found {len(lines)} vehicle actor(s).\n")
        else:
            self.selected_vehicle_var.set("")
            self._append_log("\n[VEHICLES] No vehicle actors found in the current CARLA world.\n")

    def _delete_selected_vehicle(self):
        actor_id = self._get_selected_vehicle_id()
        if actor_id is None:
            messagebox.showinfo("Delete Vehicle", "Select a vehicle from the dropdown first.")
            return
        if not messagebox.askyesno("Delete Vehicle", f"Destroy vehicle actor {actor_id}?"):
            return

        code = (
            self._carla_python_snippet_prefix()
            + f"""
actor = world.get_actor({actor_id})
if actor is None:
    print("Vehicle not found.")
else:
    actor.destroy()
    print(f"Destroyed vehicle actor {{actor.id}}.")
"""
        )
        result = self._run_short_python(code)
        self._append_log("\n[DELETE VEHICLE]\n")
        if result.stdout:
            self._append_log(result.stdout)
        if result.stderr:
            self._append_log(result.stderr)
        self._refresh_vehicles()

    def _control_selected_vehicle(self):
        if self.process is not None or self.stopping:
            self._append_log("Stop the currently running test before attaching to another vehicle.\n")
            return
        actor_id = self._get_selected_vehicle_id()
        if actor_id is None:
            messagebox.showinfo("Control Vehicle", "Select a vehicle from the dropdown first.")
            return

        python = sys.executable
        drive_code = f"""
from CarlaEnv.manual_control import control_existing_vehicle

control_existing_vehicle(
    host={self.host_var.get().strip()!r},
    port=int({self.port_var.get().strip()!r}),
    viewer_res=tuple(int(x) for x in {self.viewer_res_var.get().strip()!r}.split("x")),
    obs_res=tuple(int(x) for x in {self.obs_res_var.get().strip()!r}.split("x")),
    synchronous={self.sync_var.get()!r},
    fps=int({self.fps_var.get().strip()!r}),
    actor_id={actor_id},
)
"""
        command = [python, "-c", drive_code]
        self._append_log(f"\n[CONTROL EXISTING] {' '.join(command)}\n")
        self.process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            preexec_fn=os.setsid if os.name != "nt" else None,
        )
        self.run_button.configure(state=tk.DISABLED)
        self.stop_button.configure(state=tk.NORMAL)
        self.force_stop_button.configure(state=tk.DISABLED)
        self.status_var.set("Running")
        self.stopping = False
        threading.Thread(target=self._stream_output, daemon=True).start()
        threading.Thread(target=self._wait_for_exit, daemon=True).start()

    def _stream_output(self):
        if self.process is None or self.process.stdout is None:
            return
        for line in self.process.stdout:
            self.log_queue.put(line)

    def _wait_for_exit(self):
        if self.process is None:
            return
        return_code = self.process.wait()
        self.log_queue.put(f"\n[EXIT] Process finished with code {return_code}\n")
        self.log_queue.put("__PROCESS_FINISHED__")

    def _stop_process(self):
        if self.process is None:
            return
        if self.stopping:
            self._append_log("Stop is already in progress.\n")
            return
        self._append_log("\n[STOP] Stopping running test...\n")
        self.stopping = True
        self.status_var.set("Stopping...")
        self.stop_button.configure(state=tk.DISABLED)
        self.force_stop_button.configure(state=tk.NORMAL)
        try:
            if os.name == "nt":
                self.process.terminate()
            else:
                os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
        except Exception:
            try:
                self.process.terminate()
            except Exception as exc:
                self._append_log(f"Failed to stop process cleanly: {exc}\n")
        self.root.after(2500, self._offer_force_kill_if_needed)

    def _offer_force_kill_if_needed(self):
        if self.process is not None and self.stopping:
            self._append_log("[STOP] Process still alive. You can use Force Kill if it does not exit shortly.\n")
            self.force_stop_button.configure(state=tk.NORMAL)

    def _force_kill_process(self):
        if self.process is None:
            return
        self._append_log("\n[FORCE KILL] Killing running test...\n")
        try:
            if os.name == "nt":
                self.process.kill()
            else:
                os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
        except Exception:
            try:
                self.process.kill()
            except Exception as exc:
                self._append_log(f"Failed to force kill process: {exc}\n")

    def _drain_logs(self):
        try:
            while True:
                item = self.log_queue.get_nowait()
                if item == "__PROCESS_FINISHED__":
                    self.process = None
                    self.stopping = False
                    self.run_button.configure(state=tk.NORMAL)
                    self.stop_button.configure(state=tk.DISABLED)
                    self.force_stop_button.configure(state=tk.DISABLED)
                    self.status_var.set("Ready")
                else:
                    self._append_log(item)
        except queue.Empty:
            pass
        self.root.after(100, self._drain_logs)

    def _clear_log(self):
        self.log_widget.configure(state=tk.NORMAL)
        self.log_widget.delete("1.0", tk.END)
        self.log_widget.configure(state=tk.DISABLED)

    def _append_log(self, text):
        self.log_widget.configure(state=tk.NORMAL)
        self.log_widget.insert(tk.END, text)
        self.log_widget.see(tk.END)
        self.log_widget.configure(state=tk.DISABLED)

    def _on_close(self):
        self._stop_process()
        self.root.after(150, self.root.destroy)

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    CarlaEnvLab().run()
