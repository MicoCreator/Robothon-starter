"""Lockout Relay Rescue — tactile, vision-guided emergency isolation in MuJoCo.

This is a self-contained Robothon submission.  A 15-DOF five-finger hand mounted
on a Cartesian wrist must grasp a rotary isolator on a moving safety panel, turn
it to the safe position, press a spring-return verification latch, and seat a
tool-mounted lockout tag.  The controller uses measured MuJoCo contact forces
and a noisy panel-marker pose observation.  It evaluates feedback and open-loop
control against identical seeded panel-jitter disturbances.

Run:
    python run_lockout_rescue.py

The script generates MJCF, an MP4 demo, a benchmark CSV, full trial data,
trajectory samples, and a machine-readable metrics report in ./outputs.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "outputs"

PANEL_ORIGIN = np.array([0.72, 0.0], dtype=float)
MARKER_LOCAL = np.array([-0.26, 0.22], dtype=float)
HAND_BASE = np.array([0.40, 0.0, 0.72], dtype=float)
NOMINAL_KNOB = np.array([0.72, 0.0, 0.682], dtype=float)
FINGER_RADII = np.array(
    [[0.100, 0.000], [0.032, 0.094], [-0.071, 0.070], [-0.094, -0.035], [0.043, -0.090]],
    dtype=float,
)
TAG_LOCAL = np.array([-0.105, 0.045], dtype=float)
TAG_SLOT_LOCAL = np.array([-0.170, 0.140], dtype=float)
BUTTON_LOCAL = np.array([0.140, -0.150], dtype=float)
SIM_DURATION_S = 17.0
TARGET_ISOLATION_RAD = 1.42
FORCE_LIMIT_N = 6.0


@dataclass
class TrialResult:
    mode: str
    seed: int
    panel_jitter_x_mm: float
    panel_jitter_y_mm: float
    success: bool
    tactile_grasp: bool
    switch_isolated: bool
    latch_pressed: bool
    tag_seated: bool
    final_switch_angle_deg: float
    max_latch_travel_mm: float
    final_tag_error_mm: float
    peak_touch_force_n: float
    max_alignment_error_mm: float
    vision_corrections: int
    recovery_events: int
    contact_samples: int
    simulated_seconds: float


def _finger_xml(index: int, radial: np.ndarray) -> str:
    """Return one articulated, radial-closing finger with two flex joints."""

    direction = -radial / np.linalg.norm(radial)
    tangent = np.array([-direction[1], direction[0]], dtype=float)
    return f"""
      <body name="finger_{index}" pos="{radial[0]:.5f} {radial[1]:.5f} -0.020">
        <joint name="finger_{index}_radial" type="slide" axis="{direction[0]:.5f} {direction[1]:.5f} 0" range="0 0.062"/>
        <joint name="finger_{index}_knuckle" type="hinge" axis="{tangent[0]:.5f} {tangent[1]:.5f} 0" range="-0.10 0.65"/>
        <geom name="finger_{index}_proximal" type="capsule" fromto="0 0 0 0 0 -0.055" size="0.013" material="finger_material"/>
        <body name="finger_{index}_distal_body" pos="0 0 -0.055">
          <joint name="finger_{index}_distal" type="hinge" axis="{tangent[0]:.5f} {tangent[1]:.5f} 0" range="-0.10 0.75"/>
          <geom name="finger_{index}_distal" type="capsule" fromto="0 0 0 0 0 -0.060" size="0.012" material="finger_material"/>
          <geom name="finger_{index}_tip" type="sphere" pos="0 0 -0.070" size="0.019" material="finger_tip"/>
          <site name="finger_{index}_touch" type="sphere" pos="0 0 -0.070" size="0.017" rgba="0.20 0.95 0.95 0.40"/>
        </body>
      </body>"""


def scene_xml() -> str:
    """Build the complete MuJoCo scene from procedural primitives only."""

    fingers = "\n".join(_finger_xml(i, radial) for i, radial in enumerate(FINGER_RADII))
    finger_actuators = "\n".join(
        f"""
    <position name="finger_{i}_radial_position" joint="finger_{i}_radial" kp="135" forcerange="-26 26"/>
    <position name="finger_{i}_knuckle_position" joint="finger_{i}_knuckle" kp="55" forcerange="-10 10"/>
    <position name="finger_{i}_distal_position" joint="finger_{i}_distal" kp="42" forcerange="-8 8"/>"""
        for i in range(5)
    )
    touch_sensors = "\n".join(
        f'<touch name="finger_{i}_touch_sensor" site="finger_{i}_touch"/>' for i in range(5)
    )
    return f"""
<mujoco model="Lockout Relay Rescue">
  <compiler angle="radian" coordinate="local"/>
  <option timestep="0.004" integrator="implicitfast" gravity="0 0 -9.81"/>
  <size nconmax="600" njmax="800"/>
  <visual>
    <global offwidth="800" offheight="500" azimuth="142" elevation="-28"/>
    <quality shadowsize="4096" offsamples="4"/>
  </visual>
  <default>
    <joint damping="1.1" armature="0.018" limited="true"/>
    <geom density="420" friction="1.10 0.018 0.004" condim="4"/>
    <position kp="420" forcerange="-150 150"/>
  </default>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.055 0.080 0.110" rgb2="0.005 0.008 0.015" width="512" height="3072"/>
    <texture name="floor_tex" type="2d" builtin="checker" rgb1="0.075 0.095 0.120" rgb2="0.115 0.135 0.165" width="256" height="256"/>
    <material name="floor" texture="floor_tex" texrepeat="6 6" reflectance="0.22"/>
    <material name="panel_material" rgba="0.12 0.17 0.23 1"/>
    <material name="safety_yellow" rgba="0.98 0.68 0.06 1"/>
    <material name="relay_red" rgba="0.88 0.14 0.10 1"/>
    <material name="safe_green" rgba="0.10 0.88 0.38 1"/>
    <material name="finger_material" rgba="0.18 0.48 0.82 1"/>
    <material name="finger_tip" rgba="0.14 0.95 0.92 1"/>
    <material name="lockout_tag" rgba="0.95 0.25 0.10 1"/>
  </asset>
  <worldbody>
    <light name="key_light" pos="-0.4 -1.3 2.4" dir="0.4 0.6 -1" diffuse="1.0 0.88 0.72" castshadow="true"/>
    <light name="rim_light" pos="1.7 1.1 1.8" dir="-1 -0.7 -0.5" diffuse="0.24 0.48 1.0"/>
    <geom name="floor" type="plane" size="4 4 0.1" material="floor"/>
    <geom name="workbench" type="box" pos="0.72 0 0.18" size="0.78 0.62 0.18" rgba="0.14 0.18 0.24 1" contype="0" conaffinity="0"/>
    <geom name="bench_front_stripe" type="box" pos="0.72 -0.58 0.37" size="0.70 0.020 0.012" material="safety_yellow" contype="0" conaffinity="0"/>

    <body name="safety_panel" pos="0.72 0 0.34" gravcomp="1">
      <joint name="panel_x" type="slide" axis="1 0 0" range="-0.045 0.045"/>
      <joint name="panel_y" type="slide" axis="0 1 0" range="-0.045 0.045"/>
      <geom name="panel_chassis" type="box" pos="0 0 0.065" size="0.36 0.30 0.065" material="panel_material"/>
      <geom name="panel_inset" type="box" pos="0 0 0.133" size="0.305 0.245 0.006" rgba="0.05 0.07 0.10 1"/>
      <geom name="panel_label_safe" type="box" pos="-0.105 0.132 0.143" size="0.055 0.013 0.004" material="safe_green" contype="0" conaffinity="0"/>
      <geom name="panel_label_live" type="box" pos="0.105 0.132 0.143" size="0.055 0.013 0.004" material="relay_red" contype="0" conaffinity="0"/>
      <site name="vision_marker" type="sphere" pos="{MARKER_LOCAL[0]} {MARKER_LOCAL[1]} 0.150" size="0.013" rgba="0.15 0.92 0.55 0.90"/>

      <body name="rotary_isolator" pos="0 0 0.145" gravcomp="1">
        <joint name="isolator_angle" type="hinge" axis="0 0 1" range="0 1.5708" damping="1.8" armature="0.008"/>
        <geom name="isolator_knob" type="cylinder" pos="0 0 0.025" size="0.058 0.030" material="relay_red"/>
        <geom name="isolator_handle" type="box" pos="0.034 0 0.067" size="0.060 0.018 0.012" material="safety_yellow"/>
        <geom name="isolator_key" type="box" pos="0.050 0 0.086" size="0.016 0.011 0.011" rgba="0.96 0.96 0.98 1"/>
        <site name="isolator_center" type="sphere" pos="0 0 0.075" size="0.012" rgba="1 0.8 0.1 0.35"/>
      </body>

      <body name="verification_latch" pos="{BUTTON_LOCAL[0]} {BUTTON_LOCAL[1]} 0.135">
        <joint name="latch_travel" type="slide" axis="0 0 -1" range="0 0.032" stiffness="340" springref="0" damping="3.5"/>
        <geom name="latch_guard" type="cylinder" size="0.050 0.016" rgba="0.35 0.39 0.46 1" contype="1" conaffinity="1"/>
        <geom name="latch_button" type="cylinder" pos="0 0 0.020" size="0.030 0.020" material="safe_green"/>
        <site name="latch_site" type="sphere" pos="0 0 0.042" size="0.010" rgba="0.2 1 0.4 0.45"/>
      </body>

      <geom name="tag_slot_left" type="box" pos="{TAG_SLOT_LOCAL[0] - 0.040:.4f} {TAG_SLOT_LOCAL[1]:.4f} 0.158" size="0.010 0.055 0.028" rgba="0.42 0.46 0.55 1"/>
      <geom name="tag_slot_right" type="box" pos="{TAG_SLOT_LOCAL[0] + 0.040:.4f} {TAG_SLOT_LOCAL[1]:.4f} 0.158" size="0.010 0.055 0.028" rgba="0.42 0.46 0.55 1"/>
      <geom name="tag_slot_back" type="box" pos="{TAG_SLOT_LOCAL[0]:.4f} {TAG_SLOT_LOCAL[1] + 0.050:.4f} 0.158" size="0.050 0.008 0.028" rgba="0.42 0.46 0.55 1"/>
      <site name="tag_slot_target" type="sphere" pos="{TAG_SLOT_LOCAL[0]} {TAG_SLOT_LOCAL[1]} 0.174" size="0.010" rgba="1 0.40 0.10 0.60"/>
    </body>

    <body name="cartesian_hand" pos="{HAND_BASE[0]} {HAND_BASE[1]} {HAND_BASE[2]}" gravcomp="1">
      <joint name="hand_x" type="slide" axis="1 0 0" range="-0.28 0.54"/>
      <joint name="hand_y" type="slide" axis="0 1 0" range="-0.48 0.48"/>
      <joint name="hand_z" type="slide" axis="0 0 1" range="-0.16 0.30"/>
      <joint name="wrist_yaw" type="hinge" axis="0 0 1" range="-1.75 1.75"/>
      <geom name="palm" type="box" size="0.084 0.084 0.030" rgba="0.16 0.29 0.46 1"/>
      <geom name="palm_guard" type="cylinder" pos="0 0 0.035" size="0.050 0.020" rgba="0.10 0.15 0.22 1" contype="0" conaffinity="0"/>
      <site name="palm_pose" type="sphere" size="0.010" rgba="0.15 0.85 1 0.60"/>
      {fingers}
      <body name="tool_mounted_lockout_tag" pos="{TAG_LOCAL[0]} {TAG_LOCAL[1]} -0.175">
        <geom name="lockout_tag" type="box" size="0.032 0.045 0.010" material="lockout_tag"/>
        <geom name="tag_tab" type="capsule" fromto="0 0.035 0 0 0.072 0" size="0.008" material="lockout_tag"/>
        <site name="tag_tip" type="sphere" pos="0 0 -0.012" size="0.009" rgba="1 0.30 0.08 0.65"/>
      </body>
    </body>
    <camera name="overview" pos="1.56 -1.65 1.28" xyaxes="0.74 0.67 0 -0.28 0.31 0.91"/>
  </worldbody>
  <actuator>
    <position name="hand_x_position" joint="hand_x" kp="760" forcerange="-260 260"/>
    <position name="hand_y_position" joint="hand_y" kp="760" forcerange="-260 260"/>
    <position name="hand_z_position" joint="hand_z" kp="840" forcerange="-300 300"/>
    <position name="wrist_yaw_position" joint="wrist_yaw" kp="115" forcerange="-55 55"/>
    {finger_actuators}
    <position name="panel_x_position" joint="panel_x" kp="1200" forcerange="-280 280"/>
    <position name="panel_y_position" joint="panel_y" kp="1200" forcerange="-280 280"/>
    <motor name="wrist_coupled_isolator_torque" joint="isolator_angle" gear="1" forcerange="-8 8"/>
  </actuator>
  <sensor>
    <framepos name="vision_marker_position" objtype="site" objname="vision_marker"/>
    <framepos name="palm_position" objtype="site" objname="palm_pose"/>
    <framepos name="tag_position" objtype="site" objname="tag_tip"/>
    <framepos name="tag_slot_position" objtype="site" objname="tag_slot_target"/>
    <framepos name="isolator_position" objtype="site" objname="isolator_center"/>
    <jointpos name="isolator_angle_sensor" joint="isolator_angle"/>
    <jointvel name="isolator_velocity_sensor" joint="isolator_angle"/>
    <jointpos name="latch_travel_sensor" joint="latch_travel"/>
    {touch_sensors}
  </sensor>
</mujoco>
""".strip()


def smoothstep(value: float) -> float:
    value = float(np.clip(value, 0.0, 1.0))
    return value * value * (3.0 - 2.0 * value)


def blend(start: np.ndarray, end: np.ndarray, amount: float) -> np.ndarray:
    return (1.0 - smoothstep(amount)) * start + smoothstep(amount) * end


def name_id(model: mujoco.MjModel, object_type: mujoco.mjtObj, name: str) -> int:
    result = mujoco.mj_name2id(model, object_type, name)
    if result < 0:
        raise KeyError(f"MuJoCo object not found: {name}")
    return result


def sensor_values(model: mujoco.MjModel, data: mujoco.MjData, name: str) -> np.ndarray:
    sensor_id = name_id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
    start = int(model.sensor_adr[sensor_id])
    end = start + int(model.sensor_dim[sensor_id])
    return np.array(data.sensordata[start:end], dtype=float)


def joint_qpos(model: mujoco.MjModel, data: mujoco.MjData, name: str) -> float:
    joint_id = name_id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    return float(data.qpos[model.jnt_qposadr[joint_id]])


def joint_qvel(model: mujoco.MjModel, data: mujoco.MjData, name: str) -> float:
    joint_id = name_id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    return float(data.qvel[model.jnt_dofadr[joint_id]])


def actuator_id(model: mujoco.MjModel, name: str) -> int:
    return name_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)


def geom_name(model: mujoco.MjModel, geom_id: int) -> str:
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""


def finger_contact_count(model: mujoco.MjModel, data: mujoco.MjData) -> int:
    """Count different fingertip geoms in physical contact with the isolator."""

    contacted: set[int] = set()
    for contact_id in range(data.ncon):
        contact = data.contact[contact_id]
        names = {geom_name(model, contact.geom1), geom_name(model, contact.geom2)}
        if "isolator_knob" not in names and "isolator_handle" not in names:
            continue
        for finger_index in range(5):
            if f"finger_{finger_index}_tip" in names:
                contacted.add(finger_index)
    return len(contacted)


class LockoutController:
    """Deterministic phase policy plus sensor-driven visual/tactile residuals."""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, seed: int, feedback: bool):
        self.model = model
        self.data = data
        self.seed = seed
        self.feedback = feedback
        rng = np.random.default_rng(seed)
        self.jitter = rng.uniform(low=-0.029, high=0.029, size=2)
        self.vision_noise = rng.normal(0.0, 0.0007, size=2)
        self.grasp_acquired = False
        self.recovery_events = 0
        self.vision_corrections = 0
        self.max_latch = 0.0
        self.peak_touch = 0.0
        self.max_alignment = 0.0
        self.contact_samples = 0
        self._last_vision_correction_s = -1.0
        self._finger_overload = [False] * 5
        self.history: list[dict[str, Any]] = []
        self.controls = {name: actuator_id(model, name) for name in self._actuator_names()}

    @staticmethod
    def _actuator_names() -> list[str]:
        names = ["hand_x_position", "hand_y_position", "hand_z_position", "wrist_yaw_position", "panel_x_position", "panel_y_position", "wrist_coupled_isolator_torque"]
        for finger in range(5):
            names.extend(
                [
                    f"finger_{finger}_radial_position",
                    f"finger_{finger}_knuckle_position",
                    f"finger_{finger}_distal_position",
                ]
            )
        return names

    def observed_panel_offset(self, time_s: float) -> np.ndarray:
        nominal_marker = PANEL_ORIGIN + MARKER_LOCAL
        measured_marker = sensor_values(self.model, self.data, "vision_marker_position")[:2]
        if not self.feedback:
            return np.zeros(2, dtype=float)
        estimated = measured_marker - nominal_marker + self.vision_noise
        if float(np.linalg.norm(estimated)) > 0.002 and time_s - self._last_vision_correction_s >= 0.05:
            self.vision_corrections += 1
            self._last_vision_correction_s = time_s
        return estimated

    def phase_targets(self, time_s: float, offset: np.ndarray) -> tuple[str, np.ndarray, float, np.ndarray]:
        """Return phase label, palm target, wrist yaw, and five radial finger targets."""

        home = np.array([0.42, -0.32, 0.83])
        knob = NOMINAL_KNOB.copy()
        knob[:2] += offset
        button = np.array([PANEL_ORIGIN[0] + BUTTON_LOCAL[0] - FINGER_RADII[0, 0], PANEL_ORIGIN[1] + BUTTON_LOCAL[1], 0.682])
        button[:2] += offset
        # The tag tip is 187 mm below the palm origin; place that tip at the slot target.
        tag = np.array([PANEL_ORIGIN[0] + TAG_SLOT_LOCAL[0] - TAG_LOCAL[0], PANEL_ORIGIN[1] + TAG_SLOT_LOCAL[1] - TAG_LOCAL[1], 0.701])
        tag[:2] += offset
        open_fingers = np.zeros(5)
        closed_fingers = np.full(5, 0.057)

        if time_s < 1.8:
            return "SCAN_AND_APPROACH", blend(home, knob + np.array([0.0, 0.0, 0.105]), time_s / 1.8), 0.0, open_fingers
        if time_s < 4.8:
            close = blend(open_fingers, closed_fingers, (time_s - 1.8) / 3.0)
            return "TACTILE_GRASP_WITH_PANEL_JITTER", knob, 0.0, close
        if time_s < 8.2:
            yaw = TARGET_ISOLATION_RAD * smoothstep((time_s - 4.8) / 3.4)
            return "FORCE_GATED_SAFE_ISOLATION", knob, yaw, closed_fingers
        if time_s < 11.5:
            start = np.array([knob[0], knob[1], knob[2]])
            return "VERIFICATION_LATCH", blend(start, button, (time_s - 8.2) / 3.3), 0.0, np.array([0.050, 0.010, 0.010, 0.010, 0.010])
        # Clear the panel before translating the tool-mounted tag.  This keeps the tag's
        # trajectory physical: the hand must lift out of the rotary/latch workspace,
        # move above the slot, then descend to seat it.
        if time_s < 12.35:
            raised_button = button + np.array([0.0, 0.0, 0.150])
            return "CLEAR_PANEL_FOR_TAG", blend(button, raised_button, (time_s - 11.5) / 0.85), 0.0, open_fingers
        if time_s < 13.55:
            raised_button = button + np.array([0.0, 0.0, 0.150])
            raised_tag = tag + np.array([0.0, 0.0, 0.150])
            return "TRANSPORT_LOCKOUT_TAG", blend(raised_button, raised_tag, (time_s - 12.35) / 1.20), 0.0, open_fingers
        if time_s < 14.5:
            raised_tag = tag + np.array([0.0, 0.0, 0.150])
            return "SEAT_LOCKOUT_TAG", blend(raised_tag, tag, (time_s - 13.55) / 0.95), 0.0, open_fingers
        return "VERIFY_SAFE_STATE", tag, 0.0, open_fingers

    def step(self, time_s: float, record: bool) -> str:
        model, data = self.model, self.data
        drift_progress = smoothstep((time_s - 2.45) / 0.55)
        current_jitter = self.jitter * drift_progress
        data.ctrl[self.controls["panel_x_position"]] = float(current_jitter[0])
        data.ctrl[self.controls["panel_y_position"]] = float(current_jitter[1])

        offset = self.observed_panel_offset(time_s)
        phase, palm_target, yaw_target, radial_targets = self.phase_targets(time_s, offset)
        actual_knob = PANEL_ORIGIN + current_jitter
        alignment_error = float(np.linalg.norm(palm_target[:2] - actual_knob))
        if 1.8 <= time_s < 8.2:
            self.max_alignment = max(self.max_alignment, alignment_error)

        data.ctrl[self.controls["hand_x_position"]] = float(palm_target[0] - HAND_BASE[0])
        data.ctrl[self.controls["hand_y_position"]] = float(palm_target[1] - HAND_BASE[1])
        data.ctrl[self.controls["hand_z_position"]] = float(palm_target[2] - HAND_BASE[2])
        data.ctrl[self.controls["wrist_yaw_position"]] = float(yaw_target)

        touch_values = np.array([sensor_values(model, data, f"finger_{index}_touch_sensor")[0] for index in range(5)])
        self.peak_touch = max(self.peak_touch, float(np.max(touch_values)))
        for index in range(5):
            # Tactile back-drive: high contact force reduces the close command before the next step.
            target = float(radial_targets[index])
            overloaded = touch_values[index] > FORCE_LIMIT_N
            if self.feedback and overloaded:
                target = max(0.0, target - 0.012 * min(1.0, (touch_values[index] - FORCE_LIMIT_N) / FORCE_LIMIT_N))
                if not self._finger_overload[index]:
                    self.recovery_events += 1
            self._finger_overload[index] = overloaded
            data.ctrl[self.controls[f"finger_{index}_radial_position"]] = target
            data.ctrl[self.controls[f"finger_{index}_knuckle_position"]] = 0.07 + 0.08 * target / 0.062
            data.ctrl[self.controls[f"finger_{index}_distal_position"]] = 0.10 + 0.10 * target / 0.062

        contacts = finger_contact_count(model, data)
        if contacts:
            self.contact_samples += contacts
        if time_s > 3.65 and contacts >= 3 and alignment_error < 0.014:
            self.grasp_acquired = True

        angle = joint_qpos(model, data, "isolator_angle")
        velocity = joint_qvel(model, data, "isolator_angle")
        torque = 0.0
        if self.grasp_acquired and 4.8 <= time_s < 8.2:
            torque = float(np.clip(15.0 * (yaw_target - angle) - 1.1 * velocity, -7.0, 7.0))
        data.ctrl[self.controls["wrist_coupled_isolator_torque"]] = torque

        latch = joint_qpos(model, data, "latch_travel")
        self.max_latch = max(self.max_latch, latch)
        if record:
            tag_position = sensor_values(model, data, "tag_position")
            tag_slot = sensor_values(model, data, "tag_slot_position")
            self.history.append(
                {
                    "time_s": round(time_s, 4),
                    "phase": phase,
                    "panel_jitter_mm": [round(float(value * 1000.0), 3) for value in current_jitter],
                    "vision_offset_mm": [round(float(value * 1000.0), 3) for value in offset],
                    "contacts": contacts,
                    "touch_forces_n": [round(float(value), 4) for value in touch_values],
                    "alignment_error_mm": round((alignment_error if time_s < 8.2 else 0.0) * 1000.0, 4),
                    "isolator_angle_deg": round(math.degrees(angle), 3),
                    "latch_travel_mm": round(latch * 1000.0, 3),
                    "tag_error_mm": round(float(np.linalg.norm(tag_position - tag_slot)) * 1000.0, 3),
                    "tag_position_m": [round(float(value), 5) for value in tag_position],
                    "tag_slot_position_m": [round(float(value), 5) for value in tag_slot],
                    "grasp_gate": self.grasp_acquired,
                    "motor_torque": round(torque, 4),
                }
            )
        return phase


def run_trial(seed: int, feedback: bool, capture_video: bool = False) -> tuple[TrialResult, list[np.ndarray], list[dict[str, Any]], str]:
    """Run one physical episode.  All reported values are read from MuJoCo state."""

    xml = scene_xml()
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    controller = LockoutController(model, data, seed=seed, feedback=feedback)
    mujoco.mj_forward(model, data)

    renderer = mujoco.Renderer(model, height=500, width=800) if capture_video else None
    frames: list[np.ndarray] = []
    total_steps = int(SIM_DURATION_S / model.opt.timestep)
    video_step = max(1, int(round(1.0 / (12.0 * model.opt.timestep))) )
    history_step = max(1, int(round(0.05 / model.opt.timestep)))
    phase = "STARTUP"
    for step in range(total_steps):
        time_s = step * model.opt.timestep
        phase = controller.step(time_s, record=(step % history_step == 0))
        mujoco.mj_step(model, data)
        if renderer is not None and step % video_step == 0:
            renderer.update_scene(data, camera="overview")
            frames.append(np.array(renderer.render(), copy=True))
    if renderer is not None:
        renderer.close()

    isolator_angle = joint_qpos(model, data, "isolator_angle")
    tag_position = sensor_values(model, data, "tag_position")
    tag_slot = sensor_values(model, data, "tag_slot_position")
    tag_error = float(np.linalg.norm(tag_position - tag_slot))
    switch_isolated = isolator_angle >= 1.33
    latch_pressed = controller.max_latch >= 0.005
    tag_seated = tag_error <= 0.020
    success = bool(controller.grasp_acquired and switch_isolated and latch_pressed and tag_seated)
    result = TrialResult(
        mode="feedback" if feedback else "open_loop",
        seed=seed,
        panel_jitter_x_mm=round(float(controller.jitter[0] * 1000.0), 3),
        panel_jitter_y_mm=round(float(controller.jitter[1] * 1000.0), 3),
        success=success,
        tactile_grasp=controller.grasp_acquired,
        switch_isolated=switch_isolated,
        latch_pressed=latch_pressed,
        tag_seated=tag_seated,
        final_switch_angle_deg=round(math.degrees(isolator_angle), 3),
        max_latch_travel_mm=round(controller.max_latch * 1000.0, 3),
        final_tag_error_mm=round(tag_error * 1000.0, 3),
        peak_touch_force_n=round(controller.peak_touch, 4),
        max_alignment_error_mm=round(controller.max_alignment * 1000.0, 3),
        vision_corrections=controller.vision_corrections,
        recovery_events=controller.recovery_events,
        contact_samples=controller.contact_samples,
        simulated_seconds=SIM_DURATION_S,
    )
    return result, frames, controller.history, phase


def font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def overlay_frame(frame: np.ndarray, sample: dict[str, Any], result: TrialResult, index: int, total: int) -> np.ndarray:
    """Overlay only measurements produced during this run onto the rendered frame."""

    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image, "RGBA")
    w, h = image.size
    draw.rounded_rectangle((14, 14, 430, 136), radius=12, fill=(4, 10, 18, 220), outline=(57, 225, 202, 230), width=2)
    draw.text((30, 27), "LOCKOUT RELAY RESCUE", fill=(210, 255, 248, 255), font=font(23))
    draw.text((30, 57), sample["phase"].replace("_", " "), fill=(255, 206, 94, 255), font=font(15))
    draw.text((30, 82), f"Tactile contacts  {sample['contacts']} / 5     Alignment  {sample['alignment_error_mm']:.1f} mm", fill=(225, 236, 242, 255), font=font(14))
    draw.text((30, 104), f"Isolator  {sample['isolator_angle_deg']:.1f} deg     Latch  {sample['latch_travel_mm']:.1f} mm", fill=(158, 239, 221, 255), font=font(14))
    draw.rounded_rectangle((w - 255, 16, w - 16, 96), radius=12, fill=(4, 10, 18, 210), outline=(255, 175, 72, 220), width=2)
    draw.text((w - 236, 29), "LIVE SAFETY CHECK", fill=(255, 224, 180, 255), font=font(15))
    state = "RECOVERY ACTIVE" if sample["vision_offset_mm"] != [0.0, 0.0] and sample["phase"] == "TACTILE_GRASP_WITH_PANEL_JITTER" else "CONTROLLED"
    draw.text((w - 236, 53), state, fill=(122, 255, 180, 255), font=font(19))
    draw.rectangle((15, h - 37, w - 15, h - 22), fill=(10, 18, 28, 210))
    draw.rectangle((15, h - 37, 15 + int((w - 30) * index / max(1, total - 1)), h - 22), fill=(36, 214, 179, 230))
    return np.asarray(image)


def write_video(frames: list[np.ndarray], history: list[dict[str, Any]], result: TrialResult) -> Path:
    path = ARTIFACTS / "lockout_relay_demo.mp4"
    if not frames:
        raise RuntimeError("No frames captured for video")
    history_indices = np.linspace(0, len(history) - 1, num=len(frames)).astype(int)
    with imageio.get_writer(path, fps=12, codec="libx264", quality=8, macro_block_size=1) as writer:
        # Repeating each physical frame slows the real simulation into a ~68-second review video.
        for index, frame in enumerate(frames):
            decorated = overlay_frame(frame, history[int(history_indices[index])], result, index, len(frames))
            for _ in range(4):
                writer.append_data(decorated)
    return path


def summarize(rows: list[TrialResult]) -> dict[str, Any]:
    count = max(1, len(rows))
    return {
        "trials": len(rows),
        "successes": int(sum(row.success for row in rows)),
        "success_rate": round(float(sum(row.success for row in rows) / count), 4),
        "mean_final_switch_angle_deg": round(float(np.mean([row.final_switch_angle_deg for row in rows])), 3),
        "mean_tag_error_mm": round(float(np.mean([row.final_tag_error_mm for row in rows])), 3),
        "mean_peak_touch_force_n": round(float(np.mean([row.peak_touch_force_n for row in rows])), 3),
        "mean_max_alignment_error_mm": round(float(np.mean([row.max_alignment_error_mm for row in rows])), 3),
        "mean_vision_corrections": round(float(np.mean([row.vision_corrections for row in rows])), 2),
        "mean_recovery_events": round(float(np.mean([row.recovery_events for row in rows])), 2),
    }


def write_artifacts(feedback_rows: list[TrialResult], baseline_rows: list[TrialResult], demo_history: list[dict[str, Any]], video_path: Path, cases: int) -> None:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    scene_path = ARTIFACTS / "lockout_relay_scene.xml"
    scene_path.write_text(scene_xml(), encoding="utf-8")
    feedback_summary = summarize(feedback_rows)
    baseline_summary = summarize(baseline_rows)
    report = {
        "project": "Lockout Relay Rescue",
        "robot": "15-DOF five-finger hand on a 4-DOF Cartesian wrist",
        "task": "Contact-gated emergency relay isolation, latch verification, and lockout-tag seating",
        "simulation": {"engine": "MuJoCo", "timestep_ms": 4, "control_hz": 250, "controller_hz": 250},
        "sensors": ["five fingertip touch sensors", "panel marker pose", "isolator angle and velocity", "latch travel", "tag/slot poses"],
        "disturbance": "Seeded two-axis panel jitter introduced during tactile engagement",
        "feedback": feedback_summary,
        "open_loop_baseline": baseline_summary,
        "delta_success_rate": round(feedback_summary["success_rate"] - baseline_summary["success_rate"], 4),
        "demo_video": video_path.name,
        "benchmark_cases_per_condition": cases,
        "limitations": [
            "Simulation-only; visual pose is a noisy calibrated marker observation rather than a neural camera model.",
            "The rotary mechanism receives a contact-gated torque proxy from wrist intent; it is not a physical hardware trial.",
            "The tool-mounted lockout tag is held by the robot rather than grasped as a free object.",
        ],
    }
    (ARTIFACTS / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    paired = {"feedback": [asdict(row) for row in feedback_rows], "open_loop": [asdict(row) for row in baseline_rows]}
    (ARTIFACTS / "paired_ablation.json").write_text(json.dumps(paired, indent=2), encoding="utf-8")
    (ARTIFACTS / "demo_trajectory.json").write_text(json.dumps(demo_history, indent=2), encoding="utf-8")
    sensor_map = {
        "observation_channels": ["panel marker xyz", "five tactile contact forces", "isolator angle and velocity", "latch travel", "tag/slot xyz"],
        "action_channels": ["Cartesian x/y/z", "wrist yaw", "five radial closures", "ten finger flex joints", "contact-gated rotary torque"],
        "safety_guards": ["6 N tactile back-drive threshold", "tactile contact gate before isolation torque", "angle/latch/tag terminal checks"],
    }
    (ARTIFACTS / "sensor_policy_card.json").write_text(json.dumps(sensor_map, indent=2), encoding="utf-8")
    with (ARTIFACTS / "benchmark.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(feedback_rows[0]).keys()))
        writer.writeheader()
        for row in feedback_rows + baseline_rows:
            writer.writerow(asdict(row))
    manifest = {
        "generated_by": "python run_lockout_rescue.py",
        "artifacts": [
            "lockout_relay_demo.mp4",
            "lockout_relay_scene.xml",
            "metrics.json",
            "paired_ablation.json",
            "demo_trajectory.json",
            "sensor_policy_card.json",
            "benchmark.csv",
        ],
        "cases_per_condition": cases,
    }
    (ARTIFACTS / "artifact_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Lockout Relay Rescue MuJoCo benchmark.")
    parser.add_argument("--cases", type=int, default=36, help="Paired feedback/open-loop cases (default: 36).")
    parser.add_argument("--quick", action="store_true", help="Use 8 paired cases for a smoke test.")
    parser.add_argument("--no-video", action="store_true", help="Skip MP4 generation.")
    parser.add_argument("--demo-seed", type=int, default=19, help="Seed used by the rendered feedback demo.")
    args = parser.parse_args()
    cases = 8 if args.quick else max(1, args.cases)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)

    print("Lockout Relay Rescue: generating paired tactile-feedback benchmark")
    print(f"  Cases per condition: {cases}")
    feedback_rows: list[TrialResult] = []
    baseline_rows: list[TrialResult] = []
    for case_index in range(cases):
        seed = 1000 + case_index
        feedback_result, _, _, _ = run_trial(seed=seed, feedback=True, capture_video=False)
        baseline_result, _, _, _ = run_trial(seed=seed, feedback=False, capture_video=False)
        feedback_rows.append(feedback_result)
        baseline_rows.append(baseline_result)
        print(
            f"  seed {seed}: feedback={'PASS' if feedback_result.success else 'FAIL'} "
            f"| open-loop={'PASS' if baseline_result.success else 'FAIL'} "
            f"| jitter=({feedback_result.panel_jitter_x_mm:+.1f}, {feedback_result.panel_jitter_y_mm:+.1f}) mm"
        )

    demo_result, frames, history, _ = run_trial(seed=args.demo_seed, feedback=True, capture_video=not args.no_video)
    if args.no_video:
        video_path = ARTIFACTS / "lockout_relay_demo.mp4"
        if not video_path.exists():
            video_path.write_bytes(b"")
    else:
        video_path = write_video(frames, history, demo_result)
    write_artifacts(feedback_rows, baseline_rows, history, video_path, cases)

    feedback = summarize(feedback_rows)
    baseline = summarize(baseline_rows)
    print("\nVerified benchmark summary")
    print(f"  Feedback success:  {feedback['successes']}/{feedback['trials']} ({feedback['success_rate'] * 100:.1f}%)")
    print(f"  Open-loop success: {baseline['successes']}/{baseline['trials']} ({baseline['success_rate'] * 100:.1f}%)")
    print(f"  Demo result:       {'PASS' if demo_result.success else 'FAIL'}")
    print(f"  Artifacts:         {ARTIFACTS}")


if __name__ == "__main__":
    main()
