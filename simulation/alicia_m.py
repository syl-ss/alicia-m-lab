"""Alicia-M v1.1 position control in MuJoCo; no serial or hardware access."""

from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np


ARM_JOINTS = tuple(f"joint{i}" for i in range(1, 7))
FINGER_JOINTS = ("left_finger", "right_finger")
# Simulation gains only: these are not measured gains of the physical motors.
KP = (200, 200, 160, 80, 60, 40)
KV = (20, 20, 16, 8, 6, 4)
# Same degree-valued ready pose as the official PV example and my_examples/ex03.
INITIAL_DEG = (21, -67, -19, 8, 56, -7)


def source_model_path():
    imported = Path("D:/MuJoCo/models/Alicia_M/mjcf/Alicia_M_v1_1/Alicia_M_v1_1_follower.xml")
    if imported.is_file():
        return imported
    from synriard import get_model_path
    return Path(get_model_path("Alicia_M", version="v1_1", variant="follower", model_format="mjcf"))


def controlled_xml(source=None, gravity_comp=True, *, kp=KP, kv=KV):
    """Build a separate servo model, keeping the original official XML intact."""
    kp, kv = _joint_vector(kp), _joint_vector(kv)
    if np.any(kp <= 0) or np.any(kv < 0):
        raise ValueError("Position gains must be positive; velocity damping must be nonnegative")
    source = Path(source) if source is not None else source_model_path()
    root = ET.parse(source).getroot()
    if root.tag != "mujoco" or root.get("model") != "alicia_m_v1_1_follower":
        raise ValueError("Expected the official Alicia-M v1.1 follower MJCF")
    root.set("model", "alicia_m_v1_1_position")
    compiler = root.find("compiler")
    meshdir = (source.parent / compiler.get("meshdir", "")).resolve()
    compiler.set("meshdir", meshdir.as_posix())
    option = ET.SubElement(root, "option", timestep="0.002", integrator="implicitfast")
    option.set("gravity", "0 0 -9.81")
    ET.SubElement(root, "statistic", center="0 0 0.35", extent="0.8")
    actuators = ET.SubElement(root, "actuator")
    joints = {j.get("name"): j for j in root.findall(".//joint")}
    for index, name in enumerate(ARM_JOINTS + FINGER_JOINTS):
        joint = joints[name]
        joint.set("damping", "0.05" if index < 6 else "0.2")
        joint.set("armature", "0.01" if index < 6 else "0.001")
        if name == "right_finger":
            continue  # This finger follows the driven finger mechanically.
        ET.SubElement(
            actuators, "position", name=f"servo_{name}" if index < 6 else "servo_gripper", joint=name,
            kp=str(kp[index] if index < 6 else 1000),
            kv=str(kv[index] if index < 6 else 16),
            ctrllimited="true", ctrlrange=joint.get("range"),
            forcelimited="true", forcerange=joint.get("actuatorfrcrange"),
        )
    equality = ET.SubElement(root, "equality")
    ET.SubElement(equality, "joint", name="finger_coupling", joint1="right_finger",
                  joint2="left_finger", polycoef="0 -1 0 0 0", solref="0.004 1")
    if gravity_comp:
        # Ideal external gravity support, outside the actuator's torque limit.
        # Useful for planning tests; never use this to claim real motor capacity.
        for body in root.findall(".//body"):
            body.set("gravcomp", "1")
    initial = np.r_[np.deg2rad(INITIAL_DEG), -0.025, 0.025]
    controls = initial[:7]
    ET.SubElement(ET.SubElement(root, "keyframe"), "key", name="ready",
                  qpos=" ".join(map(str, initial)), ctrl=" ".join(map(str, controls)))
    ET.indent(root, space="  ")
    return ET.tostring(root, encoding="unicode")


def _joint_vector(values):
    q = np.asarray(values, dtype=float)
    if q.shape != (6,) or not np.isfinite(q).all():
        raise ValueError("Expected exactly six finite joint angles")
    return q


def validate_trajectory(timestamps, positions, joint_limits):
    """Reject ambiguous timing, malformed arrays and out-of-range joint angles."""
    t = np.asarray(timestamps, dtype=float)
    q = np.asarray(positions, dtype=float)
    if (t.ndim != 1 or len(t) < 2 or q.shape != (len(t), 6)
            or not np.isfinite(t).all() or not np.isfinite(q).all()
            or t[0] != 0 or np.any(np.diff(t) <= 0)):
        raise ValueError("Trajectory needs finite increasing times starting at 0, and N x 6 positions")
    limits = np.asarray(joint_limits)
    if np.any(q < limits[:, 0]) or np.any(q > limits[:, 1]):
        raise ValueError("Trajectory exceeds model joint limits")
    return t, q


def quintic_trajectory(start, goal, duration=4.0, timestep=0.002):
    """A smooth time law with zero endpoint velocity and acceleration."""
    start, goal = _joint_vector(start), _joint_vector(goal)
    if not np.isfinite([duration, timestep]).all() or duration <= 0 or timestep <= 0:
        raise ValueError("duration and timestep must be finite and positive")
    count = max(2, int(np.ceil(duration / timestep)) + 1)
    t = np.linspace(0, duration, count)
    s = t / duration
    blend = 10 * s**3 - 15 * s**4 + 6 * s**5
    return t, start + blend[:, None] * (goal - start)


class AliciaMSim:
    """Command targets, step physics, and read the six simulated arm angles.

    Default ideal gravity compensation assists trajectory-logic experiments.
    Set gravity_comp=False to test a position servo with uncompensated gravity.
    This class is not a drop-in replacement for the hardware SDK.
    """

    def __init__(self, source=None, gravity_comp=True, *, kp=KP, kv=KV):
        self.gravity_comp = bool(gravity_comp)
        self.xml = controlled_xml(source, self.gravity_comp, kp=kp, kv=kv)
        self.model = mujoco.MjModel.from_xml_string(self.xml)
        self.data = mujoco.MjData(self.model)
        joint_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
                     for name in ARM_JOINTS + FINGER_JOINTS]
        actuator_names = tuple(f"servo_{name}" for name in ARM_JOINTS) + ("servo_gripper",)
        actuator_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
                        for name in actuator_names]
        if min(joint_ids + actuator_ids) < 0:
            raise ValueError("Required named joint or actuator is missing")
        self.arm_qpos = self.model.jnt_qposadr[joint_ids[:6]].copy()
        self.arm_dofs = self.model.jnt_dofadr[joint_ids[:6]].copy()
        self.arm_actuators = np.asarray(actuator_ids[:6])
        self.finger_qpos = self.model.jnt_qposadr[joint_ids[6:]].copy()
        self.gripper_actuator = actuator_ids[6]
        self.joint_limits = self.model.jnt_range[joint_ids[:6]].copy()
        self.reset()

    def _checked_target(self, joints, joint_format):
        if joint_format not in ("rad", "deg"):
            raise ValueError("joint_format must be 'rad' or 'deg'")
        q = _joint_vector(joints)
        if joint_format == "deg":
            q = np.deg2rad(q)
        if np.any(q < self.joint_limits[:, 0]) or np.any(q > self.joint_limits[:, 1]):
            raise ValueError("Target exceeds model joint limits")
        return q

    def reset(self, initial_joints=None):
        q = self._checked_target(np.deg2rad(INITIAL_DEG) if initial_joints is None else initial_joints, "rad")
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        self.data.qpos[self.arm_qpos] = q
        self.data.ctrl[self.arm_actuators] = q
        mujoco.mj_forward(self.model, self.data)

    def set_joint_target(self, joints, joint_format="rad"):
        self.data.ctrl[self.arm_actuators] = self._checked_target(joints, joint_format)

    def set_gripper(self, opening_m):
        if not np.isfinite(opening_m) or not 0 <= opening_m <= 0.1:
            raise ValueError("Gripper opening must be within 0..0.1 metres")
        self.data.ctrl[self.gripper_actuator] = -opening_m / 2

    def set_gripper_value(self, value):
        """SDK-style 0=closed, 1000=open; linear aperture is a simulation assumption."""
        if not np.isfinite(value) or not 0 <= value <= 1000:
            raise ValueError("Gripper value must be within 0..1000")
        self.set_gripper(value * 0.1 / 1000)

    def get_joint_positions(self, joint_format="rad"):
        if joint_format not in ("rad", "deg"):
            raise ValueError("joint_format must be 'rad' or 'deg'")
        q = self.data.qpos[self.arm_qpos].copy()
        return np.rad2deg(q) if joint_format == "deg" else q

    def get_joint_state(self):
        """Read the last physics step: rad, rad/s, rad/s^2 and Nm.

        qacc is the acceleration computed for that integration step. Preserve
        it: calling mj_forward after integration with the old held target
        produces a different instantaneous acceleration, not the step result.
        """
        return {
            "time": float(self.data.time),
            "positions": self.data.qpos[self.arm_qpos].copy(),
            "velocities": self.data.qvel[self.arm_dofs].copy(),
            "accelerations": self.data.qacc[self.arm_dofs].copy(),
            "torques": self.data.actuator_force[self.arm_actuators].copy(),
        }

    def step(self, steps=1):
        if not isinstance(steps, (int, np.integer)) or steps < 1:
            raise ValueError("steps must be a positive integer")
        for _ in range(steps):
            mujoco.mj_step(self.model, self.data)
            if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all():
                raise RuntimeError("Non-finite simulation state")
        if any(w.number for w in self.data.warning):
            raise RuntimeError("MuJoCo reported a physics warning; inspect data.warning")
