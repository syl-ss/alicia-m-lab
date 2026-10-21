"""Run a joint trajectory with real MuJoCo physics and write tracking logs."""

import argparse
from contextlib import nullcontext
import csv
from datetime import datetime
import json
from pathlib import Path
import time

import mujoco
import numpy as np

from .alicia_m import AliciaMSim, KP, KV, quintic_trajectory, validate_trajectory


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TARGET_COLUMNS = [f"target_j{i}_rad" for i in range(1, 7)]


def read_trajectory(path, joint_limits):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = ["time_s"] + TARGET_COLUMNS
        if not set(required).issubset(reader.fieldnames or []):
            raise ValueError("Trajectory CSV needs: " + ",".join(required))
        rows = list(reader)
    return validate_trajectory(
        [float(row["time_s"]) for row in rows],
        [[float(row[col]) for col in TARGET_COLUMNS] for row in rows], joint_limits,
    )


def execute(sim, timestamps, positions, output, viewer=None, settle=1.0,
            target_velocities=None, target_accelerations=None, velocity_feedforward=False):
    """Use simulation time to execute arbitrary N x 6 position trajectories."""
    timestamps, positions = validate_trajectory(timestamps, positions, sim.joint_limits)
    if not np.isfinite(settle) or settle < 0:
        raise ValueError("settle must be finite and nonnegative")
    for values in (target_velocities, target_accelerations):
        if values is not None and (np.asarray(values).shape != positions.shape or not np.isfinite(values).all()):
            raise ValueError("Reference derivatives must be finite N x 6 arrays")
    if target_velocities is not None:
        target_velocities = np.asarray(target_velocities, dtype=float)
    kp = sim.model.actuator_gainprm[sim.arm_actuators, 0]
    kv = -sim.model.actuator_biasprm[sim.arm_actuators, 2]
    if velocity_feedforward:
        if target_velocities is None:
            raise ValueError("Velocity feedforward requires target_velocities")
        # Position servo: kp*(command-q)-kv*v. Set command=q_ref+kv/kp*v_ref
        # to obtain kp*(q_ref-q)+kv*(v_ref-v), without changing logged q_ref.
        validate_trajectory(timestamps, positions + (kv / kp) * target_velocities, sim.joint_limits)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with viewer.lock() if viewer is not None else nullcontext():
        sim.reset(positions[0])
    dt = sim.model.opt.timestep
    steps = int(np.ceil((timestamps[-1] + settle) / dt))
    targets, actuals = [], []
    started = time.perf_counter()
    sync_at = 0.0
    completed = True
    columns = ["time_s"] + TARGET_COLUMNS
    columns += [f"actual_j{i}_rad" for i in range(1, 7)]
    columns += [f"actual_v{i}_rad_s" for i in range(1, 7)]
    columns += [f"actual_a{i}_rad_s2" for i in range(1, 7)]
    columns += [f"actuator_tau{i}_nm" for i in range(1, 7)]
    columns += ["gripper_opening_m"]
    if target_velocities is not None:
        columns += [f"target_v{i}_rad_s" for i in range(1, 7)]
    if target_accelerations is not None:
        columns += [f"target_a{i}_rad_s2" for i in range(1, 7)]
    if velocity_feedforward:
        columns += [f"command_j{i}_rad" for i in range(1, 7)]
    with (output / "tracking.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        for _ in range(steps):
            if viewer is not None and not viewer.is_running():
                completed = False
                break
            next_time = float(sim.data.time + dt)
            target = np.array([np.interp(next_time, timestamps, positions[:, j]) for j in range(6)])
            command = target.copy()
            if velocity_feedforward and next_time < timestamps[-1]:
                velocity = np.array([np.interp(next_time, timestamps, target_velocities[:, j]) for j in range(6)])
                command += (kv / kp) * velocity
            with viewer.lock() if viewer is not None else nullcontext():
                sim.set_joint_target(command)
                sim.step()
                state = sim.get_joint_state()
                actual = state["positions"]
                fingers = sim.data.qpos[sim.finger_qpos].copy()
            targets.append(target)
            actuals.append(actual)
            row = [state["time"], *target, *actual, *state["velocities"],
                   *state["accelerations"], *state["torques"], float(fingers[1] - fingers[0])]
            for reference in (target_velocities, target_accelerations):
                if reference is not None:
                    # The target is held still after the planned motion ends.
                    row.extend([np.interp(next_time, timestamps, np.asarray(reference)[:, j])
                                if next_time < timestamps[-1] else 0.0 for j in range(6)])
            if velocity_feedforward:
                row.extend(command)
            writer.writerow(row)
            if viewer is not None:
                if sim.data.time >= sync_at:
                    viewer.sync()
                    sync_at = float(sim.data.time) + 1 / 60
                delay = float(sim.data.time) - (time.perf_counter() - started)
                if delay > 0:
                    time.sleep(delay)
    if not actuals:
        raise RuntimeError("Viewer closed before any physics step was executed")
    errors = np.rad2deg(np.asarray(actuals) - np.asarray(targets))
    summary = {
        "model": "Alicia-M v1.1 follower",
        "mujoco_version": mujoco.__version__,
        "controller": "simulation position PD",
        "velocity_feedforward": bool(velocity_feedforward),
        "ideal_external_gravity_compensation": sim.gravity_comp,
        "kp": sim.model.actuator_gainprm[sim.arm_actuators, 0].tolist(),
        "kv": (-sim.model.actuator_biasprm[sim.arm_actuators, 2]).tolist(),
        "actuator_force_limits": sim.model.actuator_forcerange.tolist(),
        "timestep_s": float(dt), "duration_s": float(sim.data.time),
        "steps": len(actuals), "completed": completed,
        "rms_error_deg_per_joint": np.sqrt(np.mean(errors**2, axis=0)).tolist(),
        "max_abs_error_deg_per_joint": np.max(np.abs(errors), axis=0).tolist(),
        "final_error_deg_per_joint": errors[-1].tolist(),
        "actual_motion_deg_per_joint": np.rad2deg(np.ptp(np.asarray(actuals), axis=0)).tolist(),
        "warning_count": sum(int(w.number) for w in sim.data.warning),
        "scope": "Simulation-only. Gains and ideal gravity support are not validated on hardware.",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    from .report import plot_joint_states
    plot_joint_states(output / "tracking.csv")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Alicia-M v1.1 simulated position control; no serial connection")
    parser.add_argument("--headless", action="store_true", help="Run physics without a window")
    parser.add_argument("--no-gravity-comp", action="store_true", help="Disable ideal external gravity support")
    parser.add_argument("--trajectory-csv", type=Path, help="time_s,target_j1_rad,...,target_j6_rad")
    parser.add_argument("--duration", type=float, default=4.0, help="Demo trajectory time in seconds")
    parser.add_argument("--settle", type=float, default=1.0, help="Hold final target before finishing log")
    parser.add_argument("--output", type=Path, help="Output folder; defaults to a timestamped logs/mujoco folder")
    parser.add_argument("--model", type=Path, help="Path to official v1.1 follower XML")
    parser.add_argument("--export-model", type=Path, help="Also save the controlled XML for native simulate.exe")
    parser.add_argument("--close", action="store_true", help="Close viewer after trajectory completes")
    args = parser.parse_args()
    sim = AliciaMSim(source=args.model, gravity_comp=not args.no_gravity_comp)
    if args.export_model:
        args.export_model.parent.mkdir(parents=True, exist_ok=True)
        args.export_model.write_text(sim.xml, encoding="utf-8")
    if args.trajectory_csv:
        t, q = read_trajectory(args.trajectory_csv, sim.joint_limits)
    else:
        goal = np.deg2rad([20, -75, -30, 10, 40, -10])
        t, q = quintic_trajectory(sim.get_joint_positions(), goal, args.duration, sim.model.opt.timestep)
    validate_trajectory(t, q, sim.joint_limits)
    output = args.output or PROJECT_ROOT / "logs" / "mujoco" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    print("Simulation only: Alicia-M v1.1; no hardware connection.", flush=True)
    print(f"Ideal external gravity compensation: {sim.gravity_comp}", flush=True)
    context = nullcontext(None)
    if not args.headless:
        import mujoco.viewer
        context = mujoco.viewer.launch_passive(sim.model, sim.data)
    with context as viewer:
        if viewer is not None:
            with viewer.lock():
                viewer.cam.lookat[:] = [0, 0, 0.35]
                viewer.cam.distance = 1.4
                viewer.cam.azimuth = 135
                viewer.cam.elevation = -25
        summary = execute(sim, t, q, output, viewer, args.settle)
        print(f"Completed: {summary['completed']}; steps: {summary['steps']}; warnings: {summary['warning_count']}", flush=True)
        print("RMS joint error (deg):", np.round(summary["rms_error_deg_per_joint"], 3), flush=True)
        print(f"Logs: {output.resolve()}", flush=True)
        if viewer is not None and not args.close:
            print("Trajectory finished; holding final target. Close the window or press Ctrl+C to exit.", flush=True)
            while viewer.is_running():
                tick = time.perf_counter()
                with viewer.lock():
                    sim.step(10)
                viewer.sync()
                time.sleep(max(0, 10 * sim.model.opt.timestep - (time.perf_counter() - tick)))
    return 0 if summary["completed"] else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Stopped.")
