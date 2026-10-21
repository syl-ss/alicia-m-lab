"""Fixed experiment runner: users only supply trajectory(t) in degrees."""

from datetime import datetime
from pathlib import Path
import time

import numpy as np

from .alicia_m import AliciaMSim, KP, KV, validate_trajectory
from .run_trajectory import execute


def run_experiment(trajectory, duration=4.0, *, headless=False, output=None,
                   gravity_comp=True, keep_open=True, reference_derivatives=None,
                   kp=KP, kv=KV, velocity_feedforward=False):
    """Run the user's six-angle function and save actual physics state.

    trajectory(t): t is seconds, returns [J1,...,J6] in DEGREES.
    It is sampled before simulation; use a pure function of t.
    MuJoCo's viewer opens during execution; result plots are saved, never popped up.
    velocity_feedforward=True also uses reference velocity in the servo command.
    """
    if not callable(trajectory) or not np.isfinite(duration) or duration <= 0:
        raise ValueError("Provide a trajectory(t) function and positive finite duration")
    sim = AliciaMSim(gravity_comp=gravity_comp, kp=kp, kv=kv)
    count = max(2, int(np.ceil(duration / sim.model.opt.timestep)) + 1)
    timestamps = np.linspace(0, duration, count)
    positions = np.deg2rad(np.asarray([trajectory(float(t)) for t in timestamps], dtype=float))
    timestamps, positions = validate_trajectory(timestamps, positions, sim.joint_limits)
    derivatives = {}
    if reference_derivatives is not None:
        samples = np.asarray([reference_derivatives(float(t)) for t in timestamps], dtype=float)
        if samples.shape != (len(timestamps), 2, 6) or not np.isfinite(samples).all():
            raise ValueError("reference_derivatives(t) must return six velocities and accelerations in degrees")
        derivatives = {"target_velocities": np.deg2rad(samples[:, 0]),
                       "target_accelerations": np.deg2rad(samples[:, 1])}
    output = Path(output) if output is not None else (
        Path(__file__).resolve().parents[1] / 'logs' / 'mujoco'
        / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    )
    if headless:
        result = execute(sim, timestamps, positions, output,
                         velocity_feedforward=velocity_feedforward, **derivatives)
    else:
        import mujoco.viewer

        with mujoco.viewer.launch_passive(sim.model, sim.data) as viewer:
            with viewer.lock():
                viewer.cam.lookat[:] = [0, 0, 0.35]
                viewer.cam.distance = 1.4
                viewer.cam.azimuth = 135
                viewer.cam.elevation = -25
            result = execute(sim, timestamps, positions, output, viewer=viewer,
                             velocity_feedforward=velocity_feedforward, **derivatives)
            print(f"Motion finished. Results: {output.resolve()}", flush=True)
            if keep_open and viewer.is_running():
                print("Holding final target; close MuJoCo window or press Ctrl+C to exit.", flush=True)
                while viewer.is_running():
                    started = time.perf_counter()
                    with viewer.lock():
                        sim.step(10)
                    viewer.sync()
                    time.sleep(max(0, 10 * sim.model.opt.timestep - (time.perf_counter() - started)))
    print(f"CSV: {output.resolve() / 'tracking.csv'}", flush=True)
    print(f"Saved plot (not opened): {output.resolve() / 'joint_states.png'}", flush=True)
    return result
