"""Plot planned and actual joint states from a completed simulation log."""

from pathlib import Path

import numpy as np


def plot_joint_states(csv_path, show=False):
    import matplotlib.pyplot as plt

    csv_path = Path(csv_path)
    values = np.genfromtxt(csv_path, delimiter=",", names=True, encoding="utf-8")
    values = np.atleast_1d(values)
    t = values["time_s"]
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(11, 9))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for j in range(1, 7):
        color = colors[(j - 1) % len(colors)]
        for ax, actual_key, target_key in (
            (axes[0], f"actual_j{j}_rad", f"target_j{j}_rad"),
            (axes[1], f"actual_v{j}_rad_s", f"target_v{j}_rad_s"),
            (axes[2], f"actual_a{j}_rad_s2", f"target_a{j}_rad_s2"),
        ):
            ax.plot(t, np.rad2deg(values[actual_key]), color=color, label=f"J{j}", lw=1.3)
            if target_key in values.dtype.names:
                ax.plot(t, np.rad2deg(values[target_key]), color=color, linestyle="--", lw=0.9, alpha=0.7)
    for ax, label in zip(axes, ("Position (deg)", "Velocity (deg/s)", "Acceleration (deg/s^2)")):
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)
    axes[0].legend(ncol=6, loc="upper left")
    axes[2].set_xlabel("Simulation time (s)")
    fig.suptitle("Alicia-M v1.1 | solid: actual physics; dashed: planned")
    fig.tight_layout()
    output = csv_path.with_name("joint_states.png")
    fig.savefig(output, dpi=150)
    if show:
        plt.show(block=False)
    else:
        plt.close(fig)
    return output
