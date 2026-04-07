from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs"
OUTPUT.mkdir(exist_ok=True)


def compact_support_potential(x: np.ndarray, height: float = 12.0, width: float = 1.0) -> np.ndarray:
    v = np.zeros_like(x)
    mask = np.abs(x) < width
    z = x[mask] / width
    v[mask] = height * np.exp(-1.0 / (1.0 - z**2))
    return v


def long_range_potential(x: np.ndarray, height: float = 9.0, scale: float = 2.0) -> np.ndarray:
    return height / (x**2 + scale**2)


def evolve(v: np.ndarray, x: np.ndarray, dt: float, t: np.ndarray, obs_x: float) -> tuple[np.ndarray, np.ndarray]:
    dx = x[1] - x[0]
    x0 = -10.0
    sigma = 0.8
    k0 = 3.0
    psi0 = np.exp(-((x - x0) ** 2) / (2.0 * sigma**2)) * np.cos(k0 * (x - x0))
    dpsi_dt0 = -np.gradient(psi0, dx)
    lap0 = (np.roll(psi0, -1) - 2.0 * psi0 + np.roll(psi0, 1)) / dx**2
    lap0[0] = 0.0
    lap0[-1] = 0.0

    psi_prev = psi0.copy()
    psi = psi0 + dt * dpsi_dt0 + 0.5 * dt**2 * (lap0 - v * psi0)
    psi[0] = 0.0
    psi[-1] = 0.0

    obs_idx = np.argmin(np.abs(x - obs_x))
    signal = [psi0[obs_idx], psi[obs_idx]]

    for _ in range(1, t.size - 1):
        lap = (np.roll(psi, -1) - 2.0 * psi + np.roll(psi, 1)) / dx**2
        psi_next = 2.0 * psi - psi_prev + dt**2 * (lap - v * psi)
        psi_next[0] = 0.0
        psi_next[-1] = 0.0
        signal.append(psi_next[obs_idx])
        psi_prev, psi = psi, psi_next

    return t[: len(signal)], np.asarray(signal)


def main() -> None:
    x_min, x_max = -40.0, 40.0
    dx = 0.05
    dt = 0.45 * dx
    t_max = 90.0

    x = np.arange(x_min, x_max + dx, dx)
    t = np.arange(0.0, t_max + dt, dt)
    obs_x = 8.0

    v_compact = compact_support_potential(x)
    v_long = long_range_potential(x)

    t1, s1 = evolve(v_compact, x, dt, t, obs_x)
    t2, s2 = evolve(v_long, x, dt, t, obs_x)

    fig, axes = plt.subplots(2, 1, figsize=(9, 8))
    axes[0].plot(x, v_compact, label="compact-support potential")
    axes[0].plot(x, v_long, label="long-range potential")
    axes[0].set_xlim(-10, 10)
    axes[0].set_title("Potentials for late-time comparison")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("V(x)")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].plot(t1, np.log10(np.abs(s1) + 1e-12), label="compact support")
    axes[1].plot(t2, np.log10(np.abs(s2) + 1e-12), label="long range")
    axes[1].set_title("Late-time behavior comparison")
    axes[1].set_xlabel("t")
    axes[1].set_ylabel(r"$\log_{10}|\psi|$")
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(OUTPUT / "tail_comparison.png", dpi=180)
    plt.close(fig)

    lines = [
        "Tail comparison summary",
        "The compact-support potential usually shows a cleaner exponentially damped window.",
        "The long-range potential is included as a toy model for the paper's warning that slow decay can modify the late-time asymptotics.",
    ]
    (OUTPUT / "tail_comparison.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

