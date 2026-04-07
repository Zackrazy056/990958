from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs"
OUTPUT.mkdir(exist_ok=True)


def compact_support_potential(x: np.ndarray, height: float = 12.0, width: float = 1.0) -> np.ndarray:
    v = np.zeros_like(x)
    mask = np.abs(x) < width
    z = x[mask] / width
    v[mask] = height * np.exp(-1.0 / (1.0 - z**2))
    return v


def damped_cosine(t: np.ndarray, amplitude: float, alpha: float, beta: float, phase: float) -> np.ndarray:
    return amplitude * np.exp(alpha * t) * np.cos(beta * t + phase)


def main() -> None:
    x_min, x_max = -30.0, 30.0
    dx = 0.05
    courant = 0.45
    dt = courant * dx
    t_max = 55.0

    x = np.arange(x_min, x_max + dx, dx)
    t = np.arange(0.0, t_max + dt, dt)
    v = compact_support_potential(x, height=12.0, width=1.0)

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

    obs_x = 8.0
    obs_idx = np.argmin(np.abs(x - obs_x))
    signal = [psi0[obs_idx], psi[obs_idx]]

    for _ in range(1, t.size - 1):
        lap = (np.roll(psi, -1) - 2.0 * psi + np.roll(psi, 1)) / dx**2
        psi_next = 2.0 * psi - psi_prev + dt**2 * (lap - v * psi)
        psi_next[0] = 0.0
        psi_next[-1] = 0.0
        signal.append(psi_next[obs_idx])
        psi_prev, psi = psi, psi_next

    signal = np.asarray(signal)
    t_signal = t[: signal.size]

    fit_mask = (t_signal >= 18.0) & (t_signal <= 36.0)
    fit_t = t_signal[fit_mask]
    fit_y = signal[fit_mask]

    p0 = [np.max(np.abs(fit_y)), -0.12, 2.0, 0.0]
    bounds = ([0.0, -3.0, 0.1, -2.0 * np.pi], [10.0, -1e-4, 10.0, 2.0 * np.pi])
    params, _ = curve_fit(damped_cosine, fit_t, fit_y, p0=p0, bounds=bounds, maxfev=20000)
    fit_curve = damped_cosine(fit_t, *params)

    fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=False)

    axes[0].plot(x, v, color="tab:blue")
    axes[0].set_title("Compact-support potential used in the time-domain evolution")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("V(x)")
    axes[0].grid(alpha=0.25)

    axes[1].plot(t_signal, signal, color="black", lw=1.0, label="signal")
    axes[1].plot(fit_t, fit_curve, color="tab:red", lw=1.2, label="ringdown fit")
    axes[1].axvspan(fit_t[0], fit_t[-1], color="tab:red", alpha=0.08)
    axes[1].set_title(f"Observed waveform at x = {x[obs_idx]:.2f}")
    axes[1].set_xlabel("t")
    axes[1].set_ylabel("psi")
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    axes[2].plot(t_signal, np.log10(np.abs(signal) + 1e-12), color="tab:green", lw=1.0)
    axes[2].axvspan(fit_t[0], fit_t[-1], color="tab:red", alpha=0.08)
    axes[2].set_title("Semi-log view of the observed signal")
    axes[2].set_xlabel("t")
    axes[2].set_ylabel(r"$\log_{10}|\psi|$")
    axes[2].grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(OUTPUT / "time_domain_wave_evolution.png", dpi=180)
    plt.close(fig)

    alpha, beta = params[1], params[2]
    lines = [
        "Time-domain wave evolution summary",
        f"observer_x = {x[obs_idx]:.3f}",
        f"fit_window = [{fit_t[0]:.3f}, {fit_t[-1]:.3f}]",
        f"fitted_alpha = {alpha:.6f}",
        f"fitted_beta = {beta:.6f}",
        f"fitted_complex_s = {alpha:+.6f}{beta:+.6f}i",
    ]
    (OUTPUT / "time_domain_wave_evolution.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

