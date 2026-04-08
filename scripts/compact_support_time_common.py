from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import curve_fit

from compact_support_qnm_common import compact_support_potential


@dataclass
class TimeDomainConfig:
    x_min: float = -30.0
    x_max: float = 30.0
    dx: float = 0.05
    courant: float = 0.45
    t_max: float = 55.0
    obs_x: float = 8.0
    pulse_x0: float = -10.0
    pulse_sigma: float = 0.8
    pulse_k0: float = 3.0


def damped_cosine(t: np.ndarray, amplitude: float, alpha: float, beta: float, phase: float) -> np.ndarray:
    return amplitude * np.exp(alpha * t) * np.cos(beta * t + phase)


def evolve_compact_support_signal(
    potential_height: float,
    potential_width: float,
    config: TimeDomainConfig | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    if config is None:
        config = TimeDomainConfig()

    dt = config.courant * config.dx
    x = np.arange(config.x_min, config.x_max + config.dx, config.dx)
    t = np.arange(0.0, config.t_max + dt, dt)
    v = compact_support_potential(x, height=potential_height, width=potential_width)

    psi0 = np.exp(-((x - config.pulse_x0) ** 2) / (2.0 * config.pulse_sigma**2))
    psi0 *= np.cos(config.pulse_k0 * (x - config.pulse_x0))
    dpsi_dt0 = -np.gradient(psi0, config.dx)

    lap0 = (np.roll(psi0, -1) - 2.0 * psi0 + np.roll(psi0, 1)) / config.dx**2
    lap0[0] = 0.0
    lap0[-1] = 0.0

    psi_prev = psi0.copy()
    psi = psi0 + dt * dpsi_dt0 + 0.5 * dt**2 * (lap0 - v * psi0)
    psi[0] = 0.0
    psi[-1] = 0.0

    obs_idx = np.argmin(np.abs(x - config.obs_x))
    signal = [psi0[obs_idx], psi[obs_idx]]

    for _ in range(1, t.size - 1):
        lap = (np.roll(psi, -1) - 2.0 * psi + np.roll(psi, 1)) / config.dx**2
        psi_next = 2.0 * psi - psi_prev + dt**2 * (lap - v * psi)
        psi_next[0] = 0.0
        psi_next[-1] = 0.0
        signal.append(psi_next[obs_idx])
        psi_prev, psi = psi, psi_next

    return x, t[: len(signal)], v, np.asarray(signal), obs_idx


def fit_ringdown_window(
    t_signal: np.ndarray,
    signal: np.ndarray,
    fit_start: float = 18.0,
    fit_end: float = 36.0,
    p0: tuple[float, float, float, float] | None = None,
    bounds: tuple[tuple[float, float, float, float], tuple[float, float, float, float]] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    fit_mask = (t_signal >= fit_start) & (t_signal <= fit_end)
    fit_t = t_signal[fit_mask]
    fit_y = signal[fit_mask]
    if p0 is None:
        p0 = [np.max(np.abs(fit_y)), -0.12, 2.0, 0.0]
    if bounds is None:
        bounds = ([0.0, -3.0, 0.1, -2.0 * np.pi], [10.0, -1e-4, 10.0, 2.0 * np.pi])
    params, cov = curve_fit(damped_cosine, fit_t, fit_y, p0=p0, bounds=bounds, maxfev=20000)
    return params, cov, fit_mask


def estimate_ringdown_window(
    t_signal: np.ndarray,
    signal: np.ndarray,
    root: complex,
    guard_time: float = 1.0,
    min_window: float = 3.0,
    max_window: float = 10.0,
) -> tuple[float, float]:
    peak_idx = int(np.argmax(np.abs(signal)))
    peak_t = t_signal[peak_idx]
    damping_scale = 1.0 / max(abs(root.real), 0.08)
    window = min(max(2.8 * damping_scale, min_window), max_window)
    fit_start = peak_t + guard_time
    fit_end = min(t_signal[-1], fit_start + window)
    if fit_end <= fit_start + 1.0:
        fit_end = min(t_signal[-1], fit_start + min_window)
    return fit_start, fit_end
