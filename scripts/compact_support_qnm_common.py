from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import minimum_filter


@dataclass
class CompactSupportConfig:
    height: float = 12.0
    width: float = 1.0
    domain: float = 3.0
    n_steps: int = 900
    match_x: float = 0.0


@dataclass
class CompactSupportEvaluator:
    config: CompactSupportConfig

    def __post_init__(self) -> None:
        self.x_left, self.x_right, self.match_index_left, self.match_index_right = make_grids(self.config)

    def wronskian(self, s: complex) -> complex:
        return wronskian_at_match(
            s,
            x_left=self.x_left,
            x_right=self.x_right,
            match_index_left=self.match_index_left,
            match_index_right=self.match_index_right,
            height=self.config.height,
            width=self.config.width,
        )

    def scan_grid(self, alpha: np.ndarray, beta: np.ndarray) -> np.ndarray:
        values = np.zeros((alpha.size, beta.size), dtype=complex)
        for i, a in enumerate(alpha):
            for j, b in enumerate(beta):
                values[i, j] = self.wronskian(a + 1j * b)
        return values


def compact_support_potential(x: np.ndarray, height: float = 12.0, width: float = 1.0) -> np.ndarray:
    y = np.zeros_like(x, dtype=float)
    mask = np.abs(x) < width
    z = x[mask] / width
    y[mask] = height * np.exp(-1.0 / (1.0 - z**2))
    return y


def rhs(x: float, y: np.ndarray, s: complex, height: float, width: float) -> np.ndarray:
    psi, dpsi = y
    v = compact_support_potential(np.array([x]), height=height, width=width)[0]
    return np.array([dpsi, (s**2 + v) * psi], dtype=complex)


def rk4_path(x_grid: np.ndarray, y0: np.ndarray, s: complex, height: float, width: float) -> np.ndarray:
    y = np.zeros((x_grid.size, 2), dtype=complex)
    y[0] = y0
    for i in range(x_grid.size - 1):
        x = x_grid[i]
        h = x_grid[i + 1] - x
        k1 = rhs(x, y[i], s, height, width)
        k2 = rhs(x + 0.5 * h, y[i] + 0.5 * h * k1, s, height, width)
        k3 = rhs(x + 0.5 * h, y[i] + 0.5 * h * k2, s, height, width)
        k4 = rhs(x + h, y[i] + h * k3, s, height, width)
        y[i + 1] = y[i] + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return y


def make_grids(config: CompactSupportConfig) -> tuple[np.ndarray, np.ndarray, int, int]:
    # The potential vanishes exactly outside |x| < width, so the asymptotic
    # boundary conditions can be imposed at x = ±width instead of at ±domain.
    x_left = np.linspace(-config.width, config.match_x, config.n_steps)
    x_right = np.linspace(config.width, config.match_x, config.n_steps)
    match_index_left = x_left.size - 1
    match_index_right = x_right.size - 1
    return x_left, x_right, match_index_left, match_index_right


def wronskian_at_match(
    s: complex,
    x_left: np.ndarray,
    x_right: np.ndarray,
    match_index_left: int,
    match_index_right: int,
    height: float,
    width: float,
) -> complex:
    left_boundary = np.array([np.exp(s * x_left[0]), s * np.exp(s * x_left[0])], dtype=complex)
    right_boundary = np.array([np.exp(-s * x_right[0]), -s * np.exp(-s * x_right[0])], dtype=complex)

    left_sol = rk4_path(x_left, left_boundary, s, height, width)
    right_sol = rk4_path(x_right, right_boundary, s, height, width)

    f_minus, df_minus = left_sol[match_index_left]
    f_plus, df_plus = right_sol[match_index_right]
    return f_minus * df_plus - df_minus * f_plus


def scan_wronskian_grid(
    alpha: np.ndarray,
    beta: np.ndarray,
    config: CompactSupportConfig,
) -> np.ndarray:
    evaluator = CompactSupportEvaluator(config)
    return evaluator.scan_grid(alpha, beta)


def find_local_minima(values: np.ndarray, alpha: np.ndarray, beta: np.ndarray, n_keep: int = 6):
    mag = np.abs(values)
    filtered = minimum_filter(mag, size=3, mode="nearest")
    mask = np.isclose(mag, filtered)
    candidates = np.argwhere(mask)
    ranked = sorted(candidates, key=lambda ij: mag[tuple(ij)])[:n_keep]
    out = []
    for ia, ib in ranked:
        out.append((alpha[ia], beta[ib], values[ia, ib], mag[ia, ib]))
    return out


def wronskian_function(s: complex, config: CompactSupportConfig) -> complex:
    evaluator = CompactSupportEvaluator(config)
    return evaluator.wronskian(s)


def complex_newton_root(
    s0: complex,
    config: CompactSupportConfig | None = None,
    evaluator: CompactSupportEvaluator | None = None,
    tol: float = 1e-9,
    max_iter: int = 25,
    diff_step: float = 1e-5,
) -> tuple[complex, complex, list[tuple[int, complex, complex]]]:
    if evaluator is None:
        if config is None:
            raise ValueError("Either config or evaluator must be provided.")
        evaluator = CompactSupportEvaluator(config)
    s = complex(s0)
    history: list[tuple[int, complex, complex]] = []
    w = evaluator.wronskian(s)
    history.append((0, s, w))

    for k in range(1, max_iter + 1):
        if abs(w) < tol:
            break

        h = diff_step * max(1.0, abs(s))
        wp = (evaluator.wronskian(s + h) - evaluator.wronskian(s - h)) / (2.0 * h)
        if abs(wp) < 1e-14:
            break

        delta = -w / wp
        damping = 1.0
        accepted = False
        best_s = s
        best_w = w
        for _ in range(12):
            s_trial = s + damping * delta
            w_trial = evaluator.wronskian(s_trial)
            if abs(w_trial) < abs(best_w):
                best_s = s_trial
                best_w = w_trial
                accepted = True
                break
            damping *= 0.5

        if not accepted:
            break

        s = best_s
        w = best_w
        history.append((k, s, w))

        if abs(delta) * damping < tol:
            break

    return s, w, history


def deduplicate_roots(roots: list[tuple[complex, complex]], tol: float = 5e-4) -> list[tuple[complex, complex]]:
    unique: list[tuple[complex, complex]] = []
    for s, w in roots:
        if not any(abs(s - s_old) < tol for s_old, _ in unique):
            unique.append((s, w))
    return unique


def initial_roots_from_scan(
    config: CompactSupportConfig,
    nroots: int,
    fast: bool,
    tol: float,
) -> list[complex]:
    evaluator = CompactSupportEvaluator(config)
    alpha = np.linspace(-2.4, -0.05, 28 if fast else 72)
    beta = np.linspace(0.1, 5.0, 38 if fast else 100)
    values = evaluator.scan_grid(alpha, beta)
    minima = find_local_minima(values, alpha, beta, n_keep=max(2 * nroots, 8))

    roots_raw: list[tuple[complex, complex]] = []
    for a, b, _, _ in minima[: max(2 * nroots, 4)]:
        root, wroot, _ = complex_newton_root(complex(a, b), evaluator=evaluator, tol=tol)
        roots_raw.append((root, wroot))

    roots = deduplicate_roots(roots_raw)
    roots = sorted(roots, key=lambda pair: (pair[0].imag, -pair[0].real))
    return [s for s, w in roots[:nroots] if abs(w) < 1e-6]


def local_reseed_and_refine(
    center: complex,
    config: CompactSupportConfig,
    tol: float,
    span_re: float = 0.12,
    span_im: float = 0.18,
) -> tuple[complex, complex]:
    evaluator = CompactSupportEvaluator(config)
    alpha = np.linspace(center.real - span_re, center.real + span_re, 7)
    beta = np.linspace(max(0.05, center.imag - span_im), center.imag + span_im, 9)
    values = evaluator.scan_grid(alpha, beta)
    minima = find_local_minima(values, alpha, beta, n_keep=4)
    best_s = center
    best_w = evaluator.wronskian(center)
    for a, b, _, _ in minima:
        root, wroot, _ = complex_newton_root(complex(a, b), evaluator=evaluator, tol=tol)
        if abs(wroot) < abs(best_w):
            best_s, best_w = root, wroot
    return best_s, best_w
