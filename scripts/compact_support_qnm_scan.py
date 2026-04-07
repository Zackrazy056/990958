import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import minimum_filter


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs"
OUTPUT.mkdir(exist_ok=True)


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Scan QNM poles for a compact-support toy potential.")
    parser.add_argument("--fast", action="store_true", help="Use a smaller complex grid.")
    parser.add_argument("--height", type=float, default=12.0)
    parser.add_argument("--width", type=float, default=1.0)
    parser.add_argument("--domain", type=float, default=3.0)
    args = parser.parse_args()

    domain = args.domain
    n_steps = 500 if args.fast else 900
    x_left = np.linspace(-domain, 0.0, n_steps)
    x_right = np.linspace(domain, 0.0, n_steps)
    match_index_left = x_left.size - 1
    match_index_right = x_right.size - 1

    alpha = np.linspace(-2.4, -0.05, 42 if args.fast else 80)
    beta = np.linspace(0.1, 5.0, 56 if args.fast else 110)
    values = np.zeros((alpha.size, beta.size), dtype=complex)

    for i, a in enumerate(alpha):
        for j, b in enumerate(beta):
            s = a + 1j * b
            values[i, j] = wronskian_at_match(
                s,
                x_left=x_left,
                x_right=x_right,
                match_index_left=match_index_left,
                match_index_right=match_index_right,
                height=args.height,
                width=args.width,
            )

    minima = find_local_minima(values, alpha, beta)

    x_plot = np.linspace(-domain, domain, 1200)
    v_plot = compact_support_potential(x_plot, height=args.height, width=args.width)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    axes[0].plot(x_plot, v_plot, color="tab:blue")
    axes[0].set_title("Compact-support toy potential")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("V(x)")
    axes[0].grid(alpha=0.25)

    image = axes[1].imshow(
        np.log10(np.abs(values).T + 1e-14),
        origin="lower",
        aspect="auto",
        extent=[alpha[0], alpha[-1], beta[0], beta[-1]],
        cmap="magma",
    )
    axes[1].set_title(r"$\log_{10}|W(s)|$ on the complex-$s$ plane")
    axes[1].set_xlabel(r"$\mathrm{Re}(s)$")
    axes[1].set_ylabel(r"$\mathrm{Im}(s)$")
    for a, b, _, _ in minima[:4]:
        axes[1].plot(a, b, "co", ms=4)
    fig.colorbar(image, ax=axes[1], pad=0.02)
    fig.tight_layout()
    fig.savefig(OUTPUT / "compact_support_qnm_scan.png", dpi=180)
    plt.close(fig)

    lines = [
        "Compact-support QNM scan summary",
        f"height = {args.height}",
        f"width = {args.width}",
        f"domain = {domain}",
        "",
        "Candidate minima of |W(s)|:",
    ]
    for idx, (a, b, wval, mag) in enumerate(minima, start=1):
        lines.append(
            f"{idx}. s = {a:+.5f}{b:+.5f}i, "
            f"W = {wval.real:+.5e}{wval.imag:+.5e}i, |W| = {mag:.5e}"
        )

    (OUTPUT / "compact_support_qnm_scan.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

