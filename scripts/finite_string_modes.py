from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs"
OUTPUT.mkdir(exist_ok=True)


def main() -> None:
    length = 1.0
    c = 1.0
    n_points = 300
    n_modes = 4

    x = np.linspace(0.0, length, n_points)
    dx = x[1] - x[0]

    interior = x[1:-1]
    size = interior.size
    lap = (
        np.diag(-2.0 * np.ones(size))
        + np.diag(np.ones(size - 1), 1)
        + np.diag(np.ones(size - 1), -1)
    ) / dx**2
    operator = -c**2 * lap

    eigvals, eigvecs = np.linalg.eigh(operator)
    omegas = np.sqrt(eigvals[:n_modes])

    fig, ax = plt.subplots(figsize=(8, 5))
    for i in range(n_modes):
        y = np.zeros_like(x)
        mode = eigvecs[:, i]
        mode /= np.max(np.abs(mode))
        y[1:-1] = mode
        ax.plot(x, y + 1.6 * i, label=f"mode {i+1}")

    ax.set_title("Finite string normal modes")
    ax.set_xlabel("x / L")
    ax.set_ylabel("eigenfunction (offset)")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT / "finite_string_modes.png", dpi=180)
    plt.close(fig)

    lines = ["Finite string mode summary", f"L = {length}", f"c = {c}", ""]
    for i, omega_num in enumerate(omegas, start=1):
        omega_exact = i * np.pi * c / length
        rel_err = abs(omega_num - omega_exact) / omega_exact
        lines.append(
            f"n={i}: omega_num={omega_num:.6f}, "
            f"omega_exact={omega_exact:.6f}, rel_err={rel_err:.3e}"
        )

    (OUTPUT / "finite_string_modes.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

