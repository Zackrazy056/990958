from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import root_scalar


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs"
OUTPUT.mkdir(exist_ok=True)

GMSUN_OVER_C3 = 4.92549095e-6


def dimensionless_m_omega(a: np.ndarray) -> np.ndarray:
    return 1.0 - 0.63 * (1.0 - a) ** 0.3


def dimensionless_tau_over_m(a: np.ndarray) -> np.ndarray:
    return 4.0 * (1.0 - a) ** 0.9 / (1.0 - 0.63 * (1.0 - a) ** 0.3)


def frequency_hz(mass_msun: np.ndarray, a: np.ndarray) -> np.ndarray:
    return dimensionless_m_omega(a) / (2.0 * np.pi * GMSUN_OVER_C3 * mass_msun)


def damping_time_s(mass_msun: np.ndarray, a: np.ndarray) -> np.ndarray:
    return dimensionless_tau_over_m(a) * GMSUN_OVER_C3 * mass_msun


def mass_from_frequency(f_obs: float, spin: float) -> float:
    return dimensionless_m_omega(spin) / (2.0 * np.pi * GMSUN_OVER_C3 * f_obs)


def tau_residual(spin: float, f_obs: float, tau_obs: float) -> float:
    mass = mass_from_frequency(f_obs, spin)
    return damping_time_s(mass, spin) - tau_obs


def main() -> None:
    a_grid = np.linspace(0.0, 0.98, 200)
    masses = [10.0, 30.0]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for mass in masses:
        axes[0].plot(a_grid, frequency_hz(mass, a_grid), label=rf"$M={mass:.0f} M_\odot$")
        axes[1].plot(a_grid, 1e3 * damping_time_s(mass, a_grid), label=rf"$M={mass:.0f} M_\odot$")

    axes[0].set_title("Ringdown frequency from the paper's Kerr fit")
    axes[0].set_xlabel("dimensionless spin a")
    axes[0].set_ylabel("f [Hz]")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].set_title("Ringdown damping time from the paper's Kerr fit")
    axes[1].set_xlabel("dimensionless spin a")
    axes[1].set_ylabel(r"$\tau$ [ms]")
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(OUTPUT / "black_hole_parameter_estimation_curves.png", dpi=180)
    plt.close(fig)

    true_mass = 18.0
    true_spin = 0.72
    f_obs = frequency_hz(true_mass, true_spin)
    tau_obs = damping_time_s(true_mass, true_spin)

    root = root_scalar(
        tau_residual,
        bracket=[0.0, 0.999],
        args=(f_obs, tau_obs),
        xtol=1e-12,
        rtol=1e-12,
    )
    spin_fit = root.root
    mass_fit = mass_from_frequency(f_obs, spin_fit)

    lines = [
        "Black-hole parameter estimation using paper equations (67) and (68)",
        f"synthetic_true_mass_msun = {true_mass:.6f}",
        f"synthetic_true_spin = {true_spin:.6f}",
        f"synthetic_observed_frequency_hz = {f_obs:.6f}",
        f"synthetic_observed_tau_ms = {1e3 * tau_obs:.6f}",
        f"recovered_mass_msun = {mass_fit:.6f}",
        f"recovered_spin = {spin_fit:.6f}",
        f"mass_abs_error = {abs(mass_fit - true_mass):.6e}",
        f"spin_abs_error = {abs(spin_fit - true_spin):.6e}",
    ]
    (OUTPUT / "black_hole_parameter_estimation.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
