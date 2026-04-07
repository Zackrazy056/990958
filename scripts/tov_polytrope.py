from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs"
OUTPUT.mkdir(exist_ok=True)


MSUN_IN_KM = 1.47662504


def eos_pressure(rho0: float, k_poly: float, gamma: float) -> float:
    return k_poly * rho0**gamma


def eos_energy_density_from_pressure(p: float, k_poly: float, gamma: float) -> float:
    if p <= 0.0:
        return 0.0
    rho0 = (p / k_poly) ** (1.0 / gamma)
    return rho0 + p / (gamma - 1.0)


def tov_rhs(r: float, y: np.ndarray, k_poly: float, gamma: float) -> np.ndarray:
    m, p = y
    if p <= 0.0:
        return np.array([0.0, 0.0])
    eps = eos_energy_density_from_pressure(p, k_poly, gamma)
    dmdr = 4.0 * np.pi * r**2 * eps
    denom = r * (r - 2.0 * m)
    dpdr = -(eps + p) * (m + 4.0 * np.pi * r**3 * p) / denom
    return np.array([dmdr, dpdr])


def surface_event(r: float, y: np.ndarray, k_poly: float, gamma: float) -> float:
    return y[1] - 1e-10


surface_event.terminal = True
surface_event.direction = -1


def integrate_star(rho_c: float, k_poly: float, gamma: float):
    p_c = eos_pressure(rho_c, k_poly, gamma)
    eps_c = eos_energy_density_from_pressure(p_c, k_poly, gamma)
    r0 = 1e-4
    m0 = 4.0 * np.pi * eps_c * r0**3 / 3.0
    y0 = np.array([m0, p_c])

    sol = solve_ivp(
        lambda r, y: tov_rhs(r, y, k_poly, gamma),
        (r0, 30.0),
        y0,
        events=lambda r, y: surface_event(r, y, k_poly, gamma),
        max_step=0.05,
        rtol=1e-6,
        atol=1e-8,
    )
    radius = sol.t_events[0][0] if sol.t_events[0].size else sol.t[-1]
    mass = np.interp(radius, sol.t, sol.y[0])
    return sol, mass, radius


def main() -> None:
    gamma = 2.0
    k_poly = 100.0
    rho_cs = np.linspace(4.0e-4, 2.2e-3, 24)

    masses = []
    radii = []
    solutions = []
    for rho_c in rho_cs:
        sol, mass, radius = integrate_star(rho_c, k_poly, gamma)
        masses.append(mass / MSUN_IN_KM)
        radii.append(radius)
        solutions.append((rho_c, sol, mass, radius))

    masses = np.asarray(masses)
    radii = np.asarray(radii)
    best_idx = np.argmax(masses)
    rho_best, sol_best, mass_best_km, radius_best = solutions[best_idx]

    p_profile = sol_best.y[1]
    eps_profile = np.array([eos_energy_density_from_pressure(p, k_poly, gamma) for p in p_profile])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    axes[0].plot(radii, masses, marker="o", ms=3)
    axes[0].set_title(rf"TOV mass-radius curve, $K={k_poly}$, $\Gamma={gamma}$")
    axes[0].set_xlabel("R [km]")
    axes[0].set_ylabel(r"M [$M_\odot$]")
    axes[0].grid(alpha=0.25)

    axes[1].plot(sol_best.t, eps_profile, label="energy density")
    axes[1].plot(sol_best.t, p_profile, label="pressure")
    axes[1].set_title("Representative stellar profile")
    axes[1].set_xlabel("r [km]")
    axes[1].set_ylabel("geometric units")
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(OUTPUT / "tov_polytrope.png", dpi=180)
    plt.close(fig)

    lines = [
        "TOV relativistic polytrope summary",
        f"K = {k_poly}",
        f"Gamma = {gamma}",
        f"max_mass = {masses[best_idx]:.6f} Msun",
        f"radius_at_max_mass = {radii[best_idx]:.6f} km",
        f"central_rest_density_at_max_mass = {rho_best:.6e}",
    ]
    (OUTPUT / "tov_polytrope.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

