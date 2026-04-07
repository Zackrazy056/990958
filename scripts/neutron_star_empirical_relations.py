from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs"
OUTPUT.mkdir(exist_ok=True)


def f_mode_khz(mass_msun: np.ndarray, radius_km: np.ndarray) -> np.ndarray:
    return 0.78 + 1.635 * np.sqrt((mass_msun / 1.4) * (10.0 / radius_km) ** 3)


def w_mode_khz(mass_msun: np.ndarray, radius_km: np.ndarray) -> np.ndarray:
    return (10.0 / radius_km) * (20.92 - 9.14 * (mass_msun / 1.4) * (10.0 / radius_km))


def main() -> None:
    mass = np.linspace(1.0, 2.2, 160)
    radius = np.linspace(8.0, 16.0, 180)
    mm, rr = np.meshgrid(mass, radius, indexing="xy")

    ff = f_mode_khz(mm, rr)
    ww = w_mode_khz(mm, rr)

    true_mass = 1.62
    true_radius = 11.4
    f_obs = f_mode_khz(true_mass, true_radius)
    w_obs = w_mode_khz(true_mass, true_radius)

    score = ((ff - f_obs) / 0.03) ** 2 + ((ww - w_obs) / 0.08) ** 2
    best_idx = np.unravel_index(np.argmin(score), score.shape)
    best_radius = radius[best_idx[0]]
    best_mass = mass[best_idx[1]]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    density_scale = np.sqrt((mass / 1.4) * (10.0 / 10.0) ** 3)
    axes[0].plot(np.sqrt((mass / 1.4)), f_mode_khz(mass, np.full_like(mass, 10.0)))
    axes[0].set_title("Paper-inspired f-mode scaling example")
    axes[0].set_xlabel(r"$\sqrt{M/1.4M_\odot}$ at fixed $R=10$ km")
    axes[0].set_ylabel(r"$f$-mode [kHz]")
    axes[0].grid(alpha=0.25)

    c1 = axes[1].contour(mm, rr, ff, levels=[f_obs], colors="tab:blue", linewidths=1.8)
    c2 = axes[1].contour(mm, rr, ww, levels=[w_obs], colors="tab:red", linewidths=1.8)
    axes[1].plot(true_mass, true_radius, "ko", label="synthetic truth")
    axes[1].plot(best_mass, best_radius, "g*", ms=10, label="best grid estimate")
    axes[1].set_title("Mass-radius inversion from f and w mode fits")
    axes[1].set_xlabel(r"M [$M_\odot$]")
    axes[1].set_ylabel("R [km]")
    axes[1].grid(alpha=0.25)
    contour_handles = [
        plt.Line2D([0], [0], color="tab:blue", lw=1.8, label="observed f-mode contour"),
        plt.Line2D([0], [0], color="tab:red", lw=1.8, label="observed w-mode contour"),
    ]
    point_handles = [
        plt.Line2D([0], [0], marker="o", color="black", lw=0, label="synthetic truth"),
        plt.Line2D([0], [0], marker="*", color="green", lw=0, markersize=10, label="best grid estimate"),
    ]
    axes[1].legend(handles=contour_handles + point_handles)

    fig.tight_layout()
    fig.savefig(OUTPUT / "neutron_star_empirical_relations.png", dpi=180)
    plt.close(fig)

    lines = [
        "Neutron-star empirical relations using paper equations (69) and (70)",
        f"synthetic_true_mass_msun = {true_mass:.6f}",
        f"synthetic_true_radius_km = {true_radius:.6f}",
        f"synthetic_f_mode_khz = {f_obs:.6f}",
        f"synthetic_w_mode_khz = {w_obs:.6f}",
        f"best_grid_mass_msun = {best_mass:.6f}",
        f"best_grid_radius_km = {best_radius:.6f}",
        f"mass_abs_error = {abs(best_mass - true_mass):.6e}",
        f"radius_abs_error = {abs(best_radius - true_radius):.6e}",
    ]
    (OUTPUT / "neutron_star_empirical_relations.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
