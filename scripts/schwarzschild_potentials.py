from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs"
OUTPUT.mkdir(exist_ok=True)


def tortoise(r: np.ndarray, m: float = 1.0) -> np.ndarray:
    return r + 2.0 * m * np.log(r / (2.0 * m) - 1.0)


def regge_wheeler_potential(r: np.ndarray, ell: int, spin: int = 2, m: float = 1.0) -> np.ndarray:
    sigma = 1 - spin**2
    f = 1.0 - 2.0 * m / r
    return f * (ell * (ell + 1.0) / r**2 + 2.0 * sigma * m / r**3)


def zerilli_potential(r: np.ndarray, ell: int, m: float = 1.0) -> np.ndarray:
    n = 0.5 * (ell - 1.0) * (ell + 2.0)
    f = 1.0 - 2.0 * m / r
    numerator = (
        2.0 * n**2 * (n + 1.0) * r**3
        + 6.0 * n**2 * m * r**2
        + 18.0 * n * m**2 * r
        + 18.0 * m**3
    )
    denominator = r**3 * (n * r + 3.0 * m) ** 2
    return f * numerator / denominator


def main() -> None:
    ell = 2
    r = np.linspace(2.02, 20.0, 2500)
    rstar = tortoise(r)

    vrw = regge_wheeler_potential(r, ell=ell)
    vz = zerilli_potential(r, ell=ell)

    peak_idx_rw = np.argmax(vrw)
    peak_idx_z = np.argmax(vz)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    axes[0].plot(r, vrw, label="Regge-Wheeler", lw=1.4)
    axes[0].plot(r, vz, label="Zerilli", lw=1.4)
    axes[0].axvline(3.0, color="black", ls="--", alpha=0.45, label="photon sphere")
    axes[0].set_title(rf"Schwarzschild potentials vs $r$ for $\ell={ell}$")
    axes[0].set_xlabel("r / M")
    axes[0].set_ylabel("V")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].plot(rstar, vrw, label="Regge-Wheeler", lw=1.4)
    axes[1].plot(rstar, vz, label="Zerilli", lw=1.4)
    axes[1].set_xlim(-10, 30)
    axes[1].set_title(rf"Schwarzschild potentials vs $r_*$ for $\ell={ell}$")
    axes[1].set_xlabel(r"$r_*$ / M")
    axes[1].set_ylabel("V")
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(OUTPUT / "schwarzschild_potentials.png", dpi=180)
    plt.close(fig)

    lines = [
        "Schwarzschild potential summary",
        f"ell = {ell}",
        f"RW peak at r = {r[peak_idx_rw]:.6f}, Vmax = {vrw[peak_idx_rw]:.6f}",
        f"Zerilli peak at r = {r[peak_idx_z]:.6f}, Vmax = {vz[peak_idx_z]:.6f}",
        "The peak sits close to the photon sphere at r = 3M.",
    ]
    (OUTPUT / "schwarzschild_potentials.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

