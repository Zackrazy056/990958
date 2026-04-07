from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs"
OUTPUT.mkdir(exist_ok=True)


def tortoise(r: np.ndarray, m: float = 1.0) -> np.ndarray:
    return r + 2.0 * m * np.log(r / (2.0 * m) - 1.0)


def regge_wheeler_potential(r: np.ndarray, ell: int, m: float = 1.0) -> np.ndarray:
    f = 1.0 - 2.0 * m / r
    sigma = -3.0
    return f * (ell * (ell + 1.0) / r**2 + 2.0 * sigma * m / r**3)


def select_physical_root(z: complex) -> complex:
    root = np.sqrt(z)
    if root.real < 0:
        root = -root
    if root.imag > 0:
        root = root.conjugate()
    return root


def main() -> None:
    ell = 2
    overtone = 0

    r = np.linspace(2.0005, 20.0, 20000)
    rstar = tortoise(r)
    v = regge_wheeler_potential(r, ell=ell)

    mask = (rstar > -25.0) & (rstar < 50.0)
    rstar = rstar[mask]
    r = r[mask]
    v = v[mask]

    rstar_uniform = np.linspace(rstar.min(), rstar.max(), 25000)
    v_uniform = np.interp(rstar_uniform, rstar, v)

    peak_idx = np.argmax(v_uniform)
    v0 = v_uniform[peak_idx]
    drs = rstar_uniform[1] - rstar_uniform[0]
    vpp = (v_uniform[peak_idx + 1] - 2.0 * v0 + v_uniform[peak_idx - 1]) / drs**2

    omega_sq = v0 - 1j * (overtone + 0.5) * np.sqrt(-2.0 * vpp)
    omega = select_physical_root(omega_sq)

    literature = 0.37367 - 0.08896j
    abs_err = abs(omega - literature)
    rel_err = abs_err / abs(literature)

    window = slice(max(peak_idx - 1200, 0), min(peak_idx + 1200, rstar_uniform.size))
    rs_local = rstar_uniform[window]
    approx = v0 + 0.5 * vpp * (rs_local - rstar_uniform[peak_idx]) ** 2
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(rstar_uniform, v_uniform, lw=1.2, label="Regge-Wheeler potential")
    ax.plot(rs_local, approx, lw=1.4, ls="--", label="parabolic WKB peak approximation")
    ax.axvline(rstar_uniform[peak_idx], color="black", alpha=0.4)
    ax.set_xlim(rstar_uniform[peak_idx] - 20.0, rstar_uniform[peak_idx] + 20.0)
    ax.set_title(rf"First-order WKB near the Schwarzschild potential peak ($\ell={ell}$)")
    ax.set_xlabel(r"$r_*$")
    ax.set_ylabel("V")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT / "wkb_schwarzschild_qnm.png", dpi=180)
    plt.close(fig)

    lines = [
        "Schwarzschild first-order WKB estimate",
        f"ell = {ell}, n = {overtone}",
        f"peak_rstar = {rstar_uniform[peak_idx]:.6f}",
        f"V0 = {v0:.6f}",
        f"V0_pp = {vpp:.6f}",
        f"WKB omega M = {omega.real:.6f}{omega.imag:+.6f}i",
        f"Literature omega M = {literature.real:.6f}{literature.imag:+.6f}i",
        f"absolute_error = {abs_err:.6e}",
        f"relative_error = {rel_err:.6e}",
    ]

    (OUTPUT / "wkb_schwarzschild_qnm.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
