from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs"
OUTPUT.mkdir(exist_ok=True)


def damped_cosine(t: np.ndarray, amplitude: float, alpha: float, beta: float, phase: float) -> np.ndarray:
    return amplitude * np.exp(alpha * t) * np.cos(beta * t + phase)


def main() -> None:
    rng = np.random.default_rng(1234)
    t = np.linspace(0.0, 30.0, 3000)

    true_params = np.array([1.0, -0.12, 2.45, 0.35])
    clean = damped_cosine(t, *true_params)
    noise = 0.03 * rng.normal(size=t.size)
    noisy = clean + noise

    fit_mask = (t >= 2.0) & (t <= 20.0)
    fit_t = t[fit_mask]
    fit_y = noisy[fit_mask]

    p0 = [0.8, -0.1, 2.2, 0.0]
    bounds = ([0.0, -2.0, 0.1, -2.0 * np.pi], [5.0, -1e-4, 6.0, 2.0 * np.pi])
    params, cov = curve_fit(damped_cosine, fit_t, fit_y, p0=p0, bounds=bounds, maxfev=20000)
    err = np.sqrt(np.diag(cov))
    fit_curve = damped_cosine(t, *params)

    fig, axes = plt.subplots(2, 1, figsize=(9, 8))
    axes[0].plot(t, noisy, color="black", lw=0.8, label="noisy signal")
    axes[0].plot(t, clean, color="tab:blue", lw=1.2, label="true signal")
    axes[0].plot(t, fit_curve, color="tab:red", lw=1.1, label="fit")
    axes[0].axvspan(fit_t[0], fit_t[-1], color="tab:red", alpha=0.08)
    axes[0].set_title("Synthetic ringdown and recovered fit")
    axes[0].set_xlabel("t")
    axes[0].set_ylabel("h(t)")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].plot(t, noisy - fit_curve, color="tab:green", lw=0.9)
    axes[1].set_title("Residual")
    axes[1].set_xlabel("t")
    axes[1].set_ylabel("data - fit")
    axes[1].grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(OUTPUT / "ringdown_fit_demo.png", dpi=180)
    plt.close(fig)

    names = ["amplitude", "alpha", "beta", "phase"]
    lines = ["Ringdown fitting demo", "True parameters vs recovered parameters:"]
    for name, true, est, sigma in zip(names, true_params, params, err):
        lines.append(f"{name}: true={true:.6f}, fit={est:.6f}, sigma_fit={sigma:.6e}")

    lines.append("")
    lines.append(
        f"Recovered complex frequency = {params[1]:+.6f}{params[2]:+.6f}i"
    )
    (OUTPUT / "ringdown_fit_demo.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

