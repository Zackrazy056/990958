from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from compact_support_time_common import damped_cosine, evolve_compact_support_signal, fit_ringdown_window


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs"
OUTPUT.mkdir(exist_ok=True)

def main() -> None:
    x, t_signal, v, signal, obs_idx = evolve_compact_support_signal(12.0, 1.0)
    params, _, fit_mask = fit_ringdown_window(t_signal, signal)
    fit_t = t_signal[fit_mask]
    fit_curve = damped_cosine(fit_t, *params)

    fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=False)

    axes[0].plot(x, v, color="tab:blue")
    axes[0].set_title("Compact-support potential used in the time-domain evolution")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("V(x)")
    axes[0].grid(alpha=0.25)

    axes[1].plot(t_signal, signal, color="black", lw=1.0, label="signal")
    axes[1].plot(fit_t, fit_curve, color="tab:red", lw=1.2, label="ringdown fit")
    axes[1].axvspan(fit_t[0], fit_t[-1], color="tab:red", alpha=0.08)
    axes[1].set_title(f"Observed waveform at x = {x[obs_idx]:.2f}")
    axes[1].set_xlabel("t")
    axes[1].set_ylabel("psi")
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    axes[2].plot(t_signal, np.log10(np.abs(signal) + 1e-12), color="tab:green", lw=1.0)
    axes[2].axvspan(fit_t[0], fit_t[-1], color="tab:red", alpha=0.08)
    axes[2].set_title("Semi-log view of the observed signal")
    axes[2].set_xlabel("t")
    axes[2].set_ylabel(r"$\log_{10}|\psi|$")
    axes[2].grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(OUTPUT / "time_domain_wave_evolution.png", dpi=180)
    plt.close(fig)

    alpha, beta = params[1], params[2]
    lines = [
        "Time-domain wave evolution summary",
        f"observer_x = {x[obs_idx]:.3f}",
        f"fit_window = [{fit_t[0]:.3f}, {fit_t[-1]:.3f}]",
        f"fitted_alpha = {alpha:.6f}",
        f"fitted_beta = {beta:.6f}",
        f"fitted_complex_s = {alpha:+.6f}{beta:+.6f}i",
    ]
    (OUTPUT / "time_domain_wave_evolution.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
