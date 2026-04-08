import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from compact_support_qnm_common import (
    CompactSupportConfig,
    CompactSupportEvaluator,
    complex_newton_root,
    initial_roots_from_scan,
    local_reseed_and_refine,
)
from compact_support_time_common import (
    TimeDomainConfig,
    damped_cosine,
    estimate_ringdown_window,
    evolve_compact_support_signal,
    fit_ringdown_window,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs"
OUTPUT.mkdir(exist_ok=True)


def format_complex(z: complex) -> str:
    return f"{z.real:+.8f}{z.imag:+.8f}i"


def refine_from_seed(seed: complex, config: CompactSupportConfig, tol: float) -> tuple[complex, float]:
    evaluator = CompactSupportEvaluator(config)
    root, wroot, _ = complex_newton_root(seed, evaluator=evaluator, tol=tol)
    if abs(wroot) > 1e-6:
        root, wroot = local_reseed_and_refine(seed, config, tol)
    return root, abs(wroot)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare frequency-domain toy QNM roots against time-domain ringdown fits."
    )
    parser.add_argument("--parameter", choices=["height", "width"], default="height")
    parser.add_argument("--start", type=float, default=8.0)
    parser.add_argument("--stop", type=float, default=14.0)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--height", type=float, default=12.0)
    parser.add_argument("--width", type=float, default=1.0)
    parser.add_argument("--domain", type=float, default=3.0)
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--tol", type=float, default=1e-9)
    parser.add_argument("--fit-start", type=float, default=None)
    parser.add_argument("--fit-end", type=float, default=None)
    args = parser.parse_args()

    values = np.linspace(args.start, args.stop, args.steps)
    n_steps = 240 if args.fast else 650
    base_config = CompactSupportConfig(
        height=values[0] if args.parameter == "height" else args.height,
        width=values[0] if args.parameter == "width" else args.width,
        domain=args.domain,
        n_steps=n_steps,
    )
    root_seed = initial_roots_from_scan(base_config, nroots=1, fast=args.fast, tol=args.tol)[0]

    roots = np.full(values.size, np.nan + 1j * np.nan, dtype=complex)
    residuals = np.full(values.size, np.nan)
    fitted = np.full(values.size, np.nan + 1j * np.nan, dtype=complex)
    fit_windows = np.full((values.size, 2), np.nan)

    sample_indices = sorted(set([0, values.size // 2, values.size - 1]))
    sample_waveforms: list[tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []

    current_seed = root_seed
    for i, value in enumerate(values):
        config = CompactSupportConfig(
            height=value if args.parameter == "height" else args.height,
            width=value if args.parameter == "width" else args.width,
            domain=args.domain,
            n_steps=n_steps,
        )
        root, res = refine_from_seed(current_seed, config, args.tol)
        roots[i] = root
        residuals[i] = res
        current_seed = root

        pot_height = value if args.parameter == "height" else args.height
        pot_width = value if args.parameter == "width" else args.width
        td_config = TimeDomainConfig(
            x_min=-20.0,
            x_max=20.0,
            dx=0.05,
            courant=0.45,
            t_max=40.0,
            obs_x=5.0,
            pulse_x0=0.0,
            pulse_sigma=0.55,
            pulse_k0=0.0,
        )
        x, t_signal, v, signal, _ = evolve_compact_support_signal(
            pot_height, pot_width, config=td_config
        )
        if args.fit_start is None or args.fit_end is None:
            fit_start, fit_end = estimate_ringdown_window(t_signal, signal, root)
        else:
            fit_start, fit_end = args.fit_start, args.fit_end
        fit_windows[i] = [fit_start, fit_end]
        p0 = (max(np.max(np.abs(signal)), 1e-6), root.real, root.imag, 0.0)
        alpha_low = max(-3.0, root.real - 0.8)
        alpha_high = min(-1e-4, root.real + 0.8)
        beta_low = max(0.1, root.imag - 0.9)
        beta_high = min(10.0, root.imag + 0.9)
        bounds = ((0.0, alpha_low, beta_low, -2.0 * np.pi), (10.0, alpha_high, beta_high, 2.0 * np.pi))
        params, _, fit_mask = fit_ringdown_window(
            t_signal, signal, fit_start=fit_start, fit_end=fit_end, p0=p0, bounds=bounds
        )
        fitted[i] = params[1] + 1j * params[2]

        if i in sample_indices:
            fit_t = t_signal[fit_mask]
            fit_curve = damped_cosine(fit_t, *params)
            sample_waveforms.append((value, t_signal, signal, fit_t, fit_curve))

    fig = plt.figure(figsize=(13, 10))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.0, 1.2])

    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])
    ax4 = fig.add_subplot(gs[2, :])

    xlab = "potential height" if args.parameter == "height" else "potential width"
    ax0.plot(values, roots.real, "-o", label="frequency-domain root")
    ax0.plot(values, fitted.real, "-s", label="time-domain fit")
    ax0.set_title(r"Comparison of $\mathrm{Re}(s)$")
    ax0.set_xlabel(xlab)
    ax0.set_ylabel(r"$\mathrm{Re}(s)$")
    ax0.grid(alpha=0.25)
    ax0.legend()

    ax1.plot(values, roots.imag, "-o", label="frequency-domain root")
    ax1.plot(values, fitted.imag, "-s", label="time-domain fit")
    ax1.set_title(r"Comparison of $\mathrm{Im}(s)$")
    ax1.set_xlabel(xlab)
    ax1.set_ylabel(r"$\mathrm{Im}(s)$")
    ax1.grid(alpha=0.25)
    ax1.legend()

    ax2.plot(roots.real, roots.imag, "-o", label="frequency-domain roots")
    ax2.plot(fitted.real, fitted.imag, "-s", label="time-domain fitted roots")
    ax2.set_title("Trajectories in the complex-s plane")
    ax2.set_xlabel(r"$\mathrm{Re}(s)$")
    ax2.set_ylabel(r"$\mathrm{Im}(s)$")
    ax2.grid(alpha=0.25)
    ax2.legend()

    mismatch = np.abs(fitted - roots)
    ax3.plot(values, mismatch, "-o", color="tab:red")
    ax3.set_title(r"Mismatch magnitude $|s_{\rm fit}-s_{\rm root}|$")
    ax3.set_xlabel(xlab)
    ax3.set_ylabel("mismatch")
    ax3.grid(alpha=0.25)

    colors = ["tab:blue", "tab:orange", "tab:green"]
    for color, (value, t_signal, signal, fit_t, fit_curve) in zip(colors, sample_waveforms):
        ax4.plot(t_signal, signal, color=color, lw=0.9, alpha=0.55, label=f"{args.parameter}={value:.2f} signal")
        ax4.plot(fit_t, fit_curve, color=color, lw=2.0, label=f"{args.parameter}={value:.2f} fit")
    ax4.set_title("Sample waveforms and fitted ringdown windows")
    ax4.set_xlabel("t")
    ax4.set_ylabel("psi")
    ax4.grid(alpha=0.25)
    ax4.legend(ncol=2, fontsize=8)

    fig.tight_layout()
    fig.savefig(OUTPUT / "compact_support_root_vs_ringdown.png", dpi=180)
    plt.close(fig)

    lines = [
        "Compact-support frequency-domain root vs time-domain ringdown comparison",
        f"parameter = {args.parameter}",
        f"range = [{args.start}, {args.stop}]",
        f"steps = {args.steps}",
        f"fit_window_mode = {'adaptive' if args.fit_start is None or args.fit_end is None else 'fixed'}",
        "",
    ]
    for i, value in enumerate(values):
        lines.append(f"{args.parameter} = {value:.8f}")
        lines.append(f"  root_s = {format_complex(roots[i])}, |W| = {residuals[i]:.8e}")
        lines.append(f"  fit_window = [{fit_windows[i,0]:.8f}, {fit_windows[i,1]:.8f}]")
        lines.append(f"  fit_s  = {format_complex(fitted[i])}")
        lines.append(f"  mismatch = {abs(fitted[i] - roots[i]):.8e}")

    (OUTPUT / "compact_support_root_vs_ringdown.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
