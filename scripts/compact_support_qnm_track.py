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


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs"
OUTPUT.mkdir(exist_ok=True)


def format_complex(z: complex) -> str:
    return f"{z.real:+.8f}{z.imag:+.8f}i"
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Track compact-support toy QNM roots as the potential height or width changes."
    )
    parser.add_argument("--parameter", choices=["height", "width"], default="height")
    parser.add_argument("--start", type=float, default=8.0)
    parser.add_argument("--stop", type=float, default=16.0)
    parser.add_argument("--steps", type=int, default=9)
    parser.add_argument("--nroots", type=int, default=2)
    parser.add_argument("--height", type=float, default=12.0, help="Baseline height when width is varied.")
    parser.add_argument("--width", type=float, default=1.0, help="Baseline width when height is varied.")
    parser.add_argument("--domain", type=float, default=3.0)
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--tol", type=float, default=1e-9)
    args = parser.parse_args()

    values = np.linspace(args.start, args.stop, args.steps)
    base_config = CompactSupportConfig(
        height=values[0] if args.parameter == "height" else args.height,
        width=values[0] if args.parameter == "width" else args.width,
        domain=args.domain,
        n_steps=260 if args.fast else 700,
    )

    roots0 = initial_roots_from_scan(base_config, args.nroots, args.fast, args.tol)
    if len(roots0) < args.nroots:
        raise RuntimeError(f"Only found {len(roots0)} initial roots, fewer than requested {args.nroots}.")

    trajectories = np.full((args.nroots, args.steps), np.nan + 1j * np.nan, dtype=complex)
    residuals = np.full((args.nroots, args.steps), np.nan, dtype=float)

    current_roots = roots0[:]
    for j, value in enumerate(values):
        config = CompactSupportConfig(
            height=value if args.parameter == "height" else args.height,
            width=value if args.parameter == "width" else args.width,
            domain=args.domain,
            n_steps=260 if args.fast else 700,
        )
        evaluator = CompactSupportEvaluator(config)

        for i in range(args.nroots):
            root, wroot, _ = complex_newton_root(current_roots[i], evaluator=evaluator, tol=args.tol)
            if abs(wroot) > 1e-6:
                root, wroot = local_reseed_and_refine(current_roots[i], config, args.tol)
            trajectories[i, j] = root
            residuals[i, j] = abs(wroot)
            current_roots[i] = root

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    colors = ["tab:blue", "tab:red", "tab:green", "tab:purple"]

    for i in range(args.nroots):
        color = colors[i % len(colors)]
        axes[0].plot(values, trajectories[i].real, "-o", color=color, label=f"mode {i+1}")
        axes[1].plot(values, trajectories[i].imag, "-o", color=color, label=f"mode {i+1}")
        axes[2].plot(trajectories[i].real, trajectories[i].imag, "-o", color=color, label=f"mode {i+1}")

    xlab = "potential height" if args.parameter == "height" else "potential width"
    axes[0].set_title(r"Tracked $\mathrm{Re}(s)$")
    axes[0].set_xlabel(xlab)
    axes[0].set_ylabel(r"$\mathrm{Re}(s)$")
    axes[0].grid(alpha=0.25)

    axes[1].set_title(r"Tracked $\mathrm{Im}(s)$")
    axes[1].set_xlabel(xlab)
    axes[1].set_ylabel(r"$\mathrm{Im}(s)$")
    axes[1].grid(alpha=0.25)

    axes[2].set_title("Trajectories in the complex-s plane")
    axes[2].set_xlabel(r"$\mathrm{Re}(s)$")
    axes[2].set_ylabel(r"$\mathrm{Im}(s)$")
    axes[2].grid(alpha=0.25)
    axes[2].legend()

    fig.tight_layout()
    fig.savefig(OUTPUT / "compact_support_qnm_track.png", dpi=180)
    plt.close(fig)

    lines = [
        "Compact-support QNM continuation tracking summary",
        f"parameter = {args.parameter}",
        f"range = [{args.start}, {args.stop}]",
        f"steps = {args.steps}",
        f"nroots = {args.nroots}",
        f"n_steps = {base_config.n_steps}",
        "",
        "Tracked roots:",
    ]
    for j, value in enumerate(values):
        lines.append(f"{args.parameter} = {value:.8f}")
        for i in range(args.nroots):
            lines.append(
                f"  mode {i+1}: s = {format_complex(trajectories[i, j])}, |W| = {residuals[i, j]:.8e}"
            )

    (OUTPUT / "compact_support_qnm_track.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
