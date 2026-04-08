import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from compact_support_qnm_common import (
    CompactSupportConfig,
    CompactSupportEvaluator,
    compact_support_potential,
    complex_newton_root,
    deduplicate_roots,
    find_local_minima,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs"
OUTPUT.mkdir(exist_ok=True)


def format_complex(z: complex) -> str:
    return f"{z.real:+.8f}{z.imag:+.8f}i"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Refine compact-support QNM roots by solving W(s)=0 with damped complex Newton iterations."
    )
    parser.add_argument("--height", type=float, default=12.0)
    parser.add_argument("--width", type=float, default=1.0)
    parser.add_argument("--domain", type=float, default=3.0)
    parser.add_argument("--fast", action="store_true", help="Use a smaller coarse scan.")
    parser.add_argument("--nseeds", type=int, default=4, help="Number of scan minima used as Newton seeds.")
    parser.add_argument("--tol", type=float, default=1e-9, help="Target |W(s)| tolerance.")
    args = parser.parse_args()

    config = CompactSupportConfig(
        height=args.height,
        width=args.width,
        domain=args.domain,
        n_steps=260 if args.fast else 700,
    )
    evaluator = CompactSupportEvaluator(config)

    alpha = np.linspace(-2.4, -0.05, 28 if args.fast else 72)
    beta = np.linspace(0.1, 5.0, 38 if args.fast else 100)
    values = evaluator.scan_grid(alpha, beta)
    minima = find_local_minima(values, alpha, beta, n_keep=max(args.nseeds * 2, 8))

    roots_raw: list[tuple[complex, complex]] = []
    histories: list[list[tuple[int, complex, complex]]] = []
    for a, b, _, _ in minima[: args.nseeds]:
        seed = complex(a, b)
        root, wroot, history = complex_newton_root(seed, evaluator=evaluator, tol=args.tol)
        roots_raw.append((root, wroot))
        histories.append(history)

    roots = deduplicate_roots(roots_raw)
    roots = sorted(roots, key=lambda pair: (abs(pair[1]), -pair[0].real, pair[0].imag))

    x_plot = np.linspace(-args.domain, args.domain, 1200)
    v_plot = compact_support_potential(x_plot, height=args.height, width=args.width)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    axes[0].plot(x_plot, v_plot, color="tab:blue")
    axes[0].set_title("Compact-support toy potential")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("V(x)")
    axes[0].grid(alpha=0.25)

    image = axes[1].imshow(
        np.log10(np.abs(values).T + 1e-14),
        origin="lower",
        aspect="auto",
        extent=[alpha[0], alpha[-1], beta[0], beta[-1]],
        cmap="magma",
    )
    axes[1].set_title(r"$\log_{10}|W(s)|$ with refined Newton roots")
    axes[1].set_xlabel(r"$\mathrm{Re}(s)$")
    axes[1].set_ylabel(r"$\mathrm{Im}(s)$")

    for idx, history in enumerate(histories[: args.nseeds]):
        path = np.array([[entry[1].real, entry[1].imag] for entry in history])
        if path.size:
            axes[1].plot(path[:, 0], path[:, 1], "-o", ms=2.5, lw=0.8, alpha=0.65)

    for s, w in roots:
        axes[1].plot(s.real, s.imag, "co", ms=5)
        axes[1].annotate(
            f"{s.real:.3f}+{s.imag:.3f}i",
            (s.real, s.imag),
            textcoords="offset points",
            xytext=(5, 4),
            fontsize=8,
            color="white",
        )

    fig.colorbar(image, ax=axes[1], pad=0.02)
    fig.tight_layout()
    fig.savefig(OUTPUT / "compact_support_qnm_rootfind.png", dpi=180)
    plt.close(fig)

    lines = [
        "Compact-support QNM root refinement summary",
        f"height = {args.height}",
        f"width = {args.width}",
        f"domain = {args.domain}",
        f"n_steps = {config.n_steps}",
        f"scan_shape = ({alpha.size}, {beta.size})",
        f"tol = {args.tol:.1e}",
        "",
        "Seeds from coarse |W(s)| minima:",
    ]
    for idx, (a, b, wval, mag) in enumerate(minima[: args.nseeds], start=1):
        lines.append(f"{idx}. seed s = {a:+.8f}{b:+.8f}i, |W| = {mag:.8e}")

    lines.append("")
    lines.append("Refined roots from damped complex Newton iterations:")
    for idx, (s, w) in enumerate(roots, start=1):
        lines.append(f"{idx}. root s = {format_complex(s)}, W(root) = {format_complex(w)}, |W| = {abs(w):.8e}")

    lines.append("")
    lines.append("Iteration histories:")
    for idx, history in enumerate(histories[: args.nseeds], start=1):
        lines.append(f"seed #{idx}:")
        for it, s, w in history:
            lines.append(f"  iter {it:02d}: s = {format_complex(s)}, |W| = {abs(w):.8e}")

    (OUTPUT / "compact_support_qnm_rootfind.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
