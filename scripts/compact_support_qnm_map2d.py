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


def refine_from_seed(seed: complex, config: CompactSupportConfig, tol: float) -> tuple[complex, float]:
    evaluator = CompactSupportEvaluator(config)
    root, wroot, _ = complex_newton_root(seed, evaluator=evaluator, tol=tol)
    if abs(wroot) > 1e-6:
        root, wroot = local_reseed_and_refine(seed, config, tol)
    return root, abs(wroot)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a 2D QNM parameter map over potential height and width using continuation."
    )
    parser.add_argument("--hstart", type=float, default=8.0)
    parser.add_argument("--hstop", type=float, default=14.0)
    parser.add_argument("--nh", type=int, default=5)
    parser.add_argument("--wstart", type=float, default=0.8)
    parser.add_argument("--wstop", type=float, default=1.3)
    parser.add_argument("--nw", type=int, default=5)
    parser.add_argument("--nroots", type=int, default=2)
    parser.add_argument("--domain", type=float, default=3.0)
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--tol", type=float, default=1e-9)
    args = parser.parse_args()

    heights = np.linspace(args.hstart, args.hstop, args.nh)
    widths = np.linspace(args.wstart, args.wstop, args.nw)
    n_steps = 240 if args.fast else 650

    base_config = CompactSupportConfig(
        height=heights[0],
        width=widths[0],
        domain=args.domain,
        n_steps=n_steps,
    )
    initial_roots = initial_roots_from_scan(base_config, args.nroots, args.fast, args.tol)
    if len(initial_roots) < args.nroots:
        raise RuntimeError(f"Only found {len(initial_roots)} initial roots.")

    roots = np.full((args.nroots, args.nh, args.nw), np.nan + 1j * np.nan, dtype=complex)
    residuals = np.full((args.nroots, args.nh, args.nw), np.nan, dtype=float)

    # First row: continue in width at fixed lowest height.
    current_row_roots = initial_roots[:]
    for j, width in enumerate(widths):
        config = CompactSupportConfig(height=heights[0], width=width, domain=args.domain, n_steps=n_steps)
        for mode in range(args.nroots):
            root, res = refine_from_seed(current_row_roots[mode], config, args.tol)
            roots[mode, 0, j] = root
            residuals[mode, 0, j] = res
            current_row_roots[mode] = root

    # Remaining rows: first continue in height at fixed width[0], then sweep across width.
    for i in range(1, args.nh):
        row_roots = [roots[mode, i - 1, 0] for mode in range(args.nroots)]
        config_first = CompactSupportConfig(
            height=heights[i], width=widths[0], domain=args.domain, n_steps=n_steps
        )
        for mode in range(args.nroots):
            root, res = refine_from_seed(row_roots[mode], config_first, args.tol)
            roots[mode, i, 0] = root
            residuals[mode, i, 0] = res
            row_roots[mode] = root

        for j in range(1, args.nw):
            config = CompactSupportConfig(
                height=heights[i], width=widths[j], domain=args.domain, n_steps=n_steps
            )
            for mode in range(args.nroots):
                root, res = refine_from_seed(row_roots[mode], config, args.tol)
                roots[mode, i, j] = root
                residuals[mode, i, j] = res
                row_roots[mode] = root

    extent = [widths[0], widths[-1], heights[0], heights[-1]]
    fig, axes = plt.subplots(args.nroots, 2, figsize=(12, 4.4 * args.nroots), squeeze=False)
    for mode in range(args.nroots):
        im0 = axes[mode, 0].imshow(
            roots[mode].real,
            origin="lower",
            aspect="auto",
            extent=extent,
            cmap="viridis",
        )
        axes[mode, 0].set_title(rf"Mode {mode+1}: $\mathrm{{Re}}(s)$")
        axes[mode, 0].set_xlabel("width")
        axes[mode, 0].set_ylabel("height")
        fig.colorbar(im0, ax=axes[mode, 0], pad=0.02)

        im1 = axes[mode, 1].imshow(
            roots[mode].imag,
            origin="lower",
            aspect="auto",
            extent=extent,
            cmap="plasma",
        )
        axes[mode, 1].set_title(rf"Mode {mode+1}: $\mathrm{{Im}}(s)$")
        axes[mode, 1].set_xlabel("width")
        axes[mode, 1].set_ylabel("height")
        fig.colorbar(im1, ax=axes[mode, 1], pad=0.02)

    fig.tight_layout()
    fig.savefig(OUTPUT / "compact_support_qnm_map2d.png", dpi=180)
    plt.close(fig)

    lines = [
        "Compact-support 2D QNM parameter map summary",
        f"height_range = [{args.hstart}, {args.hstop}], nh = {args.nh}",
        f"width_range = [{args.wstart}, {args.wstop}], nw = {args.nw}",
        f"nroots = {args.nroots}",
        f"n_steps = {n_steps}",
        "",
    ]
    for i, height in enumerate(heights):
        for j, width in enumerate(widths):
            lines.append(f"height={height:.8f}, width={width:.8f}")
            for mode in range(args.nroots):
                lines.append(
                    f"  mode {mode+1}: s = {format_complex(roots[mode, i, j])}, |W| = {residuals[mode, i, j]:.8e}"
                )

    (OUTPUT / "compact_support_qnm_map2d.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[: min(len(lines), 28)]))


if __name__ == "__main__":
    main()

