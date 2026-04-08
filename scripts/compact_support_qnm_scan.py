import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from compact_support_qnm_common import (
    CompactSupportConfig,
    compact_support_potential,
    find_local_minima,
    scan_wronskian_grid,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs"
OUTPUT.mkdir(exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Scan QNM poles for a compact-support toy potential.")
    parser.add_argument("--fast", action="store_true", help="Use a smaller complex grid.")
    parser.add_argument("--height", type=float, default=12.0)
    parser.add_argument("--width", type=float, default=1.0)
    parser.add_argument("--domain", type=float, default=3.0)
    args = parser.parse_args()

    domain = args.domain
    config = CompactSupportConfig(
        height=args.height,
        width=args.width,
        domain=domain,
        n_steps=260 if args.fast else 700,
    )

    alpha = np.linspace(-2.4, -0.05, 28 if args.fast else 72)
    beta = np.linspace(0.1, 5.0, 38 if args.fast else 100)
    values = scan_wronskian_grid(alpha, beta, config)

    minima = find_local_minima(values, alpha, beta)

    x_plot = np.linspace(-domain, domain, 1200)
    v_plot = compact_support_potential(x_plot, height=args.height, width=args.width)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
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
    axes[1].set_title(r"$\log_{10}|W(s)|$ on the complex-$s$ plane")
    axes[1].set_xlabel(r"$\mathrm{Re}(s)$")
    axes[1].set_ylabel(r"$\mathrm{Im}(s)$")
    for a, b, _, _ in minima[:4]:
        axes[1].plot(a, b, "co", ms=4)
    fig.colorbar(image, ax=axes[1], pad=0.02)
    fig.tight_layout()
    fig.savefig(OUTPUT / "compact_support_qnm_scan.png", dpi=180)
    plt.close(fig)

    lines = [
        "Compact-support QNM scan summary",
        f"height = {args.height}",
        f"width = {args.width}",
        f"domain = {domain}",
        "",
        "Candidate minima of |W(s)|:",
    ]
    for idx, (a, b, wval, mag) in enumerate(minima, start=1):
        lines.append(
            f"{idx}. s = {a:+.5f}{b:+.5f}i, "
            f"W = {wval.real:+.5e}{wval.imag:+.5e}i, |W| = {mag:.5e}"
        )

    (OUTPUT / "compact_support_qnm_scan.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
