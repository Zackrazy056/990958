from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs"
OUTPUT.mkdir(exist_ok=True)


def main() -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.axhline(0.0, color="black", lw=0.9)
    ax.axvline(0.0, color="black", lw=0.9)

    bromwich_x = 0.5
    ax.plot([bromwich_x, bromwich_x], [-4, 4], color="tab:blue", lw=2.0, label="Bromwich contour")
    ax.annotate("", xy=(bromwich_x, 3.8), xytext=(bromwich_x, -3.8), arrowprops=dict(arrowstyle="->"))

    poles = np.array([[-0.4, 0.8], [-0.6, 1.7], [-0.9, 2.8], [-0.6, -1.7], [-0.9, -2.8]])
    ax.plot(poles[:, 0], poles[:, 1], "ro", label="QNM poles")

    ax.plot([-1.6, -1.6], [-3.6, 3.6], color="tab:green", ls="--", lw=1.4, label="deformed contour")
    ax.plot([-2.2, -2.2], [-3.6, 3.6], color="tab:purple", lw=1.4, alpha=0.7, label="branch cut example")

    ax.text(0.58, 3.2, r"$\mathrm{Re}(s)=a>0$", color="tab:blue")
    ax.text(-1.45, 3.2, "move left", color="tab:green")
    ax.text(-2.52, 3.2, "cut", color="tab:purple")

    ax.set_xlim(-3.2, 1.4)
    ax.set_ylim(-4.0, 4.0)
    ax.set_xlabel(r"$\mathrm{Re}(s)$")
    ax.set_ylabel(r"$\mathrm{Im}(s)$")
    ax.set_title("Bromwich contour, QNM poles, and a branch-cut sketch")
    ax.grid(alpha=0.2)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(OUTPUT / "bromwich_poles_diagram.png", dpi=180)
    plt.close(fig)
    print("Saved outputs/bromwich_poles_diagram.png")


if __name__ == "__main__":
    main()

