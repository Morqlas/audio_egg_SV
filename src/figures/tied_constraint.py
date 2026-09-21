#!/usr/bin/env python3
"""
Conceptual schematic of the tied-gating constraint.

Illustrates why a tied gate (g_g = 1 - g_a) is costly under heavy noise: an untied
gate can place (g_a, g_g) anywhere in the unit square, while a tied gate is confined
to the anti-diagonal line. The desired low-SNR weighting (suppress audio, keep EGG)
lies far from that line -> large gap; the high-SNR near-balanced weighting lies close
-> small gap.

IMPORTANT: the plotted points are ILLUSTRATIVE positions, not measured gate values.
The caption in the thesis must state this. The only exact object here is the
constraint line g_g = 1 - g_a, which is a mathematical fact about tied gating.

Usage:
    python -m src.figures.tied_constraint
Outputs:
    tied_constraint_schematic.pdf   (for LaTeX \\includegraphics)
    tied_constraint_schematic.png   (200 dpi preview)
"""

import matplotlib
matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt


# --- thesis figure conventions -------------------------------------------------
plt.rcParams.update({
    "font.family": "DejaVu Serif",
    "font.size": 11,
    "axes.linewidth": 0.8,
})

# colourblind-safe palette (matches the rest of the thesis figures)
CB = {
    "red":   "#B85042",
    "blue":  "#065A82",
    "green": "#2C7A4B",
    "gold":  "#C9A227",
    "slate": "#1C7293",
}


def foot_on_line(gx, gy):
    """Foot of the perpendicular from (gx, gy) onto the line g_a + g_g = 1.

    This is the closest reachable point for a tied gate, i.e. the best a tied gate
    can do when the desired weighting is (gx, gy).
    """
    t = (gx + gy - 1.0) / 2.0
    return gx - t, gy - t


def main():
    fig, ax = plt.subplots(figsize=(5.2, 5.2))

    # untied feasible region = full unit square (faint fill)
    ax.add_patch(plt.Rectangle((0, 0), 1, 1, facecolor=CB["slate"],
                               alpha=0.06, edgecolor="none", zorder=0))

    # tied constraint line g_g = 1 - g_a
    ax.plot([0, 1], [1, 0], color=CB["red"], lw=2.2, zorder=3,
            label=r"tied constraint  $g_g = 1 - g_a$")

    # --- low-SNR desired point: suppress BOTH (bad audio + uninformative EGG) ---
    # The case tying cannot represent: g_a low AND g_g low, summing well below 1 and
    # so lying FAR from the line -> large gap. Tying pins g_g = 1 - g_a, forcing EGG
    # high when audio is suppressed, whether or not EGG is informative.
    lx, ly = 0.15, 0.98
    lpx, lpy = foot_on_line(lx, ly)
    ax.plot([lx, lpx], [ly, lpy], ls=(0, (3, 2)), color="0.45", lw=1.3, zorder=4)
    ax.scatter([lx], [ly], s=95, color=CB["green"], zorder=6,
               edgecolor="white", lw=1.2)
    ax.scatter([lpx], [lpy], s=55, color=CB["red"], zorder=6,
               edgecolor="white", lw=1.0)
    ax.annotate("desired low-SNR weighting \n(audio not totally suppressed)",
                xy=(lx, ly), xytext=(0.5, 0.85), fontsize=9.5, color=CB["green"],
                ha="left", va="top",
                arrowprops=dict(arrowstyle="->", color=CB["green"], lw=1.2))

    # --- high-SNR desired point: near-complementary -> ON the line, small gap ---
    hx, hy = 0.9, 0.47
    hpx, hpy = foot_on_line(hx, hy)
    ax.plot([hx, hpx], [hy, hpy], ls=(0, (3, 2)), color="0.45", lw=1.3, zorder=4)
    ax.scatter([hx], [hy], s=80, color=CB["gold"], zorder=6,
               edgecolor="white", lw=1.1)
    ax.scatter([hpx], [hpy], s=45, color=CB["red"], zorder=6,
               edgecolor="white", lw=1.0)
    ax.annotate("desired high-SNR weighting\n(high confidence on audio and EGG)",
                xy=(hx, hy), xytext=(0.4, 0.7), fontsize=9.5, color=CB["gold"],
                ha="left", va="top",
                arrowprops=dict(arrowstyle="->", color=CB["gold"], lw=1.1))

    # --- gap annotations -------------------------------------------------------
    ax.annotate("gap to tied",
                xy=((lx + lpx) / 2, (ly + lpy) / 2),
                xytext=(0.30, 0.8), fontsize=9, color="0.30",
                ha="left", va="center",
                arrowprops=dict(arrowstyle="->", color="0.45", lw=1.0))
    ax.annotate("gap to tied",
                xy=((hx + hpx) / 2, (hy + hpy) / 2),
                xytext=(0.80, 0.3), fontsize=9, color="0.30",
                ha="left", va="center",
                arrowprops=dict(arrowstyle="->", color="0.45", lw=1.0))

    # --- axes cosmetics --------------------------------------------------------
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel(r"audio gate  $g_a$")
    ax.set_ylabel(r"EGG gate  $g_g$")
    ax.set_xticks([0, 0.5, 1])
    ax.set_yticks([0, 0.5, 1])
    ax.set_aspect("equal")
    ax.text(0.8, 0.03, "untied: full square feasible",
            ha="right", va="bottom", fontsize=9, color=CB["slate"])
    ax.legend(loc="upper right", fontsize=9, frameon=True, framealpha=0.92)

    fig.tight_layout()
    fig.savefig("figures/tied_constraint_schematic.pdf", bbox_inches="tight")
    print("wrote tied_constraint_schematic.pdf")


if __name__ == "__main__":
    main()