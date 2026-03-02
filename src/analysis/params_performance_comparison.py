import matplotlib.pyplot as plt
import numpy as np


def main():
    # Hardcoded methods, parameter counts (in millions), and AbsRel values
    methods = [
        "MonoDepth2",
        "CaDepth-Net",
        "PackNet-SfM",
        "HR-Depth",
        "LiteMono",
        "Depth Hints",
        "MonoViT",
        "JEPADepth (Ours)",
    ]

    params_m = np.array([
        14,   # Monodepth2
        58,   # CaDepth-Net
        128,    # PackNet-SfM
        14,   # HR-Depth
        3,   # Lite-Mono
        34,   # Depth Hints
        27,   # MonoViT
        24,   # JEPADepth (Ours)
    ])

    absrel = np.array([
        0.110,  # Monodepth2
        0.105,  # CaDepth-Net
        0.107,  # PackNet-SfM
        0.109,  # HR-Depth
        0.107,  # Lite-Mono
        0.102,  # Depth Hints
        0.099,  # MonoViT
        0.101,  # JEPADepth (Ours)
    ])

    # Sort by params for cleaner left-to-right reading
    sort_idx = np.argsort(params_m)
    params_m = params_m[sort_idx]
    absrel = absrel[sort_idx]
    methods = [methods[i] for i in sort_idx]

    # Styling for paper
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    })

    fig, ax = plt.subplots(figsize=(7.2, 4.8))

    # Colors: highlight ours
    colors = ["#4C78A8"] * len(methods)
    ours_idx = methods.index("JEPADepth (Ours)")
    colors[ours_idx] = "#D62728"

    # Scatter points
    ax.scatter(
        params_m, absrel,
        c=colors,
        s=[85 if i != ours_idx else 120 for i in range(len(methods))],
        edgecolor="black",
        linewidth=0.6,
        zorder=3
    )

    # Annotation offsets to avoid overlap
    offsets = {
    "LiteMono": (6, -10),
    "HR-Depth": (6, 8),
    "JEPADepth (Ours)": (8, -14),
    "MonoViT": (6, 8),
    "MonoDepth2": (6, 8),
    "CaDepth-Net": (6, -12),
    "PackNet-SfM": (6, 8),
    }

    for x, y, name in zip(params_m, absrel, methods):
        dx, dy = offsets.get(name, (6, 6))
        ax.annotate(
            name,
            (x, y),
            textcoords="offset points",
            xytext=(5, 5),
            ha="left",
            va="bottom",
            fontsize=10,
            fontweight="bold" if "Ours" in name else "normal",
            color="#D62728" if "Ours" in name else "black"
        )

    # Axis labels and title
    ax.set_xlabel("Parameters (Millions)")
    ax.set_ylabel("AbsRel ↓")
    # ax.set_title("Accuracy–Efficiency Trade-off on KITTI")

    # Nice limits / ticks (dynamic so all points are visible)
    x_min, x_max = params_m.min(), params_m.max()
    x_pad = max(3.0, 0.08 * (x_max - x_min))
    ax.set_xlim(max(0, x_min - x_pad), x_max + x_pad)

    y_min, y_max = absrel.min(), absrel.max()
    y_pad = max(0.0015, 0.12 * (y_max - y_min))
    ax.set_ylim(y_min - y_pad, y_max + y_pad)

    # Dynamic y-ticks
    yticks = np.linspace(round(y_min - y_pad, 3), round(y_max + y_pad, 3), 6)
    ax.set_yticks(np.round(yticks, 3))

    # Grid and spines
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.35, zorder=1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Tight layout and save
    fig.tight_layout()
    fig.savefig("assets/params_performance_comparison.png", dpi=600, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()