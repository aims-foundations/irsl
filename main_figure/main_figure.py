import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from tueplots import bundles

# ----- data -----
np.random.seed(0)
H, W = 40, 30
data = np.random.rand(H, W)

# ----- light single-hue colormap -----
light_blue_cmap = LinearSegmentedColormap.from_list(
    "light_blue",
    ["#f7fbff", "#deebf7", "#c6dbef", "#9ecae1", "#6baed6"]
)

# ----- ICML 2024 style -----
with plt.rc_context(bundles.icml2024(usetex=True, family="serif")):
    fig, ax = plt.subplots(
        figsize=(3, 7),
        constrained_layout=True   # <-- KEY FIX
    )

    im = ax.imshow(
        data,
        cmap=light_blue_cmap,
        vmin=0,
        vmax=1,
        interpolation="nearest",
    )

    ax.set_xticks([])
    ax.set_yticks([])

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(r"Value", rotation=270, labelpad=12)

    plt.savefig(
        "random_heatmap_light_blue_icml2024.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()
