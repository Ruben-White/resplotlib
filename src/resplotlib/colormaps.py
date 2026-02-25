import glob
from pathlib import Path

import cmocean.cm as cmo  # noqa: F401
import colorcet as cc  # noqa: F401
import matplotlib as mpl
import matplotlib.pyplot as plt

DIR_PATH_PACKAGE = Path(__file__).resolve().parent
DIR_PATH_CMAPS = DIR_PATH_PACKAGE / "cmaps"


def _get_colormaps() -> list[mpl.colors.Colormap]:
    """Get custom colormaps.

    Returns:
        list[:class:`matplotlib.colors.Colormap`]: List of custom colormaps.
    """

    # Function to scale colors
    def scale_colors(colors, rgba_scales=(1, 1, 1, 1)):
        if isinstance(rgba_scales, (int, float)):
            rgba_scales = (rgba_scales, rgba_scales, rgba_scales, rgba_scales)
        colors_new = []
        for r, g, b, a in colors:
            colors_new.append((r * rgba_scales[0], g * rgba_scales[1], b * rgba_scales[2], a * rgba_scales[3]))
        return colors_new

    # Function to get colors from a colormap
    def colors_to_cmap(cmap_name):
        cmap = plt.get_cmap(cmap_name)
        colors = [cmap(i) for i in range(cmap.N)]
        return colors

    # Function to get colors from a file
    def colors_from_file(file_path):
        colors = []
        with open(file_path, "r") as file:
            for line in file:
                colors.append(tuple(float(color) for color in line.split()))
        colors = scale_colors(colors, rgba_scales=1 / 255)
        return colors

    # Function to get a colormap from colors
    def cmap_to_colors(colors, cmap_name):
        cmap = plt.cm.colors.LinearSegmentedColormap.from_list(cmap_name, colors)
        return cmap

    # Initialise colormaps
    cmaps = []
    cmaps_contours = []

    # Add colormaps from matplotlib
    cmaps_contours.append(cmap_to_colors(scale_colors(colors_to_cmap("Spectral"), rgba_scales=(0.75, 0.75, 0.75, 1)), "Spectral_contour"))
    cmaps_contours.append(cmap_to_colors(scale_colors(colors_to_cmap("RdBu"), rgba_scales=(0.75, 0.75, 0.75, 1)), "RdBu_contour"))

    # Add colormaps from text files
    for file_path_cmap in glob.glob(str(DIR_PATH_CMAPS / "*.txt")):
        cmap_name = Path(file_path_cmap).stem
        colors = colors_from_file(file_path_cmap)
        cmaps.append(cmap_to_colors(colors, cmap_name))
        cmaps_contours.append(cmap_to_colors(scale_colors(colors, rgba_scales=(0.75, 0.75, 0.75, 1)), cmap_name + "_contour"))

    # Combine colormaps
    cmaps = cmaps + cmaps_contours

    return cmaps


def register_colormaps():
    """Register custom colormaps."""

    # Get custom colormaps
    cmaps = _get_colormaps()

    # Register custom colormaps
    for cmap in cmaps:
        if cmap.name not in plt.colormaps():
            mpl.colormaps.register(cmap=cmap)
            mpl.colormaps.register(cmap=cmap.reversed())


# Register colormaps on import
__cmaps__ = register_colormaps()
