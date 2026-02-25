import warnings

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import patheffects as pe
from matplotlib.patches import FancyArrowPatch
from shapely.geometry import MultiLineString
from shapely.geometry.base import BaseGeometry


def _add_arrow(geometries: gpd.GeoSeries, ax: plt.Axes, arrow_kwargs: dict | None = None, plot_kwargs: dict | None = None) -> None:
    """Add arrow to the end of a line gseometry.

    Args:
        geometries (:class:`geopandas.GeoSeries`): GeoSeries of geometries to add arrow to.
        ax (:class:`matplotlib.axes.Axes`): Axis to plot on.
        arrow_kwargs (dict, optional): Keyword arguments used to plot the arrow. Defaults to None.
        plot_kwargs (dict, optional): Keyword arguments used to plot the geometry. Defaults to None.
    """
    # Check if all geometries are LineString or MultiLineString
    if not all(geom.geom_type in ["LineString", "MultiLineString"] for geom in geometries):
        raise ValueError("All geometries must be LineString or MultiLineString to add arrow.")

    # Convert geometries to MultiLineString
    lines = []
    for geometry in geometries:
        if geometry.geom_type == "LineString":
            lines.append(geometry)
        else:
            lines.extend(geometry.geoms)
    lines = MultiLineString(lines)

    # Set the arrow_kwargs
    if arrow_kwargs is None:
        arrow_kwargs = plot_kwargs.copy()
    else:
        arrow_kwargs = {**plot_kwargs, **arrow_kwargs}
    arrow_kwargs.setdefault("arrowstyle", "-|>")
    arrow_kwargs.setdefault("mutation_scale", 20)
    arrow_kwargs.setdefault("shrinkA", 0)
    arrow_kwargs.setdefault("shrinkB", 0)
    if "color" not in arrow_kwargs:
        arrow_kwargs.setdefault("edgecolor", "none")

    # Add arrow to the end of each line
    for line in lines.geoms:
        x1, y1, x2, y2 = line.xy[0][-2], line.xy[1][-2], line.xy[0][-1], line.xy[1][-1]
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), **arrow_kwargs))


def _add_label_to_plot(geometries: gpd.GeoSeries, labels: pd.Series, ax: plt.Axes, label_kwargs: dict | None = None) -> None:
    """Add label to the plot for a geometry.

    Args:
        geometries (:class:`geopandas.GeoSeries`): GeoSeries of geometries to add label to.
        labels (:class:`pandas.Series`): Series of labels corresponding to the geometries.
        ax (:class:`matplotlib.axes.Axes`): Axis to plot on.
        label_kwargs (dict, optional): Keyword arguments used to plot the label. Defaults to None.
    """
    # Set default keyword arguments
    if label_kwargs is None:
        label_kwargs = {}
    label_kwargs.setdefault("ha", "center")
    label_kwargs.setdefault("va", "center")
    label_kwargs.setdefault("path_effects", [pe.withStroke(linewidth=3, foreground="white")])

    for geometry, label in zip(geometries, labels):
        # Get the coordinates of the label
        if geometry.geom_type in ["Point", "MultiPoint"]:
            x, y = geometry.xy
        elif geometry.geom_type in ["LineString", "MultiLineString"]:
            x, y = geometry.interpolate(0.5, normalized=True).xy
        elif geometry.geom_type in ["Polygon", "MultiPolygon"]:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                x, y = geometry.representative_point().xy
        else:
            continue

        # Add label to the plot
        ax.text(x[0], y[0], label, **label_kwargs)


def _add_label_to_legend(geometry: BaseGeometry, ax: plt.Axes, label: str, kwargs: dict | None = None) -> None:
    """Add label to the legend for a geometry.

    Args:
        geometry (:class:`shapely.geometry.base.BaseGeometry`): Geometry to add label for.
        ax (:class:`matplotlib.axes.Axes`): Axis to plot on.
        label (str): Label to add to the legend.
        kwargs (dict, optional): Keyword arguments used to plot the geometry. Defaults to None.
    """
    # Set default keyword arguments
    if kwargs is None:
        kwargs = {}

    # Plot a point with label
    if geometry.geom_type in ["Point", "MultiPoint"]:
        if "markersize" in kwargs:
            kwargs["s"] = kwargs.pop("markersize")
        ax.scatter(np.nan, np.nan, label=label, **kwargs)

    # Plot a line with label
    elif geometry.geom_type in ["LineString", "MultiLineString"]:
        ax.plot(np.nan, np.nan, label=label, **kwargs)

    # Plot a polygon with label
    elif geometry.geom_type in ["Polygon", "MultiPolygon"]:
        ax.add_patch(plt.Polygon(np.array([[np.nan, np.nan]]), label=label, **kwargs))


def plot_gdf(gdf: gpd.GeoDataFrame, ax: plt.Axes, **kwargs) -> plt.Axes:
    """Plot a geopandas.GeoDataFrame.

    Args:
        gdf (:class:`geopandas.GeoDataFrame`): GeoDataFrame to plot.
        ax (:class:`matplotlib.axes.Axes`): Axis to plot on.
        **kwargs (dict, optional): Keyword arguments to pass to :func:`geopandas.GeoDataFrame.plot`.

    Returns:
        :class:`matplotlib.axes.Axes`: Axis with the GeoDataFrame plotted on it
    """
    # Copy GeoDataFrame
    gdf = gdf.reset_index(drop=True)

    # Combine existing keyword arguments with new keyword arguments
    if "kwargs" in gdf.columns:
        gdf["kwargs"] = gdf["kwargs"].apply(lambda x: {**x, **kwargs})
    else:
        gdf["kwargs"] = [kwargs] * len(gdf)

    # Get string representation of keyword arguments
    gdf["kwargs_str"] = gdf["kwargs"].apply(lambda x: str(x))

    # Get groups with same string representation of keyword arguments
    gdf_groups = gdf.groupby("kwargs_str")

    # Sort groups by index to ensure consistent plotting order
    gdf_groups = sorted(gdf_groups, key=lambda x: x[1].index.min())

    # Plot each group separately
    for kwargs, gdf_group in gdf_groups:
        # Get keyword arguments
        kwargs = gdf_group["kwargs"].iloc[0]

        # Seperate keyword arguments for plotting, adding arrow, and adding label
        add_arrow = kwargs.pop("add_arrow", False)
        arrow_kwargs = kwargs.pop("arrow_kwargs", None)
        label_column = kwargs.pop("label_column", None)
        label_kwargs = kwargs.pop("label_kwargs", None)
        label = kwargs.pop("label", None)

        # Plot group
        ax = gdf_group.plot(ax=ax, **kwargs)

        # Add arrow to plot
        if add_arrow:
            _add_arrow(gdf_group["geometry"], ax=ax, arrow_kwargs=arrow_kwargs, plot_kwargs=kwargs)

        # Add label to plot
        if label_column:
            _add_label_to_plot(gdf_group["geometry"], labels=gdf_group[label_column], ax=ax, label_kwargs=label_kwargs)

        # Add label to legend
        if label is not None and "column" not in kwargs:
            _add_label_to_legend(gdf_group["geometry"].iloc[0], ax=ax, label=label, kwargs=kwargs)

    return ax
