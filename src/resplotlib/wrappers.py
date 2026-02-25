from __future__ import annotations

import functools
from typing import TYPE_CHECKING

import geopandas as gpd
import ipyleaflet
import matplotlib
import matplotlib.pyplot as plt
import xarray as xr
import xugrid as xu
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pyproj import CRS as pyprojCRS
from rasterio.crs import CRS as rasterioCRS
from shapely.geometry import box

from . import map, rescale, utils

if TYPE_CHECKING:
    from .resplotclass import Resplotclass

DATA_OR_CRS_TYPE = xr.DataArray | xr.Dataset | xu.UgridDataArray | xu.UgridDataset | gpd.GeoDataFrame | pyprojCRS | rasterioCRS | str | None
PLOT_TYPE = plt.Axes | ipyleaflet.Map | map.Map


def _get_function_name(func: callable) -> str:
    """Get the original function name, even if it is wrapped by multiple decorators.

    Args:
        func (callable): The function for which to retrieve the original name.

    Returns:
        str: The original function name.
    """
    # Unwrap the function until we reach the original function
    func_ = func
    while hasattr(func_, "__wrapped__"):
        func_ = func_.__wrapped__

    # Get the function name
    func_name = getattr(func_, "__name__", func.__name__)

    return func_name


def format_args_wrapper(func: callable) -> callable:
    @functools.wraps(func)
    def wrapper(self: Resplotclass, data_or_crs: DATA_OR_CRS_TYPE = None, **kwargs) -> PLOT_TYPE:
        """Wrapper to format arguments for plotting functions.

        Args:
            self (:class:`Resplotclass <resplotlib.Resplotclass>`): The instance of the Resplotclass.
            data_or_crs (:class:`xarray.DataArray`, :class:`xarray.Dataset`, :class:`xugrid.UgridDataArray`, :class:`xugrid.UgridDataset`, :class:`geopandas.GeoDataFrame`, :class:`pyproj.CRS`, :class:`rasterio.CRS`, str, or None, optional): The data or CRS to be plotted. Defaults to None.
            **kwargs (dict, optional): Additional keyword arguments to be passed to the plotting function.

        Returns:
            :class:`matplotlib.axes.Axes`, :class:`ipyleaflet.Map`, or :class:`resplotlib.map.Map`: The result of the plotting function.
        """
        # Get function name
        func_name = _get_function_name(func)

        # Remove any keword arguments that are None
        kwargs = {key: value for key, value in kwargs.items() if value is not None}

        # Format arguments based on function name and call original function
        if func_name in ["imshow", "pcolormesh", "contour", "contourf", "grid", "explore_imshow"] and data_or_crs is None:
            if "da" in kwargs:
                data_or_crs = kwargs.pop("da")
            elif "uda" in kwargs:
                data_or_crs = kwargs.pop("uda")
            else:
                raise ValueError("Data not provided. Please provide a xarray.DataArray or xugrid.UgridDataArray using the 'da' or 'uda' keywords.")
            return func(self, data_or_crs, **kwargs)
        elif func_name in ["scatter"] and data_or_crs is None:
            if "ds" in kwargs:
                data_or_crs = kwargs.pop("ds")
            elif "uda" in kwargs:
                data_or_crs = kwargs.pop("uda")
            else:
                raise ValueError("Data not provided. Please provide a xarray.Dataset or xugrid.UgridDataArray using the 'ds' or 'uda' keywords.")
            return func(self, data_or_crs, **kwargs)
        elif func_name in ["quiver", "streamplot"] and data_or_crs is None:
            if "ds" in kwargs:
                data_or_crs = kwargs.pop("ds")
            elif "uds" in kwargs:
                data_or_crs = kwargs.pop("uds")
            else:
                raise ValueError("Data not provided. Please provide a xarray.Dataset or xugrid.UgridDataArray using the 'ds' or 'uds' keywords.")
            return func(self, data_or_crs, **kwargs)
        elif func_name in ["grid"] and data_or_crs is None:
            if "da" in kwargs:
                data_or_crs = kwargs.pop("da")
            elif "uda" in kwargs:
                data_or_crs = kwargs.pop("uda")
            else:
                raise ValueError("Data not provided. Please provide a xarray.DataArray or xugrid.UgridDataArray using the 'da' or 'uda' keywords.")
            return func(self, data_or_crs, **kwargs)
        elif func_name in ["geometries", "explore_geometries"] and data_or_crs is None:
            if "gdf" in kwargs:
                data_or_crs = kwargs.pop("gdf")
            else:
                raise ValueError("Data not provided. Please provide a geopandas.GeoDataFrame using the 'gdf' keyword.")
            return func(self, data_or_crs, **kwargs)
        elif func_name in ["basemap"] and data_or_crs is None:
            if "crs" in kwargs:
                data_or_crs = kwargs.pop("crs")
            else:
                raise ValueError("CRS not provided. Please provide a CRS using the 'crs' keyword.")
            return func(self, data_or_crs, **kwargs)
        else:
            return func(self, data_or_crs, **kwargs)

    return wrapper


def guideline_wrapper(func: callable) -> callable:
    @functools.wraps(func)
    def wrapper(
        self, data_or_crs: DATA_OR_CRS_TYPE = None, style: str = "default", extent: str | None = None, rescale_unit: str | None = None, **kwargs
    ) -> PLOT_TYPE:
        """Wrapper to merge guideline kwargs and user-provided kwargs for plotting functions.
        Args:
            self (:class:`Resplotclass <resplotlib.Resplotclass>`): The instance of the Resplotclass.
            data_or_crs (:class:`xarray.DataArray`, :class:`xarray.Dataset`, :class:`xugrid.UgridDataArray`, :class:`xugrid.UgridDataset`, :class:`geopandas.GeoDataFrame`, :class:`pyproj.CRS`, :class:`rasterio.CRS`, str, or None, optional): The data or CRS to be plotted. Defaults to None.
            style (str, optional): The style to be applied to the plot. Defaults to "default". Set to "none" to ignore style guidelines.
            extent (str or None, optional): The extent to be applied to the plot. Defaults to None.
            rescale_unit (str or None, optional): The unit to which the data or CRS should be rescaled. Defaults to None. If None, the rescale_unit specified in the guidelines will be used.

        Returns:
            :class:`matplotlib.axes.Axes`, :class:`ipyleaflet.Map`, or :class:`resplotlib.map.Map`: The result of the plotting function.
        """
        # Get function name
        func_name = _get_function_name(func)

        # Get style kwargs from guidelines
        if style != "none":
            # Get function guidelines
            styles = self.guidelines["styles"]
            if func_name not in styles:
                raise ValueError(f"Function '{func_name}' not found in guidelines. Available functions: {list(styles.keys())}")
            func_guidelines = styles[func_name]

            # Get style kwargs
            if style not in func_guidelines:
                raise ValueError(
                    f"Style '{style}' not found for function '{func_name}' in guidelines. Available styles: {list(func_guidelines.keys())}"
                )
            style_kwargs = func_guidelines[style]

            # Combine guidelines with kwargs
            kwargs = utils._combine_dicts(style_kwargs, kwargs)

        # Get extent kwargs from guidelines
        if extent is not None and not func_name.startswith("explore_"):
            # Get extent guidelines
            extents = self.guidelines["extents"]
            if extent not in extents:
                raise ValueError(f"Extent '{extent}' not found in guidelines. Available extents: {list(extents.keys())}")
            extent_kwargs = extents[extent]

            # Combine extent kwargs with existing kwargs
            kwargs = utils._combine_dicts(extent_kwargs, kwargs)

        # Get rescale_unit from guidelines if not provided
        if rescale_unit is None:
            rescale_unit = self.guidelines["project"]["rescale_unit"]

        # Get crs from guidelines if function is basemap and crs is not provided
        if func_name == "basemap" and data_or_crs is None:
            data_or_crs = self.guidelines["project"]["crs"]

        # Call original function with combined kwargs
        if data_or_crs is not None:
            return func(self, data_or_crs, rescale_unit=rescale_unit, **kwargs)
        else:
            return func(self, rescale_unit=rescale_unit, **kwargs)

    return wrapper


def initialise_fig_wrapper(func: callable) -> callable:
    @functools.wraps(func)
    def wrapper(self, data_or_crs: DATA_OR_CRS_TYPE = None, ax: plt.Axes | None = None, subplot_kwargs: dict | None = None, **kwargs) -> plt.Axes:
        """Wrapper to initialise a figure and axes for plotting functions.

        Args:
            self (:class:`Resplotclass <resplotlib.Resplotclass>`): The instance of the Resplotclass.
            data_or_crs (:class:`xarray.DataArray`, :class:`xarray.Dataset`, :class:`xugrid.UgridDataArray`, :class:`xugrid.UgridDataset`, :class:`geopandas.GeoDataFrame`, :class:`pyproj.CRS`, :class:`rasterio.CRS`, str, or None, optional): The data or CRS to be plotted. Defaults to None.
            ax (:class:`matplotlib.axes.Axes`, optional): The axes to be used for plotting. If None, a new figure and axes will be created. Defaults to None.
            subplot_kwargs (dict, optional): Additional keyword arguments to be passed to :func:`matplotlib.pyplot.subplots` when creating a new figure and axes. Defaults to None.
            **kwargs (dict, optional): Additional keyword arguments to be passed to the plotting function.

        Returns:
            :class:`matplotlib.axes.Axes`: The axes object resulting from the plotting function.
        """
        # Initialise figure and axes if not provided
        if ax is None:
            subplot_kwargs = {} if subplot_kwargs is None else subplot_kwargs
            fig, ax = plt.subplots(1, 1, **subplot_kwargs)

        return func(self, data_or_crs, ax=ax, **kwargs)

    return wrapper


def initialise_map_wrapper(func: callable) -> callable:
    @functools.wraps(func)
    def wrapper(self, data_or_crs: DATA_OR_CRS_TYPE = None, m: ipyleaflet.Map | map.Map | None = None, **kwargs) -> ipyleaflet.Map | map.Map:
        """Wrapper to initialise a map for plotting functions.

        Args:
            self (:class:`Resplotclass <resplotlib.Resplotclass>`): The instance of the Resplotclass.
            data_or_crs (:class:`xarray.DataArray`, :class:`xarray.Dataset`, :class:`xugrid.UgridDataArray`, :class:`xugrid.UgridDataset`, :class:`geopandas.GeoDataFrame`, :class:`pyproj.CRS`, :class:`rasterio.CRS`, str, or None, optional): The data or CRS to be plotted. Defaults to None.
            m (:class:`ipyleaflet.Map`, :class:`resplotlib.map.Map`, optional): The map to be used for plotting. If None, a new map will be created. Defaults to None.
            **kwargs (dict, optional): Additional keyword arguments to be passed to the plotting function.

        Returns:
            :class:`ipyleaflet.Map` or :class:`resplotlib.map.Map`: The map object resulting from the plotting function.
        """
        # Initialise map if not provided
        if m is not None:
            return func(self, data_or_crs, m=m, **kwargs)

        # Get bounds
        if isinstance(data_or_crs, xr.DataArray | xr.Dataset):
            bounds = data_or_crs.rio.bounds()
            gdf_bbox = gpd.GeoDataFrame(geometry=[box(*bounds)], crs=data_or_crs.rio.crs)
            bounds = gdf_bbox.to_crs("EPSG:4326").geometry.total_bounds

        elif isinstance(data_or_crs, xu.UgridDataArray | xu.UgridDataset):
            bounds = data_or_crs.ugrid.bounds
            gdf_bbox = gpd.GeoDataFrame(geometry=[box(*bounds)], crs=data_or_crs.ugrid.crs)
            bounds = gdf_bbox.to_crs("EPSG:4326").geometry.total_bounds
        elif isinstance(data_or_crs, gpd.GeoDataFrame):
            bounds = data_or_crs.total_bounds
            gdf_bbox = gpd.GeoDataFrame(geometry=[box(*bounds)], crs=data_or_crs.crs)
            bounds = gdf_bbox.to_crs("EPSG:4326").geometry.total_bounds
        elif isinstance(data_or_crs, pyprojCRS | rasterioCRS | str):
            bounds = (-180, -90, 180, 90)
        else:
            raise ValueError(
                "Data or CRS not provided. Please provide a xarray.DataArray, xarray.Dataset, xugrid.UgridDataArray, xugrid.UgridDataset, geopandas.GeoDataFrame, pyproj.CRS, rasterio.CRS, or a CRS string."
            )

        # Get center and zoom from bounds
        center = utils.get_center_from_bounds(bounds)
        zoom = utils.get_zoom_from_bounds(bounds)

        # Initialise map
        m = self.Map(center=center, zoom=zoom)

        return func(self, data_or_crs, m=m, **kwargs)

    return wrapper


def rescale_wrapper(func: callable) -> callable:
    @functools.wraps(func)
    def wrapper(self, data_or_crs: DATA_OR_CRS_TYPE = None, rescale_unit: str | None = None, **kwargs) -> plt.Axes:
        """Wrapper to rescale data or CRS for plotting functions.

        Args:
            self (:class:`Resplotclass <resplotlib.Resplotclass>`): The instance of the Resplotclass.
            data_or_crs (:class:`xarray.DataArray`, :class:`xarray.Dataset`, :class:`xugrid.UgridDataArray`, :class:`xugrid.UgridDataset`, :class:`geopandas.GeoDataFrame`, :class:`pyproj.CRS`, :class:`rasterio.CRS`, str, or None, optional): The data or CRS to be plotted. Defaults to None.
            rescale_unit (str, optional): The unit to which the data or CRS should be rescaled. Defaults to None.
            **kwargs (dict, optional): Additional keyword arguments to be passed to the plotting function.

        Returns:
            The result of the original function after rescaling.
        """
        # Get rescale parameters
        rescale_unit, scale_factor = utils.get_rescale_parameters(data_or_crs, rescale_unit=rescale_unit)

        # Rescale data or CRS
        if isinstance(data_or_crs, xr.DataArray | xr.Dataset):
            data_or_crs = rescale.rescale_da(data_or_crs, scale_factor)
        elif isinstance(data_or_crs, xu.UgridDataArray | xu.UgridDataset):
            data_or_crs = rescale.rescale_uda(data_or_crs, scale_factor)
        elif isinstance(data_or_crs, gpd.GeoDataFrame):
            data_or_crs = rescale.rescale_gdf(data_or_crs, scale_factor)

        # Rescale xlim and ylim if provided
        if "xlim" in kwargs and kwargs["xlim"] is not None:
            xlim = kwargs["xlim"]
            kwargs["xlim"] = (xlim[0] * scale_factor, xlim[1] * scale_factor)
        if "ylim" in kwargs and kwargs["ylim"] is not None:
            ylim = kwargs["ylim"]
            kwargs["ylim"] = (ylim[0] * scale_factor, ylim[1] * scale_factor)

        # Call original function
        return func(self, data_or_crs, rescale_unit=rescale_unit, **kwargs)

    return wrapper


def skip_and_smooth_wrapper(func: callable) -> callable:
    @functools.wraps(func)
    def wrapper(self, data_or_crs: DATA_OR_CRS_TYPE = None, skip: int = 1, smooth: int = 1, **kwargs):
        """Wrapper to skip and smooth data for plotting functions.

        Args:
            self (:class:`Resplotclass <resplotlib.Resplotclass>`): The instance of the Resplotclass.
            data_or_crs (:class:`xarray.DataArray`, :class:`xarray.Dataset`, :class:`xugrid.UgridDataArray`, :class:`xugrid.UgridDataset`, :class:`geopandas.GeoDataFrame`, :class:`pyproj.CRS`, :class:`rasterio.CRS`, str, or None, optional): The data or CRS to be plotted. Defaults to None.
            skip (int, optional): The factor by which to skip data points. Defaults to 1.
            smooth (int, optional): The factor by which to smooth data points. Defaults to 1.
            **kwargs (dict, optional): Additional keyword arguments to be passed to the plotting function.

        Returns:
            The result of the original function after skipping and smoothing.
        """
        # Skip and smooth data or CRS
        if isinstance(data_or_crs, xr.DataArray | xr.Dataset):
            if skip > 1:
                data_or_crs = data_or_crs.isel(x=slice(None, None, skip), y=slice(None, None, skip))
            if smooth > 1:
                data_or_crs = data_or_crs.rolling(x=smooth, y=smooth, center=True).mean()

        # Call original function
        return func(self, data_or_crs, **kwargs)

    return wrapper


def cbar_axis_wrapper(func: callable) -> callable:
    @functools.wraps(func)
    def wrapper(
        self, data_or_crs: DATA_OR_CRS_TYPE = None, ax: matplotlib.axes.Axes | None = None, append_axes_kwargs: dict | None = None, **kwargs
    ) -> matplotlib.axes.Axes:
        """Wrapper to create a secondary colorbar axis for plotting functions.

        Args:
            self (:class:`Resplotclass <resplotlib.Resplotclass>`): The instance of the Resplotclass.
            data_or_crs (:class:`xarray.DataArray`, :class:`xarray.Dataset`, :class:`xugrid.UgridDataArray`, :class:`xugrid.UgridDataset`, :class:`geopandas.GeoDataFrame`, :class:`pyproj.CRS`, :class:`rasterio.CRS`, str, or None, optional): The data or CRS to be plotted. Defaults to None.
            ax (:class:`matplotlib.axes.Axes`, optional): The axes to be used for plotting. Defaults to None.
            append_axes_kwargs (dict, optional): Additional keyword arguments to be passed to :func:`matplotlib.axes_grid1.axes_divider.make_axes_locatable.append_axes` for creating the colorbar axis. If None, no colorbar axis will be created. Defaults to None.
            **kwargs (dict, optional): Additional keyword arguments to be passed to the plotting function.

        Returns:
            :class:`matplotlib.axes.Axes`: The axes object resulting from the plotting function.
        """
        # Get function name
        func_name = _get_function_name(func)

        # If append_axes_kwargs is None (or function specific conditions are met), call original function
        if append_axes_kwargs is None:
            return func(self, data_or_crs, ax=ax, **kwargs)
        elif func_name == "imshow" and (("add_colorbar" in kwargs and kwargs["add_colorbar"] is False) or data_or_crs.ndim == 3):
            kwargs.pop("cbar_kwargs", None)
            return func(self, data_or_crs, ax=ax, **kwargs)
        elif func_name == "pcolormesh" and ("add_colorbar" in kwargs and kwargs["add_colorbar"] is False):
            kwargs.pop("cbar_kwargs", None)
            return func(self, data_or_crs, ax=ax, **kwargs)
        elif func_name == "contour" and ("add_colorbar" not in kwargs or kwargs["add_colorbar"] is False):
            kwargs.pop("cbar_kwargs", None)
            return func(self, data_or_crs, ax=ax, **kwargs)
        elif func_name == "contourf" and ("add_colorbar" in kwargs and kwargs["add_colorbar"] is False):
            kwargs.pop("cbar_kwargs", None)
            return func(self, data_or_crs, ax=ax, **kwargs)
        elif func_name == "scatter" and ("hue" not in kwargs or ("add_colorbar" not in kwargs or kwargs["add_colorbar"] is False)):
            kwargs.pop("cbar_kwargs", None)
            return func(self, data_or_crs, ax=ax, **kwargs)
        elif func_name == "quiver" and ("hue" not in kwargs or ("add_guide" in kwargs and kwargs["add_guide"] is False)):
            kwargs.pop("cbar_kwargs", None)
            return func(self, data_or_crs, ax=ax, **kwargs)
        elif func_name == "streamplot" and ("hue" not in kwargs or ("add_guide" in kwargs and kwargs["add_guide"] is False)):
            kwargs.pop("cbar_kwargs", None)
            return func(self, data_or_crs, ax=ax, **kwargs)
        elif func_name == "grid" and (data_or_crs.dims[0] != "mesh2d_nEdges" or ("add_colorbar" in kwargs and kwargs["add_colorbar"] is False)):
            kwargs.pop("cbar_kwargs", None)
            return func(self, data_or_crs, ax=ax, **kwargs)
        elif func_name == "geometries" and ("legend" not in kwargs or kwargs["legend"] is False):
            return func(self, data_or_crs, ax=ax, **kwargs)

        # Divide axis
        divider = make_axes_locatable(ax)

        # Create colorbar axis
        cax = divider.append_axes(**append_axes_kwargs)

        # Change keywords based on function name
        if func_name != "geometries":
            cbar_kwargs = kwargs.get("cbar_kwargs", {})
            cbar_kwargs["cax"] = cax
            kwargs["cbar_kwargs"] = cbar_kwargs
        else:
            kwargs["cax"] = cax

        # Call original function
        return func(self, data_or_crs, ax=ax, **kwargs)

    return wrapper


def format_axis_wrapper(func: callable) -> callable:
    @functools.wraps(func)
    def wrapper(
        self: Resplotclass,
        data_or_crs: DATA_OR_CRS_TYPE = None,
        rescale_unit: str | None = None,
        xlim: tuple[float, float] | None = None,
        ylim: tuple[float, float] | None = None,
        xlabel_kwargs: dict | None = None,
        ylabel_kwargs: dict | None = None,
        title_kwargs: dict | None = None,
        aspect_kwargs: dict | None = None,
        grid_kwargs: dict | None = None,
        **kwargs,
    ) -> plt.Axes:
        """Wrapper to format axes for plotting functions.

        Args:
            self (:class:`Resplotclass <resplotlib.Resplotclass>`): The instance of the Resplotclass.
            data_or_crs (:class:`xarray.DataArray`, :class:`xarray.Dataset`, :class:`xugrid.UgridDataArray`, :class:`xugrid.UgridDataset`, :class:`geopandas.GeoDataFrame`, :class:`pyproj.CRS`, :class:`rasterio.CRS`, str, or None, optional): The data or CRS to be plotted. Defaults to None.
            rescale_unit (str, optional): The unit to which the data or CRS should be rescaled. Defaults to None.
            xlim (tuple[float, float], optional): The limits for the x-axis. Defaults to None.
            ylim (tuple[float, float], optional): The limits for the y-axis. Defaults to None.
            xlabel_kwargs (dict, optional): Additional keyword arguments passed to :func:`matplotlib.axes.Axes.set_xlabel` for formatting the x-axis label. If None, the x-axis label will be determined automatically. Defaults to None.
            ylabel_kwargs (dict, optional): Additional keyword arguments passed to :func:`matplotlib.axes.Axes.set_ylabel` for formatting the y-axis label. If None, the y-axis label will be determined automatically. Defaults to None.
            title_kwargs (dict, optional): Additional keyword arguments passed to :func:`matplotlib.axes.Axes.set_title` for formatting the title. If None, the title will be set to an empty string. Defaults to None.
            aspect_kwargs (dict, optional): Additional keyword arguments passed to :func:`matplotlib.axes.Axes.set_aspect` for formatting the aspect ratio. If None, the aspect ratio will be set to 'equal'. Defaults to None.
            grid_kwargs (dict, optional): Additional keyword arguments passed to :func:`matplotlib.axes.Axes.grid` for formatting the grid. If None, the grid will be set to visible. Defaults to None.
            **kwargs (dict, optional): Additional keyword arguments to be passed to the plotting function.

        Returns:
            :class:`matplotlib.axes.Axes`: The axes object resulting from the plotting function.
        """
        # Get function name
        func_name = _get_function_name(func)

        # Get default xlabel and ylabel kwargs if not provided
        if xlabel_kwargs is None or ylabel_kwargs is None:
            # Get rescale parameters
            rescale_unit, _ = utils.get_rescale_parameters(data_or_crs, rescale_unit=rescale_unit)

            # Get x and y labels
            xlabel, ylabel = utils.get_xy_labels(data_or_crs, rescale_unit=rescale_unit)

            # Set xlabel and ylabel kwargs if not provided
            if xlabel_kwargs is None:
                xlabel_kwargs = {"xlabel": xlabel}
            if ylabel_kwargs is None:
                ylabel_kwargs = {"ylabel": ylabel}

        # Set default title, aspect, and grid kwargs if not provided
        if title_kwargs is None:
            title_kwargs = {"label": ""}
        if aspect_kwargs is None:
            aspect_kwargs = {"aspect": "equal"}
        if grid_kwargs is None:
            grid_kwargs = {"visible": True}

        # Call original function
        if func_name in ["basemap"]:
            p = func(self, data_or_crs, rescale_unit=rescale_unit, xlim=xlim, ylim=ylim, **kwargs)
        else:
            p = func(self, data_or_crs, **kwargs)

        # Format axes
        ax = p.axes
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xlabel(**xlabel_kwargs)
        ax.set_ylabel(**ylabel_kwargs)
        ax.set_title(**title_kwargs)
        ax.set_aspect(**aspect_kwargs)
        ax.grid(**grid_kwargs)

        return p

    return wrapper


def show_kwargs_wrapper(func: callable) -> callable:
    @functools.wraps(func)
    def wrapper(self, data_or_crs: DATA_OR_CRS_TYPE = None, show_kwargs: bool = False, **kwargs) -> PLOT_TYPE:
        """Wrapper to print keyword arguments passed to plotting function.

        Args:
            self (:class:`Resplotclass <resplotlib.Resplotclass>`): The instance of the Resplotclass.
            data_or_crs (:class:`xarray.DataArray`, :class:`xarray.Dataset`, :class:`xugrid.UgridDataArray`, :class:`xugrid.UgridDataset`, :class:`geopandas.GeoDataFrame`, :class:`pyproj.CRS`, :class:`rasterio.CRS`, str, or None, optional): The data or CRS to be plotted. Defaults to None.
            show_kwargs (bool, optional): Whether to print the keyword arguments passed to plotting function. Defaults to False.
            **kwargs (dict, optional): Additional keyword arguments to be passed to the plotting function.
        Returns:
            :class:`matplotlib.axes.Axes`, :class:`ipyleaflet.Map`, or :class:`resplotlib.map.Map`: The result of the original function.
        """
        # Show kwargs if requested
        if show_kwargs:
            print(f"Keywords passed to {func.__name__}: {kwargs}")

        # Call the original function
        if data_or_crs is not None:
            return func(self, data_or_crs, **kwargs)
        else:
            return func(self, **kwargs)

    return wrapper


def _debug_wrapper(func: callable) -> callable:
    @functools.wraps(func)
    def wrapper(self, data_or_crs: DATA_OR_CRS_TYPE = None, **kwargs) -> PLOT_TYPE:
        """Wrapper to print function name and arguments for debugging purposes.

        Args:
            self (:class:`Resplotclass <resplotlib.Resplotclass>`): The instance of the Resplotclass.
            data_or_crs (:class:`xarray.DataArray`, :class:`xarray.Dataset`, :class:`xugrid.UgridDataArray`, :class:`xugrid.UgridDataset`, :class:`geopandas.GeoDataFrame`, :class:`pyproj.CRS`, :class:`rasterio.CRS`, str, or None, optional): The data or CRS to be plotted. Defaults to None.
            **kwargs (dict, optional): Additional keyword arguments to be passed to the plotting function.

        Returns:
            :class:`matplotlib.axes.Axes`, :class:`ipyleaflet.Map`, or :class:`resplotlib.map.Map`: The result of the original function.
        """
        # Show function name and arguments
        print(f"Function: {func.__name__}")
        print(f"Arguments: {data_or_crs}, {kwargs}")

        # Call the original function
        if data_or_crs is not None:
            return func(self, data_or_crs, **kwargs)
        else:
            return func(self, **kwargs)

    return wrapper
