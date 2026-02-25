import copy
import gc
import time
from pathlib import Path

import geopandas as gpd
import ipyleaflet
import matplotlib.pyplot as plt
import xarray as xr
import xugrid as xu
from IPython.display import display
from pyproj import CRS as pyprojCRS
from rasterio.crs import CRS as rasterioCRS

from . import basemaps, explore, geometries, map, utils, videos, wrappers
from .guidelines import Guidelines

DIR_PATH_PACKAGE = Path(__file__).resolve().parent
FILE_PATH_DEFAULT_GLS = str(DIR_PATH_PACKAGE / "guidelines" / "default_guidelines.json")


class Resplotclass:
    """Resilient plotter class.

    This class provides methods for plotting and exploring geospatial data. It also provides methods for setting and getting guidelines for plotting, as well as support methods for creating subplots and maps, showing and saving figures, and creating videos.

    Attributes:
        guidelines (:class:`resplotlib.Guidelines`): The guidelines for plotting.

    Args:
        file_path (str, optional): File path to project guidelines. Defaults to None.
    """

    def __init__(self, file_path: str | None = None) -> None:
        """Initialise the Resplotclass.

        Args:
            file_path (str, optional): File path to project guidelines. Defaults to None.
        """
        # Set guidelines
        self.set_guidelines(file_path=file_path)

    # Guidelines methods
    def set_guidelines(self, file_path: str | None = None) -> None:
        """Set guidelines for the Resplotclass.

        Args:
            file_path (str, optional): File path to project guidelines. Defaults to None.
        """

        # Read default guidelines
        self.default_guidelines = Guidelines(file_path=FILE_PATH_DEFAULT_GLS)
        self.default_guidelines["metadata"]["file_path"] = FILE_PATH_DEFAULT_GLS

        # Read project guidelines
        if file_path is not None:
            self.project_guidelines = Guidelines(file_path=file_path)
            self.project_guidelines["metadata"]["file_path"] = file_path
        else:
            self.project_guidelines = Guidelines()

        # Combine guidelines
        self.guidelines = Guidelines(
            utils._combine_dicts(
                copy.deepcopy(self.default_guidelines),
                copy.deepcopy(self.project_guidelines),
            )
        )

        # Substitute properties in guidelines
        self.guidelines._substitute_properties()

    def get_guidelines(self) -> Guidelines:
        """Get guidelines for the Resplotclass.

        Returns:
            :class:`resplotlib.Guidelines`: The guidelines object.
        """
        return self.guidelines

    def show_guidelines(self) -> None:
        """Show guidelines for the Resplotclass."""
        self.guidelines.show()

    # Static plot methods
    @wrappers.format_args_wrapper
    @wrappers.guideline_wrapper
    @wrappers.initialise_fig_wrapper
    @wrappers.rescale_wrapper
    @wrappers.skip_and_smooth_wrapper
    @wrappers.cbar_axis_wrapper
    @wrappers.show_kwargs_wrapper
    @wrappers.format_axis_wrapper
    def imshow(
        self,
        da: xr.DataArray | xu.UgridDataArray,
        ax: plt.Axes | None = None,
        style: str | None = None,
        extent: str | None = None,
        rescale_unit: str | None = None,
        skip: int = 1,
        smooth: int = 1,
        xlim: tuple[float, float] | None = None,
        ylim: tuple[float, float] | None = None,
        subplot_kwargs: dict | None = None,
        append_axes_kwargs: dict | None = None,
        xlabel_kwargs: dict | None = None,
        ylabel_kwargs: dict | None = None,
        title_kwargs: dict | None = None,
        aspect_kwargs: dict | None = None,
        grid_kwargs: dict | None = None,
        show_kwargs: bool = False,
        **kwargs,
    ) -> plt.Axes:
        """Image plot of a xarray.DataArray or xugrid.UgridDataArray.

        Args:
            da (:class:`xarray:xarray.DataArray` or :class:`xugrid.UgridDataArray`): DataArray to plot.
            ax (:class:`matplotlib.axes.Axes`, optional): The axes to be used for plotting. If None, a new figure and axes will be created. Defaults to None.
            style (str, optional): The style to be applied to the plot. Defaults to "default". Set to "none" to ignore style guidelines.
            extent (str or None, optional): The extent to be applied to the plot. Defaults to None.
            rescale_unit (str or None, optional): The unit to which the data or CRS should be rescaled. If None, the rescale_unit specified in the guidelines will be used. Defaults to None.
            skip (int, optional): The factor by which to skip data points. Defaults to 1.
            smooth (int, optional): The factor by which to smooth data points. Defaults to 1.
            xlim (tuple[float, float], optional): The limits for the x-axis. Defaults to None.
            ylim (tuple[float, float], optional): The limits for the y-axis. Defaults to None.
            subplot_kwargs (dict, optional): Additional keyword arguments to be passed to :func:`matplotlib.pyplot.subplots` when creating a new figure and axes. Defaults to None.
            append_axes_kwargs (dict, optional): Additional keyword arguments to be passed to :func:`matplotlib.axes_grid1.axes_divider.make_axes_locatable.append_axes` for creating the colorbar axis. If None, no colorbar axis will be created. Defaults to None.
            xlabel_kwargs (dict, optional): Additional keyword arguments passed to :func:`matplotlib.axes.Axes.set_xlabel` for formatting the x-axis label. If None, the x-axis label will be determined automatically. Defaults to None.
            ylabel_kwargs (dict, optional): Additional keyword arguments passed to :func:`matplotlib.axes.Axes.set_ylabel` for formatting the y-axis label. If None, the y-axis label will be determined automatically. Defaults to None.
            title_kwargs (dict, optional): Additional keyword arguments passed to :func:`matplotlib.axes.Axes.set_title` for formatting the title. If None, the title will be set to an empty string. Defaults to None.
            aspect_kwargs (dict, optional): Additional keyword arguments passed to :func:`matplotlib.axes.Axes.set_aspect` for formatting the aspect ratio. If None, the aspect ratio will be set to 'equal'. Defaults to None.
            grid_kwargs (dict, optional): Additional keyword arguments passed to :func:`matplotlib.axes.Axes.grid` for formatting the grid. If None, the grid will be set to visible. Defaults to None.
            show_kwargs (bool, optional): Whether to print the keyword arguments passed to plotting function. Defaults to False.
            **kwargs (dict, optional): Additional keyword arguments to pass to :func:`xarray.plot.imshow` or :func:`xugrid.plot.imshow`.

        Returns:
            :class:`matplotlib.axes.Axes`: The axes object of the plot.
        """
        if isinstance(da, xr.DataArray):
            return da.plot.imshow(ax=ax, **kwargs)
        elif isinstance(da, xu.UgridDataArray):
            return da.ugrid.plot.imshow(ax=ax, **kwargs)
        else:
            raise TypeError("Data must be an xarray.DataArray or xugrid.UgridDataArray")

    @wrappers.format_args_wrapper
    @wrappers.guideline_wrapper
    @wrappers.initialise_fig_wrapper
    @wrappers.rescale_wrapper
    @wrappers.skip_and_smooth_wrapper
    @wrappers.cbar_axis_wrapper
    @wrappers.show_kwargs_wrapper
    @wrappers.format_axis_wrapper
    def pcolormesh(
        self,
        da: xr.DataArray | xu.UgridDataArray,
        ax: plt.Axes | None = None,
        style: str | None = None,
        extent: str | None = None,
        rescale_unit: str | None = None,
        skip: int = 1,
        smooth: int = 1,
        xlim: tuple[float, float] | None = None,
        ylim: tuple[float, float] | None = None,
        subplot_kwargs: dict | None = None,
        append_axes_kwargs: dict | None = None,
        xlabel_kwargs: dict | None = None,
        ylabel_kwargs: dict | None = None,
        title_kwargs: dict | None = None,
        aspect_kwargs: dict | None = None,
        grid_kwargs: dict | None = None,
        show_kwargs: bool = False,
        **kwargs,
    ) -> plt.Axes:
        """Pseudocolor plot of a xarray.DataArray or xugrid.UgridDataArray.

        Args:
            da (:class:`xarray.DataArray` or :class:`xugrid.UgridDataArray`): DataArray to plot.
            ax (:class:`matplotlib.axes.Axes`, optional): The axes to be used for plotting. If None, a new figure and axes will be created. Defaults to None.
            style (str, optional): The style to be applied to the plot. Defaults to "default". Set to "none" to ignore style guidelines.
            extent (str or None, optional): The extent to be applied to the plot. Defaults to None.
            rescale_unit (str or None, optional): The unit to which the data or CRS should be rescaled. If None, the rescale_unit specified in the guidelines will be used. Defaults to None.
            skip (int, optional): The factor by which to skip data points. Defaults to 1.
            smooth (int, optional): The factor by which to smooth data points. Defaults to 1.
            xlim (tuple[float, float], optional): The limits for the x-axis. Defaults to None.
            ylim (tuple[float, float], optional): The limits for the y-axis. Defaults to None.
            subplot_kwargs (dict, optional): Additional keyword arguments to be passed to :func:`matplotlib.pyplot.subplots` when creating a new figure and axes. Defaults to None.
            append_axes_kwargs (dict, optional): Additional keyword arguments to be passed to :func:`matplotlib.axes_grid1.axes_divider.make_axes_locatable.append_axes` for creating the colorbar axis. If None, no colorbar axis will be created. Defaults to None.
            xlabel_kwargs (dict, optional): Additional keyword arguments passed to :func:`matplotlib.axes.Axes.set_xlabel` for formatting the x-axis label. If None, the x-axis label will be determined automatically. Defaults to None.
            ylabel_kwargs (dict, optional): Additional keyword arguments passed to :func:`matplotlib.axes.Axes.set_ylabel` for formatting the y-axis label. If None, the y-axis label will be determined automatically. Defaults to None.
            title_kwargs (dict, optional): Additional keyword arguments passed to :func:`matplotlib.axes.Axes.set_title` for formatting the title. If None, the title will be set to an empty string. Defaults to None.
            aspect_kwargs (dict, optional): Additional keyword arguments passed to :func:`matplotlib.axes.Axes.set_aspect` for formatting the aspect ratio. If None, the aspect ratio will be set to 'equal'. Defaults to None.
            grid_kwargs (dict, optional): Additional keyword arguments passed to :func:`matplotlib.axes.Axes.grid` for formatting the grid. If None, the grid will be set to visible. Defaults to None.
            show_kwargs (bool, optional): Whether to print the keyword arguments passed to plotting function. Defaults to False.
            **kwargs (dict, optional): Additional keyword arguments to pass to :func:`xarray.plot.pcolormesh` or :func:`xugrid.plot.pcolormesh`.

        Returns:
            :class:`matplotlib.axes.Axes`: The axes object of the plot.
        """
        if isinstance(da, xr.DataArray):
            return da.plot.pcolormesh(ax=ax, **kwargs)
        elif isinstance(da, xu.UgridDataArray):
            return da.ugrid.plot.pcolormesh(ax=ax, **kwargs)
        else:
            raise TypeError("Data must be an xarray.DataArray or xugrid.UgridDataArray")

    @wrappers.format_args_wrapper
    @wrappers.guideline_wrapper
    @wrappers.initialise_fig_wrapper
    @wrappers.rescale_wrapper
    @wrappers.skip_and_smooth_wrapper
    @wrappers.cbar_axis_wrapper
    @wrappers.show_kwargs_wrapper
    @wrappers.format_axis_wrapper
    def contour(
        self,
        da: xr.DataArray | xu.UgridDataArray,
        ax: plt.Axes | None = None,
        style: str | None = None,
        extent: str | None = None,
        rescale_unit: str | None = None,
        skip: int = 1,
        smooth: int = 1,
        xlim: tuple[float, float] | None = None,
        ylim: tuple[float, float] | None = None,
        subplot_kwargs: dict | None = None,
        append_axes_kwargs: dict | None = None,
        xlabel_kwargs: dict | None = None,
        ylabel_kwargs: dict | None = None,
        title_kwargs: dict | None = None,
        aspect_kwargs: dict | None = None,
        grid_kwargs: dict | None = None,
        show_kwargs: bool = False,
        **kwargs,
    ) -> plt.Axes:
        """Contour plot of a xarray.DataArray or xugrid.UgridDataArray.

        Args:
            da (:class:`xarray.DataArray` or :class:`xugrid.UgridDataArray`): DataArray to plot.
            ax (:class:`matplotlib.axes.Axes`, optional): The axes to be used for plotting. If None, a new figure and axes will be created. Defaults to None.
            style (str, optional): The style to be applied to the plot. Defaults to "default". Set to "none" to ignore style guidelines.
            extent (str or None, optional): The extent to be applied to the plot. Defaults to None.
            rescale_unit (str or None, optional): The unit to which the data or CRS should be rescaled. If None, the rescale_unit specified in the guidelines will be used. Defaults to None.
            skip (int, optional): The factor by which to skip data points. Defaults to 1.
            smooth (int, optional): The factor by which to smooth data points. Defaults to 1.
            xlim (tuple[float, float], optional): The limits for the x-axis. Defaults to None.
            ylim (tuple[float, float], optional): The limits for the y-axis. Defaults to None.
            subplot_kwargs (dict, optional): Additional keyword arguments to be passed to :func:`matplotlib.pyplot.subplots` when creating a new figure and axes. Defaults to None.
            append_axes_kwargs (dict, optional): Additional keyword arguments to be passed to :func:`matplotlib.axes_grid1.axes_divider.make_axes_locatable.append_axes` for creating the colorbar axis. If None, no colorbar axis will be created. Defaults to None.
            xlabel_kwargs (dict, optional): Additional keyword arguments passed to :func:`matplotlib.axes.Axes.set_xlabel` for formatting the x-axis label. If None, the x-axis label will be determined automatically. Defaults to None.
            ylabel_kwargs (dict, optional): Additional keyword arguments passed to :func:`matplotlib.axes.Axes.set_ylabel` for formatting the y-axis label. If None, the y-axis label will be determined automatically. Defaults to None.
            title_kwargs (dict, optional): Additional keyword arguments passed to :func:`matplotlib.axes.Axes.set_title` for formatting the title. If None, the title will be set to an empty string. Defaults to None.
            aspect_kwargs (dict, optional): Additional keyword arguments passed to :func:`matplotlib.axes.Axes.set_aspect` for formatting the aspect ratio. If None, the aspect ratio will be set to 'equal'. Defaults to None.
            grid_kwargs (dict, optional): Additional keyword arguments passed to :func:`matplotlib.axes.Axes.grid` for formatting the grid. If None, the grid will be set to visible. Defaults to None.
            show_kwargs (bool, optional): Whether to print the keyword arguments passed to plotting function. Defaults to False.
            **kwargs (dict, optional): Additional keyword arguments to pass to :func:`xarray.plot.contour` or :func:`xugrid.plot.contour`.

        Returns:
            :class:`matplotlib.axes.Axes`: The axes object of the plot.
        """
        if isinstance(da, xr.DataArray):
            return da.plot.contour(ax=ax, **kwargs)
        elif isinstance(da, xu.UgridDataArray):
            return da.ugrid.plot.contour(ax=ax, **kwargs)
        else:
            raise TypeError("Data must be an xarray.DataArray or xugrid.UgridDataArray")

    @wrappers.format_args_wrapper
    @wrappers.guideline_wrapper
    @wrappers.initialise_fig_wrapper
    @wrappers.rescale_wrapper
    @wrappers.skip_and_smooth_wrapper
    @wrappers.cbar_axis_wrapper
    @wrappers.show_kwargs_wrapper
    @wrappers.format_axis_wrapper
    def contourf(
        self,
        da: xr.DataArray | xu.UgridDataArray,
        ax: plt.Axes | None = None,
        style: str | None = None,
        extent: str | None = None,
        rescale_unit: str | None = None,
        skip: int = 1,
        smooth: int = 1,
        xlim: tuple[float, float] | None = None,
        ylim: tuple[float, float] | None = None,
        subplot_kwargs: dict | None = None,
        append_axes_kwargs: dict | None = None,
        xlabel_kwargs: dict | None = None,
        ylabel_kwargs: dict | None = None,
        title_kwargs: dict | None = None,
        aspect_kwargs: dict | None = None,
        grid_kwargs: dict | None = None,
        show_kwargs: bool = False,
        **kwargs,
    ) -> plt.Axes:
        """Filled contour plot of a xarray.DataArray or xugrid.UgridDataArray.

        Args:
            da (:class:`xarray.DataArray` or :class:`xugrid.UgridDataArray`): DataArray to plot.
            ax (:class:`matplotlib.axes.Axes`, optional): The axes to be used for plotting. If None, a new figure and axes will be created. Defaults to None.
            style (str, optional): The style to be applied to the plot. Defaults to "default". Set to "none" to ignore style guidelines.
            extent (str or None, optional): The extent to be applied to the plot. Defaults to None.
            rescale_unit (str or None, optional): The unit to which the data or CRS should be rescaled. If None, the rescale_unit specified in the guidelines will be used. Defaults to None.
            skip (int, optional): The factor by which to skip data points. Defaults to 1.
            smooth (int, optional): The factor by which to smooth data points. Defaults to 1.
            xlim (tuple[float, float], optional): The limits for the x-axis. Defaults to None.
            ylim (tuple[float, float], optional): The limits for the y-axis. Defaults to None.
            subplot_kwargs (dict, optional): Additional keyword arguments to be passed to :func:`matplotlib.pyplot.subplots` when creating a new figure and axes. Defaults to None.
            append_axes_kwargs (dict, optional): Additional keyword arguments to be passed to :func:`matplotlib.axes_grid1.axes_divider.make_axes_locatable.append_axes` for creating the colorbar axis. If None, no colorbar axis will be created. Defaults to None.
            xlabel_kwargs (dict, optional): Additional keyword arguments passed to :func:`matplotlib.axes.Axes.set_xlabel` for formatting the x-axis label. If None, the x-axis label will be determined automatically. Defaults to None.
            ylabel_kwargs (dict, optional): Additional keyword arguments passed to :func:`matplotlib.axes.Axes.set_ylabel` for formatting the y-axis label. If None, the y-axis label will be determined automatically. Defaults to None.
            title_kwargs (dict, optional): Additional keyword arguments passed to :func:`matplotlib.axes.Axes.set_title` for formatting the title. If None, the title will be set to an empty string. Defaults to None.
            aspect_kwargs (dict, optional): Additional keyword arguments passed to :func:`matplotlib.axes.Axes.set_aspect` for formatting the aspect ratio. If None, the aspect ratio will be set to 'equal'. Defaults to None.
            grid_kwargs (dict, optional): Additional keyword arguments passed to :func:`matplotlib.axes.Axes.grid` for formatting the grid. If None, the grid will be set to visible. Defaults to None.
            show_kwargs (bool, optional): Whether to print the keyword arguments passed to plotting function. Defaults to False.
            **kwargs (dict, optional): Additional keyword arguments to pass to :func:`xarray.plot.contourf` or :func:`xugrid.plot.contourf`.

        Returns:
            :class:`matplotlib.axes.Axes`: The axes object of the plot.
        """
        if isinstance(da, xr.DataArray):
            return da.plot.contourf(ax=ax, **kwargs)
        elif isinstance(da, xu.UgridDataArray):
            return da.ugrid.plot.contourf(ax=ax, **kwargs)
        else:
            raise TypeError("Data must be an xarray.DataArray or xugrid.UgridDataArray")

    @wrappers.format_args_wrapper
    @wrappers.guideline_wrapper
    @wrappers.initialise_fig_wrapper
    @wrappers.rescale_wrapper
    @wrappers.skip_and_smooth_wrapper
    @wrappers.cbar_axis_wrapper
    @wrappers.show_kwargs_wrapper
    @wrappers.format_axis_wrapper
    def scatter(
        self,
        ds: xr.Dataset | xu.UgridDataArray,
        ax: plt.Axes | None = None,
        style: str | None = None,
        extent: str | None = None,
        rescale_unit: str | None = None,
        skip: int = 1,
        smooth: int = 1,
        xlim: tuple[float, float] | None = None,
        ylim: tuple[float, float] | None = None,
        subplot_kwargs: dict | None = None,
        append_axes_kwargs: dict | None = None,
        xlabel_kwargs: dict | None = None,
        ylabel_kwargs: dict | None = None,
        title_kwargs: dict | None = None,
        aspect_kwargs: dict | None = None,
        grid_kwargs: dict | None = None,
        show_kwargs: bool = False,
        **kwargs,
    ) -> plt.Axes:
        """Scatter plot of a xarray.Dataset or xugrid.UgridDataArray.

        Args:
            ds (:class:`xarray.Dataset` or :class:`xugrid.UgridDataArray`): Dataset or DataArray to plot.
            ax (:class:`matplotlib.axes.Axes`, optional): The axes to be used for plotting. If None, a new figure and axes will be created. Defaults to None.
            style (str, optional): The style to be applied to the plot. Defaults to "default". Set to "none" to ignore style guidelines.
            extent (str or None, optional): The extent to be applied to the plot. Defaults to None.
            rescale_unit (str or None, optional): The unit to which the data or CRS should be rescaled. If None, the rescale_unit specified in the guidelines will be used. Defaults to None.
            skip (int, optional): The factor by which to skip data points. Defaults to 1.
            smooth (int, optional): The factor by which to smooth data points. Defaults to 1.
            xlim (tuple[float, float], optional): The limits for the x-axis. Defaults to None.
            ylim (tuple[float, float], optional): The limits for the y-axis. Defaults to None.
            subplot_kwargs (dict, optional): Additional keyword arguments to be passed to :func:`matplotlib.pyplot.subplots` when creating a new figure and axes. Defaults to None.
            append_axes_kwargs (dict, optional): Additional keyword arguments to be passed to :func:`matplotlib.axes_grid1.axes_divider.make_axes_locatable.append_axes` for creating the colorbar axis. If None, no colorbar axis will be created. Defaults to None.
            xlabel_kwargs (dict, optional): Additional keyword arguments passed to :func:`matplotlib.axes.Axes.set_xlabel` for formatting the x-axis label. If None, the x-axis label will be determined automatically. Defaults to None.
            ylabel_kwargs (dict, optional): Additional keyword arguments passed to :func:`matplotlib.axes.Axes.set_ylabel` for formatting the y-axis label. If None, the y-axis label will be determined automatically. Defaults to None.
            title_kwargs (dict, optional): Additional keyword arguments passed to :func:`matplotlib.axes.Axes.set_title` for formatting the title. If None, the title will be set to an empty string. Defaults to None.
            aspect_kwargs (dict, optional): Additional keyword arguments passed to :func:`matplotlib.axes.Axes.set_aspect` for formatting the aspect ratio. If None, the aspect ratio will be set to 'equal'. Defaults to None.
            grid_kwargs (dict, optional): Additional keyword arguments passed to :func:`matplotlib.axes.Axes.grid` for formatting the grid. If None, the grid will be set to visible. Defaults to None.
            show_kwargs (bool, optional): Whether to print the keyword arguments passed to plotting function. Defaults to False.
            **kwargs (dict, optional): Additional keyword arguments to pass to :func:`xarray.plot.scatter` or :func:`xugrid.plot.scatter`.

        Returns:
            :class:`matplotlib.axes.Axes`: The axes object of the plot.
        """

        if isinstance(ds, xr.Dataset):
            if "hue" not in kwargs and "color" not in kwargs:
                kwargs.setdefault("color", "black")
            return ds.plot.scatter(**kwargs)
        elif isinstance(ds, xu.UgridDataArray):
            return ds.ugrid.plot.scatter(**kwargs)
        else:
            raise TypeError("Data must be an xarray.Dataset or xugrid.UgridDataArray")

    @wrappers.format_args_wrapper
    @wrappers.guideline_wrapper
    @wrappers.initialise_fig_wrapper
    @wrappers.rescale_wrapper
    @wrappers.skip_and_smooth_wrapper
    @wrappers.cbar_axis_wrapper
    @wrappers.show_kwargs_wrapper
    @wrappers.format_axis_wrapper
    def quiver(
        self,
        ds: xr.Dataset,
        ax: plt.Axes | None = None,
        style: str | None = None,
        extent: str | None = None,
        rescale_unit: str | None = None,
        skip: int = 1,
        smooth: int = 1,
        xlim: tuple[float, float] | None = None,
        ylim: tuple[float, float] | None = None,
        subplot_kwargs: dict | None = None,
        append_axes_kwargs: dict | None = None,
        xlabel_kwargs: dict | None = None,
        ylabel_kwargs: dict | None = None,
        title_kwargs: dict | None = None,
        aspect_kwargs: dict | None = None,
        grid_kwargs: dict | None = None,
        show_kwargs: bool = False,
        **kwargs,
    ) -> plt.Axes:
        """Quiver plot of a xarray.Dataset.

        Args:
            ds (:class:`xarray.Dataset`): Dataset to plot.
            ax (:class:`matplotlib.axes.Axes`, optional): The axes to be used for plotting. If None, a new figure and axes will be created. Defaults to None.
            style (str, optional): The style to be applied to the plot. Defaults to "default". Set to "none" to ignore style guidelines.
            extent (str or None, optional): The extent to be applied to the plot. Defaults to None.
            rescale_unit (str or None, optional): The unit to which the data or CRS should be rescaled. If None, the rescale_unit specified in the guidelines will be used. Defaults to None.
            skip (int, optional): The factor by which to skip data points. Defaults to 1.
            smooth (int, optional): The factor by which to smooth data points. Defaults to 1.
            xlim (tuple[float, float], optional): The limits for the x-axis. Defaults to None.
            ylim (tuple[float, float], optional): The limits for the y-axis. Defaults to None.
            subplot_kwargs (dict, optional): Additional keyword arguments to be passed to :func:`matplotlib.pyplot.subplots` when creating a new figure and axes. Defaults to None.
            append_axes_kwargs (dict, optional): Additional keyword arguments to be passed to :func:`matplotlib.axes_grid1.axes_divider.make_axes_locatable.append_axes` for creating the colorbar axis. If None, no colorbar axis will be created. Defaults to None.
            xlabel_kwargs (dict, optional): Additional keyword arguments passed to :func:`matplotlib.axes.Axes.set_xlabel` for formatting the x-axis label. If None, the x-axis label will be determined automatically. Defaults to None.
            ylabel_kwargs (dict, optional): Additional keyword arguments passed to :func:`matplotlib.axes.Axes.set_ylabel` for formatting the y-axis label. If None, the y-axis label will be determined automatically. Defaults to None.
            title_kwargs (dict, optional): Additional keyword arguments passed to :func:`matplotlib.axes.Axes.set_title` for formatting the title. If None, the title will be set to an empty string. Defaults to None.
            aspect_kwargs (dict, optional): Additional keyword arguments passed to :func:`matplotlib.axes.Axes.set_aspect` for formatting the aspect ratio. If None, the aspect ratio will be set to 'equal'. Defaults to None.
            grid_kwargs (dict, optional): Additional keyword arguments passed to :func:`matplotlib.axes.Axes.grid` for formatting the grid. If None, the grid will be set to visible. Defaults to None.
            show_kwargs (bool, optional): Whether to print the keyword arguments passed to plotting function. Defaults to False.
            **kwargs (dict, optional): Additional keyword arguments to pass to :func:`xarray.plot.quiver`.

        Returns:
            :class:`matplotlib.axes.Axes`: The axes object of the plot.
        """
        if isinstance(ds, xr.Dataset):
            # Set default color to black if neither hue nor color is specified
            if "hue" not in kwargs and "color" not in kwargs:
                kwargs.setdefault("color", "black")

            # Quiver requires that x and y must be the first two dimensions
            xy_dims = [kwargs.get("x", "x"), kwargs.get("y", "y")]
            dims = xy_dims + [dim for dim in ds.dims if dim not in xy_dims]
            ds = ds.transpose(*dims)

            return ds.plot.quiver(ax=ax, **kwargs)
        else:
            raise TypeError("Data must be an xarray.Dataset")

    @wrappers.format_args_wrapper
    @wrappers.guideline_wrapper
    @wrappers.initialise_fig_wrapper
    @wrappers.rescale_wrapper
    @wrappers.skip_and_smooth_wrapper
    @wrappers.cbar_axis_wrapper
    @wrappers.show_kwargs_wrapper
    @wrappers.format_axis_wrapper
    def streamplot(
        self,
        ds: xr.Dataset,
        ax: plt.Axes | None = None,
        style: str | None = None,
        extent: str | None = None,
        rescale_unit: str | None = None,
        skip: int = 1,
        smooth: int = 1,
        xlim: tuple[float, float] | None = None,
        ylim: tuple[float, float] | None = None,
        subplot_kwargs: dict | None = None,
        append_axes_kwargs: dict | None = None,
        xlabel_kwargs: dict | None = None,
        ylabel_kwargs: dict | None = None,
        title_kwargs: dict | None = None,
        aspect_kwargs: dict | None = None,
        grid_kwargs: dict | None = None,
        show_kwargs: bool = False,
        **kwargs,
    ) -> plt.Axes:
        """Streamline plot of a xarray.Dataset.

        Args:
            ds (:class:`xarray.Dataset`): Dataset to plot.
            ax (:class:`matplotlib.axes.Axes`, optional): The axes to be used for plotting. If None, a new figure and axes will be created. Defaults to None.
            style (str, optional): The style to be applied to the plot. Defaults to "default". Set to "none" to ignore style guidelines.
            extent (str or None, optional): The extent to be applied to the plot. Defaults to None.
            rescale_unit (str or None, optional): The unit to which the data or CRS should be rescaled. If None, the rescale_unit specified in the guidelines will be used. Defaults to None.
            skip (int, optional): The factor by which to skip data points. Defaults to 1.
            smooth (int, optional): The factor by which to smooth data points. Defaults to 1.
            xlim (tuple[float, float], optional): The limits for the x-axis. Defaults to None.
            ylim (tuple[float, float], optional): The limits for the y-axis. Defaults to None.
            subplot_kwargs (dict, optional): Additional keyword arguments to be passed to :func:`matplotlib.pyplot.subplots` when creating a new figure and axes. Defaults to None.
            append_axes_kwargs (dict, optional): Additional keyword arguments to be passed to :func:`matplotlib.axes_grid1.axes_divider.make_axes_locatable.append_axes` for creating the colorbar axis. If None, no colorbar axis will be created. Defaults to None.
            xlabel_kwargs (dict, optional): Additional keyword arguments passed to :func:`matplotlib.axes.Axes.set_xlabel` for formatting the x-axis label. If None, the x-axis label will be determined automatically. Defaults to None.
            ylabel_kwargs (dict, optional): Additional keyword arguments passed to :func:`matplotlib.axes.Axes.set_ylabel` for formatting the y-axis label. If None, the y-axis label will be determined automatically. Defaults to None.
            title_kwargs (dict, optional): Additional keyword arguments passed to :func:`matplotlib.axes.Axes.set_title` for formatting the title. If None, the title will be set to an empty string. Defaults to None.
            aspect_kwargs (dict, optional): Additional keyword arguments passed to :func:`matplotlib.axes.Axes.set_aspect` for formatting the aspect ratio. If None, the aspect ratio will be set to 'equal'. Defaults to None.
            grid_kwargs (dict, optional): Additional keyword arguments passed to :func:`matplotlib.axes.Axes.grid` for formatting the grid. If None, the grid will be set to visible. Defaults to None.
            show_kwargs (bool, optional): Whether to print the keyword arguments passed to plotting function. Defaults to False.
            **kwargs (dict, optional): Additional keyword arguments to pass to :func:`xarray.plot.streamplot`.

        Returns:
            :class:`matplotlib.axes.Axes`: The axes object of the plot.
        """
        if isinstance(ds, xr.Dataset):
            # Set default color to black if neither hue nor color is specified
            if "hue" not in kwargs and "color" not in kwargs:
                kwargs.setdefault("color", "black")

            # Streamplot requires that x and y must be the first two dimensions and y must be strictly increasing
            ds = ds.sortby("y")

            return ds.plot.streamplot(ax=ax, **kwargs)
        else:
            raise TypeError("Data must be an xarray.Dataset")

    @wrappers.format_args_wrapper
    @wrappers.guideline_wrapper
    @wrappers.initialise_fig_wrapper
    @wrappers.rescale_wrapper
    @wrappers.skip_and_smooth_wrapper
    @wrappers.cbar_axis_wrapper
    @wrappers.show_kwargs_wrapper
    @wrappers.format_axis_wrapper
    def grid(
        self,
        da: xr.DataArray | xu.UgridDataArray,
        ax: plt.Axes | None = None,
        style: str | None = None,
        extent: str | None = None,
        rescale_unit: str | None = None,
        skip: int = 1,
        smooth: int = 1,
        xlim: tuple[float, float] | None = None,
        ylim: tuple[float, float] | None = None,
        subplot_kwargs: dict | None = None,
        append_axes_kwargs: dict | None = None,
        xlabel_kwargs: dict | None = None,
        ylabel_kwargs: dict | None = None,
        title_kwargs: dict | None = None,
        aspect_kwargs: dict | None = None,
        grid_kwargs: dict | None = None,
        show_kwargs: bool = False,
        **kwargs,
    ) -> plt.Axes:
        """Grid plot of a xarray.DataArray or xugrid.UgridDataArray.

        Args:
            da (:class:`xarray.DataArray` or :class:`xugrid.UgridDataArray`): DataArray to plot.
            ax (:class:`matplotlib.axes.Axes`, optional): The axes to be used for plotting. If None, a new figure and axes will be created. Defaults to None.
            style (str, optional): The style to be applied to the plot. Defaults to "default". Set to "none" to ignore style guidelines.
            extent (str or None, optional): The extent to be applied to the plot. Defaults to None.
            rescale_unit (str or None, optional): The unit to which the data or CRS should be rescaled. If None, the rescale_unit specified in the guidelines will be used. Defaults to None.
            skip (int, optional): The factor by which to skip data points. Defaults to 1.
            smooth (int, optional): The factor by which to smooth data points. Defaults to 1.
            xlim (tuple[float, float], optional): The limits for the x-axis. Defaults to None.
            ylim (tuple[float, float], optional): The limits for the y-axis. Defaults to None.
            subplot_kwargs (dict, optional): Additional keyword arguments to be passed to :func:`matplotlib.pyplot.subplots` when creating a new figure and axes. Defaults to None.
            append_axes_kwargs (dict, optional): Additional keyword arguments to be passed to :func:`matplotlib.axes_grid1.axes_divider.make_axes_locatable.append_axes` for creating the colorbar axis. If None, no colorbar axis will be created. Defaults to None.
            xlabel_kwargs (dict, optional): Additional keyword arguments passed to :func:`matplotlib.axes.Axes.set_xlabel` for formatting the x-axis label. If None, the x-axis label will be determined automatically. Defaults to None.
            ylabel_kwargs (dict, optional): Additional keyword arguments passed to :func:`matplotlib.axes.Axes.set_ylabel` for formatting the y-axis label. If None, the y-axis label will be determined automatically. Defaults to None.
            title_kwargs (dict, optional): Additional keyword arguments passed to :func:`matplotlib.axes.Axes.set_title` for formatting the title. If None, the title will be set to an empty string. Defaults to None.
            aspect_kwargs (dict, optional): Additional keyword arguments passed to :func:`matplotlib.axes.Axes.set_aspect` for formatting the aspect ratio. If None, the aspect ratio will be set to 'equal'. Defaults to None.
            grid_kwargs (dict, optional): Additional keyword arguments passed to :func:`matplotlib.axes.Axes.grid` for formatting the grid. If None, the grid will be set to visible. Defaults to None.
            show_kwargs (bool, optional): Whether to print the keyword arguments passed to plotting function. Defaults to False.
            **kwargs (dict, optional): Additional keyword arguments to pass to :func:`xarray.plot.line` or :func:`xugrid.plot.line`.

        Returns:
            :class:`matplotlib.axes.Axes`: The axes object of the plot.
        """
        if isinstance(da, xr.DataArray):
            # Convert to UgridDataArray for plotting
            uda = xu.UgridDataArray.from_structured2d(da)
            uda.ugrid.set_crs(da.rio.crs)

            # Set default color to black if color is not specified
            if "color" not in kwargs:
                kwargs.setdefault("color", "black")

            return uda.ugrid.plot.line(ax=ax, **kwargs)
        elif isinstance(da, xu.UgridDataArray):
            if da.dims[0] != "mesh2d_nEdges" and "color" not in kwargs:
                kwargs.setdefault("color", "black")
            return da.ugrid.plot.line(ax=ax, **kwargs)

        else:
            raise TypeError("Data must be an xarray.DataArray or xugrid.UgridDataArray")

    @wrappers.format_args_wrapper
    @wrappers.guideline_wrapper
    @wrappers.initialise_fig_wrapper
    @wrappers.rescale_wrapper
    @wrappers.cbar_axis_wrapper
    @wrappers.show_kwargs_wrapper
    @wrappers.format_axis_wrapper
    def geometries(
        self,
        gdf: gpd.GeoDataFrame,
        ax: plt.Axes | None = None,
        style: str | None = None,
        extent: str | None = None,
        rescale_unit: str | None = None,
        xlim: tuple[float, float] | None = None,
        ylim: tuple[float, float] | None = None,
        subplot_kwargs: dict | None = None,
        append_axes_kwargs: dict | None = None,
        xlabel_kwargs: dict | None = None,
        ylabel_kwargs: dict | None = None,
        title_kwargs: dict | None = None,
        aspect_kwargs: dict | None = None,
        grid_kwargs: dict | None = None,
        show_kwargs: bool = False,
        **kwargs,
    ) -> plt.Axes:
        """Plot a geopandas.GeoDataFrame.

        Args:
            gdf (:class:`geopandas.GeoDataFrame`): GeoDataFrame to plot.
            ax (:class:`matplotlib.axes.Axes`, optional): The axes to be used for plotting. If None, a new figure and axes will be created. Defaults to None.
            style (str, optional): The style to be applied to the plot. Defaults to "default". Set to "none" to ignore style guidelines.
            extent (str or None, optional): The extent to be applied to the plot. Defaults to None.
            rescale_unit (str or None, optional): The unit to which the data or CRS should be rescaled. If None, the rescale_unit specified in the guidelines will be used. Defaults to None.
            xlim (tuple[float, float], optional): The limits for the x-axis. Defaults to None.
            ylim (tuple[float, float], optional): The limits for the y-axis. Defaults to None.
            subplot_kwargs (dict, optional): Additional keyword arguments to be passed to :func:`matplotlib.pyplot.subplots` when creating a new figure and axes. Defaults to None.
            append_axes_kwargs (dict, optional): Additional keyword arguments to be passed to :func:`matplotlib.axes_grid1.axes_divider.make_axes_locatable.append_axes` for creating the colorbar axis. If None, no colorbar axis will be created. Defaults to None.
            xlabel_kwargs (dict, optional): Additional keyword arguments passed to :func:`matplotlib.axes.Axes.set_xlabel` for formatting the x-axis label. If None, the x-axis label will be determined automatically. Defaults to None.
            ylabel_kwargs (dict, optional): Additional keyword arguments passed to :func:`matplotlib.axes.Axes.set_ylabel` for formatting the y-axis label. If None, the y-axis label will be determined automatically. Defaults to None.
            title_kwargs (dict, optional): Additional keyword arguments passed to :func:`matplotlib.axes.Axes.set_title` for formatting the title. If None, the title will be set to an empty string. Defaults to None.
            aspect_kwargs (dict, optional): Additional keyword arguments passed to :func:`matplotlib.axes.Axes.set_aspect` for formatting the aspect ratio. If None, the aspect ratio will be set to 'equal'. Defaults to None.
            grid_kwargs (dict, optional): Additional keyword arguments passed to :func:`matplotlib.axes.Axes.grid` for formatting the grid. If None, the grid will be set to visible. Defaults to None.
            show_kwargs (bool, optional): Whether to print the keyword arguments passed to plotting function. Defaults to False.
            **kwargs (dict, optional): Additional keyword arguments to pass to :func:`plot_gdf <resplotlib.geometries.plot_gdf>`.

        Returns:
            :class:`matplotlib.axes.Axes`: The axes object of the plot.
        """
        return geometries.plot_gdf(gdf, ax=ax, **kwargs)

    @wrappers.format_args_wrapper
    @wrappers.guideline_wrapper
    @wrappers.initialise_fig_wrapper
    @wrappers.rescale_wrapper
    @wrappers.show_kwargs_wrapper
    @wrappers.format_axis_wrapper
    def basemap(
        self,
        crs: pyprojCRS | rasterioCRS | str | None,
        ax: plt.Axes | None = None,
        style: str | None = None,
        extent: str | None = None,
        rescale_unit: str | None = None,
        xlim: tuple[float, float] | None = None,
        ylim: tuple[float, float] | None = None,
        subplot_kwargs: dict | None = None,
        xlabel_kwargs: dict | None = None,
        ylabel_kwargs: dict | None = None,
        title_kwargs: dict | None = None,
        aspect_kwargs: dict | None = None,
        grid_kwargs: dict | None = None,
        show_kwargs: bool = False,
        **kwargs,
    ) -> plt.Axes:
        """Plot a basemap.

        Args:
            crs (:class:`pyproj.CRS`, :class:`rasterio.crs.CRS`, str or None): Coordinate reference system of the basemap.
            ax (:class:`matplotlib.axes.Axes`, optional): The axes to be used for plotting. If None, a new figure and axes will be created. Defaults to None.
            style (str, optional): The style to be applied to the plot. Defaults to "default". Set to "none" to ignore style guidelines.
            extent (str or None, optional): The extent to be applied to the plot. Defaults to None.
            rescale_unit (str or None, optional): The unit to which the data or CRS should be rescaled. If None, the rescale_unit specified in the guidelines will be used. Defaults to None.
            xlim (tuple[float, float], optional): The limits for the x-axis. Defaults to None.
            ylim (tuple[float, float], optional): The limits for the y-axis. Defaults to None.
            subplot_kwargs (dict, optional): Additional keyword arguments to be passed to :func:`matplotlib.pyplot.subplots` when creating a new figure and axes. Defaults to None.
            xlabel_kwargs (dict, optional): Additional keyword arguments passed to :func:`matplotlib.axes.Axes.set_xlabel` for formatting the x-axis label. If None, the x-axis label will be determined automatically. Defaults to None.
            ylabel_kwargs (dict, optional): Additional keyword arguments passed to :func:`matplotlib.axes.Axes.set_ylabel` for formatting the y-axis label. If None, the y-axis label will be determined automatically. Defaults to None.
            title_kwargs (dict, optional): Additional keyword arguments passed to :func:`matplotlib.axes.Axes.set_title` for formatting the title. If None, the title will be set to an empty string. Defaults to None.
            aspect_kwargs (dict, optional): Additional keyword arguments passed to :func:`matplotlib.axes.Axes.set_aspect` for formatting the aspect ratio. If None, the aspect ratio will be set to 'equal'. Defaults to None.
            grid_kwargs (dict, optional): Additional keyword arguments passed to :func:`matplotlib.axes.Axes.grid` for formatting the grid. If None, the grid will be set to visible. Defaults to None.
            show_kwargs (bool, optional): Whether to print the keyword arguments passed to plotting function. Defaults to False.
            **kwargs (dict, optional): Additional keyword arguments to pass to :func:`plot_basemap <resplotlib.basemaps.plot_basemap>`.

        Returns:
            :class:`matplotlib.axes.Axes`: The axes object of the plot.
        """

        return basemaps.plot_basemap(crs, ax=ax, rescale_unit=rescale_unit, xlim=xlim, ylim=ylim, **kwargs)

    # Interactive plot methods
    @wrappers.format_args_wrapper
    @wrappers.guideline_wrapper
    @wrappers.initialise_map_wrapper
    @wrappers.show_kwargs_wrapper
    def explore_data(
        self,
        da: xr.DataArray | xu.UgridDataArray,
        m: ipyleaflet.Map | map.Map | None = None,
        show_kwargs: bool = False,
        **kwargs,
    ) -> ipyleaflet.Map:
        """Explore a xarray.DataArray or xugrid.UgridDataArray in an interactive map.

        Args:
            da (:class:`xarray.DataArray` or :class:`xugrid.UgridDataArray`): DataArray to explore.
            m (:class:`ipyleaflet.Map`, :class:`resplotlib.map.Map`, optional): The map to be used for plotting. If None, a new map will be created. Defaults to None.
            style (str, optional): The style to be applied to the plot. Defaults to "default". Set to "none" to ignore style guidelines.
            show_kwargs (bool, optional): Whether to print the keyword arguments passed to plotting function. Defaults to False.
            **kwargs (dict, optional): Additional keyword arguments to pass to :func:`explore_da <resplotlib.interactive.explore_da>` or :func:`explore_uda <resplotlib.interactive.explore_uda>`.

        Returns:
            :class:`ipyleaflet.Map`: The interactive map object.
        """

        if isinstance(da, xr.DataArray):
            return explore.explore_da(da, m=m, **kwargs)
        elif isinstance(da, xu.UgridDataArray):
            return explore.explore_uda(da, m=m, **kwargs)

    @wrappers.format_args_wrapper
    @wrappers.guideline_wrapper
    @wrappers.initialise_map_wrapper
    @wrappers.show_kwargs_wrapper
    def explore_geometries(
        self, gdf: gpd.GeoDataFrame, m: ipyleaflet.Map | map.Map | None = None, show_kwargs: bool = False, **kwargs
    ) -> ipyleaflet.Map:
        """Explore a geopandas.GeoDataFrame in an interactive map.

        Args:
            gdf (:class:`geopandas:geopandas.GeoDataFrame`): GeoDataFrame to explore.
            m (:class:`ipyleaflet.Map`, :class:`resplotlib.map.Map`, optional): The map to be used for plotting. If None, a new map will be created. Defaults to None.
            show_kwargs (bool, optional): Whether to print the keyword arguments passed to plotting function. Defaults to False.
            **kwargs (dict, optional): Additional keyword arguments to pass to :func:`explore_gdf <resplotlib.interactive.explore_gdf>`.

        Returns:
            :class:`ipyleaflet.Map`: The interactive map object.
        """
        return explore.explore_gdf(gdf, m=m, **kwargs)

    @wrappers.format_args_wrapper
    @wrappers.guideline_wrapper
    @wrappers.initialise_map_wrapper
    @wrappers.show_kwargs_wrapper
    def explore_basemap(self, m: ipyleaflet.Map | map.Map | None = None, show_kwargs: bool = False, **kwargs) -> ipyleaflet.Map:
        """Explore a basemap in an interactive map.

        Args:
            m (:class:`ipyleaflet.Map`, :class:`resplotlib.map.Map`, optional): The map to be used for plotting. If None, a new map will be created. Defaults to None.
            show_kwargs (bool, optional): Whether to print the keyword arguments passed to plotting function. Defaults to False.
            **kwargs (dict, optional): Additional keyword arguments to pass to :func:`explore_basemap <resplotlib.interactive.explore_basemap>`.

        Returns:
            :class:`ipyleaflet.Map`: The interactive map object.
        """
        return explore.explore_basemap(m=m, **kwargs)

    # Support methods
    def subplots(self, *args, **kwargs) -> tuple[plt.Figure, plt.Axes]:
        """Create a figure and a set of subplots.

        Args:
            *args (args, optional): Positional arguments to pass to :func:`matplotlib.pyplot.subplots`.
            **kwargs (dict, optional): Additional keyword arguments to pass to :func:`matplotlib.pyplot.subplots`.

        Returns:
            tuple[:class:`matplotlib.figure.Figure`, :class:`matplotlib.axes.Axes`]: A tuple containing the figure and axes object(s).
        """
        return plt.subplots(*args, **kwargs)

    def Map(self, gdf: gpd.GeoDataFrame | None = None, clear_on_draw: bool = False, geoman_draw: bool = True, **kwargs) -> map.Map:
        """Create an interactive map.

        Args:
            gdf (gpd.GeoDataFrame, optional): GeoDataFrame containing the geometries to add to the map. Defaults to None.
            clear_on_draw (bool, optional): Clear drawn geometries from the map when a new geometry is drawn. Defaults to False.
            geoman_draw (bool, optional): Use GeomanDrawControl to draw geometries on the map. Defaults to True.
            **kwargs (dict, optional): Additional keyword arguments to pass to :func:`Map <resplotlib.map.Map>`.

        Returns:
            :class:`resplotlib.map.Map`: The interactive map object.
        """
        return map.Map(gdf=gdf, clear_on_draw=clear_on_draw, geoman_draw=geoman_draw, **kwargs)

    def show(self, fig: plt.Figure | ipyleaflet.Map, **kwargs) -> None:
        """Display a figure or interactive map.

        Args:
            fig (:class:`matplotlib.figure.Figure` or :class:`ipyleaflet.Map`): The figure or map to display.
            **kwargs (dict, optional): Additional keyword arguments. For matplotlib figures, ``tight_layout`` (bool, default True)
                controls whether to apply tight layout before showing.

        Returns:
            None
        """
        if isinstance(fig, plt.Figure):
            if kwargs.pop("tight_layout", True):
                fig.tight_layout()
            plt.show(**kwargs)
        elif isinstance(fig, ipyleaflet.Map):
            display(fig)
            time.sleep(1)  # Sleep for a second to ensure the map is displayed before any subsequent code is executed
        else:
            raise TypeError("Figure must be a matplotlib.figure.Figure or ipyleaflet.Map")

    def save(self, fig, file_path, **kwargs) -> None:
        """Save a figure or interactive map to a file.

        Args:
            fig (:class:`matplotlib.figure.Figure` or :class:`ipyleaflet.Map`): The figure or map to save.
            file_path (str): The file path where the figure or map should be saved.
            **kwargs (dict, optional): Additional keyword arguments. For matplotlib figures, ``tight_layout`` (bool, default True)
                controls whether to apply tight layout before saving.

        Returns:
            None
        """
        if isinstance(fig, plt.Figure):
            if kwargs.pop("tight_layout", True):
                fig.tight_layout()
            kwargs.setdefault("dpi", 300)
            kwargs.setdefault("bbox_inches", "tight")
            fig.savefig(file_path, **kwargs)
        elif isinstance(fig, ipyleaflet.Map):
            width = fig.layout.width
            height = fig.layout.height
            fig.layout.width = "100%"
            fig.layout.height = "calc(100vh - 16px)"
            fig.save(file_path, **kwargs)
            fig.layout.width = width
            fig.layout.height = height
        else:
            raise TypeError("Figure must be a matplotlib.figure.Figure or ipyleaflet.Map")

    def close(self, fig, **kwargs) -> None:
        """Close a figure or interactive map.

        Args:
            fig (:class:`matplotlib.figure.Figure` or :class:`ipyleaflet.Map`): The figure or map to close.
            **kwargs (dict, optional): Additional keyword arguments to pass to :func:`matplotlib.pyplot.close`.

        Returns:
            None
        """
        if isinstance(fig, plt.Figure):
            plt.close(fig, **kwargs)
        elif isinstance(fig, ipyleaflet.Map):
            pass
        else:
            raise TypeError("Figure must be a matplotlib.figure.Figure or ipyleaflet.Map")
        del fig
        gc.collect()

    def video(self, file_path_images: list[str], file_path_video: str, fps: int = 5, **kwargs) -> None:
        """Create a video from a list of images.

        Args:
            file_path_images (list[str]): List of file paths to images.
            file_path_video (str): File path for the output video.
            fps (int, optional): Frames per second for the video. Defaults to 5.
            **kwargs: Additional keyword arguments to pass to :func:`cv2.VideoWriter`.
        """
        videos.create_video(file_path_images, file_path_video, fps, **kwargs)

    def gif(self, file_path_images: list[str], file_path_gif: str, fps: int = 5, **kwargs) -> None:
        """Create a GIF from a list of images.

        Args:
            file_path_images (list[str]): List of file paths to images.
            file_path_gif (str): File path for the output GIF.
            fps (int, optional): Frames per second for the GIF. Defaults to 5.
            **kwargs: Additional keyword arguments to pass to :func:`imageio.mimsave`.
        """
        videos.create_gif(file_path_images, file_path_gif, fps, **kwargs)


# Instance of resilient plotter class
rpc = Resplotclass()
