import geopandas as gpd
import ipyleaflet
import matplotlib as mpl
import numpy as np
import odc.geo.xr  # noqa: F401
import xarray as xr
import xugrid as xu
from branca import colormap as cm

from . import map


def _mpl_cmap_to_branca_cmap(cmap_name: str, n: int = 256) -> cm.LinearColormap:
    """Convert a matplotlib colormap to a branca LinearColormap.

    Args:
        cmap_name (str): Name of the matplotlib colormap to convert.
        n (int, optional): Number of colors to sample from the matplotlib colormap. Default is 256.

    Returns:
        :class:`branca.colormap.LinearColormap`: Converted Branca LinearColormap.
    """
    mpl_cmap = mpl.colormaps.get_cmap(cmap_name)

    colors = [mpl.colors.to_hex(mpl_cmap(i / (n - 1))) for i in range(n)]

    return cm.LinearColormap(colors)


def explore_geojson(gdf: gpd.GeoDataFrame, m: ipyleaflet.Map | map.Map, **kwargs) -> ipyleaflet.Map | map.Map:
    """Explore a geopandas.GeoDataFrame as a GeoJSON layer on a map.

    Args:
        gdf (:class:`geopandas.GeoDataFrame`): GeoDataFrame to explore.
        m (:class:`ipyleaflet.Map` or :class:`Map <resplotlib.map.Map>`): Map on which to display the GeoJSON layer.
        **kwargs: Additional keyword arguments to pass to :class:`ipyleaflet.GeoJSON`.

    Returns:
        :class:`ipyleaflet.Map` or :class:`Map <resplotlib.map.Map>`: Map with the GeoDataFrame added as a GeoJSON layer.
    """
    # Set point style to ensure circle markers are used instead of default markers
    kwargs.setdefault("point_style", {"radius": 10})

    # Get geojson data
    geo_data = gdf.__geo_interface__

    # Create GeoJSON layer
    layer = ipyleaflet.GeoJSON(data=geo_data, **kwargs)

    # Add layer to map
    m.add(layer)

    return m


def explore_choropleth(gdf: gpd.GeoDataFrame, m: ipyleaflet.Map | map.Map, column: str, **kwargs) -> ipyleaflet.Map | map.Map:
    """Explore a geopandas.GeoDataFrame as a choropleth layer on a map.

    Args:
        gdf (:class:`geopandas.GeoDataFrame`): GeoDataFrame to explore.
        m (:class:`ipyleaflet.Map` or :class:`Map <resplotlib.map.Map>`): Map on which to display the choropleth layer.
        column (str): Column to use for the choropleth values.
        **kwargs: Additional keyword arguments to pass to :class:`ipyleaflet.Choropleth`.

    Returns:
        :class:`ipyleaflet.Map` or :class:`Map <resplotlib.map.Map>`: Map with the GeoDataFrame added as a choropleth layer.
    """
    # Set point style to ensure circle markers are used
    kwargs.setdefault("point_style", {"radius": 10})

    # Set style callback to ensure lines are styled with the choropleth colors
    def style_callback(feature, colormap, value):
        return {"color": colormap(value)}

    kwargs.setdefault("style_callback", style_callback)

    # Check if column is numeric
    if gdf[column].dtype.kind not in "biufc":
        raise ValueError(f"Column '{column}' must be numeric for choropleth.")

    # Get data for choropleth
    geo_data = gdf.__geo_interface__
    choro_data = gdf[column].to_dict()
    colormap = _mpl_cmap_to_branca_cmap(kwargs.pop("cmap", "Spectral_r"))
    value_min = kwargs.pop("vmin", None)
    value_max = kwargs.pop("vmax", None)

    # Create choropleth layer
    layer = ipyleaflet.Choropleth(
        geo_data=geo_data,
        choro_data=choro_data,
        colormap=colormap,
        value_min=value_min,
        value_max=value_max,
        **kwargs,
    )

    # Add layer to map
    m.add(layer)

    if "legend" in kwargs and kwargs["legend"]:
        # Create colormap control
        colormap_control = _ColormapControl(
            caption=kwargs.get("label", column),
            colormap=colormap,
            value_min=layer.value_min,
            value_max=layer.value_max,
            position="bottomright",
        )

        # Add colormap control to map
        m.add_control(colormap_control)

    return m


def explore_gdf(gdf: gpd.GeoDataFrame, m: ipyleaflet.Map | map.Map | None = None, column: str | None = None, **kwargs) -> ipyleaflet.Map | map.Map:
    """Explore a geopandas.GeoDataFrame in an interactive map.

    Args:
        gdf (:class:`geopandas.GeoDataFrame`): GeoDataFrame to explore.
        m (:class:`ipyleaflet.Map` or :class:`Map <resplotlib.map.Map>`, optional): Map on which to display the GeoDataFrame. If None, a new map will be created. Default is None.
        column (str, optional): Column to use for choropleth values. If None, geometries will be plotted without a choropleth. Default is None.
        **kwargs: Additional keyword arguments to pass to the underlying functions for plotting the GeoDataFrame.

    Returns:
        :class:`ipyleaflet.Map` or :class:`Map <resplotlib.map.Map>`: Map with the GeoDataFrame added as a layer.
    """
    # Create map if not provided
    if m is None:
        m = ipyleaflet.Map()

    # Reproject geodataframe to EPSG:4326
    gdf = gdf.to_crs("EPSG:4326")

    # Reset index and convert to string to ensure compatibility with ipyleaflet
    gdf = gdf.reset_index(drop=True)
    gdf.index = gdf.index.astype(str)

    # Set style and hover style kwargs
    if "style_kwargs" in kwargs:
        kwargs["style"] = kwargs.pop("style_kwargs")
    if "hover_style_kwargs" in kwargs:
        kwargs["hover_style"] = kwargs.pop("hover_style_kwargs")

    # Plot geometries
    if column is None:
        m = explore_geojson(gdf, m, **kwargs)
    else:
        m = explore_choropleth(gdf, m, column, **kwargs)

    return m


def explore_da(da: xr.DataArray, m: ipyleaflet.Map | map.Map | None = None, **kwargs) -> ipyleaflet.Map | map.Map:
    """Explore a xarray.DataArray in an interactive map.

    Args:
        da (:class:`xarray.DataArray`): DataArray to explore.
        m (:class:`ipyleaflet.Map` or :class:`Map <resplotlib.map.Map>`, optional): Map on which to display the DataArray. If None, a new map will be created. Default is None.
        **kwargs: Additional keyword arguments to pass to :class:`odc.geo.xr.DataArrayAccessor.add_to`.

    Returns:
        :class:`ipyleaflet.Map` or :class:`Map <resplotlib.map.Map>`: Map with the DataArray added as a layer.
    """
    # Create map if not provided
    if m is None:
        m = ipyleaflet.Map()

    # Set vmin and vmax to 2nd and 98th percentiles if "robust" is True in kwargs
    robust = kwargs.get("robust", False)
    kwargs["vmin"] = da.quantile(0.02).item() if robust else da.min().item()
    kwargs["vmax"] = da.quantile(0.98).item() if robust else da.max().item()

    # Convert to RGBA if data array has 3 dimensions and "band" is one of the dimensions with size 3
    rgb = len(da.dims) == 3 and "band" in da.dims and da.sizes["band"] == 3
    if rgb:
        ds = da.to_dataset(dim="band")
        da = ds.odc.to_rgba(bands=da.band.values.tolist(), vmin=kwargs["vmin"], vmax=kwargs["vmax"])

    # Change module of Map class to ipyleaflet to trick odc.add_to
    if isinstance(m, map.Map):
        m.__class__.__module__ = "ipyleaflet"

    # Add data array to map
    da.odc.add_to(m, **kwargs)

    if "legend" in kwargs and kwargs["legend"] and not rgb:
        # Get data for colormap control
        caption = f"{da.attrs.get('long_name', '')} [{da.attrs.get('units', '')}]"
        colormap = _mpl_cmap_to_branca_cmap(kwargs.get("cmap", "Spectral_r"))
        value_min = np.round(kwargs["vmin"], 1)
        value_max = np.round(kwargs["vmax"], 1)

        # Create colormap control
        colormap_control = _ColormapControl(
            caption=caption,
            colormap=colormap,
            value_min=value_min,
            value_max=value_max,
            position="bottomright",
        )

        # Add colormap control to map
        m.add_control(colormap_control)

    # Change module of Map class back to resplotlib
    if isinstance(m, map.Map):
        m.__class__.__module__ = "src.resplotlib.interactive"

    return m


def explore_uda(uda: xu.UgridDataArray, m: ipyleaflet.Map | map.Map = None, **kwargs) -> ipyleaflet.Map | map.Map:
    """Explore a xugrid.UgridDataArray with an unstructured grid in an interactive map.

    Args:
        uda (:class:`xugrid.UgridDataArray`): DataArray with an unstructured grid to explore.
        m (:class:`ipyleaflet.Map` or :class:`Map <resplotlib.map.Map>`, optional): Map on which to display the DataArray. If None, a new map will be created. Default is None.
        **kwargs: Additional keyword arguments to pass to :class:`resplotlib.explore.explore_gdf`.

    Returns:
        :class:`ipyleaflet.Map` or :class:`Map <resplotlib.map.Map>`: Map with the DataArray added as a layer.
    """
    # Set default style kwargs to ensure filled geometries with no borders
    kwargs.setdefault("style_kwargs", {"weight": 0, "fillOpacity": 1})

    # Create map if not provided
    if m is None:
        m = ipyleaflet.Map()

    # Convert uda to geodataframe
    gdf = uda.ugrid.to_geodataframe()

    # Plot geodataframe
    m = explore_gdf(gdf, m, column=uda.name, **kwargs)

    return m


def explore_basemap(m: ipyleaflet.Map | map.Map, source: str = "OpenStreetMap.Mapnik", **kwargs) -> ipyleaflet.Map | map.Map:
    """Add a basemap layer to an interactive map.

    Args:
        m (:class:`ipyleaflet.Map` or :class:`Map <resplotlib.map.Map>`): Map to which to add the basemap layer.
        source (str, optional): Basemap source in the format "provider.layer_name". Default is "OpenStreetMap.Mapnik".
        **kwargs: Additional keyword arguments to pass to :class:`ipyleaflet.basemap_to_tiles`.

    Returns:
        :class:`ipyleaflet.Map` or :class:`Map <resplotlib.map.Map>`: Map with the basemap layer added.
    """
    # Split source string into provider and layer
    provider, layer_name = source.split(".")

    # Get basemap layer
    layer = ipyleaflet.basemaps[provider][layer_name]
    layer = ipyleaflet.basemap_to_tiles(layer, **kwargs)
    layer.base = True  # Set layer as basemap to ensure it is plotted below other layers

    # Add basemap layer to map
    m.add(layer)

    return m


# Class below was adapted from ipyleaflet's ColorMapControl to allow for a slightly nicer display of the colormap
# (https://ipyleaflet.readthedocs.io/en/latest/_modules/ipyleaflet/leaflet.html#ColormapControl)
from branca.colormap import ColorMap, linear  # noqa: E402
from ipyleaflet import WidgetControl  # noqa: E402
from IPython.display import display  # noqa: E402
from ipywidgets import Output  # noqa: E402
from traitlets import CFloat, Instance, Unicode, default  # noqa: E402


class _ColormapControl(WidgetControl):
    """ColormapControl class, with WidgetControl as parent class.

    A control which contains a colormap.

    Attributes
    ----------
    caption : str, default 'caption'
        The caption of the colormap.
    colormap: branca.colormap.ColorMap instance, default linear.OrRd_06
        The colormap used for the effect.
    value_min : float, default 0.0
        The minimal value taken by the data to be represented by the colormap.
    value_max : float, default 1.0
        The maximal value taken by the data to be represented by the colormap.
    """

    caption = Unicode("caption")
    colormap = Instance(ColorMap, default_value=linear.OrRd_06)
    value_min = CFloat(0.0)
    value_max = CFloat(1.0)

    @default("widget")
    def _default_widget(self):
        widget = Output(layout={"height": "40px", "width": "450px", "margin": "0px 10px"})  # Adjusted height and width to better fit colormap
        with widget:
            colormap = self.colormap.scale(self.value_min, self.value_max)
            colormap.caption = self.caption
            display(colormap)

        return widget
