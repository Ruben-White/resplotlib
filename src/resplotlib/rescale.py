import geopandas as gpd
import numpy as np
import xarray as xr
import xugrid as xu
from matplotlib.ticker import AutoLocator, Locator, ScalarFormatter
from pyproj import CRS as pyprojCRS
from rasterio.crs import CRS as rasterioCRS

DATA_OR_CRS_TYPE = xr.DataArray | xr.Dataset | xu.UgridDataArray | xu.UgridDataset | gpd.GeoDataFrame | pyprojCRS | rasterioCRS | str | None


class RescaleLocator(Locator):
    """Generate ticks as if the axis coordinates were scaled."""

    def __init__(self, scale=1.0, origin=0.0):
        super().__init__()
        self.scale = scale
        self.origin = origin
        self._locator = AutoLocator()

    def tick_values(self, vmin, vmax):
        scaled_vmin = (vmin - self.origin) * self.scale
        scaled_vmax = (vmax - self.origin) * self.scale

        scaled_ticks = self._locator.tick_values(
            scaled_vmin,
            scaled_vmax,
        )

        return scaled_ticks / self.scale + self.origin

    def __call__(self):
        vmin, vmax = self.axis.get_view_interval()
        return self.tick_values(vmin, vmax)


class RecaledFormatter(ScalarFormatter):
    """Format ticks as if the axis coordinates were scaled."""

    def __init__(self, scale=1.0, origin=0.0, **kwargs):
        self.scale = scale
        self.origin = origin
        super().__init__(**kwargs)

    def set_locs(self, locs):
        self._scaled_locs = [(x - self.origin) * self.scale for x in locs]

        super().set_locs(self._scaled_locs)

    def __call__(self, x, pos=None):
        scaled_x = (x - self.origin) * self.scale
        return super().__call__(scaled_x, pos)


def rescale_axis(ax, scale_factor=1.0, origin=(0.0, 0.0)):
    ax.xaxis.set_major_locator(RescaleLocator(scale=scale_factor, origin=origin[0]))
    ax.xaxis.set_major_formatter(RecaledFormatter(scale=scale_factor, origin=origin[0]))
    ax.yaxis.set_major_locator(RescaleLocator(scale=scale_factor, origin=origin[1]))
    ax.yaxis.set_major_formatter(RecaledFormatter(scale=scale_factor, origin=origin[1]))
    return ax


def get_rescale_parameters(
    data_or_crs: DATA_OR_CRS_TYPE = None,
    rescale_unit: str | None = None,
) -> tuple[str | None, float]:
    """Get rescale parameters from a coordinate reference system (CRS).

    Args:
        data_or_crs (:class:`xarray.DataArray` | :class:`xarray.Dataset` | :class:`xugrid.UgridDataArray` | :class:`xugrid.UgridDataset` | :class:`geopandas.GeoDataFrame` | :class:`pyproj.CRS` | :class:`rasterio.crs.CRS` | str | None): Data or coordinate reference system.
        rescale_unit (str, optional): Desired rescale unit. Set to "none" to disable rescaling. Defaults to None.

    Returns:
        tuple[str | None, float]: Rescale unit and scale factor.
    """
    # Define abbreviations and scale factors
    UNIT_ABBREVIATIONS = {
        "millimetre": "mm",
        "centimetre": "cm",
        "decimetre": "dm",
        "metre": "m",
        "decametre": "dam",
        "hectometre": "hm",
        "kilometre": "km",
        "feet": "ft",
        "inch": "in",
        "yard": "yd",
        "mile": "mi",
        "nautical_mile": "nmi",
        "degree": "deg",
        "arcsecond": "arcsec",
        "arcminute": "arcmin",
        "radian": "rad",
        "pixel": "px",
        "unknown": "-",
    }
    SCALE_METRES = {
        "mm": 1000,
        "cm": 100,
        "dm": 10,
        "m": 1,
        "dam": 0.1,
        "hm": 0.01,
        "km": 0.001,
        "ft": 3.28084,
        "in": 39.3701,
        "yd": 1.09361,
        "mi": 0.000621371,
        "nmi": 0.000539957,
    }
    SCALE_DEGREES = {"deg": 1, "arcsec": 3600, "arcmin": 60, "rad": np.pi / 180}
    SCALE_OTHERS = ["px", "-"]

    # Get crs from data or CRS
    crs = get_crs_from_data_or_crs(data_or_crs)

    # Get crs unit
    crs_unit = crs.axis_info[0].unit_name.lower() if crs is not None else None
    if crs_unit in UNIT_ABBREVIATIONS:
        crs_unit = UNIT_ABBREVIATIONS[crs_unit]

    # Get rescale unit
    if crs_unit is None or crs_unit in SCALE_OTHERS:
        rescale_unit = None
    elif rescale_unit == "none":
        rescale_unit = crs_unit
    elif crs_unit in SCALE_METRES:
        rescale_unit = rescale_unit or "km"
    elif crs_unit in SCALE_DEGREES:
        rescale_unit = rescale_unit or "deg"
    else:
        raise ValueError(f"CRS unit '{crs_unit}' not recognised for rescaling")

    # Get scale factor
    if rescale_unit is None and (crs_unit is None or crs_unit in SCALE_OTHERS):
        scale_factor = 1.0
    elif crs_unit in SCALE_METRES and rescale_unit in SCALE_METRES:
        scale_factor = SCALE_METRES[rescale_unit] / SCALE_METRES[crs_unit]
    elif crs_unit in SCALE_DEGREES and rescale_unit in SCALE_DEGREES:
        scale_factor = SCALE_DEGREES[rescale_unit] / SCALE_DEGREES[crs_unit]
    else:
        raise ValueError(f"Cannot rescale from '{crs_unit}' to '{rescale_unit}'")

    return rescale_unit, scale_factor


def get_xy_labels(data_or_crs: DATA_OR_CRS_TYPE = None, rescale_unit: str | None = None) -> tuple[str, str]:
    """Get x and y axis labels from a coordinate reference system (CRS).

    Args:
        data_or_crs (:class:`xarray.DataArray` | :class:`xarray.Dataset` | :class:`xugrid.UgridDataArray` | :class:`xugrid.UgridDataset` | :class:`geopandas.GeoDataFrame` | :class:`pyproj.CRS` | :class:`rasterio.crs.CRS` | str | None): Data or coordinate reference system.
        rescale_unit (str, optional): Rescale unit. Defaults to None.

    Returns:
        tuple[str, str]: x and y axis labels.
    """
    # Get crs from data or CRS
    crs = get_crs_from_data_or_crs(data_or_crs)

    # Rescale unit
    rescale_unit = "-" if rescale_unit is None else rescale_unit

    # Get x and y labels from crs
    if crs is not None:
        x_label = f"{crs.axis_info[0].name} {crs.name} [{rescale_unit}]"
        y_label = f"{crs.axis_info[1].name} {crs.name} [{rescale_unit}]"
    else:
        x_label = f"x [{rescale_unit}]"
        y_label = f"y [{rescale_unit}]"

    return x_label, y_label


def get_crs_from_data_or_crs(data_or_crs: DATA_OR_CRS_TYPE = None) -> pyprojCRS | None:
    """Get coordinate reference system (CRS) from data or CRS.

    Args:
        data_or_crs (:class:`xarray.DataArray` | :class:`xarray.Dataset` | :class:`xugrid.UgridDataArray` | :class:`xugrid.UgridDataset` | :class:`geopandas.GeoDataFrame` | :class:`pyproj.CRS` | :class:`rasterio.crs.CRS` | str | None): Data or coordinate reference system.

    Returns:
        pyproj.CRS | None: Coordinate reference system.
    """
    # Get crs from data or CRS
    if isinstance(data_or_crs, xr.DataArray | xr.Dataset):
        crs = data_or_crs.rio.crs
    elif isinstance(data_or_crs, xu.UgridDataArray | xu.UgridDataset):
        crs = data_or_crs.grid.crs
    elif isinstance(data_or_crs, gpd.GeoDataFrame):
        crs = data_or_crs.crs
    else:
        crs = data_or_crs

    # Convert the crs to a pyproj.CRS
    if isinstance(crs, pyprojCRS):
        crs = crs  # noqa: PLW0127
    elif isinstance(crs, rasterioCRS):
        crs = pyprojCRS.from_string(crs.to_string())
    elif isinstance(crs, str):
        crs = pyprojCRS.from_string(crs)

    return crs
