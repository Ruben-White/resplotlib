import geopandas as gpd
import numpy as np
import xarray as xr
import xugrid as xu
from pyproj import CRS as pyprojCRS
from rasterio.crs import CRS as rasterioCRS

DATA_OR_CRS_TYPE = xr.DataArray | xr.Dataset | xu.UgridDataArray | xu.UgridDataset | gpd.GeoDataFrame | pyprojCRS | rasterioCRS | str | None


def _combine_dicts(dict1: dict, dict2: dict, max_depth: int = 4) -> dict:
    """Recursively combine dictionaries, prioritising dictionary 2.

    Args:
        dict1 (dict): First dictionary.
        dict2 (dict): Second dictionary.
        max_depth (int): Maximum recursion depth.

    Returns:
        dict: Combined dictionary.
    """

    # Maximum depth reached -> return dictionary 2
    if max_depth == 0:
        return dict2

    # Both dictionaries are not dictionaries -> return dictionary 2
    if not isinstance(dict1, dict) and not isinstance(dict2, dict):
        return dict2

    # Dictionary 1 is not a dictionary -> return dictionary 2
    elif not isinstance(dict1, dict):
        return dict2

    # Dictionary 2 is not a dictionary -> return dictionary 1
    elif not isinstance(dict2, dict):
        return dict1

    # Both dictionaries are dictionaries -> combine dictionaries
    keys = list(dict1.keys()) + list(dict2.keys())
    dict_combined = {}
    for key in keys:
        # Key not in dictionary 1 -> use dictionary 2
        if key not in dict1.keys():
            dict_combined[key] = dict2[key]

        # Key not in dictionary 2 -> use dictionary 1
        elif key not in dict2.keys():
            dict_combined[key] = dict1[key]

        # Key in both dictionaries -> combine dictionaries
        else:
            dict_combined[key] = _combine_dicts(dict1[key], dict2[key], max_depth=max_depth - 1)

    return dict_combined


def _substitute_str_in_dict(dict1: dict, str1: str, str2: str) -> dict:
    """Recursively substitutes strings in a dictionary.

    Args:
        dict1 (dict): Dictionary to substitute strings in.
        str1 (str): String to substitute.
        str2 (str): String to substitute with.

    Returns:
        dict: Dictionary with substituted strings.
    """

    # Check if strings are None
    if str1 is None or str2 is None:
        return dict1

    # Substitute string in dictionary
    for key, value in dict1.items():
        if isinstance(value, dict):
            dict1[key] = _substitute_str_in_dict(value, str1, str2)
        elif isinstance(value, str):
            dict1[key] = value.replace(str1, str2)

    return dict1


def _substitute_inherit_str_in_dicts(dict1: dict, max_depth: int = 4) -> dict:
    """Recursively apply inheritance in a dictionary of dictionaries, substituting the "@inherits" key with the corresponding dictionary in the same parent dictionary.

    Args:
        dict1 (dict): Dictionary to apply inheritance in.
        max_depth (int): Maximum recursion depth.

    Returns:
        dict: Dictionary with applied inheritance.
    """

    # Maximum depth reached -> return dictionary
    if max_depth == 0:
        return dict1

    for key1 in dict1.keys():
        # Check if value is a dictionary
        if not isinstance(dict1[key1], dict):
            continue

        # Recursively apply inheritance in child dictionary
        dict2 = _substitute_inherit_str_in_dicts(dict1[key1], max_depth=max_depth - 1)

        # Check if "inherits" key is present
        if "inherits" not in dict2:
            continue

        # Get reference and available references
        reference = dict2.pop("inherits").lstrip("@")
        available_references = [k for k in dict1.keys() if k != key1]

        # Apply inheritance if reference is available
        if reference in available_references:
            dict2 = _combine_dicts(dict1[reference], dict2)
            dict2 = dict(sorted(dict2.items()))
            dict1[key1] = dict2
        else:
            raise KeyError(
                f"An error occurred while processing guidelines. Inheritance reference '{reference}' in '{key1}' not found. Available references: {available_references}"
            )

    return dict1


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
        crs = crs
    elif isinstance(crs, rasterioCRS):
        crs = pyprojCRS.from_string(crs.to_string())
    elif isinstance(crs, str):
        crs = pyprojCRS.from_string(crs)

    return crs


def get_rescale_parameters(
    data_or_crs: DATA_OR_CRS_TYPE = None,
    rescale_unit: str | None = None,
) -> tuple[str | None, float]:
    """Get rescale parameters from a coordinate reference system (CRS).

    Args:
        data_or_crs (:class:`xarray.DataArray` | :class:`xarray.Dataset` | :class:`xugrid.UgridDataArray` | :class:`xugrid.UgridDataset` | :class:`geopandas.GeoDataFrame` | :class:`pyproj.CRS` | :class:`rasterio.crs.CRS` | str | None): Data or coordinate reference system.
        rescale_unit (str, optional): Desired rescale unit. Defaults to None.

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

    # Get crs from data or CRS
    crs = get_crs_from_data_or_crs(data_or_crs)

    # Get crs unit
    crs_unit = UNIT_ABBREVIATIONS[crs.axis_info[0].unit_name.lower()] if crs is not None else None

    # Get rescale unit
    if crs_unit is None:
        rescale_unit = None
    elif crs_unit in SCALE_METRES:
        rescale_unit = rescale_unit or "km"
    elif crs_unit in SCALE_DEGREES:
        rescale_unit = rescale_unit or "deg"
    else:
        raise ValueError(f"CRS unit '{crs_unit}' not recognised for rescaling")

    # Get scale factor
    if rescale_unit is None and crs_unit is None:
        scale_factor = 1.0
    if crs_unit in SCALE_METRES and rescale_unit in SCALE_METRES:
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


def get_center_from_bounds(bounds: tuple[float, float, float, float]) -> tuple[float, float]:
    """Get center from bounds.

    Args:
        bounds (tuple[float, float, float, float]): Bounds (minx, miny, maxx, maxy).

    Returns:
        tuple[float, float]: Center of the bounds (latitude, longitude).
    """
    minx, miny, maxx, maxy = bounds
    center_lat = (miny + maxy) / 2
    center_lon = (minx + maxx) / 2
    return center_lat, center_lon


def get_zoom_from_bounds(bounds: tuple[float, float, float, float]) -> int:
    """Get approximate zoom level from bounds.

    Args:
        bounds (tuple[float, float, float, float]): Bounds (minx, miny, maxx, maxy).

    Returns:
        int: Approximate zoom level of the bounds.
    """
    minx, miny, maxx, maxy = bounds
    span = max(maxx - minx, maxy - miny)
    zoom = int(np.log2(360 / span)) + 1
    return zoom
