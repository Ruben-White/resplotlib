import numpy as np
import xarray as xr
import xugrid as xu

from . import rescale


def from_structured(da: xr.DataArray | xr.Dataset, **kwargs) -> xu.UgridDataArray | xu.UgridDataset:
    """Convert a structured xarray.DataArray or xarray.Dataset to an unstructured xugrid.UgridDataArray or xugrid.UgridDataset.

    Args:
        da (:class:`xarray.DataArray` or :class:`xarray.Dataset`): The structured DataArray or Dataset to convert.
    Returns:
        :class:`xugrid.UgridDataArray` or :class:`xugrid.UgridDataset`: The converted unstructured UgridDataArray or UgridDataset.
    """
    # Convert structured data to unstructured data
    if isinstance(da, xr.DataArray):
        uda = xu.UgridDataArray.from_structured2d(da, **kwargs)
    elif isinstance(da, xr.Dataset):
        uda = xu.UgridDataset.from_structured2d(da, **kwargs)

    # Set coordinate reference system
    uda.ugrid.set_crs(da.rio.crs)

    return uda


def to_structured(
    uda: xu.UgridDataArray | xu.UgridDataset,
    template: xr.DataArray = None,
    bounds: tuple[float, float, float, float] = None,
    resolution: float = None,
) -> xr.DataArray | xr.Dataset:
    """Convert an unstructured xugrid.UgridDataArray or xugrid.UgridDataset to a structured xarray.DataArray.

    Args:
        uda (:class:`xugrid.UgridDataArray` or :class:`xugrid.UgridDataset`): The unstructured UgridDataArray or UgridDataset to convert.
        template (:class:`xarray.DataArray`, optional): A template DataArray to use for the conversion. If provided, the x and y coordinates will be taken from the template. Defaults to None.
        bounds (tuple, optional): A tuple of the form (xmin, ymin, xmax, ymax) defining the bounds of the output structured data. Required if template is not provided. Defaults to None.
        resolution (float, optional): The resolution of the output structured data. Required if template is not provided. Defaults to None.
    Returns:
        :class:`xarray.DataArray`: The converted structured DataArray.
    """

    # Get x and y coordinates from template or create them from bounds and resolution
    if template is not None:
        xs = template["x"]
        ys = template["y"]
    elif bounds is not None and resolution is not None:
        xmin, ymin, xmax, ymax = bounds
        xs = np.linspace(xmin, xmax, int((xmax - xmin) / resolution) + 1)
        ys = np.linspace(ymin, ymax, int((ymax - ymin) / resolution) + 1)
    else:
        raise ValueError("Either template or bounds and resolution must be provided.")

    # Create x and y grids
    xG, yG = np.meshgrid(xs, ys)

    # Use rescale to rename the dimensions of the data
    uda = rescale.rescale_uda(uda, 1)

    # Remove data variables that do not have nmesh2d_face dimension
    if isinstance(uda, xu.UgridDataset):
        for var in uda.data_vars:
            if "mesh2d_nFaces" not in uda[var].dims:
                uda = uda.drop_vars(var)

    # Remove coordinates that contain _index, _x, or _y
    for coord in uda.coords:
        if "_index" in coord or "_x" in coord or "_y" in coord:
            uda = uda.drop_vars(coord)

    # Remove dimensions that contain nodes or edges
    for dim in uda.dims:
        if "node" in dim or "edge" in dim or "Node" in dim or "Edge" in dim:
            uda = uda.drop_dims(dim)

    # Get dataset
    ds = uda.ugrid.sel_points(x=xG.flatten(), y=yG.flatten(), out_of_bounds="ignore")

    # Remove mesh2d_ from data variables
    if isinstance(ds, xu.UgridDataset):
        for var in ds.data_vars:
            if "mesh2d_" in var:
                ds = ds.rename({var: var.replace("mesh2d_", "")})

    # Remove coordinates that contain _index, _x, or _y
    for coord in ds.coords:
        if "_index" in coord or "_x" in coord or "_y" in coord:
            ds = ds.drop_vars(coord)

    # Replace mesh2d_nFaces dimension with idx dimension
    ds = ds.rename({"mesh2d_nFaces": "idx"})
    ds["idx"] = range(len(ds["idx"]))

    # Add x, and y coordinates
    ds = ds.assign_coords(x=("idx", xG.flatten()), y=("idx", yG.flatten()))

    # Set index
    ds = ds.set_index(idx=("x", "y"))

    # Unstack index
    ds = ds.unstack("idx")

    # Transpose dimensions
    ds = ds.transpose("y", "x", ...)

    # Set coordinate reference system
    ds = ds.rio.write_crs(uda.grid.crs)

    # Return the rasterised data
    return ds
