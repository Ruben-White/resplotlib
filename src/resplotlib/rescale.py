import geopandas as gpd
import numpy as np
import xarray as xr
import xugrid as xu


def rescale_uda(uda: xu.UgridDataArray | xu.UgridDataset, scale_factor: float) -> xu.UgridDataArray:
    """Rescale the coordinates of a xugrid.UgridDataArray or xugrid.UgridDataset by a specified scale factor.

    Args:
        uda (:class:`xugrid.UgridDataArray` | :class:`xugrid.UgridDataset`): DataArray or Dataset to be rescaled.
        scale_factor (float): Factor by which to rescale the coordinates.

    Returns:
        :class:`xugrid.UgridDataArray` | :class:`xugrid.UgridDataset`: Rescaled UgridDataArray or UgridDataset.
    """

    # Function to rename the dimensions of the data
    def _rename_dims(uda):
        # Define the new dimension names
        NEW_DIMS_1D = {
            "node": "network1d_nNodes",
            "edge": "network1d_nEdges",
            "face": "network1d_nFaces",
            "Node": "network1d_nNodes",
            "Edge": "network1d_nEdges",
            "Face": "network1d_nFaces",
        }
        NEW_DIMS_2D = {
            "node": "mesh2d_nNodes",
            "edge": "mesh2d_nEdges",
            "face": "mesh2d_nFaces",
            "Node": "mesh2d_nNodes",
            "Edge": "mesh2d_nEdges",
            "Face": "mesh2d_nFaces",
        }

        # Get the new dimension names
        if isinstance(uda.grid, xu.Ugrid1d):
            NEW_DIMS = NEW_DIMS_1D
        elif isinstance(uda.grid, xu.Ugrid2d):
            NEW_DIMS = NEW_DIMS_2D

        # Rename the dimensions of the data
        for dim in list(uda.indexes):
            for new_dim in NEW_DIMS.keys():
                if new_dim in dim:
                    uda = uda.rename({dim: NEW_DIMS[new_dim]})

        return uda

    # Rescale the coordinates of the data
    def _rescale_coords(da, scale_factor):
        # Define the coordinates to rescale
        COORD_NAMES = [
            "x",
            "y",
            "node_x",
            "node_y",
            "edge_x",
            "edge_y",
            "face_x",
            "face_y",
            "X",
            "Y",
            "Node_x",
            "Node_y",
            "Edge_x",
            "Edge_y",
            "Face_x",
            "Face_y",
        ]

        # Rescale the coordinates
        coord_names = [coord_name for coord_name in list(da.coords) if coord_name not in list(da.dims)]

        # Rescale the coordinates
        coords = {}
        for coord_name in coord_names:
            if np.any([COORD_NAME in coord_name for COORD_NAME in COORD_NAMES]):
                coords[coord_name] = xr.DataArray(da[coord_name].values * scale_factor, dims=da[coord_name].dims, attrs=da[coord_name].attrs)

        # Assign the rescaled coordinates to the data
        da = da.assign_coords(coords)

        return da

    # Function to rescale the grid
    def _rescale_grid(grid, scale_factor):
        # Rescale the x and y dimensions of 1D grid
        if isinstance(grid, xu.Ugrid1d):
            grid = xu.Ugrid1d(
                node_x=grid.node_x * scale_factor,
                node_y=grid.node_y * scale_factor,
                fill_value=grid.fill_value,
                edge_node_connectivity=grid.edge_node_connectivity,
            )

        # Rescale x and y dimensions of 2D grid
        elif isinstance(grid, xu.Ugrid2d):
            grid = xu.Ugrid2d(
                node_x=grid.node_x * scale_factor,
                node_y=grid.node_y * scale_factor,
                fill_value=grid.fill_value,
                face_node_connectivity=grid.face_node_connectivity,
                edge_node_connectivity=grid.edge_node_connectivity,
            )

        return grid

    # Rename the dimensions of the data
    da = _rename_dims(uda)

    # Rescale the coordinates of the data
    da = _rescale_coords(da, scale_factor)

    # Assign the coordinate arrays to the data
    if isinstance(uda, xu.UgridDataArray):
        grid = _rescale_grid(uda.grid, scale_factor)
        uda_rescaled = xu.UgridDataArray(obj=xr.DataArray(da), grid=grid)
    elif isinstance(uda, xu.UgridDataset):
        grids = [_rescale_grid(grid, scale_factor) for grid in uda.grids]
        uda_rescaled = xu.UgridDataset(obj=xr.Dataset(da), grids=grids)

    # Assign the coordinate reference system to the data
    uda_rescaled.grid.set_crs(uda.grid.crs)

    return uda_rescaled


def rescale_da(da: xr.DataArray | xr.Dataset, scale_factor: float) -> xr.DataArray | xr.Dataset:
    """Rescale the coordinates of a xarray.DataArray or xarray.Dataset by a specified scale factor.

    Args:
        da (:class:`xarray.DataArray` | :class:`xarray.Dataset`): DataArray or Dataset to be rescaled.
        scale_factor (float): Factor by which to rescale the coordinates.

    Returns:
        :class:`xarray.DataArray` | :class:`xarray.Dataset`: Rescaled DataArray or Dataset.
    """
    # Rescale the x and y dimensions
    x_coord = xr.DataArray(da["x"].values * scale_factor, dims=["x"], attrs=da.x.attrs)
    y_coord = xr.DataArray(da["y"].values * scale_factor, dims=["y"], attrs=da.y.attrs)

    # Assign the coordinate arrays
    da = da.assign_coords(x=x_coord, y=y_coord)

    return da


def rescale_gdf(gdf: gpd.GeoDataFrame, scale_factor: float) -> gpd.GeoDataFrame:
    """Rescale the coordinates of a geopandas.GeoDataFrame by a specified scale factor.

    Args:
        gdf (:class:`geopandas.GeoDataFrame`): GeoDataFrame to be rescaled.
        scale_factor (float): Factor by which to rescale the geometries.

    Returns:
        :class:`geopandas.GeoDataFrame`: Rescaled GeoDataFrame.
    """
    # Copy the GeoDataFrame
    gdf = gdf.copy()

    # Rescale geometries
    gdf["geometry"] = gdf["geometry"].scale(scale_factor, scale_factor, origin=(0, 0))

    return gdf
