import geopandas as gpd
import ipyleaflet
import numpy as np
import shapely.geometry
from ipywidgets import HTML

from resplotlib import utils

# The following CSS is used to set the background color of the Leaflet controls to white and make them fully opaque.
# It does not have to be supplied anywhere but for some reason, without it, the controls are transparent.
css = HTML("""
<style>
    .leaflet-control { 
        background-color: white !important; 
        opacity: 1 !important; 
    }
</style>
""")


class Map(ipyleaflet.Map):
    """
    A class for creating an interactive map using ipyleaflet.

    Attributes:
        drawn (bool): A flag indicating whether geometries have been drawn on the map.
        style_kwargs (dict): A dictionary containing the style options for the geometries drawn on the map.
        draw_control (ipyleaflet.DrawControl or ipyleaflet.GeomanDrawControl): The draw control used for drawing geometries on the map.

    Args:
        gdf (:class:`geopandas.GeoDataFrame`, optional): GeoDataFrame containing draw geometries to add to the map. Defaults to None.
        clear_on_draw (bool, optional): Clear drawn geometries from the map when a new geometry is drawn. Defaults to False.
        geoman_draw (bool, optional): Use GeomanDrawControl to draw geometries on the map. Defaults to True.
        **kwargs: Additional keyword arguments to pass to :class:`ipyleaflet.Map`.
    """

    _style_kwargs_red = {"weight": 2, "color": "red", "fillColor": "red", "fillOpacity": 0.5, "radius": 10}
    _style_kwargs_green = {"weight": 2, "color": "green", "fillColor": "green", "fillOpacity": 0.5, "radius": 10}
    _hover_style_kwargs_red = {"weight": 2, "color": "red", "fillColor": "red", "fillOpacity": 0.7, "radius": 10}
    _hover_style_kwargs_green = {"weight": 2, "color": "green", "fillColor": "green", "fillOpacity": 0.7, "radius": 10}
    _clear_on_draw = None

    def __init__(self, gdf: gpd.GeoDataFrame | None = None, clear_on_draw: bool = False, geoman_draw: bool = True, **kwargs) -> None:
        """Initialise the map.

        Args:
            gdf (gpd.GeoDataFrame, optional): GeoDataFrame containing the geometries to add to the map. Defaults to None.
            clear_on_draw (bool, optional): Clear drawn geometries from the map when a new geometry is drawn. Defaults to False.
            geoman_draw (bool, optional): Use GeomanDrawControl to draw geometries on the map. Defaults to True.
            **kwargs: Additional keyword arguments to pass to :class:`ipyleaflet.Map`.
        """
        # Convert gdf to EPSG:4326
        if gdf is not None and not gdf.empty:
            gdf = gdf.to_crs("EPSG:4326")

        # Get center from geometries if not provided
        if "center" not in kwargs and gdf is not None and not gdf.empty:
            kwargs["center"] = utils.get_center_from_bounds(gdf.total_bounds)

        # Get approximate zoom level from geometries if not provided
        if "zoom" not in kwargs and gdf is not None and not gdf.empty:
            kwargs["zoom"] = utils.get_zoom_from_bounds(gdf.total_bounds)

        # Set default scroll wheel zoom and layout kwargs
        kwargs.setdefault("scroll_wheel_zoom", True)
        kwargs.setdefault("layout", {"width": "100%", "height": "600px"})

        # Initialize map superclass
        super().__init__(**kwargs)

        # Set clear_on_draw attribute
        self._clear_on_draw = clear_on_draw

        # Add draw control
        draw_control_kwargs = {
            "polyline": {"shapeOptions": self._style_kwargs_red},
            "rectangle": {"shapeOptions": self._style_kwargs_red},
            "polygon": {"shapeOptions": self._style_kwargs_red},
            "circle": {"shapeOptions": self._style_kwargs_red},
            "circlemarker": {"shapeOptions": self._style_kwargs_red},
        }
        if geoman_draw:
            self.draw_control = ipyleaflet.GeomanDrawControl(**draw_control_kwargs)
        else:
            self.draw_control = ipyleaflet.DrawControl(**draw_control_kwargs)
        self.add_control(self.draw_control)
        self.draw_control.on_draw(self._handle_draw)

        # Add draw layer (to store drawn geometries separately from draw control)
        if self._clear_on_draw:
            draw_layer_kwargs = {
                "data": {"type": "FeatureCollection", "features": []},
                "style": self._style_kwargs_green,
                "hover_style": self._hover_style_kwargs_green,
                "point_style": self._style_kwargs_green,
                "name": "Drawn Geometries",
            }
            self.draw_layer = ipyleaflet.GeoJSON(**draw_layer_kwargs)
            self.add_layer(self.draw_layer)

        # Add additional controls to map
        self.add_control(ipyleaflet.FullScreenControl())
        self.add_control(ipyleaflet.LayersControl(position="topright"))

        # Set geometries to map
        if gdf is not None and not gdf.empty:
            self.set_drawn_geometries(gdf)

    def _handle_draw(self, target: ipyleaflet.Widget, action: str, geo_json: dict) -> None:
        """Handle draw events from the draw control.

        Args:
            target (ipyleaflet.Widget): The draw control widget that triggered the event.
            action (str): The type of draw event (e.g., "created", "edited", "deleted").
            geo_json (dict): The GeoJSON representation of the drawn geometry.
        """
        # If clear_on_draw is False, do not clear drawn geometries from map
        if not self._clear_on_draw:
            return

        # Clear drawn geometries from map
        self.clear_drawn_geometries()

        # Convert drawn geometry to GeoDataFrame
        if isinstance(geo_json, dict):
            geo_json = [geo_json]
        gdf_drawn = gpd.GeoDataFrame.from_features(geo_json, crs="EPSG:4326")

        # Set drawn geometry to map
        self.set_drawn_geometries(gdf_drawn)

    def clear_drawn_geometries(self) -> None:
        """Clear geometries from the map."""
        # Clear geometries from draw control and draw layer
        self.draw_control.clear()
        if self._clear_on_draw:
            self.draw_layer.clear()

    def set_drawn_geometries(self, gdf: gpd.GeoDataFrame) -> None:
        """Set geometries to the map.

        Args:
            gdf (:class:`geopandas.GeoDataFrame`): GeoDataFrame containing the geometries to add to the map.
        """
        # Add style to GeoDataFrame
        gdf["style"] = [self._style_kwargs_red] * len(gdf)
        gdf["hover_style"] = [self._hover_style_kwargs_red] * len(gdf)

        # Add type to GeoDataFrame
        gdf["type"] = None
        for idx, row in gdf.iterrows():
            if row.geometry.geom_type in ["Point", "MultiPoint"]:
                gdf.at[idx, "type"] = "circlemarker"
            elif row.geometry.geom_type in ["LineString", "MultiLineString"]:
                gdf.at[idx, "type"] = "polyline"
            elif row.geometry.geom_type in ["Polygon", "MultiPolygon"]:
                gdf.at[idx, "type"] = "polygon"

        # Add geometries to draw control or draw layer
        if not self._clear_on_draw:
            self.draw_control.data = list(gdf.iterfeatures())
        else:
            self.draw_layer.data = {"type": "FeatureCollection", "features": list(gdf.iterfeatures())}

    def get_drawn_geometries(self, crs: str = "EPSG:4326") -> gpd.GeoDataFrame:
        """Get drawn geometries from the map.

        Args:
            crs (str, optional): Coordinate reference system to reproject the geometries to. Defaults to "EPSG:4326".

        Returns:
            :class:`geopandas.GeoDataFrame`: GeoDataFrame containing the drawn geometries from the map.
        """
        # Get geometries from the map
        if not self._clear_on_draw and self.draw_control.data:
            gdf = gpd.GeoDataFrame.from_features(self.draw_control.data, crs="EPSG:4326").drop(columns="style")
        elif self._clear_on_draw and self.draw_layer.data["features"]:
            gdf = gpd.GeoDataFrame.from_features(self.draw_layer.data["features"], crs="EPSG:4326").drop(columns="style")
        else:
            gdf = gpd.GeoDataFrame(columns=["geometry"], crs="EPSG:4326")

        # Move geometry column to end
        columns = [col for col in gdf.columns if col != "geometry"] + ["geometry"]
        gdf = gdf[columns]

        # Drop type column if it exists
        if "type" in gdf.columns:
            gdf = gdf.drop(columns="type")

        # Reproject geometries
        gdf = gdf.to_crs(crs)

        return gdf

    def get_drawn_points(self) -> gpd.GeoDataFrame:
        """Get drawn points from the map.

        Returns:
            :class:`geopandas.GeoDataFrame`: GeoDataFrame containing the drawn points from the map.
        """
        gdf = self.get_drawn_geometries()
        gdf_points = gdf[gdf.geometry.type.isin(["Point", "MultiPoint"])].reset_index(drop=True)
        return gdf_points

    def get_drawn_lines(self) -> gpd.GeoDataFrame:
        """Get drawn lines from the map.

        Returns:
            :class:`geopandas.GeoDataFrame`: GeoDataFrame containing the drawn lines from the map.
        """
        gdf = self.get_drawn_geometries()
        gdf_lines = gdf[gdf.geometry.type.isin(["LineString", "MultiLineString"])].reset_index(drop=True)
        return gdf_lines

    def get_drawn_polygons(self) -> gpd.GeoDataFrame:
        """Get drawn polygons from the map.

        Returns:
            :class:`geopandas.GeoDataFrame`: GeoDataFrame containing the drawn polygons from the map.
        """
        gdf = self.get_drawn_geometries()
        gdf_polygons = gdf[gdf.geometry.type.isin(["Polygon", "MultiPolygon"])].reset_index(drop=True)
        return gdf_polygons

    def get_drawn_boxes(self) -> gpd.GeoDataFrame:
        """Get drawn boxes from the map.

        Returns:
            :class:`geopandas.GeoDataFrame`: GeoDataFrame containing the drawn box geometries from the map.
        """
        gdf = self.get_drawn_geometries()
        boxes = []
        for geom in gdf.geometry:
            if geom.geom_type == "Polygon":
                minx, miny, maxx, maxy = geom.bounds
                box = shapely.geometry.box(minx, miny, maxx, maxy)
                if geom.equals(box):
                    boxes.append(geom)
        gdf_boxes = gpd.GeoDataFrame(geometry=boxes, crs=gdf.crs).reset_index(drop=True)
        return gdf_boxes

    def get_view_bounds(self) -> tuple[float, float, float, float]:
        """Get view bounds of the map.

        Returns:
            tuple[float, float, float, float]: View bounds in the format (minx, miny, maxx, maxy).
        """
        # If bounds are not set, return nan values
        if len(self.bounds) != 2:
            return (np.nan, np.nan, np.nan, np.nan)

        # Get bounds of map view
        south, west = self.bounds[0]
        north, east = self.bounds[1]

        # Clip bounds to valid lat/lon ranges
        minx = max(west, -180)
        miny = max(south, -90)
        maxx = min(east, 180)
        maxy = min(north, 90)

        return (minx, miny, maxx, maxy)

    def get_view_zoom(self) -> int:
        """Get view zoom level of the map.

        Returns:
            int: View zoom level of the map.
        """
        # Get zoom level of map view
        return self.zoom
