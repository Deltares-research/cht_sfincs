"""SFINCS thin dam polyline reader/writer.

Provides the SfincsThinDams class for managing thin-dam line features that
impose impermeable barriers at cell edges in the SFINCS computational grid.
"""

import os

import geopandas as gpd
import pandas as pd
import shapely
from cht_utils.fileio.pli_file import gdf2pli, pli2gdf


class SfincsThinDams:
    """SFINCS thin dam polylines.

    Manages thin-dam line features that impose impermeable barriers at
    cell edges in the SFINCS computational grid.

    Parameters
    ----------
    sf : SFINCS
        The parent SFINCS model instance.
    """

    def __init__(self, sf: "SFINCS") -> None:
        self.model = sf
        self.gdf = gpd.GeoDataFrame()

    def read(self) -> None:
        """Read thin dam polylines from the thd file.

        Returns
        -------
        None
        """
        # Read in all thin dams

        if not self.model.input.variables.thdfile:
            return

        filename = os.path.join(self.model.path, self.model.input.variables.thdfile)

        # Read the thd file
        self.gdf = pli2gdf(filename, crs=self.model.crs)

    def write(self) -> None:
        """Write thin dam polylines to the thd file.

        Returns
        -------
        None
        """
        if len(self.gdf.index) == 0:
            return

        if not self.model.input.variables.thdfile:
            self.model.input.variables.thdfile = "sfincs.thd"

        filename = os.path.join(self.model.path, self.model.input.variables.thdfile)

        gdf2pli(self.gdf, filename)

    def add(self, thin_dam) -> None:
        """Append a thin dam GeoDataFrame to the collection.

        Parameters
        ----------
        thin_dam : geopandas.GeoDataFrame
            Thin dam polyline(s) to add.

        Returns
        -------
        None
        """
        # Thin dam may be a gdf or shapely geometry
        # Assume for now a gdf
        thin_dam.set_crs(self.model.crs)
        self.gdf = pd.concat([self.gdf, thin_dam], ignore_index=True)

    def add_xy(self, x: list, y: list) -> None:
        """Add a thin dam by providing x and y coordinate lists.

        Parameters
        ----------
        x : list[float]
            X-coordinates of the thin dam polyline.
        y : list[float]
            Y-coordinates of the thin dam polyline.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If *x* and *y* have different lengths.
        """
        # Add a thin dam by providing x and y coordinates
        # x and y are lists of the same length
        # Create linestring geometry
        if len(x) != len(y):
            raise ValueError("x and y must be the same length")
        thin_dam = gpd.GeoDataFrame(
            geometry=[shapely.geometry.LineString(zip(x, y))], crs=self.model.crs
        )
        # Create a new row in the gdf
        self.gdf = pd.concat([self.gdf, thin_dam], ignore_index=True)

    def delete(self, index: int) -> None:
        """Delete a thin dam by row index.

        Parameters
        ----------
        index : int
            Zero-based row index of the thin dam to remove.

        Returns
        -------
        None
        """
        if len(self.gdf.index) < index + 1:
            print("Index exceeds length!")
        self.gdf = self.gdf.drop(index).reset_index(drop=True)
        return

    def clear(self) -> None:
        """Remove all thin dams.

        Returns
        -------
        None
        """
        self.gdf = gpd.GeoDataFrame()

    def snap_to_grid(self):
        snap_gdf = self.model.grid.snap_to_grid(self.gdf)
        return snap_gdf

    def list_names(self):
        names = []
        for index, row in self.gdf.iterrows():
            names.append(str(index + 1))
        return names
