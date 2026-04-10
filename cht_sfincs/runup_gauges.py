"""SFINCS runup gauge transect reader/writer.

Provides the SfincsRunupGauges class for managing runup gauge polylines
(rug file) used to compute wave runup along transects in the SFINCS domain.
"""

import os

import geopandas as gpd
import pandas as pd
import shapely
from cht_utils.fileio.pli_file import gdf2pli, pli2gdf


class SfincsRunupGauges:
    """SFINCS runup gauge transect lines.

    Manages runup gauge polylines (rug file) used to compute wave runup
    along transects in the SFINCS domain.

    Parameters
    ----------
    sf : SFINCS
        The parent SFINCS model instance.
    """

    def __init__(self, sf: "SFINCS") -> None:
        self.model = sf
        self.gdf = gpd.GeoDataFrame()

    def read(self) -> None:
        """Read runup gauge transects from the rug file.

        Returns
        -------
        None
        """
        # Read in all runup gauges

        if not self.model.input.variables.rugfile:
            return

        filename = os.path.join(self.model.path, self.model.input.variables.rugfile)

        # Read the crs file
        self.gdf = pli2gdf(filename, crs=self.model.crs)

    def write(self, file_name: str | None = None) -> None:
        """Write runup gauge transects to the rug file.

        Parameters
        ----------
        file_name : str, optional
            Override path for the output file.

        Returns
        -------
        None
        """
        if not self.model.input.variables.rugfile:
            return
        if len(self.gdf.index) == 0:
            return

        filename = os.path.join(self.model.path, self.model.input.variables.rugfile)

        gdf2pli(self.gdf, filename)

    def add(self, runup_gauge) -> None:
        """Append a runup gauge GeoDataFrame to the collection.

        Parameters
        ----------
        runup_gauge : geopandas.GeoDataFrame
            Runup gauge transect(s) to add.

        Returns
        -------
        None
        """
        runup_gauge.set_crs(self.model.crs)
        self.gdf = pd.concat([self.gdf, runup_gauge], ignore_index=True)

    def add_xy(self, name: str, x0: float, y0: float, x1: float, y1: float) -> None:
        """Add a runup gauge transect from two endpoints.

        Parameters
        ----------
        name : str
            Name label for the runup gauge.
        x0 : float
            X-coordinate of the transect start point.
        y0 : float
            Y-coordinate of the transect start point.
        x1 : float
            X-coordinate of the transect end point.
        y1 : float
            Y-coordinate of the transect end point.

        Returns
        -------
        None
        """
        line = shapely.geometry.LineString([(x0, y0), (x1, y1)])
        gdf = gpd.GeoDataFrame({"name": [name], "geometry": [line]}).set_crs(
            self.model.crs
        )
        self.gdf = pd.concat([self.gdf, gdf], ignore_index=True)

    def delete(self, name_or_index: str | int) -> None:
        """Delete a runup gauge by name or row index.

        Parameters
        ----------
        name_or_index : str or int
            Name string or zero-based integer row index.

        Returns
        -------
        None
        """
        if isinstance(name_or_index, str):
            name = name_or_index
            for index, row in self.gdf.iterrows():
                if row["name"] == name:
                    self.gdf = self.gdf.drop(index).reset_index(drop=True)
                    return
            print(f"Run-up gauge {name} not found!")
        else:
            index = name_or_index
            if len(self.gdf.index) < index + 1:
                print("Index exceeds length!")
            self.gdf = self.gdf.drop(index).reset_index(drop=True)
            return

    def clear(self) -> None:
        """Remove all runup gauges.

        Returns
        -------
        None
        """
        self.gdf = gpd.GeoDataFrame()

    def list_names(self) -> list:
        """Return a list of all runup gauge names.

        Returns
        -------
        list[str]
            Runup gauge names in GeoDataFrame order.
        """
        names = []
        for index, row in self.gdf.iterrows():
            names.append(row["name"])
        return names
