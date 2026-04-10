"""SFINCS cross-section polyline reader/writer.

Provides the SfincsCrossSections class for managing cross-section polylines
used to compute transect discharges in SFINCS post-processing.
"""

import os

import geopandas as gpd
import pandas as pd
from cht_utils.fileio.pli_file import gdf2pli, pli2gdf


class SfincsCrossSections:
    """SFINCS cross-section definitions.

        Manages cross-section polylines that can be used to compute
    transect discharge in SFINCS post-processing.

        Parameters
        ----------
        hw : SFINCS
            The parent SFINCS model instance.
    """

    def __init__(self, hw: "SFINCS") -> None:
        self.model = hw
        self.gdf = gpd.GeoDataFrame()

    def read(self) -> None:
        """Read cross-section polylines from the crs file.

        Returns
        -------
        None
        """
        # Read in all cross sections

        if not self.model.input.variables.crsfile:
            return

        filename = os.path.join(self.model.path, self.model.input.variables.crsfile)

        # Read the crs file
        self.gdf = pli2gdf(filename, crs=self.model.crs)

    def write(self, file_name: str | None = None) -> None:
        """Write cross-section polylines to the crs file.

        Parameters
        ----------
        file_name : str, optional
            Override path for the output file.

        Returns
        -------
        None
        """
        if len(self.gdf.index) == 0:
            return

        if not self.model.input.variables.crsfile:
            self.model.input.variables.crsfile = "sfincs.crs"

        filename = os.path.join(self.model.path, self.model.input.variables.crsfile)

        gdf2pli(self.gdf, filename)

    def add(self, cross_section) -> None:
        """Append a cross-section GeoDataFrame to the collection.

        Parameters
        ----------
        cross_section : geopandas.GeoDataFrame
            Cross-section polyline(s) to add.

        Returns
        -------
        None
        """
        cross_section.set_crs(self.model.crs)
        self.gdf = pd.concat([self.gdf, cross_section], ignore_index=True)

    def delete(self, name_or_index: str | int) -> None:
        """Delete a cross-section by name or row index.

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
            print(f"Cross section {name} not found!")
        else:
            index = name_or_index
            if len(self.gdf.index) < index + 1:
                print("Index exceeds length!")
            self.gdf = self.gdf.drop(index).reset_index(drop=True)
            return

    def clear(self) -> None:
        """Remove all cross sections.

        Returns
        -------
        None
        """
        self.gdf = gpd.GeoDataFrame()

    def list_names(self) -> list:
        """Return a list of all cross-section names.

        Returns
        -------
        list[str]
            Cross-section names in GeoDataFrame order.
        """
        names = []
        for index, row in self.gdf.iterrows():
            names.append(row["name"])
        return names
