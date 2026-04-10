"""SFINCS point source/sink utilities (stub).

Provides the SfincsPointSources class for managing point source boundary
conditions; the read/write methods are not yet fully implemented.
"""

import os

import geopandas as gpd
import pandas as pd
import shapely


class SfincsPointSources:
    """SFINCS point source/sink manager (stub).

    Placeholder class for point source boundary conditions; the read/write
    methods are not yet fully implemented.

    Parameters
    ----------
    hw : SFINCS
        The parent SFINCS model instance.
    """

    def __init__(self, hw: "SFINCS") -> None:
        self.model = hw
        self.gdf = gpd.GeoDataFrame()

    def read(self) -> None:
        """Read point source data (not yet implemented).

        Returns
        -------
        None
        """
        # Read in all observation points
        return

        if not self.model.input.variables.obsfile:
            return

        file_name = os.path.join(self.model.path, self.model.input.variables.obsfile)

        # Read the bnd file
        df = pd.read_csv(
            file_name,
            index_col=False,
            header=None,
            delim_whitespace=True,
            names=["x", "y", "name"],
        )

        gdf_list = []
        # Loop through points
        for ind in range(len(df.x.values)):
            name = df.name.values[ind]
            x = df.x.values[ind]
            y = df.y.values[ind]
            point = shapely.geometry.Point(x, y)
            d = {"name": name, "long_name": None, "geometry": point}
            gdf_list.append(d)
        self.gdf = gpd.GeoDataFrame(gdf_list, crs=self.model.crs)

    def write(self, file_name: str | None = None) -> None:
        """Write point source data (not yet implemented).

        Parameters
        ----------
        file_name : str, optional
            Override path for the output file.

        Returns
        -------
        None
        """
        return

        if len(self.gdf.index) == 0:
            return

        if not file_name:
            if not self.model.input.variables.obsfile:
                return
            file_name = os.path.join(
                self.model.path, self.model.input.variables.obsfile
            )

        if self.model.crs.is_geographic:
            with open(file_name, "w") as fid:
                for index, row in self.gdf.iterrows():
                    x = row["geometry"].coords[0][0]
                    y = row["geometry"].coords[0][1]
                    name = row["name"]
                    string = f'{x:12.6f}{y:12.6f}  "{name}"\n'
                    fid.write(string)
        else:
            with open(file_name, "w") as fid:
                for index, row in self.gdf.iterrows():
                    x = row["geometry"].coords[0][0]
                    y = row["geometry"].coords[0][1]
                    name = row["name"]
                    string = f'{x:12.1f}{y:12.1f}  "{name}"\n'
                    fid.write(string)

    def add_point(self, x: float, y: float, name: str) -> None:
        """Add a point source at (x, y).

        Parameters
        ----------
        x : float
            X-coordinate of the point source.
        y : float
            Y-coordinate of the point source.
        name : str
            Name label for the point source.

        Returns
        -------
        None
        """
        point = shapely.geometry.Point(x, y)
        gdf_list = []
        d = {"name": name, "long_name": None, "geometry": point}
        gdf_list.append(d)
        gdf_new = gpd.GeoDataFrame(gdf_list, crs=self.model.crs)
        self.gdf = pd.concat([self.gdf, gdf_new], ignore_index=True)

    def delete_point(self, name_or_index: str | int) -> None:
        """Delete a point source by name or row index.

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
            print(f"Point {name} not found!")
        else:
            index = name_or_index
            if len(self.gdf.index) < index + 1:
                print("Index exceeds length!")
            self.gdf = self.gdf.drop(index).reset_index(drop=True)
            return

    def clear(self) -> None:
        """Remove all point sources.

        Returns
        -------
        None
        """
        self.gdf = gpd.GeoDataFrame()

    def list_observation_points(self) -> list:
        """Return a list of all point source names.

        Returns
        -------
        list[str]
            Point source names in GeoDataFrame order.
        """
        names = []
        for index, row in self.gdf.iterrows():
            names.append(row["name"])
        return names
