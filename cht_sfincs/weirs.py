"""SFINCS weir structure stub.

Provides the SfincsWeirs class as a placeholder for future weir support;
read/write methods are not yet implemented.
"""

import os

import geopandas as gpd
import pandas as pd
import shapely


class SfincsWeirs:
    """SFINCS weir structures (stub).

    The read/write methods are not yet implemented; this class is a
    placeholder for future weir support in SFINCS.

    Parameters
    ----------
    sf : SFINCS
        The parent SFINCS model instance.
    """

    def __init__(self, sf: "SFINCS") -> None:
        self.model = sf
        self.gdf = gpd.GeoDataFrame()

    def read(self) -> None:
        """Read weir data (not yet implemented).

        Returns
        -------
        None
        """
        # Not yet implemented
        return

        if not self.model.input.variables.thdfile:
            return

        file_name = os.path.join(self.model.path, self.model.input.variables.thdfile)

        self.gdf = gpd.GeoDataFrame()
        return

        # Read the thd file
        gdf = tek2gdf(file_name, shape="line")
        self.gdf = gdf
        return

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

    def write(self) -> None:
        """Write weir data (not yet implemented).

        Returns
        -------
        None
        """
        return

        if not self.model.input.variables.thdfile:
            return
        if len(self.gdf.index) == 0:
            return

        file_name = os.path.join(self.model.path, self.model.input.variables.thdfile)

        if self.model.crs.is_geographic:
            gdf2tek(self.gdf, file_name)
            # fid = open(file_name, "w")
            # for index, row in self.gdf.iterrows():
            #     x = row["geometry"].coords[0][0]
            #     y = row["geometry"].coords[0][1]
            #     name = row["name"]
            #     string = f'{x:12.6f}{y:12.6f}  "{name}"\n'
            #     fid.write(string)
            # fid.close()
        else:
            with open(file_name, "w") as fid:
                for index, row in self.gdf.iterrows():
                    x = row["geometry"].coords[0][0]
                    y = row["geometry"].coords[0][1]
                    name = row["name"]
                    string = f'{x:12.1f}{y:12.1f}  "{name}"\n'
                    fid.write(string)

    def add(self, thin_dam) -> None:
        """Append a weir GeoDataFrame to the collection.

        Parameters
        ----------
        thin_dam : geopandas.GeoDataFrame
            Weir polyline(s) to add.

        Returns
        -------
        None
        """
        thin_dam.set_crs(self.model.crs)
        self.gdf = pd.concat([self.gdf, thin_dam], ignore_index=True)

    def delete(self, index: int) -> None:
        """Delete a weir by row index.

        Parameters
        ----------
        index : int
            Zero-based row index of the weir to remove.

        Returns
        -------
        None
        """
        if len(self.gdf.index) < index + 1:
            print("Index exceeds length!")
        self.gdf = self.gdf.drop(index).reset_index(drop=True)
        return

    def clear(self) -> None:
        """Remove all weirs.

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
