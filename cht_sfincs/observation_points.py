"""SFINCS observation point reader/writer.

Provides the SfincsObservationPoints class for managing named observation
point locations written to the obs file; SFINCS writes time series output
at each active observation point.
"""

import os

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from matplotlib import path


class SfincsObservationPoints:
    """SFINCS output observation points.

    Manages named observation point locations written to the obs file;
    SFINCS writes time series output at each active observation point.

    Parameters
    ----------
    hw : SFINCS
        The parent SFINCS model instance.
    """

    def __init__(self, hw: "SFINCS") -> None:
        self.model = hw
        self.gdf = gpd.GeoDataFrame()

    def read(self) -> None:
        """Read observation point locations from the obs file.

        Returns
        -------
        None
        """
        # Read in all observation points

        if not self.model.input.variables.obsfile:
            return

        file_name = os.path.join(self.model.path, self.model.input.variables.obsfile)

        # Read the bnd file
        df = pd.read_csv(
            file_name,
            index_col=False,
            header=None,
            sep=r"\s+",
            names=["x", "y", "name"],
        )

        # there are any floats or ints in the name column, convert them to str
        df["name"] = df["name"].astype(str)

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
        """Write observation point locations to the obs file.

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

        if not file_name:
            if not self.model.input.variables.obsfile:
                self.model.input.variables.obsfile = "sfincs.obs"
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
                    string = f'{x:12.1f}{y:12.2f}  "{name}"\n'
                    fid.write(string)

    def add_point(self, x: float, y: float, name: str) -> None:
        """Add an observation point at (x, y).

        Parameters
        ----------
        x : float
            X-coordinate of the new observation point.
        y : float
            Y-coordinate of the new observation point.
        name : str
            Name label for the observation point.

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

    def add_points(self, gdf, name: str = "name") -> None:
        """Add multiple observation points from a GeoDataFrame, clipped to the model domain.

        Parameters
        ----------
        gdf : geopandas.GeoDataFrame
            Points to add; re-projected to the model CRS internally.
        name : str, optional
            Column in *gdf* used as the point name.  Defaults to ``"name"``.

        Returns
        -------
        None
        """
        exterior = self.model.grid.exterior.unary_union
        gdf = gdf.to_crs(self.model.crs)
        x = np.empty((len(gdf)))
        y = np.empty((len(gdf)))
        for index, row in gdf.iterrows():
            x[index] = row["geometry"].coords[0][0]
            y[index] = row["geometry"].coords[0][1]
        inpol = inpolygon(x, y, exterior)
        gdf_list = []
        for index, row in gdf.iterrows():
            if inpol[index]:
                d = {
                    "name": row[name],
                    "long_name": None,
                    "geometry": shapely.geometry.Point(x[index], y[index]),
                }
                gdf_list.append(d)
        gdf_new = gpd.GeoDataFrame(gdf_list, crs=self.model.crs)
        self.gdf = pd.concat([self.gdf, gdf_new], ignore_index=True)

    def delete_point(self, name_or_index: str | int) -> None:
        """Delete an observation point by name or row index.

        Parameters
        ----------
        name_or_index : str or int
            Name string or zero-based integer row index of the point to remove.

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
        """Remove all observation points.

        Returns
        -------
        None
        """
        self.gdf = gpd.GeoDataFrame()

    def list_names(self) -> list:
        """Return a list of all observation point names.

        Returns
        -------
        list[str]
            Observation point names in GeoDataFrame order.
        """
        names = []
        for index, row in self.gdf.iterrows():
            names.append(row["name"])
        return names


def inpolygon(xq: np.ndarray, yq: np.ndarray, p) -> np.ndarray:
    """Test which query points lie inside a Shapely polygon or multipolygon.

    Parameters
    ----------
    xq : numpy.ndarray
        X-coordinates of the query points.
    yq : numpy.ndarray
        Y-coordinates of the query points.
    p : shapely.geometry.Polygon or shapely.geometry.MultiPolygon
        Polygon to test against.

    Returns
    -------
    numpy.ndarray
        Boolean array with the same shape as *xq*; ``True`` where the point
        is inside *p*.
    """
    shape = xq.shape
    xq = xq.reshape(-1)
    yq = yq.reshape(-1)
    q = [(xq[i], yq[i]) for i in range(xq.shape[0])]
    # Check if p is a polygon or a multipolygon
    if isinstance(p, shapely.geometry.MultiPolygon):
        # Loop through each polygon in the multipolygon
        inpol = np.zeros((len(q),), dtype=bool)
        for poly in p.geoms:
            p = path.Path(
                [(crds[0], crds[1]) for i, crds in enumerate(poly.exterior.coords)]
            )
            inpol |= p.contains_points(q).reshape(shape)
    else:
        p = path.Path([(crds[0], crds[1]) for i, crds in enumerate(p.exterior.coords)])
        inpol = p.contains_points(q).reshape(shape)

    return inpol
