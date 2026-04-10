"""SFINCS discharge (source) point reader/writer and helper utilities.

Provides the SfincsDischargePoints class for managing point source locations
(src file) and associated discharge time series (dis file), together with
module-level helper functions for reading and writing timeseries files.
"""

import os

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
import xarray as xr
from tabulate import tabulate


class SfincsDischargePoints:
    """SFINCS source/discharge points.

    Manages point source locations (src file) and associated discharge
    time series (dis file).

    Parameters
    ----------
    hw : SFINCS
        The parent SFINCS model instance.
    """

    def __init__(self, hw: "SFINCS") -> None:
        self.model = hw
        self.gdf = gpd.GeoDataFrame()

    def read(self) -> None:
        """Read both src and dis files.

        Returns
        -------
        None
        """
        # Read in both disc points
        self.read_src()
        self.read_dis()

    def read_src(self) -> None:
        """Read source point locations from the src file.

        Returns
        -------
        None
        """
        if not self.model.input.variables.srcfile:
            return

        filename = os.path.join(self.model.path, self.model.input.variables.srcfile)

        # Read the bnd file
        df = pd.read_csv(
            filename, index_col=False, header=None, sep=r"\s+", names=["x", "y", "name"]
        )

        self.gdf = gpd.GeoDataFrame()
        # Loop through points to add them to the gdf
        for ind in range(len(df.x.values)):
            # name = df.name.values[ind]
            name = str(ind + 1)
            x = df.x.values[ind]
            y = df.y.values[ind]
            self.add_point(x, y, name, q=0.0)

    def read_dis(self) -> None:
        """Read discharge time series from the dis file.

        Returns
        -------
        None
        """
        # Check that gdf is not empty
        if len(self.gdf.index) == 0:
            # No points defined
            return
        if not self.model.input.variables.disfile:
            return
        filename = os.path.join(self.model.path, self.model.input.variables.disfile)
        # Check if file exists
        if not os.path.exists(filename):
            print(f"Warning! File {filename} does not exist!")
            return

        # Time
        tref = self.model.input.variables.tref

        dffile = read_timeseries_file(filename, tref)
        times = dffile.index

        # Loop through boundary points
        for ip, point in self.gdf.iterrows():
            point["timeseries"] = pd.DataFrame()
            point["timeseries"]["time"] = times
            point["timeseries"]["q"] = dffile.iloc[:, ip].values
            point["timeseries"].set_index("time", inplace=True)

    def write(self) -> None:
        """Write both src and dis files.

        Returns
        -------
        None
        """
        # Read in both disc points
        self.write_src()
        self.write_dis()

    def write_src(self, filename: str | None = None) -> None:
        """Write source point locations to the src file.

        Parameters
        ----------
        filename : str, optional
            Override output path.

        Returns
        -------
        None
        """

        if len(self.gdf.index) == 0:
            # No points defined
            return

        if not filename:
            # File name not provided
            if not self.model.input.variables.srcfile:
                # And it is not in the input file, so set it now
                self.model.input.variables.srcfile = "sfincs.src"
            filename = os.path.join(self.model.path, self.model.input.variables.srcfile)

        if self.model.crs.is_geographic:
            with open(filename, "w") as fid:
                for index, row in self.gdf.iterrows():
                    x = row["geometry"].coords[0][0]
                    y = row["geometry"].coords[0][1]
                    name = row["name"]
                    string = f'{x:12.6f}{y:12.6f}  "{name}"\n'
                    fid.write(string)
        else:
            with open(filename, "w") as fid:
                for index, row in self.gdf.iterrows():
                    x = row["geometry"].coords[0][0]
                    y = row["geometry"].coords[0][1]
                    name = row["name"]
                    string = f'{x:12.1f}{y:12.1f}  "{name}"\n'
                    fid.write(string)

    def write_dis(self, filename: str | None = None) -> None:
        """Write discharge time series to the dis file.

        Parameters
        ----------
        filename : str, optional
            Override output path.

        Returns
        -------
        None
        """

        if len(self.gdf.index) == 0:
            # No points defined
            return
        if len(self.gdf.loc[0]["timeseries"].index) == 0:
            # No time series data
            return

        if not filename:
            # File name not provided
            if not self.model.input.variables.disfile:
                # And it is not in the input file, so set it now
                self.model.input.variables.disfile = "sfincs.dis"
            filename = os.path.join(self.model.path, self.model.input.variables.disfile)

        # First get times from the first point (times in other points should be identical)
        time = self.gdf.loc[0]["timeseries"].index
        tref = self.model.input.variables.tref
        dt = (time - tref).total_seconds()

        # Build a new DataFrame
        df = pd.DataFrame()
        for ip, point in self.gdf.iterrows():
            df = pd.concat([df, point["timeseries"]["q"]], axis=1)
        df.index = dt
        to_fwf(df, filename)

    def write_to_netcdf(self) -> None:
        """Write discharge point data to a NetCDF file.

        Returns
        -------
        None
        """
        if len(self.gdf.index) == 0:
            # No boundary points
            return

        if len(self.gdf.loc[0]["timeseries"].index) == 0:
            # No time series data
            return

        # if not self.model.input.variables.netbndbzsbzifile:
        #     self.model.input.variables.netbndbzsbzifile = "sfincs_boundary_conditions.nc"
        self.model.input.variables.netsrcdisfile = "sfincs_discharges.nc"
        file_name = os.path.join(
            self.model.path, self.model.input.variables.netsrcdisfile
        )

        nrp = len(self.gdf.index)
        times = self.gdf.loc[0]["timeseries"].index
        nt = len(times)
        float_minutes = (
            (times - np.datetime64("1970-01-01T00:00:00")) / pd.Timedelta(seconds=60)
        ).to_numpy()
        x = np.empty(nrp, dtype=float)
        y = np.empty(nrp, dtype=float)
        q = np.empty((nt, nrp), dtype=float)

        # Loop through boundary points and obtain data from timeseries
        for ip, point in self.gdf.iterrows():
            x[ip] = point["geometry"].x
            y[ip] = point["geometry"].y
            df = point["timeseries"]
            q[:, ip] = df["q"].values

        # Make XArray Dataset
        ds = xr.Dataset()

        ds["time"] = xr.DataArray(float_minutes, dims=["time"])
        ds["time"].attrs["units"] = "minutes since 1970-01-01 00:00:00.0 +0000"
        ds["x"] = xr.DataArray(x, dims=["stations"])
        ds["y"] = xr.DataArray(y, dims=["stations"])
        ds["discharge"] = xr.DataArray(
            q,
            dims=["time", "stations"],
            attrs={"units": "m3 s-1", "long_name": "discharge at source points"},
        )

        # Add attributes
        ds.attrs["description"] = "SFINCS discharge points"

        # Write netcdf file
        ds.to_netcdf(file_name, mode="w", format="NETCDF4", engine="netcdf4")

    def add_point(self, x: float, y: float, name: str, q: float = 0.0) -> None:
        """Add a discharge point at (x, y) with constant discharge *q*.

        Parameters
        ----------
        x : float
            X-coordinate of the discharge point.
        y : float
            Y-coordinate of the discharge point.
        name : str
            Name label for the discharge point.
        q : float, optional
            Constant discharge (m3/s).  Defaults to ``0.0``.

        Returns
        -------
        None
        """
        point = shapely.geometry.Point(x, y)
        df = pd.DataFrame()

        new = True
        if len(self.gdf.index) > 0:
            new = False

        if new:
            # Start and stop time
            time = [self.model.input.variables.tstart, self.model.input.variables.tstop]
        else:
            # Get times from first point
            time = self.gdf.loc[0]["timeseries"].index
        nt = len(time)

        q = [q] * nt

        df["time"] = time
        df["q"] = q
        df = df.set_index("time")

        gdf_list = []
        d = {"name": name, "timeseries": df, "geometry": point}
        gdf_list.append(d)
        gdf_new = gpd.GeoDataFrame(gdf_list, crs=self.model.crs)
        self.gdf = pd.concat([self.gdf, gdf_new], ignore_index=True)

    def delete_point(self, name_or_index: str | int) -> None:
        """Delete a discharge point by name or row index.

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
        """Remove all discharge points.

        Returns
        -------
        None
        """
        self.gdf = gpd.GeoDataFrame()

    def list_discharge_points(self) -> list:
        """Return a list of all discharge point names.

        Returns
        -------
        list[str]
            Discharge point names in GeoDataFrame order.
        """
        names = []
        for index, row in self.gdf.iterrows():
            names.append(row["name"])
        return names

    def check_times(self) -> tuple:
        """Check that discharge forcing covers the full simulation period.

        Returns
        -------
        tuple
            ``(True, "")`` if times are valid, or ``(False, message)`` with a
            descriptive warning message.
        """
        t0_model = self.model.input.variables.tstart
        t1_model = self.model.input.variables.tstop
        # Boundary conditions
        if len(self.gdf) > 0:
            # Get first and last time of first bc point
            df = self.gdf.loc[0]["timeseries"]
            t0_bc = df.index[0]
            t1_bc = df.index[-1]
            if t0_bc > t0_model or t1_bc < t1_model:
                return (
                    False,
                    "Discharge data does not fully cover simulation time. Please consider extending the discharge time series.",
                )
        return True, ""


def read_timeseries_file(file_name: str, ref_date) -> pd.DataFrame:
    """Read a space-separated timeseries file into a DataFrame.

    Parameters
    ----------
    file_name : str
        Path to the file. First column is time in seconds since *ref_date*;
        remaining columns are data values.
    ref_date : datetime.datetime
        Reference date used to convert elapsed seconds to absolute timestamps.

    Returns
    -------
    pandas.DataFrame
        DataFrame indexed by absolute timestamps.
    """
    df = pd.read_csv(file_name, index_col=0, header=None, sep=r"\s+")
    ts = ref_date + pd.to_timedelta(df.index, unit="s")
    df.index = ts
    return df


def to_fwf(df: pd.DataFrame, fname: str, floatfmt: str = ".3f") -> None:
    """Write a DataFrame to a fixed-width formatted text file.

    Parameters
    ----------
    df : pandas.DataFrame
        Data to write. The index is prepended as the first column.
    fname : str
        Output file path.
    floatfmt : str, optional
        Float format string passed to :func:`tabulate`. Defaults to ``".3f"``.

    Returns
    -------
    None
    """
    indx = df.index.tolist()
    vals = df.values.tolist()
    for it, t in enumerate(vals):
        t.insert(0, indx[it])
    content = tabulate(vals, [], tablefmt="plain", floatfmt=floatfmt)
    with open(fname, "w") as f:
        f.write(content)
