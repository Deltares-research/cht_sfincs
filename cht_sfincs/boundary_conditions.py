"""SFINCS boundary condition reader/writer and helper utilities.

Provides the SfincsBoundaryConditions class for managing open-boundary water
level time series (bzs), astronomical constituents (bca), and the boundary
point locations (bnd), together with module-level helper functions for reading
and writing fixed-width formatted timeseries files.
"""

import os

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
import xarray as xr
from cht_tide.tide_predict import predict
from cht_utils.fileio.deltares_ini import IniStruct
from pyproj import Transformer

# from pandas.tseries.offsets import DateOffset
from tabulate import tabulate


class SfincsBoundaryConditions:
    """SFINCS open-boundary conditions manager.

    Manages boundary point locations (bnd file), water level time series
    (bzs file), and astronomical constituent data (bca file).

    Parameters
    ----------
    sf : SFINCS
        The parent SFINCS model instance.
    """

    def __init__(self, sf: "SFINCS") -> None:
        self.model = sf
        self.forcing = "timeseries"
        self.gdf = gpd.GeoDataFrame()
        self.times = []

    def read(self) -> None:
        """Read SFINCS boundary conditions (*.bnd, *.bzs, *.bca) files.

        Returns
        -------
        None
        """
        # Read in all boundary data
        self.read_boundary_points()
        self.read_boundary_conditions_timeseries()
        if self.model.input.variables.bcafile:
            self.read_boundary_conditions_astro()

    def write(self) -> None:
        """Write all boundary condition files.

        Returns
        -------
        None
        """
        # Write all boundary data
        self.write_boundary_points()
        self.write_boundary_conditions_timeseries()
        # # Alternatively, write netcdf file
        # self.write_to_netcdf()

    def read_boundary_points(self) -> None:
        """Read boundary point locations from the bnd file.

        Returns
        -------
        None
        """
        # Read bnd file
        if not self.model.input.variables.bndfile:
            return

        file_name = os.path.join(self.model.path, self.model.input.variables.bndfile)

        if not os.path.exists(file_name):
            print(f"Warning! File {file_name} does not exist!")
            return

        # Read the bnd file
        df = pd.read_csv(
            file_name, index_col=False, header=None, names=["x", "y"], sep=r"\s+"
        )

        gdf_list = []
        # Loop through points
        for ind in range(len(df.x.values)):
            name = str(ind + 1).zfill(4)
            x = df.x.values[ind]
            y = df.y.values[ind]
            point = shapely.geometry.Point(x, y)
            d = {
                "name": name,
                "timeseries": pd.DataFrame(),
                "astro": pd.DataFrame(),
                "geometry": point,
            }
            gdf_list.append(d)
        self.gdf = gpd.GeoDataFrame(gdf_list, crs=self.model.crs)

    def write_boundary_points(self) -> None:
        # Write bnd file

        if len(self.gdf.index) == 0:
            return

        if not self.model.input.variables.bndfile:
            self.model.input.variables.bndfile = "sfincs.bnd"

        file_name = os.path.join(self.model.path, self.model.input.variables.bndfile)

        if self.model.crs.is_geographic:
            with open(file_name, "w") as fid:
                for index, row in self.gdf.iterrows():
                    x = row["geometry"].coords[0][0]
                    y = row["geometry"].coords[0][1]
                    string = f"{x:12.6f}{y:12.6f}\n"
                    fid.write(string)
        else:
            with open(file_name, "w") as fid:
                for index, row in self.gdf.iterrows():
                    x = row["geometry"].coords[0][0]
                    y = row["geometry"].coords[0][1]
                    string = f"{x:12.1f}{y:12.1f}\n"
                    fid.write(string)

    def add_point(self, x: float, y: float, wl: float) -> None:
        """Add a boundary point at (x, y) with constant water level *wl*.

        Parameters
        ----------
        x : float
            X-coordinate of the boundary point.
        y : float
            Y-coordinate of the boundary point.
        wl : float
            Constant water level (m) to initialise the time series.

        Returns
        -------
        None
        """
        # Add point

        nrp = len(self.gdf.index)
        name = str(nrp + 1).zfill(4)
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

        wl = [wl] * nt

        df["time"] = time
        df["wl"] = wl
        df = df.set_index("time")

        gdf_list = []
        d = {"name": name, "timeseries": df, "astro": pd.DataFrame(), "geometry": point}
        gdf_list.append(d)
        gdf_new = gpd.GeoDataFrame(gdf_list, crs=self.model.crs)
        self.gdf = pd.concat([self.gdf, gdf_new], ignore_index=True)

    def delete_point(self, index: int) -> None:
        """Delete a boundary point by row index.

        Parameters
        ----------
        index : int
            Zero-based row index of the point to remove.

        Returns
        -------
        None
        """
        # Delete boundary point by index
        if len(self.gdf.index) == 0:
            return
        if index < len(self.gdf.index):
            self.gdf = self.gdf.drop(index).reset_index(drop=True)

    def clear(self) -> None:
        """Remove all boundary points and time series data.

        Returns
        -------
        None
        """
        self.gdf = gpd.GeoDataFrame()

    def read_boundary_conditions_timeseries(self) -> None:
        """Read water level time series from the bzs file.

        Returns
        -------
        None
        """
        # Read SFINCS bzs file

        if not self.model.input.variables.bzsfile:
            return
        if len(self.gdf.index) == 0:
            return

        tref = self.model.input.variables.tref

        # Time

        # WL
        file_name = os.path.join(self.model.path, self.model.input.variables.bzsfile)

        # Check if file exists
        if not os.path.exists(file_name):
            print(f"Warning! File {file_name} does not exist!")
            return

        dffile = read_timeseries_file(file_name, tref)

        # Loop through boundary points
        for ip, point in self.gdf.iterrows():
            point["timeseries"]["time"] = dffile.index
            point["timeseries"]["wl"] = dffile.iloc[:, ip].values
            point["timeseries"].set_index("time", inplace=True)

    def check_times(self) -> tuple:
        """Check that boundary forcing covers the full simulation period.

        Returns
        -------
        tuple
            ``(True, "")`` if the times are valid, or
            ``(False, message)`` with a descriptive warning message.
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
                    "Boundary forcing does not fully cover simulation time. Please consider extending the boundary forcing.",
                )
        return True, ""

    def write_boundary_conditions_timeseries(self) -> None:
        """Write water level time series to the bzs file.

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

        # First get times from the first point (times in other points should be identical)
        time = self.gdf.loc[0]["timeseries"].index
        tref = self.model.input.variables.tref
        dt = (time - tref).total_seconds()

        # WL
        if not self.model.input.variables.bzsfile:
            self.model.input.variables.bzsfile = "sfincs.bzs"
        file_name = os.path.join(self.model.path, self.model.input.variables.bzsfile)
        # Build a new DataFrame
        df = pd.DataFrame()
        for ip, point in self.gdf.iterrows():
            df = pd.concat([df, point["timeseries"]["wl"]], axis=1)
        df.index = dt
        # df.to_csv(file_name,
        #           index=True,
        #           sep=" ",
        #           header=False,
        #           float_format="%.3f")
        to_fwf(df, file_name)

    def write_to_netcdf(self) -> None:
        """Write boundary conditions to a NetCDF file.

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
        self.model.input.variables.netbndbzsbzifile = "sfincs_boundary_conditions.nc"
        file_name = os.path.join(
            self.model.path, self.model.input.variables.netbndbzsbzifile
        )

        nrp = len(self.gdf.index)
        times = self.gdf.loc[0]["timeseries"].index
        nt = len(times)
        float_minutes = (
            (times - np.datetime64("1970-01-01T00:00:00")) / pd.Timedelta(seconds=60)
        ).to_numpy()
        x = np.empty(nrp, dtype=float)
        y = np.empty(nrp, dtype=float)
        zs = np.empty((nt, nrp), dtype=float)

        # Loop through boundary points and obtain data from timeseries
        for ip, point in self.gdf.iterrows():
            x[ip] = point["geometry"].x
            y[ip] = point["geometry"].y
            df = point["timeseries"]
            wl = df["wl"].values
            zs[:, ip] = wl

        # Make XArray Dataset
        ds = xr.Dataset()

        ds["time"] = xr.DataArray(float_minutes, dims=["time"])
        ds["time"].attrs["units"] = "minutes since 1970-01-01 00:00:00.0 +0000"
        ds["x"] = xr.DataArray(x, dims=["stations"])
        ds["y"] = xr.DataArray(y, dims=["stations"])
        ds["zs"] = xr.DataArray(
            zs,
            dims=["time", "stations"],
            attrs={"units": "m", "long_name": "water level at boundary points"},
        )

        # Add attributes
        ds.attrs["description"] = "SFINCS boundary conditions"

        # Write netcdf file
        ds.to_netcdf(file_name, mode="w", format="NETCDF4", engine="netcdf4")

    def read_boundary_conditions_astro(self) -> None:
        """Read astronomical constituent data from the bca file.

        Returns
        -------
        None
        """
        if len(self.gdf.index) == 0:
            # No boundary points
            return

        # if len(self.gdf.loc[0]["timeseries"].index) == 0:
        #     # No time series data
        #     return

        if not self.model.input.variables.bcafile:
            self.model.input.variables.bcafile = "sfincs.bca"
        file_name = os.path.join(self.model.path, self.model.input.variables.bcafile)

        d = IniStruct(filename=file_name)
        # Loop through boundary points
        for ip, point in self.gdf.iterrows():
            # Set data in row of gdf
            self.gdf.at[ip, "astro"] = d.section[ip].data

    def write_boundary_conditions_astro(self) -> None:
        """Write astronomical constituent data to the bca file.

        Returns
        -------
        None
        """
        if not self.model.input.variables.bcafile:
            # No file name
            return

        if len(self.gdf.index) == 0:
            # No points
            return

        # WL
        filename = os.path.join(self.model.path, self.model.input.variables.bcafile)

        with open(filename, "w") as fid:
            for ip, point in self.gdf.iterrows():
                astro = point["astro"]
                # name is like "sfincs_0001"
                name = f"sfincs_{ip + 1:04d}"
                fid.write("[forcing]\n")
                fid.write(f"Name                            = {name}\n")
                fid.write("Function                        = astronomic\n")
                fid.write("Quantity                        = astronomic component\n")
                fid.write("Unit                            = -\n")
                fid.write("Quantity                        = waterlevelbnd amplitude\n")
                fid.write("Unit                            = m\n")
                fid.write("Quantity                        = waterlevelbnd phase\n")
                fid.write("Unit                            = deg\n")
            for constituent, row in astro.iterrows():
                fid.write(
                    f"{constituent:6s}{row['amplitude']:10.5f}{row['phase']:10.2f}\n"
                )
                fid.write("\n")

    def generate_bzs_from_bca(
        self,
        dt: float = 600.0,
        offset: float = 0.0,
        write_file: bool = True,
    ) -> None:
        """Generate water level time series from astronomical constituents.

        Parameters
        ----------
        dt : float, optional
            Time step in seconds. Defaults to ``600.0``.
        offset : float, optional
            Constant offset (m) added to the predicted tide. Defaults to ``0.0``.
        write_file : bool, optional
            If ``True``, write the generated time series to the bzs file.
            Defaults to ``True``.

        Returns
        -------
        None
        """
        if len(self.gdf.index) == 0:
            return

        if not self.model.input.variables.bzsfile:
            self.model.input.variables.bzsfile = "sfincs.bzs"

        times = pd.date_range(
            start=self.model.input.variables.tstart,
            end=self.model.input.variables.tstop,
            freq=pd.tseries.offsets.DateOffset(seconds=dt),
        )

        # Make boundary conditions based on bca file
        for icol, point in self.gdf.iterrows():
            v = predict(point.astro, times) + offset
            ts = pd.Series(v, index=times)
            # Convert this pandas series to a DataFrame
            df = pd.DataFrame()
            df["time"] = ts.index
            df["wl"] = ts.values
            df = df.set_index("time")
            self.gdf.at[icol, "timeseries"] = df

        if write_file:
            self.write_boundary_conditions_timeseries()

    def get_boundary_points_from_mask(
        self,
        min_dist: float | None = None,
        bnd_dist: float = 5000.0,
    ) -> None:
        """Derive boundary points from the grid mask open-boundary cells.

        Parameters
        ----------
        min_dist : float, optional
            Minimum distance (m) between consecutive boundary points along a
            polyline.  Defaults to ``2 * dx`` of the grid.
        bnd_dist : float, optional
            Target spacing (m) between boundary condition points placed along
            each open-boundary polyline.  Defaults to ``5000.0``.

        Returns
        -------
        None
        """

        if min_dist is None:
            # Set minimum distance between to grid boundary points on polyline to 2 * dx
            min_dist = self.model.grid.data.attrs["dx"] * 2

        mask = self.model.grid.data["mask"]
        ibnd = np.where(mask == 2)
        xz, yz = self.model.grid.face_coordinates()
        xp = xz[ibnd]
        yp = yz[ibnd]

        # Make boolean array for points that are include in a polyline
        used = np.full(xp.shape, False, dtype=bool)

        # Make list of polylines. Each polyline is a list of indices of boundary points.
        polylines = []

        while True:
            if np.all(used):
                # All boundary grid points have been used. We can stop now.
                break

            # Find first the unused points
            i1 = np.where(~used)[0][0]

            # Set this point to used
            used[i1] = True

            # Start new polyline with index i1
            polyline = [i1]

            while True:
                # Compute distances to all points that have not been used
                xpunused = xp[~used]
                ypunused = yp[~used]
                # Get all indices of unused points
                unused_indices = np.where(~used)[0]

                dst = np.sqrt((xpunused - xp[i1]) ** 2 + (ypunused - yp[i1]) ** 2)
                if np.all(np.isnan(dst)):
                    break
                inear = np.nanargmin(dst)
                inearall = unused_indices[inear]
                if dst[inear] < min_dist:
                    # Found next point along polyline
                    polyline.append(inearall)
                    used[inearall] = True
                    i1 = inearall
                else:
                    # Last point found
                    break

            # Now work the other way
            # Start with first point of polyline
            i1 = polyline[0]
            while True:
                if np.all(used):
                    # All boundary grid points have been used. We can stop now.
                    break
                # Now we go in the other direction
                xpunused = xp[~used]
                ypunused = yp[~used]
                unused_indices = np.where(~used)[0]
                dst = np.sqrt((xpunused - xp[i1]) ** 2 + (ypunused - yp[i1]) ** 2)
                inear = np.nanargmin(dst)
                inearall = unused_indices[inear]
                if dst[inear] < min_dist:
                    # Found next point along polyline
                    polyline.insert(0, inearall)
                    used[inearall] = True
                    # Set index of next point
                    i1 = inearall
                else:
                    # Last nearby point found
                    break

            if len(polyline) > 1:
                polylines.append(polyline)

        gdf_list = []
        ip = 0
        # Transform to web mercator to get distance in metres
        if self.model.crs.is_geographic:
            transformer = Transformer.from_crs(self.model.crs, 3857, always_xy=True)
        # Loop through polylines
        for polyline in polylines:
            x = xp[polyline]
            y = yp[polyline]
            points = [(x, y) for x, y in zip(x.ravel(), y.ravel())]
            line = shapely.geometry.LineString(points)
            if self.model.crs.is_geographic:
                # Line in web mercator (to get length in metres)
                xm, ym = transformer.transform(x, y)
                pointsm = [(xm, ym) for xm, ym in zip(xm.ravel(), ym.ravel())]
                linem = shapely.geometry.LineString(pointsm)
                num_points = int(linem.length / bnd_dist) + 2
            else:
                num_points = int(line.length / bnd_dist) + 2
            # Interpolate to new points
            new_points = [
                line.interpolate(i / float(num_points - 1), normalized=True)
                for i in range(num_points)
            ]
            # Loop through points in polyline
            for point in new_points:
                name = str(ip + 1).zfill(4)
                d = {
                    "name": name,
                    "timeseries": pd.DataFrame(),
                    "astro": pd.DataFrame(),
                    "geometry": point,
                }
                gdf_list.append(d)
                ip += 1

        self.gdf = gpd.GeoDataFrame(gdf_list, crs=self.model.crs)

    def set_timeseries(
        self,
        shape: str = "constant",
        timestep: float = 600.0,
        offset: float = 0.0,
        amplitude: float = 1.0,
        phase: float = 0.0,
        period: float = 43200.0,
        peak: float = 1.0,
        tpeak: float = 86400.0,
        duration: float = 43200.0,
    ) -> None:
        """Set synthetic water level time series for all boundary points.

        Parameters
        ----------
        shape : str, optional
            Waveform shape: ``"constant"``, ``"sine"``, ``"gaussian"``, or
            ``"astronomical"``. Defaults to ``"constant"``.
        timestep : float, optional
            Time step in seconds. Defaults to ``600.0``.
        offset : float, optional
            Constant offset or mean water level (m). Defaults to ``0.0``.
        amplitude : float, optional
            Wave amplitude (m); used for ``"sine"`` shape. Defaults to ``1.0``.
        phase : float, optional
            Phase angle (degrees); used for ``"sine"`` shape. Defaults to ``0.0``.
        period : float, optional
            Wave period (s); used for ``"sine"`` shape. Defaults to ``43200.0``.
        peak : float, optional
            Peak water level (m); used for ``"gaussian"`` shape. Defaults to ``1.0``.
        tpeak : float, optional
            Time of peak (s since tref); used for ``"gaussian"`` shape.
            Defaults to ``86400.0``.
        duration : float, optional
            Duration (s) of the Gaussian pulse; used for ``"gaussian"`` shape.
            Defaults to ``43200.0``.

        Returns
        -------
        None
        """
        # Applies time series boundary conditions for each point
        # Create numpy datetime64 array for time series with python datetime.datetime objects

        if shape == "astronomical":
            # Use existing method
            self.generate_bzs_from_bca(dt=timestep, offset=offset, write_file=False)
            return

        t0 = np.datetime64(self.model.input.variables.tstart)
        t1 = np.datetime64(self.model.input.variables.tstop)
        if shape == "constant":
            dt = np.timedelta64(int((t1 - t0).astype(float) / 1e6), "s")
        else:
            dt = np.timedelta64(int(timestep), "s")
        time = np.arange(t0, t1 + dt, dt)
        dtsec = dt.astype(float)
        # Convert time to seconds since tref
        tsec = (
            (time - np.datetime64(self.model.input.variables.tref))
            .astype("timedelta64[s]")
            .astype(float)
        )
        nt = len(tsec)
        if shape == "constant":
            wl = [offset] * nt
        elif shape == "sine":
            wl = offset + amplitude * np.sin(
                2 * np.pi * tsec / period + phase * np.pi / 180
            )
        elif shape == "gaussian":
            wl = offset + peak * np.exp(-(((tsec - tpeak) / (0.25 * duration)) ** 2))
        elif shape == "astronomical":
            # Not implemented
            return

        times = pd.date_range(
            start=self.model.input.variables.tstart,
            end=self.model.input.variables.tstop,
            freq=pd.tseries.offsets.DateOffset(seconds=dtsec),
        )

        for index, point in self.gdf.iterrows():
            df = pd.DataFrame()
            df["time"] = times
            df["wl"] = wl
            df = df.set_index("time")
            self.gdf.at[index, "timeseries"] = df


def read_timeseries_file(file_name: str, ref_date) -> pd.DataFrame:
    """Read a space-separated timeseries file into a DataFrame.

    Parameters
    ----------
    file_name : str
        Path to the timeseries file. The first column contains time in seconds
        since *ref_date*; remaining columns are data values.
    ref_date : datetime.datetime
        Reference date used to convert the elapsed-seconds index to absolute
        timestamps.

    Returns
    -------
    pandas.DataFrame
        DataFrame indexed by absolute timestamps with one column per data
        channel.
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
        Data to write.  The index is included as the first column.
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
