"""SFINCS output readers for his-file and map-file data.

Provides the SfincsOutput class for reading time series from ``sfincs_his.nc``
and flood maps from ``sfincs_map.nc``.
"""

import os
import warnings

import numpy as np
import pandas as pd
import xarray as xr

np.warnings = warnings


class SfincsOutput:
    """SFINCS output readers for his-file and map-file data.

    Provides methods for reading time series from ``sfincs_his.nc``
    and flood maps / maximum water levels from ``sfincs_map.nc``.

    Parameters
    ----------
    sf : SFINCS
        The parent SFINCS model instance.
    """

    def __init__(self, sf: "SFINCS") -> None:
        self.model = sf

    def read_his_file(
        self,
        station=None,
        file_name: str | None = None,
        ensemble_member=None,
        parameter: str = "point_zs",
        nodata=None,
    ):
        """Read a SFINCS history file and return a DataFrame of time series.

        Parameters
        ----------
        station : str or list[str], optional
            Station name or list of names to extract.  ``None`` returns all
            stations.
        file_name : str, optional
            Path to the his-file.  Defaults to ``<model.path>/sfincs_his.nc``.
        ensemble_member : int, optional
            Ensemble index to extract when the his-file contains ensemble data.
        parameter : str, optional
            Variable name to read.  Common aliases (``"waterlevel"``,
            ``"zs"``, etc.) are resolved automatically.  Defaults to
            ``"point_zs"``.
        nodata : float, optional
            If provided, NaN values are replaced with this value.

        Returns
        -------
        pandas.DataFrame
            Time series DataFrame with station names as columns.
        """

        if parameter == "waterlevel" or parameter == "water_level" or parameter == "zs":
            parameter = "point_zs"
        if parameter == "bedlevel" or parameter == "bed_level" or parameter == "zb":
            parameter = "point_zb"

        # NetCDF output
        if not file_name:
            file_name = os.path.join(self.model.path, "sfincs_his.nc")

        # Open netcdf file
        ddd = xr.open_dataset(file_name)
        stations = ddd.station_name.values
        all_stations = []
        for ist, st in enumerate(stations):
            all_stations.append(st.decode().strip())

        if "time" in ddd[parameter].coords:
            times = ddd[parameter].coords["time"].values

        if station is None:
            # If name_list is empty, add all points
            name_list = []
            for st in all_stations:
                name_list.append(st)
        elif type(station) is str:
            name_list = [station]
        else:
            name_list = station

        if "time" in ddd[parameter].coords:
            df = pd.DataFrame(index=times, columns=name_list)
        else:
            df = pd.DataFrame(columns=name_list)

        for station in name_list:
            for ist, st in enumerate(all_stations):
                if station == st:
                    if ensemble_member is not None:
                        wl = (
                            ddd[parameter]
                            .isel(stations=ist, ensemble=ensemble_member)
                            .values
                        )
                    else:
                        wl = ddd[parameter].isel(stations=ist).values

                    if "time" in ddd[parameter].coords:
                        df[st] = wl
                    else:
                        df[st] = pd.Series([wl])

                    df[st] = wl

                    break

        ddd.close()

        if nodata is not None:
            # Replace NaN values with nodata
            df = df.replace(np.nan, nodata)

        return df

    def read_zsmax(
        self,
        time_range=None,
        zsmax_file: str | None = None,
        output: str = "grid",
        varname: str = "zsmax",
        fmt: str = "numpy",
    ):
        """Read maximum water level data from the map output file.

        Parameters
        ----------
        time_range : list, optional
            ``[t_start, t_end]`` datetime range to extract.  ``None`` uses
            the full output time axis.
        zsmax_file : str, optional
            Path to the map output file.  Auto-detected from model settings.
        output : str, optional
            Output type selector (reserved for future use).
        varname : str, optional
            Variable name in the map file.  Defaults to ``"zsmax"``.
        fmt : str, optional
            Return format: ``"numpy"`` (default) or ``"xarray"``/``"xr"``.

        Returns
        -------
        numpy.ndarray or xarray.DataArray
            Maximum water level array, shape dependent on grid type.
        """
        # Returns xarray data set (not yet) or numpy array with maximum water levels (yet)
        if not zsmax_file:
            if self.model.input.variables.outputformat[0:3] == "net":
                zsmax_file = os.path.join(self.model.path, "sfincs_map.nc")
            else:
                zsmax_file = os.path.join(self.model.path, "zsmax.dat")

        if self.model.input.variables.outputformat[0:3] == "net":
            dsin = xr.open_dataset(zsmax_file)

            output_times = dsin.timemax.values
            if time_range is None:
                t0 = (
                    pd.to_datetime(str(output_times[0]))
                    .replace(tzinfo=None)
                    .to_pydatetime()
                )
                t1 = (
                    pd.to_datetime(str(output_times[-1]))
                    .replace(tzinfo=None)
                    .to_pydatetime()
                )
                time_range = [t0, t1]

            it0 = -1
            for it, time in enumerate(output_times):
                time = pd.to_datetime(str(time)).replace(tzinfo=None).to_pydatetime()
                if time >= time_range[0] and it0 < 0:
                    it0 = it
                if time <= time_range[1]:
                    it1 = it

            if fmt == "xarray" or fmt == "xr":
                # Find out the shape of the data
                if "mesh2d_face_nodes" in dsin:
                    # Quadtree grid
                    zsmax = dsin[varname][it0 : it1 + 1, :].max(dim="timemax")
                else:
                    # Regular grid
                    zsmax = dsin[varname][it0 : it1 + 1, :, :].max(dim="timemax")

                return zsmax

            else:
                # Find out the shape of the data
                if "mesh2d_face_nodes" in dsin:
                    # Quadtree grid
                    zsmax = np.nanmax(dsin[varname].values[it0 : it1 + 1, :], axis=0)
                else:
                    # Regular grid
                    zsmax = np.nanmax(dsin[varname].values[it0 : it1 + 1, :, :], axis=0)

            dsin.close()

            return zsmax

        else:
            # Are we still supporting this ?

            ind_file = os.path.join(
                self.model.path, self.model.input.variables.indexfile
            )

            freqstr = str(self.model.input.dtmaxout) + "S"
            output_times = (
                pd.date_range(
                    start=self.input.variables.tstart,
                    end=self.input.variables.tstop,
                    freq=freqstr,
                )
                .to_pydatetime()
                .tolist()
            )
            nt = len(output_times)

            if time_range is None:
                time_range = [
                    self.model.input.variables.tstart,
                    self.model.input.variables.tstop,
                ]

            for it, time in enumerate(output_times):
                if time <= time_range[0]:
                    it0 = it
                if time <= time_range[1]:
                    it1 = it

            # Get maximum values
            nmax = self.input.variables.nmax
            mmax = self.input.variables.mmax

            # Read sfincs.ind
            data_ind = np.fromfile(ind_file, dtype="i4")
            npoints = data_ind[0]
            data_ind = np.squeeze(data_ind[1:])

            # Read zsmax file
            data_zs = np.fromfile(zsmax_file, dtype="f4")
            data_zs = np.reshape(data_zs, [nt, npoints + 2])[it0 : it1 + 1, 1:-1]
            data_zs = np.amax(data_zs, axis=0)

            if output == "grid":
                zs_da = np.full([nmax * mmax], np.nan)
                zs_da[data_ind - 1] = np.squeeze(data_zs)
                zs_da = np.where(zs_da == -999, np.nan, zs_da)
                zs_da = np.transpose(np.reshape(zs_da, [mmax, nmax]))
                return zs_da
            else:
                return data_zs

    def read_cumulative_precipitation(
        self,
        time_range: list | None = None,
        file_name: str | None = None,
        output: str = "grid",
    ) -> "np.ndarray | None":
        """Read cumulative precipitation from the output file.

        Parameters
        ----------
        time_range : list[datetime], optional
            ``[t_start, t_end]`` window to sum over.  Defaults to the full
            output period.
        file_name : str, optional
            Path to the precipitation file.  Defaults to
            ``<model_path>/cumprcp.dat``.
        output : str, optional
            Output format flag (currently unused).  Defaults to ``"grid"``.

        Returns
        -------
        numpy.ndarray or None
            2-D array of cumulative precipitation totals summed over the
            requested time window, or ``None`` if the format is not NetCDF.
        """
        if not file_name:
            file_name = os.path.join(self.model.path, "cumprcp.dat")

        if self.input.outputformat[0:3] == "net":
            ddd = xr.open_dataset(file_name)

            # freqstr = str(self.input.dtmaxout) + "S"
            # output_times = pd.date_range(start=self.input.tstart,
            #                              end=self.input.tstop,
            #                              freq=freqstr).to_pydatetime().tolist()

            #           output_times = ddd.timemax.values
            #            output_times = ddd.timemax.values.astype(datetime.datetime)
            #            nt = len(output_times)

            output_times = ddd.timemax.values
            if time_range is None:
                t0 = (
                    pd.to_datetime(str(output_times[0]))
                    .replace(tzinfo=None)
                    .to_pydatetime()
                )
                t1 = (
                    pd.to_datetime(str(output_times[-1]))
                    .replace(tzinfo=None)
                    .to_pydatetime()
                )
                time_range = [t0, t1]

            # if time_range is None:
            #     time_range = [self.input.tstart, self.input.tstop]

            for it, time in enumerate(output_times):
                t = pd.to_datetime(str(time)).replace(tzinfo=None).to_pydatetime()
                if t <= time_range[0]:
                    it0 = it
                if t <= time_range[1]:
                    it1 = it

            #            pall = ddd.cumprcp.values[it0:it1,:,:]
            #            psum =
            p = np.transpose(np.sum(ddd.cumprcp.values[it0:it1, :, :], axis=0))

            return p


#     def write_hmax_geotiff(self, dem_file, index_file, hmax_file, time_range=None, zsmax_file=None):

#         no_datavalue = -9999

#         zs_da = self.read_zsmax(time_range=time_range, zsmax_file=zsmax_file)
#         zs_da = 100 * zs_da

#         # Read indices for DEM and resample SFINCS max. water levels on DEM grid
#         dem_ind   = np.fromfile(index_file, dtype="i4")
#         ndem      = dem_ind[0]
#         mdem      = dem_ind[1]
#         indices   = dem_ind[2:]
#         zsmax_dem = np.zeros_like(indices)
#         zsmax_dem = np.where(zsmax_dem == 0, np.nan, 0)
#         valid_indices = np.where(indices > 0)
#         indices = np.where(indices == 0, 1, indices)
#         indices = indices - 1  # correct for python start counting at 0 (matlab at 1)
#         zsmax_dem[valid_indices] = zs_da[indices][valid_indices]
#         zsmax_dem = np.flipud(zsmax_dem.reshape(mdem, ndem).transpose())

#         # Open DEM file
#         dem_ds = gdal.Open(dem_file)
#         band = dem_ds.GetRasterBand(1)
#         dem = band.ReadAsArray()
#         # calculate max. flood depth as difference between water level zs and dem, do not allow for negative values
#         hmax_dem = zsmax_dem - dem  ## just for testing
#         hmax_dem = np.where(hmax_dem < 0, 0, hmax_dem)
#         # set no data value to -9999
#         hmax_dem = np.where(np.isnan(hmax_dem), no_datavalue, hmax_dem)
#         # convert cm to m
#         hmax_dem = hmax_dem/100

#         # write max. flood depth (in m) to geotiff
#         [cols, rows] = dem.shape
#         driver = gdal.GetDriverByName("GTiff")
#         outdata = driver.Create(hmax_file, rows, cols, 1, gdal.GDT_Float32)
#         outdata.SetGeoTransform(dem_ds.GetGeoTransform())  ## sets same geotransform as input
#         outdata.SetProjection(dem_ds.GetProjection())      ## sets same projection as input
#         outdata.GetRasterBand(1).WriteArray(hmax_dem)
#         outdata.GetRasterBand(1).SetNoDataValue(no_datavalue)  ## if you want these values transparent
# #        outdata.SetMetadata({k: str(v) for k, v in scenarioDict.items()})

#         outdata.FlushCache()  ## saves to disk!!
#         outdata = None
#         band = None
#         dem_ds = None
