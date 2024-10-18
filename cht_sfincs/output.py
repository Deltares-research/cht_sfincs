# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 17:24:49 2022

@author: ormondt
"""
import os
import numpy as np
import pandas as pd
import xarray as xr
import warnings
np.warnings = warnings

class SfincsOutput:

    def __init__(self, sf):
        self.model = sf

    def read_his_file(self,
                    station=None,
                    file_name=None,
                    ensemble_member=None,
                    parameter="point_zs",
                    nodata=None):
        """Reads a SFINCS history file and returns a DataFrame with timeseries"""
                    
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
            times   = ddd[parameter].coords["time"].values

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
                        wl = ddd[parameter].isel(stations=ist, ensemble=ensemble_member).values
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

    def read_zsmax(self, time_range=None, zsmax_file=None, output="grid", varname="zsmax"):
        # Returns xarray data set (not yet) or numpy array with maximum water levels (yet)    
        if not zsmax_file:
            if self.model.input.outputformat[0:3] == "net":
                zsmax_file = os.path.join(self.model.path, "sfincs_map.nc")
            else:
                zsmax_file = os.path.join(self.model.path, "zsmax.dat")
            

        if self.model.input.variables.outputformat[0:3] == "net":

            dsin = xr.open_dataset(zsmax_file)

            output_times = dsin.timemax.values
            if time_range is None:
                t0 = pd.to_datetime(str(output_times[0])).replace(tzinfo=None).to_pydatetime()
                t1 = pd.to_datetime(str(output_times[-1])).replace(tzinfo=None).to_pydatetime()
                time_range = [t0, t1]

            it0 = -1
            for it, time in enumerate(output_times):
                time = pd.to_datetime(str(time)).replace(tzinfo=None).to_pydatetime()
                if time>=time_range[0] and it0<0:
                    it0 = it
                if time<=time_range[1]:
                    it1 = it

            # Find out the shape of the data
            # If dsin[varname].shape is (nt, ny, nx) then we have a regular grid
            # If dsin[varname].shape is (nt, n) then we have a quadtree grid
            if len(dsin[varname].shape) == 3:
                zsmax = np.nanmax(dsin[varname].values[it0:it1 + 1,:,:], axis=0)
            else:
                zsmax = np.nanmax(dsin[varname].values[it0:it1 + 1,:], axis=0)
            # if self.model.input.variables.qtrfile:
            #     zsmax = np.nanmax(dsin[varname].values[it0:it1 + 1,:], axis=0)
            # else:                
            #     zsmax = np.transpose(np.nanmax(dsin[varname].values[it0:it1 + 1,:,:], axis=0))

            dsin.close()

            return zsmax

        else:

            # Are we still supporting this ?
        
            ind_file = os.path.join(self.model.path, self.model.input.variables.indexfile)
    
            freqstr = str(self.model.input.dtmaxout) + "S"
            output_times = pd.date_range(start=self.input.variables.tstart,
                                         end=self.input.variables.tstop,
                                         freq=freqstr).to_pydatetime().tolist()
            nt = len(output_times)
            
            if time_range is None:
                time_range = [self.model.input.variables.tstart, self.model.input.variables.tstop]
            
            for it, time in enumerate(output_times):
                if time<=time_range[0]:
                    it0 = it
                if time<=time_range[1]:
                    it1 = it
    
            # Get maximum values
            nmax = self.input.variables.nmax
            mmax = self.input.variables.mmax
                            
            # Read sfincs.ind
            data_ind = np.fromfile(ind_file, dtype="i4")
            npoints  = data_ind[0]
            data_ind = np.squeeze(data_ind[1:])
            
            # Read zsmax file
            data_zs = np.fromfile(zsmax_file, dtype="f4")
            data_zs = np.reshape(data_zs,[nt, npoints + 2])[it0:it1+1, 1:-1]
            data_zs = np.amax(data_zs, axis=0)
            
            if output=="grid":
                zs_da = np.full([nmax*mmax], np.nan)        
                zs_da[data_ind - 1] = np.squeeze(data_zs)
                zs_da = np.where(zs_da == -999, np.nan, zs_da)
                zs_da = np.transpose(np.reshape(zs_da, [mmax, nmax]))
                return zs_da
            else:
                return data_zs
            
    def read_cumulative_precipitation(self, time_range=None, file_name=None, output="grid"):
    
        if not file_name:
            file_name = os.path.join(self.model.path, "cumprcp.dat")
            

        if self.input.outputformat[0:3] == "net":

            ddd=xr.open_dataset(file_name)
            
            # freqstr = str(self.input.dtmaxout) + "S"
            # output_times = pd.date_range(start=self.input.tstart,
            #                              end=self.input.tstop,
            #                              freq=freqstr).to_pydatetime().tolist()
            
 #           output_times = ddd.timemax.values
#            output_times = ddd.timemax.values.astype(datetime.datetime)
#            nt = len(output_times)


            output_times = ddd.timemax.values
            if time_range is None:

                t0 = pd.to_datetime(str(output_times[0])).replace(tzinfo=None).to_pydatetime()
                t1 = pd.to_datetime(str(output_times[-1])).replace(tzinfo=None).to_pydatetime()
                time_range = [t0, t1]


            
            # if time_range is None:
            #     time_range = [self.input.tstart, self.input.tstop]
            
            for it, time in enumerate(output_times):
                t = pd.to_datetime(str(time)).replace(tzinfo=None).to_pydatetime()
                if t<=time_range[0]:
                    it0 = it
                if t<=time_range[1]:
                    it1 = it
            
#            pall = ddd.cumprcp.values[it0:it1,:,:]            
#            psum = 
            p = np.transpose(np.sum(ddd.cumprcp.values[it0:it1,:,:], axis=0))

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
