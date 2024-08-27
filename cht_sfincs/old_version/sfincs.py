# -*- coding: utf-8 -*-
"""
Created on Sat May 15 08:08:40 2021

@author: ormondt
"""
import sys
import os
import pandas as pd
import datetime
#import gdal
#import subprocess
import numpy as np
#from matplotlib import pyplot as plt
from pandas.tseries.offsets import DateOffset
import xarray as xr

import math
from pyproj import CRS
from pyproj import Transformer

from cht.misc.geometry import RegularGrid
from cht.misc.geometry import Point
from cht.misc.deltares_ini import IniStruct
from cht.tiling import tiling

class SFINCS:
    
    def __init__(self, input_file=None, crs=None):
 
#        self.epsg                     = epsg
        self.input                    = SfincsInput()
        self.crs                      = crs
        self.grid                     = None
        self.mask                     = None
        self.flow_boundary_point      = []
        self.wave_boundary_point      = []
        self.observation_point        = []
        self.obstacle                 = []
        
        if input_file:
            self.path = os.path.dirname(input_file)
            self.load(input_file)

    def load(self, inputfile):
        # Reads sfincs.inp and attribute files
        self.read_input_file(inputfile)
        self.read_attribute_files()
    
    def read_input_file(self, inputfile):
        
        # Reads sfincs.inp
        
        # Get the path of sfincs.inp
        self.path = os.path.dirname(inputfile)
        
        fid = open(inputfile, 'r')
        lines = fid.readlines()
        fid.close()
        for line in lines:
            str = line.split("=")
            if len(str)==1:
               # Empty line
               continue
            name = str[0].strip()
            val  = str[1].strip()
            try:
                # First try to convert to int
                val = int(val)
            except ValueError:
                try:
                    # Now try to convert to float
                    val = float(val)
                except:
                    pass
            if name == "tref":
                try:
                    val = datetime.datetime.strptime(val.rstrip(), '%Y%m%d %H%M%S')
                except:
                    val = None
            if name == "tstart":
                try:
                    val = datetime.datetime.strptime(val.rstrip(), '%Y%m%d %H%M%S')
                except:
                    val = None
            if name == "tstop":
                try:
                    val = datetime.datetime.strptime(val.rstrip(), '%Y%m%d %H%M%S')
                except:
                    val = None
            setattr(self.input, name, val)

    def write_input_file(self, input_file=None):

        if not input_file:
            input_file = os.path.join(self.path, "sfincs.inp")
            
        fid = open(input_file, "w")
        for key, value in self.input.__dict__.items():
            if not value is None:
                if type(value) == "float":
                    string = f'{key.ljust(20)} = {float(value)}\n'
                elif type(value) == "int":
                    string = f'{key.ljust(20)} = {int(value)}\n'
                elif type(value) == list:
                    valstr = ""
                    for v in value:
                        valstr += str(v) + " "
                    string = f'{key.ljust(20)} = {valstr}\n'
                elif isinstance(value, datetime.date):
                    dstr = value.strftime("%Y%m%d %H%M%S")
                    string = f'{key.ljust(20)} = {dstr}\n'
                else:
                    string = f'{key.ljust(20)} = {value}\n'                
                fid.write(string)
        fid.close()    
                       
    def read_attribute_files(self):

        # Grid
#        self.grid = SfincsGrid()

        # Flow boundary conditions
        self.read_flow_boundary_points()
        self.read_flow_boundary_conditions()
        self.read_astro_boundary_conditions()

        # Wave boundary conditions
        self.read_wave_boundary_points()
        # self.read_wave_boundary_conditions()

        # Observation points
        self.read_observation_points()

#        self.grid.compute_coordinates(x0,y0,dx,dy,nx,ny,rotation)
    
    def read_index_file(self):
        pass
        
    def read_depth_file(self):
        pass

    def read_mask_file(self):
        pass
    
    def write_index_file(self):
        pass

    def write_depth_file(self):
        pass

    def write_mask_file(self):
        pass

##### Flow Boundary points #####
    
    def read_flow_boundary_points(self):
        
        # Read SFINCS bnd file
        
        self.flow_boundary_point = []
        
        if not self.input.bndfile:
            return
                    
        bnd_file = os.path.join(self.path,
                                self.input.bndfile)

        if not os.path.exists(bnd_file):
            return
        
        # Read the bnd file
        df = pd.read_csv(bnd_file, index_col=False, header=None,
             delim_whitespace=True, names=['x', 'y'])
        
        # Loop through points
        for ind in range(len(df.x.values)):
            name = str(ind + 1).zfill(4)
            point = FlowBoundaryPoint(df.x.values[ind],
                                      df.y.values[ind],
                                      name=name)
            self.flow_boundary_point.append(point)

    def write_flow_boundary_points(self, file_name=None):

        # Write SFINCS bnd file
        if not file_name:
            if not self.input.bndfile:
                return
            file_name = os.path.join(self.path,
                                     self.input.bndfile)
            
        if not file_name:
            return
            
        fid = open(file_name, "w")
        for point in self.flow_boundary_point:
            string = f'{point.geometry.x:12.1f}{point.geometry.y:12.1f}\n'
            fid.write(string)
        fid.close()    

    ### Flow Boundary conditions ###
    
    def read_flow_boundary_conditions(self, file_name=None):

        # Read SFINCS bzs file
        
        if not file_name:
            if not self.input.bzsfile:
                return
            file_name = os.path.join(self.path,
                                     self.input.bzsfile)
            
        if not file_name:
            return
        
        if not os.path.exists(file_name):
            return
        
        if not self.input.tref:
            # tref has not yet been defined
            return

        df = read_timeseries_file(file_name, self.input.tref)

        ts  = df.index
        for icol, point in enumerate(self.flow_boundary_point):
            point.data = pd.Series(df.iloc[:,icol].values, index=ts)
        
    def write_flow_boundary_conditions(self, file_name=None):

        # Write SFINCS bzs file
        if not file_name and not self.input.bzsfile:
            return
        if not file_name:
            file_name = os.path.join(self.path, self.input.bzsfile)
                    
        # Build a new DataFrame
        df = pd.DataFrame()
        for point in self.flow_boundary_point:
            df = pd.concat([df, point.data], axis=1)
        tmsec = pd.to_timedelta(df.index.values - self.input.tref, unit="s")
        df.index = tmsec.total_seconds()
        df.to_csv(file_name,
                  index=True,
                  sep=" ",
                  header=False,
                  float_format="%0.3f")

    def read_astro_boundary_conditions(self, file_name=None):

        # Read SFINCS bca file
        if not file_name:
            if not self.input.bcafile:
                return
            file_name = os.path.join(self.path,
                                     self.input.bcafile)
            
        if not file_name:
            return
        
        if not os.path.exists(file_name):
            return

        d = IniStruct(filename=file_name)
        for ind, point in enumerate(self.flow_boundary_point):
            point.astro = d.section[ind].data

    def generate_bzs_from_bca(self,
                              file_name=None,
                              dt=600.0,
                              offset=0.0,
                              write_file=True):
#        sys.path.append(r'd:/checkouts/OpenEarthTools/trunk/python/applications/DeltaShell/applications/CoDeS/CoDeS_2.0/Tide/pytides/')
        
        if not self.flow_boundary_point:
            print("No boundary points found!")
            return

        from cht.tide.tide_predict import predict

        if not file_name:
            if not self.input.bzsfile:
                self.input.bzsfile = "sfincs.bzs"
            file_name = self.input.bzsfile

        times = pd.date_range(start=self.input.tstart,
                              end=self.input.tstop,
                              freq=DateOffset(seconds=dt))
                              

        # Make boundary conditions based on bca file
        for point in self.flow_boundary_point:
            v = predict(point.astro, times) + offset
            ts = pd.Series(v, index=times)
            point.data = pd.Series(v, index=times)
                    
        if write_file:            
            self.write_flow_boundary_conditions()

    ### Wave boundary points

    def read_wave_boundary_points(self):
        
        # Read SFINCS bnd file
        
        self.wave_boundary_point = []
        
        if not self.input.snapwave_bndfile:
            return
                    
        bnd_file = os.path.join(self.path,
                                self.input.snapwave_bndfile)

        if not os.path.exists(bnd_file):
            return
        
        # Read the bnd file
        df = pd.read_csv(bnd_file, index_col=False, header=None,
             delim_whitespace=True, names=['x', 'y'])
        
        # Loop through points
        for ind in range(len(df.x.values)):
            name = str(ind + 1).zfill(4)
            point = WaveBoundaryPoint(df.x.values[ind],
                                      df.y.values[ind],
                                      name=name)
            self.wave_boundary_point.append(point)

    def write_wave_boundary_points(self, file_name=None):

        # Write SFINCS bnd file
        if not file_name:
            if not self.input.snapwave_bndfile:
                return
            file_name = os.path.join(self.path,
                                     self.input.snapwave_bndfile)
            
        if not file_name:
            return
            
        fid = open(file_name, "w")
        for point in self.wave_boundary_point:
            string = f'{point.geometry.x:12.1f}{point.geometry.y:12.1f}\n'
            fid.write(string)
        fid.close()    

    def write_wavemaker_forcing_points(self, file_name=None):

        # Write SFINCS bfp file
        if not file_name:
            if not self.input.wfpfile:
                return
            file_name = os.path.join(self.path,
                                     self.input.wfpfile)
            
        if not file_name:
            return
            
        fid = open(file_name, "w")
        for point in self.wavemaker_forcing_point:
            string = f'{point.geometry.x:12.1f}{point.geometry.y:12.1f}\n'
            fid.write(string)
        fid.close()    

    def write_wave_boundary_conditions(self, path=None):
        
        # Hm0, Tp, etc given (probably forced with SnapWave)
        self.write_bhs_file(path=path)
        self.write_btp_file(path=path)
        self.write_bwd_file(path=path)
        self.write_bds_file(path=path)            

    def write_wavemaker_forcing_conditions(self, path=None):
        
        # Hm0_ig given (probably forced with BEWARE, or something)
        self.write_whi_file(path=path)
        self.write_wti_file(path=path)
        self.write_wst_file(path=path)

            
    def write_bhs_file(self, file_name=None, path=None):
        # Hm0
        if not path:
            path=self.path
        if not file_name:
            if not self.input.snapwave_bhsfile:
                return
            file_name = os.path.join(path,
                                      self.input.snapwave_bhsfile)
        df = pd.DataFrame()
        for point in self.wave_boundary_point:
            df = pd.concat([df, point.data["hm0"]], axis=1)
        tmsec = pd.to_timedelta(df.index.values - self.input.tref, unit="s")
        df.index = tmsec.total_seconds()
        df.to_csv(file_name,
                  index=True,
                  sep=" ",
                  header=False,
                  float_format="%0.3f")

    def write_btp_file(self, file_name=None, path=None):
        # Tp
        if not path:
            path=self.path
        if not file_name:
            if not self.input.snapwave_btpfile:
                return
            file_name = os.path.join(path,
                                      self.input.snapwave_btpfile)
        df = pd.DataFrame()
        for point in self.wave_boundary_point:
            df = pd.concat([df, point.data["tp"]], axis=1)
        tmsec = pd.to_timedelta(df.index.values - self.input.tref, unit="s")
        df.index = tmsec.total_seconds()
        df.to_csv(file_name,
                  index=True,
                  sep=" ",
                  header=False,
                  float_format="%0.1f")

    def write_bwd_file(self, file_name=None, path=None):
        # WavDir
        if not path:
            path=self.path
        if not file_name:
            if not self.input.snapwave_bwdfile:
                return
            file_name = os.path.join(path,
                                      self.input.snapwave_bwdfile)
        df = pd.DataFrame()
        for point in self.wave_boundary_point:
            df = pd.concat([df, point.data["wavdir"]], axis=1)
        tmsec = pd.to_timedelta(df.index.values - self.input.tref, unit="s")
        df.index = tmsec.total_seconds()
        df.to_csv(file_name,
                  index=True,
                  sep=" ",
                  header=False,
                  float_format="%0.1f")

    def write_bds_file(self, file_name=None, path=None):
        # DirSpr
        if not path:
            path=self.path
        if not file_name:
            if not self.input.snapwave_bdsfile:
                return
            file_name = os.path.join(path,
                                      self.input.snapwave_bdsfile)
        df = pd.DataFrame()
        for point in self.wave_boundary_point:
            df = pd.concat([df, point.data["dirspr"]], axis=1)
        tmsec = pd.to_timedelta(df.index.values - self.input.tref, unit="s")
        df.index = tmsec.total_seconds()
        df.to_csv(file_name,
                  index=True,
                  sep=" ",
                  header=False,
                  float_format="%0.1f")

            


    def write_whi_file(self, file_name=None, path=None):
        # Hm0 ig
        if not path:
            path=self.path
        if not file_name:
            if not self.input.whifile:
                return
            file_name = os.path.join(path,
                                      self.input.whifile)
        df = pd.DataFrame()
        for point in self.wavemaker_forcing_point:
            df = pd.concat([df, point.data["hm0_ig"]], axis=1)
        tmsec = pd.to_timedelta(df.index.values - self.input.tref, unit="s")
        df.index = tmsec.total_seconds()
        df.to_csv(file_name,
                  index=True,
                  sep=" ",
                  header=False,
                  float_format="%0.3f")

    def write_wti_file(self, file_name=None, path=None):
        # Tp ig
        if not path:
            path=self.path
        if not file_name:
            if not self.input.wtifile:
                return
            file_name = os.path.join(path,
                                      self.input.wtifile)
        df = pd.DataFrame()
        for point in self.wavemaker_forcing_point:
            df = pd.concat([df, point.data["tp_ig"]], axis=1)
        tmsec = pd.to_timedelta(df.index.values - self.input.tref, unit="s")
        df.index = tmsec.total_seconds()
        df.to_csv(file_name,
                  index=True,
                  sep=" ",
                  header=False,
                  float_format="%0.1f")

    def write_wst_file(self, file_name=None, path=None):
        # Set-up
        if not path:
            path=self.path
        if not file_name:
            if not self.input.wstfile:
                return
            file_name = os.path.join(path,
                                      self.input.wstfile)
        df = pd.DataFrame()
        for point in self.wavemaker_forcing_point:
            df = pd.concat([df, point.data["setup"]], axis=1)
        tmsec = pd.to_timedelta(df.index.values - self.input.tref, unit="s")
        df.index = tmsec.total_seconds()
        df.to_csv(file_name,
                  index=True,
                  sep=" ",
                  header=False,
                  float_format="%0.3f")

    ### Observation points ###

    def add_observation_point(self, x, y, name):
                
        self.observation_point.append(ObservationPoint(x, y, name, crs=None))

    def read_observation_points(self, file_name=None):
        
        self.observation_point = []

        if not file_name:
            if not self.input.obsfile:
                return
            file_name = os.path.join(self.path,
                                     self.input.obsfile)
                            
        if not os.path.exists(file_name):
            print("Warning : file " + file_name + " does not exist !")
            return
        
        # Loop through points
        df = pd.read_csv(file_name, index_col=False, header=None,
             delim_whitespace=True, names=['x', 'y', 'name'])
        
        for ind in range(len(df.x.values)):
            point = ObservationPoint(df.x.values[ind],
                                     df.y.values[ind],
                                     name=str(df.name.values[ind]))
            self.observation_point.append(point)

    def write_observation_points(self, file_name=None):

        if not file_name:
            file_name = os.path.join(self.path,
                                     self.input.obsfile)
        if self.input.crsgeo == 0:
            fid = open(file_name, "w")
            for point in self.observation_point:
                string = f'{point.geometry.x:12.1f}{point.geometry.y:12.1f}  "{point.name}"\n'
                fid.write(string)
            fid.close()
        else:
            fid = open(file_name, "w")
            for point in self.observation_point:
                string = f'{point.geometry.x:12.6f}{point.geometry.y:12.6f}  "{point.name}"\n'
                fid.write(string)
            fid.close()
        
            
    ### Output ###

    def read_timeseries_output(
            self,
            path=None, 
            file_name=None,
            name_list=None,
            parameter="point_zs"
    ):

        if path is None and hasattr(self, "path"):
            path = self.path
        elif path is None:
            path = os.getcwd()

        # Returns a dataframe with timeseries
        if self.input.outputformat[0:3] == "bin":
            
            # ASCII output
    
            if not file_name:
                file_name = os.path.join(path, "zst.txt")
        
            if not self.observation_point:
                # First read observation points
                self.read_observation_points()
            
            columns = []
            for point in self.observation_point:
                columns.append(point.name)
                
            df = read_timeseries_file(file_name, self.input.tref)
            
            # Add column names
            df.columns = columns
                
            if name_list:
                df = df[name_list]
            
        else:
            
            # NetCDF output
            if not file_name:
                file_name = "sfincs_his.nc"
            file_name = os.path.join(path, file_name)
                    
            # Open netcdf file
            ddd = xr.open_dataset(file_name)
#            stations=ddd.point_zs.coords["station_name"].values
            stations=ddd.station_name.values
            all_stations = []
            for ist, st in enumerate(stations):
#                st=str(st)[2:-1]
                all_stations.append(st.decode().strip())
            

            times   = ddd[parameter].coords["time"].values
    
            # If name_list is empty, add all points    
            if not name_list:
                name_list = []
                for st in all_stations:
                    name_list.append(st)
            
            df = pd.DataFrame(index=times, columns=name_list)

            for station in name_list:
                for ist, st in enumerate(all_stations):
                    if station == st:
                        wl = ddd[parameter].isel(stations=ist).values
                        wl[np.isnan(wl)] = -999.0
                        df[st]=wl
                        break            
    
            ddd.close()
                    
        return df    

    def read_zsmax(self, time_range=None, zsmax_file=None, output="grid", varname='zsmax'):
    
        if not zsmax_file:
            if self.input.outputformat[0:3] == "net":
                zsmax_file = os.path.join(self.path, "sfincs_map.nc")
            else:
                zsmax_file = os.path.join(self.path, "zsmax.dat")
            

        if self.input.outputformat[0:3] == "net":
#            ddd=xr.open_dataset(zsmax_file)
#            zsmx=ddd.zsmax.values
#            zsmax=np.transpose(np.nanmax(ddd.zsmax.values, axis=0))
            

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

            # Check if dimension nmesh2d_face exists (in which case this is a quadtree mesh)
            if 'nmesh2d_face' in dsin[varname].dims:
                zsmax = np.nanmax(dsin[varname].values[it0:it1 + 1,:], axis=0)
            else:                
                zsmax = np.transpose(np.nanmax(dsin[varname].values[it0:it1 + 1,:,:], axis=0))
                zsmax = np.nanmax(dsin[varname].values[it0:it1 + 1,:,:], axis=0)
            dsin.close()

            return zsmax



        else:
        
            ind_file = os.path.join(self.path, self.input.indexfile)
    
            freqstr = str(self.input.dtmaxout) + "S"
            output_times = pd.date_range(start=self.input.tstart,
                                         end=self.input.tstop,
                                         freq=freqstr).to_pydatetime().tolist()
            nt = len(output_times)
            
            if time_range is None:
                time_range = [self.input.tstart, self.input.tstop]
            
            for it, time in enumerate(output_times):
                if time<=time_range[0]:
                    it0 = it
                if time<=time_range[1]:
                    it1 = it
    
            # Get maximum values
            nmax = self.input.nmax
            mmax = self.input.mmax
                            
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
            file_name = os.path.join(self.path, "cumprcp.dat")
            

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

            it0 = 0
            it1 = len(output_times)
            for it, time in enumerate(output_times):
                t = pd.to_datetime(str(time)).replace(tzinfo=None).to_pydatetime()
                if t<=time_range[0]:
                    it0 = it
                if t<=time_range[1]:
                    it1 = it
            
            # NOTE since november 2022, SFINCS does not recet cumprcp after dtmaxout 
            if self.input.qtrfile:
                p = ddd.cumprcp.values[it1,:] - ddd.cumprcp.values[it0,:]
            else:  
                p = np.transpose(ddd.cumprcp.values[it1,:,:]-ddd.cumprcp.values[it0,:,:])

            return p

        # else:
        
        #     ind_file = os.path.join(self.path, self.input.indexfile)
    
        #     freqstr = str(self.input.dtmaxout) + "S"
        #     output_times = pd.date_range(start=self.input.tstart,
        #                                  end=self.input.tstop,
        #                                  freq=freqstr).to_pydatetime().tolist()
        #     nt = len(output_times)
            
        #     if time_range is None:
        #         time_range = [self.input.tstart, self.input.tstop]
            
        #     for it, time in enumerate(output_times):
        #         if time<=time_range[0]:
        #             it0 = it
        #         if time<=time_range[1]:
        #             it1 = it
    
        #     # Get maximum values
        #     nmax = self.input.nmax
        #     mmax = self.input.mmax
                            
        #     # Read sfincs.ind
        #     data_ind = np.fromfile(ind_file, dtype="i4")
        #     npoints  = data_ind[0]
        #     data_ind = np.squeeze(data_ind[1:])
            
        #     # Read zsmax file
        #     data_zs = np.fromfile(zsmax_file, dtype="f4")
        #     data_zs = np.reshape(data_zs,[nt, npoints + 2])[it0:it1+1, 1:-1]
        #     data_zs = np.amax(data_zs, axis=0)
            
        #     if output=="grid":
        #         zs_da = np.full([nmax*mmax], np.nan)        
        #         zs_da[data_ind - 1] = np.squeeze(data_zs)
        #         zs_da = np.where(zs_da == -999, np.nan, zs_da)
        #         zs_da = np.transpose(np.reshape(zs_da, [mmax, nmax]))
        #         return zs_da
        #     else:
        #         return data_zs
        
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

    def grid_coordinates(self, loc='cor'):

        cosrot = math.cos(self.input.rotation*math.pi/180)
        sinrot = math.sin(self.input.rotation*math.pi/180)
        if loc=="cor":
            xx     = np.linspace(0.0,
                                 self.input.mmax*self.input.dx,
                                 num=self.input.mmax + 1)
            yy     = np.linspace(0.0,
                                 self.input.nmax*self.input.dy,
                                 num=self.input.nmax + 1)
        else:
            xx     = np.linspace(0.5*self.input.dx,
                                 self.input.mmax*self.input.dx - 0.5*self.input.dx,
                                 num=self.input.mmax)
            yy     = np.linspace(0.5*self.input.dy,
                                 self.input.nmax*self.input.dy - 0.5*self.input.dy,
                                 num=self.input.nmax)
            
        xg0, yg0 = np.meshgrid(xx, yy)
        xg = self.input.x0 + xg0*cosrot - yg0*sinrot
        yg = self.input.y0 + xg0*sinrot + yg0*cosrot

        return xg, yg
    
    def bounding_box(self, crs=None):

        xg, yg = self.grid_coordinates(loc='cor')
        
        if crs:
            transformer = Transformer.from_crs(self.crs,
                                               crs,
                                               always_xy=True)
            xg, yg = transformer.transform(xg, yg)
        
        x_range = [np.min(np.min(xg)), np.max(np.max(xg))]
        y_range = [np.min(np.min(yg)), np.max(np.max(yg))]
        
        return x_range, y_range

    def outline(self, crs=None):

        xg, yg = self.grid_coordinates(loc='cor')
        
        if crs:
            transformer = Transformer.from_crs(self.crs,
                                               crs,
                                               always_xy=True)
            xg, yg = transformer.transform(xg, yg)
        
        xp = [ xg[0,0], xg[0,-1], xg[-1,-1], xg[-1,0], xg[0,0] ]
        yp = [ yg[0,0], yg[0,-1], yg[-1,-1], yg[-1,0], yg[0,0] ]
        
        return xp, yp
        
    def make_index_tiles(self, path, zoom_range=None, format=0):

        if self.input.qtrfile:
            from .quadtree import QuadtreeGrid
            quadtree = QuadtreeGrid(crs=self.crs) 
            quadtree.load(os.path.join(self.path, self.input.qtrfile))
            quadtree.make_index_tiles(path, zoom_range=zoom_range)
            return
        
        from cht.tiling.tiling import deg2num
        from cht.tiling.tiling import num2deg
        import cht.misc.fileops as fo
        
        if not zoom_range:
            zoom_range = [0, 13]

        npix = 256
        
        # Compute lon/lat range
        lon_range, lat_range = self.bounding_box(crs=CRS.from_epsg(4326))
        
        cosrot = math.cos(-self.input.rotation*math.pi/180)
        sinrot = math.sin(-self.input.rotation*math.pi/180)       
        
        transformer_a = Transformer.from_crs(CRS.from_epsg(4326),
                                             CRS.from_epsg(3857),
                                             always_xy=True)
        transformer_b = Transformer.from_crs(CRS.from_epsg(3857),
                                             self.crs,
                                             always_xy=True)
        
        for izoom in range(zoom_range[0], zoom_range[1] + 1):
            
            print("Processing zoom level " + str(izoom))
        
            zoom_path = os.path.join(path, str(izoom))
        
            dxy = (40075016.686/npix) / 2 ** izoom
            xx = np.linspace(0.0, (npix - 1)*dxy, num=npix)
            yy = xx[:]
            xv, yv = np.meshgrid(xx, yy)
        
            ix0, iy0 = deg2num(lat_range[0], lon_range[0], izoom)
            ix1, iy1 = deg2num(lat_range[1], lon_range[1], izoom)
        
            for i in range(ix0, ix1 + 1):
            
                path_okay = False
                zoom_path_i = os.path.join(zoom_path, str(i))
            
                for j in range(iy0, iy1 + 1):
            
                    file_name = os.path.join(zoom_path_i, str(j) + ".dat")
            
                    # Compute lat/lon at ll corner of tile
                    lat, lon = num2deg(i, j, izoom)
            
                    # Convert to Global Mercator
                    xo, yo   = transformer_a.transform(lon,lat)
            
                    # Tile grid on local mercator
                    x        = xv[:] + xo + 0.5*dxy
                    y        = yv[:] + yo + 0.5*dxy
            
                    # Convert tile grid to crs of SFINCS model
                    x,y      = transformer_b.transform(x,y)
                    
                    # Now rotate around origin of SFINCS model
                    x00 = x - self.input.x0
                    y00 = y - self.input.y0
                    xg  = x00*cosrot - y00*sinrot
                    yg  = x00*sinrot + y00*cosrot
                    
                    iind = np.floor(xg/self.input.dx).astype(int)
                    jind = np.floor(yg/self.input.dy).astype(int)
                    ind  = iind*self.input.nmax + jind
                    ind[iind<0]   = -999
                    ind[jind<0]   = -999
                    ind[iind>=self.input.mmax] = -999
                    ind[jind>=self.input.nmax] = -999
#                    ind           = np.ascontiguousarray(np.transpose(ind))

                    # if i==142 and j==305:
                    
                    #     from matplotlib import pyplot as plt
                    #     fig, ax = plt.subplots(1,1)                                            
                    #     ax.plot(x,y)
                    #     ax.plot(x.transpose(),y.transpose())
                    #     ax.axis('equal')
                    #     x_range, y_range = self.bounding_box()
                    #     xp = [x_range[0], x_range[1],x_range[1],x_range[0],x_range[0]]
                    #     yp = [y_range[0], y_range[0],y_range[1],y_range[1],y_range[0]]
                    #     ax.plot(xp,yp)
                    #     xout, yout = self.outline()
                    #     ax.plot(xout,yout)
                    #     fig, ax = plt.subplots(1,1)                                            
                    #     ax.pcolor(ind.reshape([256, 256]))
                    #     xxx=1
                    
                    if np.any(ind>=0):
                        
                        if not path_okay:
                            if not os.path.exists(zoom_path_i):
                                fo.mkdir(zoom_path_i)
                                path_okay = True
                             
                        # And write indices to file
                        fid = open(file_name, "wb")
                        fid.write(ind)
                        fid.close()

class SfincsInput():
    def __init__(self):
        self.mmax = 0
        self.nmax = 0
        self.dx   = 10.0
        self.dy   = 10.0
        self.x0   = 0.0
        self.y0   = 0.0
        self.rotation = 0.0
        self.latitude = 0.0
        tnow = datetime.datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        self.tref=tnow
        self.tstart=tnow
        self.tstop=tnow + datetime.timedelta(days=1)
        self.tspinup=60.0
        self.t0out=None
        self.dtmapout=3600.0
        self.dthisout=600.0
        self.dtrstout=0.0
        self.dtmaxout=0.0
        self.trstout=-999.0
        self.dtwnd=1800.0
        self.alpha=0.5
        self.theta=1.0
        self.nuvisc=-999.0
        self.huthresh=0.01
        self.manning=0.04
        self.manning_land=0.04
        self.manning_sea=0.02
        self.rgh_lev_land=0.0
        self.zsini=0.0
        self.qinf=0.0
        self.igperiod=120.0
        self.rhoa=1.25
        self.rhow=1024.0
        self.dtmax=999.0
        self.maxlev=999.0
        self.bndtype=1
        self.advection=0
        self.baro=0
        self.pavbnd=0
        self.gapres=101200.0
        self.advlim=9999.9
        self.stopdepth=1000.0
        self.crsgeo=0
        
        self.qtrfile=None
        self.depfile=None
        self.mskfile=None
        self.indexfile=None
        self.qtrfile=None
        self.cstfile=None
        self.bndfile=None
        self.bzsfile=None
        self.bzifile=None
        self.bwvfile=None
        self.bhsfile=None
#        self.bhifile=None
#        self.bstfile=None
        self.btpfile=None
        self.bwdfile=None
        self.bdsfile=None
        self.bcafile=None
        self.corfile=None
        self.srcfile=None
        self.disfile=None
        self.inifile=None
        self.sbgfile=None        
        self.spwfile=None
        self.amufile=None
        self.amvfile=None
        self.ampfile=None
        self.amprfile=None
        self.wndfile=None
        self.precipfile=None
        self.obsfile=None
        self.crsfile=None
        self.thdfile=None
        self.manningfile=None
        self.scsfile=None
        self.rstfile=None
        self.wfpfile=None
        self.whifile=None
        self.wtifile=None
        self.wstfile=None
        
        self.inputformat="bin"
        self.outputformat="net"
        
        self.cdnrb=3
        self.cdwnd=[0.0,28.0,50.0]
        self.cdval=[0.001,0.0025,0.0015]

        self.dtwave=None
        self.snapwave=None
        self.snapwave_gamma=None
        self.snapwave_dtheta=None
        self.snapwave_hmin=None
        self.snapwave_fw0=None
        self.snapwave_crit=None
        self.snapwave_igwaves=None
        self.snapwave_nrsweeps=None
        self.snapwave_bhsfile=None
        self.snapwave_btpfile=None
        self.snapwave_bwdfile=None
        self.snapwave_bdsfile=None
        self.snapwave_encfile=None
        self.snapwave_bndfile=None

class SfincsGrid():

    def __init__(self, x0, y0, dx, dy, nx, ny, rotation):
        self.geometry = RegularGrid(x0, y0, dx, dy, nx, ny, rotation)

    # def plot(self,ax):
    #     self.geometry.plot(ax)

    # def corner_coordinates(self):
    #     x,y = self.geometry.grid_coordinates_corners()
    #     return x, y

    # def centre_coordinates(self):
    #     x,y = self.geometry.grid_coordinates_centres()
    #     return x, y

class SfincsDepth():
    def __init__(self):
        self.value = []
        self.geometry = []
    def plot(self,ax):
        pass
    def read(self):
        pass

class SfincsMask():
    def __init__(self):
        self.msk = []
    def plot(self,ax):
        pass

class SfincsFlowBoundaryConditions():
    
    def __init__(self):
        self.geometry = []

    def read(self, bndfile, bzsfile):
        self.read_points(bndfile)
        self.read_time_series(bzsfile)

    def read_points(self, file_name):
        pass

    def read_time_series(self, file_name):
        pass
    
    def set_xy(self, x, y):
        self.geometry.x = x
        self.geometry.y = y
        pass
    
    def plot(self,ax):
        pass

class SfincsWaveBoundaryConditions():
    
    def __init__(self):
        self.geometry = []

    def read(self, bndfile, bzsfile):
        self.read_points(bndfile)
        self.read_time_series(bzsfile)

    def read_points(self, file_name):
        pass

    def read_time_series(self, file_name):
        pass
    
    def set_xy(self, x, y):
        self.geometry.x = x
        self.geometry.y = y
        pass
    
    def plot(self,ax):
        pass

class FlowBoundaryPoint():

    def __init__(self, x, y, name=None, crs=None, data=None, astro=None):
        
        self.name                   = name
        self.geometry               = Point(x, y, crs=crs)
        self.data                   = data
        self.astro                  = astro

class WaveBoundaryPoint():

    def __init__(self, x, y, name=None, crs=None, data=None):
        
        self.name                   = name
        self.geometry               = Point(x, y, crs=crs)
        self.data                   = data

class WaveMakerForcingPoint():

    def __init__(self, x, y, name=None, crs=None, data=None):
        
        self.name                   = name
        self.geometry               = Point(x, y, crs=crs)
        self.data                   = data

class ObservationPoint():

    def __init__(self, x, y, name, crs=None):
        
        self.name     = name
        self.geometry = Point(x, y, crs=crs)

                    
def read_timeseries_file(file_name, ref_date):
    
    # Returns a dataframe with time series for each of the columns

    df = pd.read_csv(file_name, index_col=0, header=None,
                    delim_whitespace=True)
    ts = ref_date + pd.to_timedelta(df.index, unit="s")
    df.index = ts
    
    return df


# class QuadtreeGrid():

#     def __init__(self):
#         self.geometry = RegularGrid(x0, y0, dx, dy, nx, ny, rotation)
        
#     def read(self, file_name):
    

# fid=fopen(buqfile,'r');

# % Number of blocks
# np=fread(fid,1,'integer*4');

# % Nr levels
# nlev=fread(fid,1,'integer*1');

# % Grid stuff
# buq.x0=fread(fid,1,'real*4');
# buq.y0=fread(fid,1,'real*4');
# buq.dx=fread(fid,1,'real*4');
# buq.dy=fread(fid,1,'real*4');
# buq.rotation=fread(fid,1,'real*4');

# % Levels
# buq.level=fread(fid,np,'integer*1');

# % N
# buq.n=fread(fid,np,'integer*4');

# % M
# buq.m=fread(fid,np,'integer*4');

# % NU
# buq.nu=fread(fid,np,'integer*1');
# buq.nu1=fread(fid,np,'integer*4');
# buq.nu2=fread(fid,np,'integer*4');
# % MU
# buq.mu=fread(fid,np,'integer*1');
# buq.mu1=fread(fid,np,'integer*4');
# buq.mu2=fread(fid,np,'integer*4');
# % ND
# buq.nd=fread(fid,np,'integer*1');
# buq.nd1=fread(fid,np,'integer*4');
# buq.nd2=fread(fid,np,'integer*4');
# % MD
# buq.md=fread(fid,np,'integer*1');
# buq.md1=fread(fid,np,'integer*4');
# buq.md2=fread(fid,np,'integer*4');

# fclose(fid);

# buq.level=buq.level+1;
# nlev=nlev+1;
# buq.nmax=0;
# buq.mmax=0;

# for nm = 1:np
#       n    = buq.n(nm);
#       m    = buq.m(nm);
#       iref = buq.level(nm);
#       buq.nmax = max(buq.nmax, floor( (1.0*(n - 1) + 0.01) / (2^(iref - 1))) + 2);
#       buq.mmax = max(buq.mmax, floor( (1.0*(m - 1) + 0.01) / (2^(iref - 1))) + 2);
# end

# buq.first_point_per_level = 0;
# buq.last_point_per_level = 0;
# buq.nm_indices = 0;
# % First count
# %
# ireflast = 0;
# %
# for ip = 1: np
#     %
#     iref = buq.level(ip);
#     n    = buq.n(ip);
#     m    = buq.m(ip);
#     nmx  = buq.nmax*2^(iref - 1);
#     nm   = (m - 1)*nmx + n;
#     %
#     buq.nm_indices(ip) = nm;
#     %
#     if iref>ireflast
#         %
#         % Found new level
#         %
#         buq.first_point_per_level(iref) = ip;
#         ireflast = iref;
#         %
#     end
#     %
#     buq.last_point_per_level(iref) = ip;
#     %
# end
# disp('done')    