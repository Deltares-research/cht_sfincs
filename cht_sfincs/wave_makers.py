# -*- coding: utf-8 -*-
"""
Created on Sat Jun 18 09:03:08 2022
@author: ormondt
"""
import os
import geopandas as gpd
import shapely
import pandas as pd

from cht_utils.pli_file import pli2gdf, gdf2pli

class SfincsWaveMakers:
    def __init__(self, hw):
        self.model = hw
        self.gdf  = gpd.GeoDataFrame()

    def read(self):
        # Read in wave makers from file
        if not self.model.input.variables.wvmfile:
            return
        file_name = os.path.join(self.model.path, self.model.input.variables.wvmfile)
        self.gdf = pli2gdf(file_name, crs=self.model.crs)

    def write(self):
        if not self.model.input.variables.wvmfile:
            return
        if len(self.gdf.index)==0:
            return
        file_name = os.path.join(self.model.path, self.model.input.variables.wvmfile)
        gdf2pli(self.gdf, file_name)

    def add_point(self, gdf_to_add):
        pass
        # point = shapely.geometry.Point(x, y)
        # gdf_list = []
        # d = {"name": name, "long_name": None, "geometry": point}
        # gdf_list.append(d)
        # gdf_new = gpd.GeoDataFrame(gdf_list, crs=self.model.crs)
        # self.gdf = pd.concat([self.gdf, gdf_new], ignore_index=True)

    def delete_polyline(self, index):
        if len(self.gdf.index) < index + 1:
            print("Index exceeds length!")    
        self.gdf = self.gdf.drop(index).reset_index(drop=True)
        return
        
    def clear(self):
        self.gdf  = gpd.GeoDataFrame()

    def list_names(self):
        names = []
        for index, row in self.gdf.iterrows():
            names.append(str(index + 1))
        return names


    ### Wave boundary points

    # def read_wave_boundary_points(self):
        
    #     # Read SFINCS bnd file
        
    #     self.wave_boundary_point = []
        
    #     if not self.input.bwvfile:
    #         return
                    
    #     bnd_file = os.path.join(self.path,
    #                             self.input.bwvfile)

    #     if not os.path.exists(bnd_file):
    #         return
        
    #     # Read the bnd file
    #     df = pd.read_csv(bnd_file, index_col=False, header=None,
    #          delim_whitespace=True, names=['x', 'y'])
        
    #     # Loop through points
    #     for ind in range(len(df.x.values)):
    #         name = str(ind + 1).zfill(4)
    #         point = WaveBoundaryPoint(df.x.values[ind],
    #                                   df.y.values[ind],
    #                                   name=name)
    #         self.wave_boundary_point.append(point)

    # def write_wave_boundary_points(self, file_name=None):

    #     # Write SFINCS bnd file
    #     if not file_name:
    #         if not self.input.bwvfile:
    #             return
    #         file_name = os.path.join(self.path,
    #                                  self.input.bwvfile)
            
    #     if not file_name:
    #         return
            
    #     fid = open(file_name, "w")
    #     for point in self.wave_boundary_point:
    #         string = f'{point.geometry.x:12.1f}{point.geometry.y:12.1f}\n'
    #         fid.write(string)
    #     fid.close()    

    # def write_wavemaker_forcing_points(self, file_name=None):

    #     # Write SFINCS bfp file
    #     if not file_name:
    #         if not self.input.wfpfile:
    #             return
    #         file_name = os.path.join(self.path,
    #                                  self.input.wfpfile)
            
    #     if not file_name:
    #         return
            
    #     fid = open(file_name, "w")
    #     for point in self.wavemaker_forcing_point:
    #         string = f'{point.geometry.x:12.1f}{point.geometry.y:12.1f}\n'
    #         fid.write(string)
    #     fid.close()    

    # def write_wave_boundary_conditions(self):
        
    #     # Hm0, Tp, etc given (probably forced with SnapWave)
    #     self.write_bhs_file()
    #     self.write_btp_file()
    #     self.write_bwd_file()
    #     self.write_bds_file()            

    # def write_wavemaker_forcing_conditions(self):
        
    #     # Hm0_ig given (probably forced with BEWARE, or something)
    #     self.write_whi_file()
    #     self.write_wti_file()
    #     self.write_wst_file()

            
    # def write_bhs_file(self, file_name=None):
    #     # Hm0
    #     if not file_name:
    #         if not self.input.bhsfile:
    #             return
    #         file_name = os.path.join(self.path,
    #                                   self.input.bhsfile)
    #     df = pd.DataFrame()
    #     for point in self.wave_boundary_point:
    #         df = pd.concat([df, point.data["hm0"]], axis=1)
    #     tmsec = pd.to_timedelta(df.index.values - self.input.tref, unit="s")
    #     df.index = tmsec.total_seconds()
    #     df.to_csv(file_name,
    #               index=True,
    #               sep=" ",
    #               header=False,
    #               float_format="%0.3f")

    # def write_btp_file(self, file_name=None):
    #     # Tp
    #     if not file_name:
    #         if not self.input.btpfile:
    #             return
    #         file_name = os.path.join(self.path,
    #                                   self.input.btpfile)
    #     df = pd.DataFrame()
    #     for point in self.wave_boundary_point:
    #         df = pd.concat([df, point.data["tp"]], axis=1)
    #     tmsec = pd.to_timedelta(df.index.values - self.input.tref, unit="s")
    #     df.index = tmsec.total_seconds()
    #     df.to_csv(file_name,
    #               index=True,
    #               sep=" ",
    #               header=False,
    #               float_format="%0.1f")

    # def write_bwd_file(self, file_name=None):
    #     # WavDir
    #     if not file_name:
    #         if not self.input.bwdfile:
    #             return
    #         file_name = os.path.join(self.path,
    #                                   self.input.bwdfile)
    #     df = pd.DataFrame()
    #     for point in self.wave_boundary_point:
    #         df = pd.concat([df, point.data["wavdir"]], axis=1)
    #     tmsec = pd.to_timedelta(df.index.values - self.input.tref, unit="s")
    #     df.index = tmsec.total_seconds()
    #     df.to_csv(file_name,
    #               index=True,
    #               sep=" ",
    #               header=False,
    #               float_format="%0.1f")

    # def write_bds_file(self, file_name=None):
    #     # DirSpr
    #     if not file_name:
    #         if not self.input.bdsfile:
    #             return
    #         file_name = os.path.join(self.path,
    #                                   self.input.bdsfile)
    #     df = pd.DataFrame()
    #     for point in self.wave_boundary_point:
    #         df = pd.concat([df, point.data["dirspr"]], axis=1)
    #     tmsec = pd.to_timedelta(df.index.values - self.input.tref, unit="s")
    #     df.index = tmsec.total_seconds()
    #     df.to_csv(file_name,
    #               index=True,
    #               sep=" ",
    #               header=False,
    #               float_format="%0.1f")

            


    # def write_whi_file(self, file_name=None):

    #     # Hm0 ig
    #     if not file_name:
    #         if not self.input.whifile:
    #             return
    #         file_name = os.path.join(self.path,
    #                                   self.input.whifile)
    #     df = pd.DataFrame()
    #     for point in self.wavemaker_forcing_point:
    #         df = pd.concat([df, point.data["hm0_ig"]], axis=1)
    #     tmsec = pd.to_timedelta(df.index.values - self.input.tref, unit="s")
    #     df.index = tmsec.total_seconds()
    #     df.to_csv(file_name,
    #               index=True,
    #               sep=" ",
    #               header=False,
    #               float_format="%0.3f")

    # def write_wti_file(self, file_name=None):

    #     # Tp ig
    #     if not file_name:
    #         if not self.input.wtifile:
    #             return
    #         file_name = os.path.join(self.path,
    #                                   self.input.wtifile)
    #     df = pd.DataFrame()
    #     for point in self.wavemaker_forcing_point:
    #         df = pd.concat([df, point.data["tp_ig"]], axis=1)
    #     tmsec = pd.to_timedelta(df.index.values - self.input.tref, unit="s")
    #     df.index = tmsec.total_seconds()
    #     df.to_csv(file_name,
    #               index=True,
    #               sep=" ",
    #               header=False,
    #               float_format="%0.1f")

    # def write_wst_file(self, file_name=None):

    #     # Set-up
    #     if not file_name:
    #         if not self.input.wstfile:
    #             return
    #         file_name = os.path.join(self.path,
    #                                   self.input.wstfile)
    #     df = pd.DataFrame()
    #     for point in self.wavemaker_forcing_point:
    #         df = pd.concat([df, point.data["setup"]], axis=1)
    #     tmsec = pd.to_timedelta(df.index.values - self.input.tref, unit="s")
    #     df.index = tmsec.total_seconds()
    #     df.to_csv(file_name,
    #               index=True,
    #               sep=" ",
    #               header=False,
    #               float_format="%0.3f")
