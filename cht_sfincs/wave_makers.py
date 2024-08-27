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
