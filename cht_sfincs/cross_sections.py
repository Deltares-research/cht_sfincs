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

class SfincsCrossSections:
    def __init__(self, hw):
        self.model = hw
        self.gdf  = gpd.GeoDataFrame()

    def read(self):

        # Read in all cross sections

        if not self.model.input.variables.crsfile:
            return

        filename = os.path.join(self.model.path, self.model.input.variables.crsfile)

        # Read the crs file
        self.gdf = pli2gdf(filename, crs=self.model.crs)

    def write(self, file_name=None):

        if not self.model.input.variables.crsfile:
            return
        if len(self.gdf.index)==0:
            return

        filename = os.path.join(self.model.path, self.model.input.variables.crsfile)

        gdf2pli(self.gdf, filename)

    def add(self, cross_section):
        cross_section.set_crs(self.model.crs)
        self.gdf = pd.concat([self.gdf, cross_section], ignore_index=True)

    def delete(self, name_or_index):
        if type(name_or_index) == str:
            name = name_or_index
            for index, row in self.gdf.iterrows():
                if row["name"] == name:
                    self.gdf = self.gdf.drop(index).reset_index(drop=True)
                    return
            print("Cross section " + name + " not found!")    
        else:
            index = name_or_index
            if len(self.gdf.index) < index + 1:
                print("Index exceeds length!")    
            self.gdf = self.gdf.drop(index).reset_index(drop=True)
            return
        
    def clear(self):
        self.gdf  = gpd.GeoDataFrame()

    def list_names(self):
        names = []
        for index, row in self.gdf.iterrows():
            names.append(row["name"])
        return names
