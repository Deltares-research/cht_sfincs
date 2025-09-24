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

class SfincsThinDams:
    def __init__(self, sf):
        self.model = sf
        self.gdf  = gpd.GeoDataFrame()

    def read(self):
        # Read in all observation points

        if not self.model.input.variables.thdfile:
            return

        filename = os.path.join(self.model.path, self.model.input.variables.thdfile)

        # Read the thd file
        self.gdf = pli2gdf(filename, crs=self.model.crs)

    def write(self):

        if len(self.gdf.index)==0:
            return

        if not self.model.input.variables.thdfile:
            self.model.input.variables.thdfile = "sfincs.thd"

        filename = os.path.join(self.model.path, self.model.input.variables.thdfile)

        gdf2pli(self.gdf, filename)
        
    def add(self, thin_dam):
        # Thin dam may be a gdf or shapely geometry
        # Assume for now a gdf
        thin_dam.set_crs(self.model.crs)
        self.gdf = pd.concat([self.gdf, thin_dam], ignore_index=True)

    def add_xy(self, x, y):
        # Add a thin dam by providing x and y coordinates
        # x and y are lists of the same length
        # Create linestring geometry
        if len(x) != len(y):
            raise ValueError("x and y must be the same length")
        thin_dam = gpd.GeoDataFrame(geometry=[shapely.geometry.LineString(zip(x, y))],
                                    crs=self.model.crs)
        # Create a new row in the gdf
        self.gdf = pd.concat([self.gdf, thin_dam], ignore_index=True)


    def delete(self, index):
        if len(self.gdf.index) < index + 1:
            print("Index exceeds length!")    
        self.gdf = self.gdf.drop(index).reset_index(drop=True)
        return
        
    def clear(self):
        self.gdf  = gpd.GeoDataFrame()

    def snap_to_grid(self):
        snap_gdf = self.model.grid.snap_to_grid(self.gdf)
        return snap_gdf

    def list_names(self):
        names = []
        for index, row in self.gdf.iterrows():
            names.append(str(index + 1))
        return names
