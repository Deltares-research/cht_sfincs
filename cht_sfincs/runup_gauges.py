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

class SfincsRunupGauges:
    def __init__(self, sf):
        self.model = sf
        self.gdf  = gpd.GeoDataFrame()

    def read(self):

        # Read in all cross sections

        if not self.model.input.variables.rugfile:
            return

        filename = os.path.join(self.model.path, self.model.input.variables.rugfile)

        # Read the crs file
        self.gdf = pli2gdf(filename, crs=self.model.crs)

    def write(self, file_name=None):

        if not self.model.input.variables.rugfile:
            return
        if len(self.gdf.index)==0:
            return

        filename = os.path.join(self.model.path, self.model.input.variables.rugfile)

        gdf2pli(self.gdf, filename)

    def add(self, runup_gauge):
        runup_gauge.set_crs(self.model.crs)
        self.gdf = pd.concat([self.gdf, runup_gauge], ignore_index=True)

    def add_xy(self, name, x0, y0, x1, y1):
        """
        Add a runup gauge to the list of runup gauges.
        Parameters:
        name : str
            The name of the runup gauge.
        x0 : float
            The x-coordinate of the start point of the runup gauge.
        y0 : float
            The y-coordinate of the start point of the runup gauge.
        x1 : float
            The x-coordinate of the end point of the runup gauge.
        y1 : float
            The y-coordinate of the end point of the runup gauge.
        """
        line = shapely.geometry.LineString([(x0, y0), (x1, y1)])
        gdf = gpd.GeoDataFrame({"name": [name], "geometry": [line]}).set_crs(self.model.crs)
        self.gdf = pd.concat([self.gdf, gdf], ignore_index=True)

    def delete(self, name_or_index):
        if type(name_or_index) is str:
            name = name_or_index
            for index, row in self.gdf.iterrows():
                if row["name"] == name:
                    self.gdf = self.gdf.drop(index).reset_index(drop=True)
                    return
            print("Run-up gauge " + name + " not found!")    
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
