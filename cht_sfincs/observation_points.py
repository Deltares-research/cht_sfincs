# -*- coding: utf-8 -*-
"""
Created on Sat Jun 18 09:03:08 2022
@author: ormondt
"""
import os
import geopandas as gpd
import shapely
import pandas as pd

class SfincsObservationPoints:
    def __init__(self, hw):
        self.model = hw
        self.gdf  = gpd.GeoDataFrame()

    def read(self):
        # Read in all observation points

        if not self.model.input.variables.obsfile:
            return

        file_name = os.path.join(self.model.path, self.model.input.variables.obsfile)

        # Read the bnd file
        df = pd.read_csv(file_name,
                         index_col=False,
                         header=None,
                         sep="\s+",
                         names=['x', 'y', 'name'])

        gdf_list = []
        # Loop through points
        for ind in range(len(df.x.values)):
            name = df.name.values[ind]
            x = df.x.values[ind]
            y = df.y.values[ind]
            point = shapely.geometry.Point(x, y)
            d = {"name": name, "long_name": None, "geometry": point}
            gdf_list.append(d)
        self.gdf = gpd.GeoDataFrame(gdf_list, crs=self.model.crs)

    def write(self, file_name=None):

        if len(self.gdf.index)==0:
            return

        if not file_name:
            if not self.model.input.variables.obsfile:
                return
            file_name = os.path.join(self.model.path, self.model.input.variables.obsfile)
        
        if self.model.crs.is_geographic:
            fid = open(file_name, "w")
            for index, row in self.gdf.iterrows():
                x = row["geometry"].coords[0][0]
                y = row["geometry"].coords[0][1]
                name = row["name"]
                string = f'{x:12.6f}{y:12.6f}  "{name}"\n'
                fid.write(string)
            fid.close()
        else:
            fid = open(file_name, "w")
            for index, row in self.gdf.iterrows():
                x = row["geometry"].coords[0][0]
                y = row["geometry"].coords[0][1]
                name = row["name"]
                string = f'{x:12.1f}{y:12.1f}  "{name}"\n'
                fid.write(string)
            fid.close()

    def add_point(self, x, y, name):
        point = shapely.geometry.Point(x, y)
        gdf_list = []
        d = {"name": name, "long_name": None, "geometry": point}
        gdf_list.append(d)
        gdf_new = gpd.GeoDataFrame(gdf_list, crs=self.model.crs)
        self.gdf = pd.concat([self.gdf, gdf_new], ignore_index=True)

    def delete_point(self, name_or_index):
        if type(name_or_index) == str:
            name = name_or_index
            for index, row in self.gdf.iterrows():
                if row["name"] == name:
                    self.gdf = self.gdf.drop(index).reset_index(drop=True)
                    return
            print("Point " + name + " not found!")    
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
