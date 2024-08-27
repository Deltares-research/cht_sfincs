# -*- coding: utf-8 -*-
"""
Created on Sat Jun 18 09:03:08 2022
@author: ormondt
"""
import os
import geopandas as gpd
import shapely
import pandas as pd

class SfincsThinDams:
    def __init__(self, sf):
        self.model = sf
        self.gdf  = gpd.GeoDataFrame()

    def read(self):
        # Read in all observation points

        if not self.model.input.variables.thdfile:
            return

        file_name = os.path.join(self.model.path, self.model.input.variables.thdfile)

        self.gdf = gpd.GeoDataFrame()
        return

        # Read the thd file
        gdf = tek2gdf(file_name, shape="line")
        self.gdf = gdf
        return

        df = pd.read_csv(file_name, index_col=False, header=None,
             delim_whitespace=True, names=['x', 'y', 'name'])

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

    def write(self):

        if not self.model.input.variables.thdfile:
            return
        if len(self.gdf.index)==0:
            return

        file_name = os.path.join(self.model.path, self.model.input.variables.thdfile)
        
        if self.model.crs.is_geographic:
            gdf2tek(self.gdf, file_name)
            # fid = open(file_name, "w")
            # for index, row in self.gdf.iterrows():
            #     x = row["geometry"].coords[0][0]
            #     y = row["geometry"].coords[0][1]
            #     name = row["name"]
            #     string = f'{x:12.6f}{y:12.6f}  "{name}"\n'
            #     fid.write(string)
            # fid.close()
        else:
            fid = open(file_name, "w")
            for index, row in self.gdf.iterrows():
                x = row["geometry"].coords[0][0]
                y = row["geometry"].coords[0][1]
                name = row["name"]
                string = f'{x:12.1f}{y:12.1f}  "{name}"\n'
                fid.write(string)
            fid.close()

    def add(self, thin_dam):
        # Thin dam may be a gdf or shapely geometry
        # Assume for now a gdf
        thin_dam.set_crs(self.model.crs)
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
