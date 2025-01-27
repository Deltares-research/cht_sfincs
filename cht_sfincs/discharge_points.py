# -*- coding: utf-8 -*-
"""
Created on Sat Jun 18 09:03:08 2022
@author: ormondt
"""
import os
import geopandas as gpd
import shapely
import pandas as pd
from tabulate import tabulate

class SfincsDischargePoints:
    def __init__(self, hw):
        self.model = hw
        self.gdf  = gpd.GeoDataFrame()

    def read(self):
        """Read both src and dis files"""
        # Read in both disc points
        self.read_src()
        self.read_dis()
    
    def read_src(self):
        if not self.model.input.variables.srcfile:
            return
        
        filename = os.path.join(self.model.path, self.model.input.variables.srcfile)

        # Read the bnd file
        df = pd.read_csv(filename, index_col=False, header=None,
             sep="\s+", names=['x', 'y', 'name'])

        self.gdf = gpd.GeoDataFrame()
        # Loop through points to add them to the gdf
        for ind in range(len(df.x.values)):
            # name = df.name.values[ind]
            name = str(ind + 1)
            x = df.x.values[ind]
            y = df.y.values[ind]
            self.add_point(x, y, name, q=0.0)

    def read_dis(self):
        # Check that gdf is not empty
        if len(self.gdf.index)==0:
            # No points defined
            return        
        if not self.model.input.variables.disfile:
            return
        filename = os.path.join(self.model.path, self.model.input.variables.disfile)
        # Check if file exists
        if not os.path.exists(filename):
            print(f"Warning! File {filename} does not exist!")
            return

        # Time
        tref = self.model.input.variables.tref

        dffile = read_timeseries_file(filename, tref)
        times = dffile.index

        # Loop through boundary points
        for ip, point in self.gdf.iterrows():
            point["timeseries"] = pd.DataFrame()
            point["timeseries"]["time"] = times
            point["timeseries"]["q"] = dffile.iloc[:, ip].values
            point["timeseries"].set_index("time", inplace=True)

    def write(self):
        """Write both src and dis files"""
        # Read in both disc points
        self.write_src()
        self.write_dis()

    def write_src(self, filename=None):
        """Write src file"""

        if len(self.gdf.index)==0:
            # No points defined
            return

        if not filename:
            # File name not provided
            if not self.model.input.variables.srcfile:
                # And it is not in the input file, so set it now
                self.model.input.variables.srcfile = "sfincs.src"
            filename = os.path.join(self.model.path, self.model.input.variables.srcfile)
        
        if self.model.crs.is_geographic:
            fid = open(filename, "w")
            for index, row in self.gdf.iterrows():
                x = row["geometry"].coords[0][0]
                y = row["geometry"].coords[0][1]
                name = row["name"]
                string = f'{x:12.6f}{y:12.6f}  "{name}"\n'
                fid.write(string)
            fid.close()
        else:
            fid = open(filename, "w")
            for index, row in self.gdf.iterrows():
                x = row["geometry"].coords[0][0]
                y = row["geometry"].coords[0][1]
                name = row["name"]
                string = f'{x:12.1f}{y:12.1f}  "{name}"\n'
                fid.write(string)
            fid.close()

    def write_dis(self, filename=None):
        """Write dis file"""

        if len(self.gdf.index)==0:
            # No points defined
            return
        if len(self.gdf.loc[0]["timeseries"].index) == 0:
            # No time series data
            return

        if not filename:
            # File name not provided
            if not self.model.input.variables.disfile:
                # And it is not in the input file, so set it now
                self.model.input.variables.disfile = "sfincs.dis"
            filename = os.path.join(self.model.path, self.model.input.variables.disfile)
        
        # First get times from the first point (times in other points should be identical)
        time = self.gdf.loc[0]["timeseries"].index
        tref = self.model.input.variables.tref
        dt   = (time - tref).total_seconds()
        
        # Build a new DataFrame
        df = pd.DataFrame()
        for ip, point in self.gdf.iterrows():
            df = pd.concat([df, point["timeseries"]["q"]], axis=1)
        df.index = dt
        to_fwf(df, filename)

    def add_point(self, x, y, name, q=0.0):

        point = shapely.geometry.Point(x, y)
        df = pd.DataFrame()     
                    
        new = True
        if len(self.gdf.index)>0:
            new = False
            
        if new:
            # Start and stop time
            time = [self.model.input.variables.tstart, self.model.input.variables.tstop]
        else:
            # Get times from first point
            time = self.gdf.loc[0]["timeseries"].index    
        nt = len(time)

        q = [q] * nt

        df["time"] = time
        df["q"]   = q
        df = df.set_index("time")
            
        gdf_list = []
        d = {"name": name, "timeseries": df, "geometry": point}
        gdf_list.append(d)
        gdf_new = gpd.GeoDataFrame(gdf_list, crs=self.model.crs)        
        self.gdf = pd.concat([self.gdf, gdf_new], ignore_index=True)

    def delete_point(self, name_or_index):
        if isinstance(name_or_index, str):
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

    def list_discharge_points(self):
        names = []
        for index, row in self.gdf.iterrows():
            names.append(row["name"])
        return names

def read_timeseries_file(file_name, ref_date):
    # Returns a dataframe with time series for each of the columns
    df = pd.read_csv(file_name, index_col=0, header=None,
                     sep="\s+")
    ts = ref_date + pd.to_timedelta(df.index, unit="s")
    df.index = ts
    return df

def to_fwf(df, fname, floatfmt=".3f"):
    indx = df.index.tolist()
    vals = df.values.tolist()
    for it, t in enumerate(vals):
        t.insert(0, indx[it])
    content = tabulate(vals, [], tablefmt="plain", floatfmt=floatfmt)
    open(fname, "w").write(content)
