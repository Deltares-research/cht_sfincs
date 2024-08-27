# -*- coding: utf-8 -*-
"""
Created on Sat Jun 18 09:03:08 2022
@author: ormondt
"""
import os
import numpy as np
import geopandas as gpd
import shapely
import pandas as pd
from tabulate import tabulate
from pyproj import Transformer

class SfincsBoundaryConditions:

    def __init__(self, sf):
        self.model = sf
        self.forcing = "timeseries"
        self.gdf = gpd.GeoDataFrame()
        self.times = []

    def read(self):
        # Read in all boundary data
        self.read_boundary_points()
        self.read_boundary_conditions_time_series()


    def write(self):
        # Write all boundary data
        self.write_boundary_points()
        self.write_boundary_conditions_timeseries()


    def read_boundary_points(self):
        # Read bnd file
        if not self.model.input.variables.bndfile:
            return

        file_name = os.path.join(self.model.path, self.model.input.variables.bndfile)

        # Read the bnd file
        df = pd.read_csv(file_name, index_col=False, header=None,
                         delim_whitespace=True, names=['x', 'y'])

        gdf_list = []
        # Loop through points
        for ind in range(len(df.x.values)):
            name = str(ind + 1).zfill(4)
            x = df.x.values[ind]
            y = df.y.values[ind]
            point = shapely.geometry.Point(x, y)
            d = {"name": name, "timeseries": pd.DataFrame(), "geometry": point}
            gdf_list.append(d)
        self.gdf = gpd.GeoDataFrame(gdf_list, crs=self.model.crs)


    def write_boundary_points(self):
        # Write bnd file

        if len(self.gdf.index)==0:
            return

        if not self.model.input.variables.bndfile:
            self.model.input.variables.bndfile = "sfincs.bnd"

        file_name = os.path.join(self.model.path, self.model.input.variables.bndfile)

        if self.model.crs.is_geographic:
            fid = open(file_name, "w")
            for index, row in self.gdf.iterrows():
                x = row["geometry"].coords[0][0]
                y = row["geometry"].coords[0][1]
                string = f'{x:12.6f}{y:12.6f}\n'
                fid.write(string)
            fid.close()
        else:
            fid = open(file_name, "w")
            for index, row in self.gdf.iterrows():
                x = row["geometry"].coords[0][0]
                y = row["geometry"].coords[0][1]
                string = f'{x:12.1f}{y:12.1f}\n'
                fid.write(string)
            fid.close()

    def add_point(self, x, y, wl):
        # Add point

        nrp = len(self.gdf.index)
        name = str(nrp + 1).zfill(4)
        point = shapely.geometry.Point(x, y)
        df = pd.DataFrame()     
       
        if not self.model.input.variables.bndfile:
            self.model.input.variables.bndfile = "sfincs.bnd"
        if not self.model.input.variables.bzsfile:
            self.model.input.variables.bzsfile = "sfincs.bzs"
                    
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

        wl = [wl] * nt

        df["time"] = time
        df["wl"]   = wl
        df = df.set_index("time")
            
        gdf_list = []
        d = {"name": name, "timeseries": df, "geometry": point}
        gdf_list.append(d)
        gdf_new = gpd.GeoDataFrame(gdf_list, crs=self.model.crs)        
        self.gdf = pd.concat([self.gdf, gdf_new], ignore_index=True)


    def delete_point(self, index):
        # Delete boundary point by index
        if len(self.gdf.index)==0:
            return
        if index<len(self.gdf.index):
            self.gdf = self.gdf.drop(index).reset_index(drop=True)
        

    def clear(self):
        self.gdf  = gpd.GeoDataFrame()


    def read_boundary_conditions_time_series(self):
        # Read SFINCS bzs file

        if not self.model.input.variables.bzsfile:
            return
        if len(self.gdf.index)==0:
            return

        tref = self.model.input.variables.tref

        # Time
        
        # WL      
        file_name = os.path.join(self.model.path, self.model.input.variables.bzsfile)
        dffile = read_timeseries_file(file_name, tref)
        # Loop through boundary points
        for ip, point in self.gdf.iterrows():
            point["timeseries"]["time"] = dffile.index
            point["timeseries"]["wl"] = dffile.iloc[:, ip].values
            point["timeseries"].set_index("time", inplace=True)


    def write_boundary_conditions_timeseries(self):
        if len(self.gdf.index)==0:
            return
        # First get times from the first point (times in other points should be identical)
        time = self.gdf.loc[0]["timeseries"].index
        tref = self.model.input.variables.tref
        dt   = (time - tref).total_seconds()
        
        # WL
        if not self.model.input.variables.bzsfile:
            self.model.input.variables.bzsfile = "sfincs.bzs"            
        file_name = os.path.join(self.model.path, self.model.input.variables.bzsfile)
        # Build a new DataFrame
        df = pd.DataFrame()
        for ip, point in self.gdf.iterrows():
            df = pd.concat([df, point["timeseries"]["wl"]], axis=1)
        df.index = dt
        # df.to_csv(file_name,
        #           index=True,
        #           sep=" ",
        #           header=False,
        #           float_format="%.3f")
        to_fwf(df, file_name)
    
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
        
        if not self.boundary_conditions:
            print("No boundary points found!")
            return

        from cht.tide.tide_predict import predict

        if not file_name:
            if not self.model.input.variables.bzsfile:
                self.model.input.variables.bzsfile = "sfincs.bzs"
            file_name = self.model.input.variables.bzsfile

        times = pd.date_range(start=self.model.input.variables.tstart,
                              end=self.model.input.variables.tstop,
                              freq=DateOffset(seconds=dt))
                              

        # Make boundary conditions based on bca file
        for point in self.flow_boundary_point:
            v = predict(point.astro, times) + offset
            ts = pd.Series(v, index=times)
            point.data = pd.Series(v, index=times)
                    
        if write_file:            
            self.write_flow_boundary_conditions()


    def get_boundary_points_from_mask(self, min_dist=None, bnd_dist=5000.0):

        if min_dist is None:
            # Set minimum distance between to grid boundary points on polyline to 2 * dx
            min_dist = self.model.grid.data.attrs["dx"] * 2 

        # # Get coordinates of boundary points
        # if self.model.grid_type == "regular":
        #     da_mask = self.model.grid.ds["mask"]
        #     ibnd = np.where(da_mask.values == 2)
        #     xp = da_mask["xc"].values[ibnd]
        #     yp = da_mask["yc"].values[ibnd]
        # else:
        mask = self.model.grid.data["mask"]
        ibnd = np.where(mask == 2)
        xz, yz = self.model.grid.face_coordinates()
        xp = xz[ibnd]
        yp = yz[ibnd]


        # Make boolean array for points that are include in a polyline 
        used = np.full(xp.shape, False, dtype=bool)

        polylines = []

        while True:

            if np.all(used):
                # All boundary grid points have been used. We can stop now.
                break

            # Find first of the unused points
            i1 = np.where(used == False)[0][0]

            # Set this point to used
            used[i1] = True

            polyline = [i1] 

            while True:
                # Started new polyline
                dst = np.sqrt((xp - xp[i1])**2 + (yp - yp[i1])**2)
                dst[polyline] = np.nan
                if np.all(np.isnan(dst)):
                    break
                inear = np.nanargmin(dst)
                if dst[inear] < min_dist:
                    # Found next point along polyline
                    polyline.append(inear)
                    used[inear] = True
                    i1 = inear
                else:
                    # Last point found
                    break    

            i1 = polyline[0]
            while True:
                if np.all(used):
                    # All boundary grid points have been used. We can stop now.
                    break
                # Now we go in the other direction            
                dst = np.sqrt((xp - xp[i1])**2 + (yp - yp[i1])**2)
                dst[polyline] = np.nan
                inear = np.nanargmin(dst)
                if dst[inear] < min_dist:
                    # Found next point along polyline
                    polyline.insert(0, inear)
                    used[inear] = True
                    i1 = inear
                else:
                    # Last point found
                    # On to the next polyline
                    break    

            if len(polyline) > 1:  
                polylines.append(polyline)

        gdf_list = []
        ip = 0
        # Transform to web mercator to get distance in metres
        if self.model.crs.is_geographic:
            transformer = Transformer.from_crs(self.model.crs,
                                            3857,
                                            always_xy=True)
        # Loop through polylines 
        for polyline in polylines:
            x = xp[polyline]
            y = yp[polyline]
            points = [(x,y) for x,y in zip(x.ravel(),y.ravel())]
            line = shapely.geometry.LineString(points)
            if self.model.crs.is_geographic:
                # Line in web mercator (to get length in metres)
                xm, ym = transformer.transform(x, y)
                pointsm = [(xm,ym) for xm,ym in zip(xm.ravel(),ym.ravel())]
                linem = shapely.geometry.LineString(pointsm)
                num_points = int(linem.length / bnd_dist) + 2
            else:
                num_points = int(line.length / bnd_dist) + 2
            # Interpolate to new points
            new_points = [line.interpolate(i/float(num_points - 1), normalized=True) for i in range(num_points)]
            # Loop through points in polyline
            for point in new_points:
                name = str(ip + 1).zfill(4)
                d = {"name": name, "timeseries": pd.DataFrame(), "geometry": point}
                gdf_list.append(d)
                ip += 1

        self.gdf = gpd.GeoDataFrame(gdf_list, crs=self.model.crs)

    def set_timeseries_uniform(self, wl):
        # Applies uniform time series boundary conditions for each point
        time = [self.model.input.variables.tstart, self.model.input.variables.tstop]
        nt = len(time)
        wl = [wl] * nt
        for index, point in self.gdf.iterrows():
            df = pd.DataFrame()     
            df["time"] = time
            df["wl"] = wl
            df = df.set_index("time")
            self.gdf.at[index, "timeseries"] = df


def read_timeseries_file(file_name, ref_date):
    # Returns a dataframe with time series for each of the columns
    df = pd.read_csv(file_name, index_col=0, header=None,
                     delim_whitespace=True)
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
    