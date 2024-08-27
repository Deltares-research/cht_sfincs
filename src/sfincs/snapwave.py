# -*- coding: utf-8 -*-
"""
Created on Sat Jun 18 09:03:08 2022
@author: ormondt
"""
import os
import geopandas as gpd
import shapely
import pandas as pd
import numpy as np
from tabulate import tabulate
from matplotlib import path
import xugrid as xu
import xarray as xr

from cht.misc.pli_file import pol2gdf, gdf2pol


class SnapWaveMask:
    def __init__(self, model):
        self.model       = model

    def build(self,
              zmin=99999.0,
              zmax=-99999.0,
              include_polygon=None,
              exclude_polygon=None,
              include_zmin=-99999.0,
              include_zmax= 99999.0,
              exclude_zmin=-99999.0,
              exclude_zmax= 99999.0,
              quiet=True):

        if not quiet:
            print("Building SnapWave mask ...")

        mask = np.zeros(self.model.grid.nr_cells, dtype=np.int8)
        x, y = self.model.grid.face_coordinates()
        z = self.model.grid.data["z"] 

        if zmin>=zmax:
            # Do not include any points initially
            if include_polygon is None:
                print("WARNING: Entire mask set to zeros! Please ensure zmax is greater than zmin, or provide include polygon(s) !")
                return
        else:
            if z is not None:                
                # Set initial mask based on zmin and zmax
                iok = np.where((z>=zmin) & (z<=zmax))
                mask[iok] = 1
            else:
                print("WARNING: Entire mask set to zeros! No depth values found on grid.")
                        
        # Include polygons
        if include_polygon is not None:
            for ip, polygon in include_polygon.iterrows():
                inpol = inpolygon(x, y, polygon["geometry"])
                iok   = np.where((inpol) & (z>=include_zmin) & (z<=include_zmax))
                mask[iok] = 1

        # Exclude polygons
        if exclude_polygon is not None:
            for ip, polygon in exclude_polygon.iterrows():
                inpol = inpolygon(x, y, polygon["geometry"])
                iok   = np.where((inpol) & (z>=exclude_zmin) & (z<=exclude_zmax))
                mask[iok] = 0

        # Now add the data arrays
        ugrid2d = self.model.grid.data.grid
        self.model.grid.data["snapwave_mask"] = xu.UgridDataArray(xr.DataArray(data=mask, dims=[ugrid2d.face_dimension]), ugrid2d)


    def to_gdf(self, option="all"):        
        nr_cells = self.model.grid.nr_cells
        xz, yz = self.model.grid.face_coordinates()
        mask = self.model.grid.data["snapwave_mask"] 
        gdf_list = []
        okay = np.zeros(mask.shape, dtype=int)
        if option == "all":
            iok = np.where((mask > 0))
        elif option == "include":
            iok = np.where((mask == 1))
        # elif option == "open":
        #     iok = np.where((mask == 2))
        # elif option == "outflow":
        #     iok = np.where((mask == 3))
        else:
            iok = np.where((mask > -999))
        okay[iok] = 1
        for icel in range(nr_cells):
            if okay[icel] == 1:
                point = shapely.geometry.Point(xz[icel], yz[icel])
                d = {"geometry": point}
                gdf_list.append(d)

        if gdf_list:
            gdf = gpd.GeoDataFrame(gdf_list, crs=self.model.crs)
        else:
            # Cannot set crs of gdf with empty list
            gdf = gpd.GeoDataFrame(gdf_list)

        return gdf

    def has_open_boundaries(self):
        mask = self.model.grid.data["snapwave_mask"]
        if mask is None:
            return False
        if np.any(mask == 2):
            return True
        else:
            return False




class SnapWaveBoundaryEnclosure:
    def __init__(self, model):
        self.model = model
        self.gdf = gpd.GeoDataFrame()

    def read(self):
        # Read in enclosure from file
        if not self.model.input.variables.snapwave_encfile:
            return
        file_name = os.path.join(self.model.path, self.model.input.variables.snapwave_encfile)
        self.gdf = pol2gdf(file_name, crs=self.model.crs, header=False)

    def write(self):
        if not self.model.input.variables.snapwave_encfile:
            return
        if len(self.gdf.index)==0:
            return
        file_name = os.path.join(self.model.path, self.model.input.variables.snapwave_encfile)
        gdf2pol(self.gdf, file_name, header=False)

class SnapWaveBoundaryConditions:
    def __init__(self, model):
        self.model = model
        self.gdf = gpd.GeoDataFrame()
        self.times = []

    def read(self):
        # Read in all boundary data
        self.read_boundary_points()
        self.read_boundary_time_series()


    def write(self):
        # Write all boundary data
        self.write_boundary_points()
        self.write_boundary_conditions_timeseries()


    def read_boundary_points(self):
        # Read bnd file
        if not self.model.input.variables.snapwave_bndfile:
            return

        file_name = os.path.join(self.model.path, self.model.input.variables.snapwave_bndfile)

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
            d = {"name": name, "timeseries": pd.DataFrame(), "spectra": None, "geometry": point}
            gdf_list.append(d)
        self.gdf = gpd.GeoDataFrame(gdf_list, crs=self.model.crs)


    def write_boundary_points(self):
        # Write bnd file

        if len(self.gdf.index)==0:
            return

        if not self.model.input.variables.snapwave_bndfile:
            self.model.input.variables.snapwave_bndfile = "snapwave.bnd"

        file_name = os.path.join(self.model.path, self.model.input.variables.snapwave_bndfile)

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

    def set_timeseries_uniform(self, hs, tp, wd, ds):
        # Applies uniform time series boundary conditions for each point
        time = [self.model.input.variables.tstart, self.model.input.variables.tstop]
        nt = len(time)
        hs = [hs] * nt
        tp = [tp] * nt
        wd = [wd] * nt
        ds = [ds] * nt
        for index, point in self.gdf.iterrows():
            df = pd.DataFrame()     
            df["time"] = time
            df["hs"] = hs
            df["tp"] = tp
            df["wd"] = wd
            df["ds"] = ds
            df = df.set_index("time")
            self.gdf.at[index, "timeseries"] = df

    def set_conditions_at_point(self, index, par, val):
        df = self.gdf["timeseries"].loc[index]
        df[par] = val

    def add_point(self, x, y, hs=None, tp=None, wd=None, ds=None, sp=None):
        # Add point
        nrp = len(self.gdf.index)
        name = str(nrp + 1).zfill(4)
        point = shapely.geometry.Point(x, y)
        df = pd.DataFrame()     

        if hs:
            # Forcing by time series        
            if not self.model.input.variables.snapwave_bndfile:
                self.model.input.variables.snapwave_bndfile = "snapwave.bnd"
            if not self.model.input.variables.snapwave_bhsfile:
                self.model.input.variables.snapwave_bhsfile = "snapwave.bhs"
            if not self.model.input.variables.snapwave_btpfile:
                self.model.input.variables.snapwave_btpfile = "snapwave.btp"
            if not self.model.input.variables.snapwave_bwdfile:
                self.model.input.variables.snapwave_bwdfile = "snapwave.bwd"
            if not self.model.input.variables.snapwave_bdsfile:
                self.model.input.variables.snapwave_bdsfile = "snapwave.bds"
                        
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

            hs = [hs] * nt
            tp = [tp] * nt
            wd = [wd] * nt
            ds = [ds] * nt

            df["time"] = time
            df["hs"] = hs
            df["tp"] = tp
            df["wd"] = wd
            df["ds"] = ds
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
        # Rename points    
        for index, point in self.gdf.iterrows():
            self.gdf.at[index, "name"] = str(index + 1).zfill(4)
        

    def clear(self):
        self.gdf  = gpd.GeoDataFrame()


    def read_boundary_time_series(self):
        # Read SnapWave bhs, btp, bwd and bds files

        if not self.model.input.variables.snapwave_bhsfile:
            return
        if len(self.gdf.index)==0:
            return

        tref = self.model.input.variables.tref

        # Time
        
        # Hs        
        file_name = os.path.join(self.model.path, self.model.input.variables.snapwave_bhsfile)
        dffile = read_timeseries_file(file_name, tref)
        # Loop through boundary points
        for ip, point in self.gdf.iterrows():
            point["timeseries"]["time"] = dffile.index
            point["timeseries"]["hs"] = dffile.iloc[:, ip].values
            point["timeseries"].set_index("time", inplace=True)

        # Tp       
        file_name = os.path.join(self.model.path, self.model.input.variables.snapwave_btpfile)
        dffile = read_timeseries_file(file_name, tref)
        for ip, point in self.gdf.iterrows():
            point["timeseries"]["tp"] = dffile.iloc[:, ip].values

        # Wd
        file_name = os.path.join(self.model.path, self.model.input.variables.snapwave_bwdfile)
        dffile = read_timeseries_file(file_name, tref)
        for ip, point in self.gdf.iterrows():
            point["timeseries"]["wd"] = dffile.iloc[:, ip].values

        # Ds
        file_name = os.path.join(self.model.path, self.model.input.variables.snapwave_bdsfile)
        dffile = read_timeseries_file(file_name, tref)
        for ip, point in self.gdf.iterrows():
            point["timeseries"]["ds"] = dffile.iloc[:, ip].values


    def write_boundary_conditions_timeseries(self):
        if len(self.gdf.index)==0:
            return
        # First get times from the first point (times in other points should be identical)
        time = self.gdf.loc[0]["timeseries"].index
        tref = self.model.input.variables.tref
        dt   = (time - tref).total_seconds()
        
        # Hs
        if not self.model.input.variables.snapwave_bhsfile:
            self.model.input.variables.bhsfile = "snapwave.bhs"            
        file_name = os.path.join(self.model.path, self.model.input.variables.snapwave_bhsfile)
        # Build a new DataFrame
        df = pd.DataFrame()
        for ip, point in self.gdf.iterrows():
            df = pd.concat([df, point["timeseries"]["hs"]], axis=1)
        df.index = dt
        # df.to_csv(file_name,
        #           index=True,
        #           sep=" ",
        #           header=False,
        #           float_format="%.3f")
        to_fwf(df, file_name)
    
        # Tp
        if not self.model.input.variables.snapwave_btpfile:
            self.model.input.variables.btpfile = "snapwave.btp"            
        file_name = os.path.join(self.model.path, self.model.input.variables.snapwave_btpfile)
        # Build a new DataFrame
        df = pd.DataFrame()
        for ip, point in self.gdf.iterrows():
            df = pd.concat([df, point["timeseries"]["tp"]], axis=1)
        df.index = dt
        # df.to_csv(file_name,
        #           index=True,
        #           sep=" ",
        #           header=False,
        #           float_format="%.3f")
        to_fwf(df, file_name)

        # Wd
        if not self.model.input.variables.snapwave_bwdfile:
            self.model.input.variables.bwdfile = "snapwave.bwd"            
        file_name = os.path.join(self.model.path, self.model.input.variables.snapwave_bwdfile)
        # Build a new DataFrame
        df = pd.DataFrame()
        for ip, point in self.gdf.iterrows():
            df = pd.concat([df, point["timeseries"]["wd"]], axis=1)
        df.index = dt
        # df.to_csv(file_name,
        #           index=True,
        #           sep=" ",
        #           header=False,
        #           float_format="%.3f")
        to_fwf(df, file_name)

        # Ds
        if not self.model.input.variables.snapwave_bdsfile:
            self.model.input.variables.bdsfile = "snapwave.bds"            
        file_name = os.path.join(self.model.path, self.model.input.variables.snapwave_bdsfile)
        # Build a new DataFrame
        df = pd.DataFrame()
        for ip, point in self.gdf.iterrows():
            df = pd.concat([df, point["timeseries"]["ds"]], axis=1)
        df.index = dt
        # df.to_csv(file_name,
        #           index=True,
        #           sep=" ",
        #           header=False,
        #           float_format="%.3f")
        to_fwf(df, file_name)

class SfincsSnapWave:
    def __init__(self, model):
        self.model = model
        self.mask                = SnapWaveMask(model)
        self.boundary_enclosure  = SnapWaveBoundaryEnclosure(model)
        self.boundary_conditions = SnapWaveBoundaryConditions(model)
    def read(self):
        # Read in all SnapWave data
        self.boundary_enclosure.read()
        self.boundary_conditions.read()
    def write(self):
        self.boundary_enclosure.write()
        self.boundary_conditions.write()


    # def get_boundary_points_from_mask(self, min_dist=None, bnd_dist=50000.0):

    #     if min_dist is None:
    #         # Set minimum distance between to grid boundary points on polyline to 2 * dx
    #         min_dist = self.model.grid.dx * 2 

    #     # Get coordinates of boundary points
    #     da_mask = self.model.grid.ds["mask"]
    #     ibnd = np.where(da_mask.values == 2)
    #     xp = da_mask["x"].values[ibnd]
    #     yp = da_mask["y"].values[ibnd]



    #     # Make boolean array for points that are include in a polyline 
    #     used = np.full(xp.shape, False, dtype=bool)

    #     polylines = []

    #     while True:

    #         if np.all(used):
    #             # All boundary grid points have been used. We can stop now.
    #             break

    #         # Find first of the unused points
    #         i1 = np.where(used==False)[0][0]

    #         # Set this point to used
    #         used[i1] = True

    #         polyline = [i1] 

    #         while True:
    #             if np.all(used):
    #                 # All boundary grid points have been used. We can stop now.
    #                 break
    #             # Started new polyline
    #             dst = np.sqrt((xp - xp[i1])**2 + (yp - yp[i1])**2)
    #             dst[polyline] = np.nan
    #             inear = np.nanargmin(dst)
    #             if dst[inear] < min_dist:
    #                 # Found next point along polyline
    #                 polyline.append(inear)
    #                 used[inear] = True
    #                 i1 = inear
    #             else:
    #                 # Last point found
    #                 break    

    #         i1 = polyline[0]
    #         while True:
    #             if np.all(used):
    #                 # All boundary grid points have been used. We can stop now.
    #                 break
    #             # Now we go in the other direction            
    #             dst = np.sqrt((xp - xp[i1])**2 + (yp - yp[i1])**2)
    #             dst[polyline] = np.nan
    #             inear = np.nanargmin(dst)
    #             if dst[inear] < min_dist:
    #                 # Found next point along polyline
    #                 polyline.insert(0, inear)
    #                 used[inear] = True
    #                 i1 = inear
    #             else:
    #                 # Last point found
    #                 # On to the next polyline
    #                 break    

    #         if len(polyline) > 1:  
    #             polylines.append(polyline)

    #     gdf_list = []
    #     ip = 0

    #     # If geographic, convert to Web Mercator
    #     if self.model.crs.is_geographic:
    #         transformer = Transformer.from_crs(self.model.crs,
    #                                            3857,
    #                                            always_xy=True)

    #     # Loop through polylines 
    #     for polyline in polylines:
    #         x = xp[polyline]
    #         y = yp[polyline]
    #         points = [(x,y) for x,y in zip(x.ravel(),y.ravel())]                
    #         line = shapely.geometry.LineString(points)
    #         if self.model.crs.is_geographic:
    #             # Line in web mercator (to get length in metres)
    #             xm, ym = transformer.transform(x, y)
    #             pointsm = [(xm,ym) for xm,ym in zip(xm.ravel(),ym.ravel())]
    #             linem = shapely.geometry.LineString(pointsm)
    #             num_points = int(linem.length / bnd_dist) + 2
    #         else:
    #             num_points = int(line.length / bnd_dist) + 2
    #         # If geographic, convert to Web Mercator
    #         new_points = [line.interpolate(i/float(num_points - 1), normalized=True) for i in range(num_points)]
    #         # Loop through points in polyline
    #         for point in new_points:
    #             name = str(ip + 1).zfill(4)
    #             d = {"name": name, "timeseries": pd.DataFrame(), "spectra": None, "geometry": point}
    #             gdf_list.append(d)
    #             ip += 1

    #     self.gdf = gpd.GeoDataFrame(gdf_list, crs=self.model.crs)

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
    
def inpolygon(xq, yq, p):
    shape = xq.shape
    xq = xq.reshape(-1)
    yq = yq.reshape(-1)
#    xv = xv.reshape(-1)
#    yv = yv.reshape(-1)
    q = [(xq[i], yq[i]) for i in range(xq.shape[0])]
#    q = [Point(xq[i], yq[i]) for i in range(xq.shape[0])]
#    mp = MultiPoint(q)
    p = path.Path([(crds[0], crds[1]) for i, crds in enumerate(p.exterior.coords)])
#    p = path.Path([(xv[i], yv[i]) for i in range(xv.shape[0])])
    return p.contains_points(q).reshape(shape)
#    return mp.within(p)

