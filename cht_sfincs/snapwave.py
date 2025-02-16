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
from pyproj import Transformer
import datashader as ds
import datashader.transfer_functions as tf
from datashader.utils import export_image

# from cht_utils.pli_file import pol2gdf, gdf2pol

class SnapWaveMask:
    def __init__(self, model):
        self.model       = model
        # For plotting map overlay (This is the only data that is stored in the object! All other data is stored in the model.grid.data["snapwave_mask"])
        self.datashader_dataframe = pd.DataFrame()

    def build(self,
              zmin=99999.0,
              zmax=-99999.0,
              include_polygon=None,
              exclude_polygon=None,
              include_zmin=-99999.0,
              include_zmax= 99999.0,
              exclude_zmin=-99999.0,
              exclude_zmax= 99999.0,
              open_boundary_polygon=None,
              neumann_boundary_polygon=None,
              open_boundary_zmin=-99999.0,
              open_boundary_zmax= 99999.0,
              neumann_boundary_zmin=-99999.0,
              neumann_boundary_zmax= 99999.0,
              quiet=True,
              update_datashader_dataframe=False):

        if not quiet:
            print("Building SnapWave mask ...")

        mask = np.zeros(self.model.grid.nr_cells, dtype=np.int8)
        x, y = self.model.grid.face_coordinates()
        z = self.model.grid.data["z"] 

        # Indices are 1-based in SFINCS so subtract 1 for python 0-based indexing
        mu    = self.model.grid.data["mu"].values[:]
        mu1   = self.model.grid.data["mu1"].values[:] - 1
        mu2   = self.model.grid.data["mu2"].values[:] - 1
        nu    = self.model.grid.data["nu"].values[:]
        nu1   = self.model.grid.data["nu1"].values[:] - 1
        nu2   = self.model.grid.data["nu2"].values[:] - 1
        md    = self.model.grid.data["md"].values[:]
        md1   = self.model.grid.data["md1"].values[:] - 1
        md2   = self.model.grid.data["md2"].values[:] - 1
        nd    = self.model.grid.data["nd"].values[:]
        nd1   = self.model.grid.data["nd1"].values[:] - 1 
        nd2   = self.model.grid.data["nd2"].values[:] - 1

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

        # Open boundary polygons
        if open_boundary_polygon is not None:
            for ip, polygon in open_boundary_polygon.iterrows():
                inpol = inpolygon(x, y, polygon["geometry"])
                # Only consider points that are:
                # 1) Inside the polygon
                # 2) Have a mask > 0
                # 3) z>=zmin
                # 4) z<=zmax
                iok   = np.where((inpol) & (mask>0) & (z>=open_boundary_zmin) & (z<=open_boundary_zmax))
                for ic in iok[0]:
                    okay = False
                    # Check neighbors, cell must have at least one inactive neighbor
                    # Left
                    if md[ic]<=0:
                        # Coarser or equal to the left
                        if md1[ic]>=0:
                            # Cell has neighbor to the left
                            if mask[md1[ic]]==0:
                                # And it's inactive
                                okay = True
                        else:
                            # No neighbor, so set mask = 2
                            okay = True
                    else:
                        # Finer to the left
                        if md1[ic]>=0:
                            # Cell has neighbor to the left
                            if mask[md1[ic]]==0:
                                # And it's inactive
                                okay = True
                        else:
                            # No neighbor, so set mask = 2
                            okay = True
                        if md2[ic]>=0:
                            # Cell has neighbor to the left
                            if mask[md2[ic]]==0:
                                # And it's inactive
                                okay = True
                        else:
                            # No neighbor, so set mask = 2
                            okay = True
                        
                    # Below
                    if nd[ic]<=0:
                        # Coarser or equal below
                        if nd1[ic]>=0:
                            # Cell has neighbor below
                            if mask[nd1[ic]]==0:
                                # And it's inactive
                                okay = True
                        else:
                            # No neighbor, so set mask = 2
                            okay = True
                    else:
                        # Finer below
                        if nd1[ic]>=0:
                            # Cell has neighbor below
                            if mask[nd1[ic]]==0:
                                # And it's inactive
                                okay = True
                        else:
                            # No neighbor, so set mask = 2
                            okay = True
                        if nd2[ic]>=0:
                            # Cell has neighbor below
                            if mask[nd2[ic]]==0:
                                # And it's inactive
                                okay = True
                        else:
                            # No neighbor, so set mask = 2
                            okay = True

                    # Right
                    if mu[ic]<=0:
                        # Coarser or equal to the right
                        if mu1[ic]>=0:
                            # Cell has neighbor to the right
                            if mask[mu1[ic]]==0:
                                # And it's inactive
                                okay = True
                        else:
                            # No neighbor, so set mask = 2
                            okay = True
                    else:
                        # Finer to the left
                        if mu1[ic]>=0:
                            # Cell has neighbor to the right
                            if mask[mu1[ic]]==0:
                                # And it's inactive
                                okay = True
                        else:
                            # No neighbor, so set mask = 2
                            okay = True
                        if mu2[ic]>=0:
                            # Cell has neighbor to the right
                            if mask[mu2[ic]]==0:
                                # And it's inactive
                                okay = True
                        else:
                            # No neighbor, so set mask = 2
                            okay = True

                    # Above
                    if nu[ic]<=0:
                        # Coarser or equal above
                        if nu1[ic]>=0:
                            # Cell has neighbor above
                            if mask[nu1[ic]]==0:
                                # And it's inactive
                                okay = True
                        else:
                            # No neighbor, so set mask = 2
                            okay = True
                    else:
                        # Finer below
                        if nu1[ic]>=0:
                            # Cell has neighbor above
                            if mask[nu1[ic]]==0:
                                # And it's inactive
                                okay = True
                        else:
                            # No neighbor, so set mask = 2
                            okay = True
                        if nu2[ic]>=0:
                            # Cell has neighbor above
                            if mask[nu2[ic]]==0:
                                # And it's inactive
                                okay = True
                        else:
                            # No neighbor, so set mask = 2
                            okay = True
                        
                    if okay:
                        mask[ic] = 2

        # neumann boundary polygons
        if neumann_boundary_polygon is not None:
            for ip, polygon in neumann_boundary_polygon.iterrows():
                inpol = inpolygon(x, y, polygon["geometry"])
                # Only consider points that are:
                # 1) Inside the polygon
                # 2) Have a mask > 0
                # 3) z>=zmin
                # 4) z<=zmax
                iok   = np.where((inpol) & (mask>0) & (z>=neumann_boundary_zmin) & (z<=neumann_boundary_zmax))
                for ic in iok[0]:
                    okay = False
                    # Check neighbors, cell must have at least one inactive neighbor
                    # Left
                    if md[ic]<=0:
                        # Coarser or equal to the left
                        if md1[ic]>=0:
                            # Cell has neighbor to the left
                            if mask[md1[ic]]==0:
                                # And it's inactive
                                okay = True
                        else:
                            # No neighbor, so set mask = 3
                            okay = True
                    else:
                        # Finer to the left
                        if md1[ic]>=0:
                            # Cell has neighbor to the left
                            if mask[md1[ic]]==0:
                                # And it's inactive
                                okay = True
                        else:
                            # No neighbor, so set mask = 3
                            okay = True
                        if md2[ic]>=0:
                            # Cell has neighbor to the left
                            if mask[md2[ic]]==0:
                                # And it's inactive
                                okay = True
                        else:
                            # No neighbor, so set mask = 3
                            okay = True
                        
                    # Below
                    if nd[ic]<=0:
                        # Coarser or equal below
                        if nd1[ic]>=0:
                            # Cell has neighbor below
                            if mask[nd1[ic]]==0:
                                # And it's inactive
                                okay = True
                        else:
                            # No neighbor, so set mask = 3
                            okay = True
                    else:
                        # Finer below
                        if nd1[ic]>=0:
                            # Cell has neighbor below
                            if mask[nd1[ic]]==0:
                                # And it's inactive
                                okay = True
                        else:
                            # No neighbor, so set mask = 3
                            okay = True
                        if nd2[ic]>=0:
                            # Cell has neighbor below
                            if mask[nd2[ic]]==0:
                                # And it's inactive
                                okay = True
                        else:
                            # No neighbor, so set mask = 3
                            okay = True

                    # Right
                    if mu[ic]<=0:
                        # Coarser or equal to the right
                        if mu1[ic]>=0:
                            # Cell has neighbor to the right
                            if mask[mu1[ic]]==0:
                                # And it's inactive
                                okay = True
                        else:
                            # No neighbor, so set mask = 3
                            okay = True
                    else:
                        # Finer to the left
                        if mu1[ic]>=0:
                            # Cell has neighbor to the right
                            if mask[mu1[ic]]==0:
                                # And it's inactive
                                okay = True
                        else:
                            # No neighbor, so set mask = 3
                            okay = True
                        if mu2[ic]>=0:
                            # Cell has neighbor to the right
                            if mask[mu2[ic]]==0:
                                # And it's inactive
                                okay = True
                        else:
                            # No neighbor, so set mask = 3
                            okay = True

                    # Above
                    if nu[ic]<=0:
                        # Coarser or equal above
                        if nu1[ic]>=0:
                            # Cell has neighbor above
                            if mask[nu1[ic]]==0:
                                # And it's inactive
                                okay = True
                        else:
                            # No neighbor, so set mask = 3
                            okay = True
                    else:
                        # Finer below
                        if nu1[ic]>=0:
                            # Cell has neighbor above
                            if mask[nu1[ic]]==0:
                                # And it's inactive
                                okay = True
                        else:
                            # No neighbor, so set mask = 3
                            okay = True
                        if nu2[ic]>=0:
                            # Cell has neighbor above
                            if mask[nu2[ic]]==0:
                                # And it's inactive
                                okay = True
                        else:
                            # No neighbor, so set mask = 3
                            okay = True                        
                    if okay:
                        mask[ic] = 3

        # Now add the data arrays
        ugrid2d = self.model.grid.data.grid
        self.model.grid.data["snapwave_mask"] = xu.UgridDataArray(xr.DataArray(data=mask, dims=[ugrid2d.face_dimension]), ugrid2d)

        if update_datashader_dataframe:
            # For use in DelftDashboard
            self.get_datashader_dataframe()

    def get_datashader_dataframe(self):
        # Create a dataframe with points elements
        # Coordinates of cell centers
        x = self.model.grid.data.grid.face_coordinates[:,0]
        y = self.model.grid.data.grid.face_coordinates[:,1]
        # Check if grid crosses the dateline
        cross_dateline = False
        if self.model.crs.is_geographic:
            if np.max(x) > 180.0:
                cross_dateline = True
        mask = self.model.grid.data["snapwave_mask"].values[:]
        # Get rid of cells with mask = 0
        iok = np.where(mask>0)
        x = x[iok]
        y = y[iok]
        mask = mask[iok]
        if np.size(x) == 0:
            # Return empty dataframe
            self.datashader_dataframe = pd.DataFrame()
            return
        # Transform all to 3857 (web mercator)
        transformer = Transformer.from_crs(self.model.crs,
                                            3857,
                                            always_xy=True)
        x, y = transformer.transform(x, y)
        if cross_dateline:
            x[x < 0] += 40075016.68557849

        self.datashader_dataframe = pd.DataFrame(dict(x=x, y=y, mask=mask))

    def clear_datashader_dataframe(self):
        # Called in model.grid.build method
        self.datashader_dataframe = pd.DataFrame()

    def map_overlay(self,
                    file_name,
                    xlim=None,
                    ylim=None,
                    active_color="yellow",
                    boundary_color="red",
                    neumann_color="green",
                    px=2,
                    width=800):

        if self.model.grid.data is None:
            # No mask points (yet)
            return False
        try:

            # Mask is empty, return False    
            if self.datashader_dataframe.empty:
                return False

            transformer = Transformer.from_crs(4326,
                                        3857,
                                        always_xy=True)
            xl0, yl0 = transformer.transform(xlim[0], ylim[0])
            xl1, yl1 = transformer.transform(xlim[1], ylim[1])
            if xl0 > xl1:
                xl1 += 40075016.68557849
            xlim = [xl0, xl1]
            ylim = [yl0, yl1]
            ratio = (ylim[1] - ylim[0]) / (xlim[1] - xlim[0])
            height = int(width * ratio)

            cvs = ds.Canvas(x_range=xlim, y_range=ylim, plot_height=height, plot_width=width)

            # With this approach we can still see colors of mask 2 and 3 even if they are not there
            # color_key = {1: active_color, 2: boundary_color, 3: neumann_color}
            # agg = cvs.points(self.datashader_dataframe, 'x', 'y', ds.min("mask"))
            # # img = tf.shade(tf.spread(agg, px=px), cmap=active_color)
            # img = tf.shade(tf.spread(agg, px=px), color_key=color_key, rescale_discrete_levels=True)
            # img = tf.stack(img, tf.shade(agg, cmap=["black"]))

            # Instead, we can create separate images for each mask and stack them
            dfact = self.datashader_dataframe[self.datashader_dataframe["mask"]==1]
            dfbnd = self.datashader_dataframe[self.datashader_dataframe["mask"]==2]
            dfout = self.datashader_dataframe[self.datashader_dataframe["mask"]==3]
            img_a = tf.shade(tf.spread(cvs.points(dfact, 'x', 'y', ds.any()), px=px), cmap=active_color)
            img_b = tf.shade(tf.spread(cvs.points(dfbnd, 'x', 'y', ds.any()), px=px), cmap=boundary_color)
            img_o = tf.shade(tf.spread(cvs.points(dfout, 'x', 'y', ds.any()), px=px), cmap=neumann_color)
            img   = tf.stack(img_a, img_b, img_o)

            path = os.path.dirname(file_name)
            if not path:
                path = os.getcwd()
            name = os.path.basename(file_name)
            name = os.path.splitext(name)[0]
            export_image(img, name, export_path=path)
            return True

        except Exception as e:
            print(e)
            return False

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
        # elif option == "neumann":
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
        if not self.model.input.variables.bwvfile:
            return

        file_name = os.path.join(self.model.path, self.model.input.variables.bwvfile)

        # Read the bnd file
        df = pd.read_csv(file_name, index_col=False, header=None,
                         sep="\s+", names=['x', 'y'])

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

        if not self.model.input.variables.bwvfile:
            self.model.input.variables.bwvfile = "snapwave.bnd"

        file_name = os.path.join(self.model.path, self.model.input.variables.bwvfile)

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
            if not self.model.input.variables.bwvfile:
                self.model.input.variables.bwvfile = "sfincs.bwv"
            if not self.model.input.variables.bhsfile:
                self.model.input.variables.bhsfile = "sfincs.bhs"
            if not self.model.input.variables.btpfile:
                self.model.input.variables.btpfile = "sfincs.btp"
            if not self.model.input.variables.bwdfile:
                self.model.input.variables.bwdfile = "sfincs.bwd"
            if not self.model.input.variables.bdsfile:
                self.model.input.variables.bdsfile = "sfincs.bds"
                        
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

        if not self.model.input.variables.bhsfile:
            return
        if len(self.gdf.index)==0:
            return

        tref = self.model.input.variables.tref

        # Time
        
        # Hs        
        file_name = os.path.join(self.model.path, self.model.input.variables.bhsfile)
        dffile = read_timeseries_file(file_name, tref)
        # Loop through boundary points
        for ip, point in self.gdf.iterrows():
            point["timeseries"]["time"] = dffile.index
            point["timeseries"]["hs"] = dffile.iloc[:, ip].values
            point["timeseries"].set_index("time", inplace=True)

        # Tp       
        file_name = os.path.join(self.model.path, self.model.input.variables.btpfile)
        dffile = read_timeseries_file(file_name, tref)
        for ip, point in self.gdf.iterrows():
            point["timeseries"]["tp"] = dffile.iloc[:, ip].values

        # Wd
        file_name = os.path.join(self.model.path, self.model.input.variables.bwdfile)
        dffile = read_timeseries_file(file_name, tref)
        for ip, point in self.gdf.iterrows():
            point["timeseries"]["wd"] = dffile.iloc[:, ip].values

        # Ds
        file_name = os.path.join(self.model.path, self.model.input.variables.bdsfile)
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
        if not self.model.input.variables.bhsfile:
            self.model.input.variables.bhsfile = "sfincs.bhs"            
        file_name = os.path.join(self.model.path, self.model.input.variables.bhsfile)
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
        if not self.model.input.variables.btpfile:
            self.model.input.variables.btpfile = "sfincs.btp"            
        file_name = os.path.join(self.model.path, self.model.input.variables.btpfile)
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
        if not self.model.input.variables.bwdfile:
            self.model.input.variables.bwdfile = "sfincs.bwd"            
        file_name = os.path.join(self.model.path, self.model.input.variables.bwdfile)
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
        if not self.model.input.variables.bdsfile:
            self.model.input.variables.bdsfile = "sfincs.bds"            
        file_name = os.path.join(self.model.path, self.model.input.variables.bdsfile)
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
        mask = self.model.grid.data["snapwave_mask"]
        ibnd = np.where(mask == 2)
        xz, yz = self.model.grid.face_coordinates()
        xp = xz[ibnd]
        yp = yz[ibnd]


        # Make boolean array for points that are include in a polyline 
        used = np.full(xp.shape, False, dtype=bool)

        # Make list of polylines. Each polyline is a list of indices of boundary points.
        polylines = []

        while True:

            if np.all(used):
                # All boundary grid points have been used. We can stop now.
                break

            # Find first the unused points
            i1 = np.where(~used)[0][0]

            # Set this point to used
            used[i1] = True

            # Start new polyline with index i1
            polyline = [i1] 

            while True:
                # Compute distances to all points that have not been used
                xpunused = xp[~used]
                ypunused = yp[~used]
                # Get all indices of unused points
                unused_indices = np.where(~used)[0]

                dst = np.sqrt((xpunused - xp[i1])**2 + (ypunused - yp[i1])**2)
                if np.all(np.isnan(dst)):
                    break
                inear = np.nanargmin(dst)
                inearall = unused_indices[inear]
                if dst[inear] < min_dist:
                    # Found next point along polyline
                    polyline.append(inearall)
                    used[inearall] = True
                    i1 = inearall
                else:
                    # Last point found
                    break
            
            # Now work the other way
            # Start with first point of polyline
            i1 = polyline[0]
            while True:
                if np.all(used):
                    # All boundary grid points have been used. We can stop now.
                    break
                # Now we go in the other direction            
                xpunused = xp[~used]
                ypunused = yp[~used]
                unused_indices = np.where(~used)[0]
                dst = np.sqrt((xpunused - xp[i1])**2 + (ypunused - yp[i1])**2)
                inear = np.nanargmin(dst)
                inearall = unused_indices[inear]
                if dst[inear] < min_dist:
                    # Found next point along polyline
                    polyline.insert(0, inearall)
                    used[inearall] = True
                    # Set index of next point
                    i1 = inearall
                else:
                    # Last nearby point found
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



class SfincsSnapWave:
    def __init__(self, model):
        self.model = model
        self.mask                = SnapWaveMask(model)
        # self.boundary_enclosure  = SnapWaveBoundaryEnclosure(model)
        self.boundary_conditions = SnapWaveBoundaryConditions(model)
    def read(self):
        # Read in all SnapWave data
        # self.boundary_enclosure.read()
        self.boundary_conditions.read()
    def write(self):
        # self.boundary_enclosure.write()
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
    df = pd.read_csv(file_name, index_col=0, header=None, sep="\s+")
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

