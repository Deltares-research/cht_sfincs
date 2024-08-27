# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 17:24:49 2022

@author: ormondt
"""
import time
import os
import numpy as np
from matplotlib import path
# import matplotlib.pyplot as plt
from pyproj import CRS, Transformer
import shapely

import xugrid as xu
import xarray as xr
#from .to_xugrid import xug
import warnings
np.warnings = warnings

import geopandas as gpd
import pandas as pd

import datashader as ds
import datashader.transfer_functions as tf
from datashader.utils import export_image

from cht.misc.misc_tools import interp2

class SfincsMask:
    def __init__(self, model):
        self.model       = model

    def build(self,
                 zmin=99999.0,
                 zmax=-99999.0,
                 include_polygon=None,
                 exclude_polygon=None,
                 open_boundary_polygon=None,
                 outflow_boundary_polygon=None,
                 include_zmin=-99999.0,
                 include_zmax= 99999.0,
                 exclude_zmin=-99999.0,
                 exclude_zmax= 99999.0,
                 open_boundary_zmin=-99999.0,
                 open_boundary_zmax= 99999.0,
                 outflow_boundary_zmin=-99999.0,
                 outflow_boundary_zmax= 99999.0,
                 quiet=True):

        if not quiet:
            print("Building mask ...")

        mask = np.zeros(self.model.grid.nr_cells, dtype=np.int8)
        x, y = self.model.grid.face_coordinates()
        z    = self.model.grid.data["z"].values[:]

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

        # Outflow boundary polygons
        if outflow_boundary_polygon is not None:
            for ip, polygon in outflow_boundary_polygon.iterrows():
                inpol = inpolygon(x, y, polygon["geometry"])
                # Only consider points that are:
                # 1) Inside the polygon
                # 2) Have a mask > 0
                # 3) z>=zmin
                # 4) z<=zmax
                iok   = np.where((inpol) & (mask>0) & (z>=outflow_boundary_zmin) & (z<=outflow_boundary_zmax))
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
        self.model.grid.data["mask"] = xu.UgridDataArray(xr.DataArray(data=mask, dims=[ugrid2d.face_dimension]), ugrid2d)

    # def read(self, file_name=None):
    #     pass

    # def write(self):
    #     mask = self.model.grid.data["mask"].values[:]
    #     file_name = os.path.join(self.model.path, self.model.input.variables.mskfile)
    #     file = open(file_name, "wb")
    #     file.write(np.int8(mask))
    #     file.close()

    def to_gdf(self, option="all"):
        nr_cells = self.model.grid.nr_cells
        if nr_cells == 0:
            # Return empty geodataframe
            return gpd.GeoDataFrame()
        xz, yz = self.model.grid.face_coordinates()
        mask = self.model.grid.data["mask"] 
        gdf_list = []
        okay = np.zeros(mask.shape, dtype=int)
        if option == "all":
            iok = np.where((mask > 0))
        elif option == "include":
            iok = np.where((mask == 1))
        elif option == "open":
            iok = np.where((mask == 2))
        elif option == "outflow":
            iok = np.where((mask == 3))
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

    def build_mask_snapwave(self,
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
            print("Building SnapWave quadtree mask ...")

        mask_snapwave = np.zeros(self.nr_cells, dtype=int)

        if zmin>=zmax:
            # Do not include any points initially
            if include_polygon is None:
                print("WARNING: Entire mask set to zeros! Please ensure zmax is greater than zmin, or provide include polygon(s) !")
                return
        else:
            if z is not None:                
                # Set initial mask based on zmin and zmax
                iok = np.where((z>=zmin) & (z<=zmax))
                mask_snapwave[iok] = 1
            else:
                print("WARNING: Entire mask set to zeros! No depth values found on grid.")
                        
        # Include polygons
        if include_polygon is not None:
            for ip, polygon in include_polygon.iterrows():
                inpol = inpolygon(x, y, polygon["geometry"])
                iok   = np.where((inpol) & (z>=include_zmin) & (z<=include_zmax))
                mask_snapwave[iok] = 1

        # Exclude polygons
        if exclude_polygon is not None:
            for ip, polygon in exclude_polygon.iterrows():
                inpol = inpolygon(x, y, polygon["geometry"])
                iok   = np.where((inpol) & (z>=exclude_zmin) & (z<=exclude_zmax))
                mask_snapwave[iok] = 0

    def read(self):
        # Read in index file, mask file and dep file
        msk  = np.full([self.model.input.variables.nmax*self.model.input.variables.mmax], 0)
        ind  = np.fromfile(self.model.input.variables.indexfile, dtype="i4")
        npoints  = ind[0]
        ind = np.squeeze(ind[1:]) - 1
        mskv = np.fromfile(self.model.input.variables.mskfile, dtype="i1")
        msk[ind] = mskv
        dep = np.full([self.model.input.variables.nmax*self.model.input.variables.mmax], 0.0)
        if self.model.input.variables.depfile:
            depv  = np.fromfile(self.model.input.variables.depfile, dtype="f4")
            dep[ind] = depv
        self.model.grid.data["mask"].values[:] = msk
        self.model.grid.data["z"].values[:] = dep

    def write(self):
        """Write msk, ind, and dep files"""
        mskv = self.model.grid.data["mask"].values
        ind  = np.where(mskv>0)
        mskv = mskv[ind]
        depv = self.model.grid.data["z"].values[ind]

        # Add 1 because indices in SFINCS start with 1, not 0
        ind = ind[0] + 1

        # Write index file
        self.model.input.variables.indexfile = "sfincs.ind"
        file = open(self.model.input.variables.indexfile, "wb")
        file.write(np.int32(np.size(ind)))
        file.write(np.int32(ind))
        file.close()
        
        # Write mask file
        self.model.input.variables.mskfile = "sfincs.msk"
        file = open(self.model.input.variables.mskfile, "wb")
        file.write(np.int8(mskv))
        file.close()

        # Write dep file
        self.model.input.variables.depfile = "sfincs.dep"
        file = open(self.model.input.variables.depfile, "wb")
        file.write(np.float32(depv))
        file.close()

#     def write_msk_file_snapwave(self):
#         file_name = os.path.join(self.model.path, self.model.input.variables.snapwave_mskfile)
#         file = open(file_name, "wb")
#         file.write(np.int8(mask_snapwave))
#         file.close()

#     def mask_to_gdf_snapwave(self, option="all"):
#         # xz = self.ds["x"].values[:]
#         # yz = self.ds["y"].values[:]
#         xz = x
#         yz = y
# #        mask = self.ds["mask"].values[:]
#         mask = mask_snapwave
#         gdf_list = []
#         okay = np.zeros(mask.shape, dtype=int)
#         if option == "all":
#             iok = np.where((mask > 0))
#         elif option == "include":
#             iok = np.where((mask == 1))
#         # elif option == "open":
#         #     iok = np.where((mask == 2))
#         # elif option == "outflow":
#         #     iok = np.where((mask == 3))
#         else:
#             iok = np.where((mask > -999))
#         okay[iok] = 1
#         for icel in range(self.nr_cells):
#             if okay[icel] == 1:
#                 point = shapely.geometry.Point(xz[icel], yz[icel])
#                 d = {"geometry": point}
#                 gdf_list.append(d)

#         if gdf_list:
#             gdf = gpd.GeoDataFrame(gdf_list, crs=self.model.crs)
#         else:
#             # Cannot set crs of gdf with empty list
#             gdf = gpd.GeoDataFrame(gdf_list)

#         return gdf


    def has_open_boundaries(self):
        mask = self.model.grid.data["mask"]
        if mask is None:
            return False
        if np.any(mask == 2):
            return True
        else:
            return False


def get_neighbors_in_larger_cell(n, m):    
    nnbr = [-1, -1, -1, -1]
    mnbr = [-1, -1, -1, -1]
    if not odd(n) and not odd(m):
        # lower left
        nnbr[0] = n + 1
        mnbr[0] = m
        nnbr[1] = n
        mnbr[1] = m + 1
        nnbr[2] = n + 1
        mnbr[2] = m + 1
    elif not odd(n) and odd(m):
        # lower right
        nnbr[1] = n
        mnbr[1] = m - 1
        nnbr[2] = n + 1
        mnbr[2] = m - 1
        nnbr[3] = n + 1
        mnbr[3] = m
    elif odd(n) and not odd(m):
        # upper left
        nnbr[1] = n - 1
        mnbr[1] = m
        nnbr[2] = n - 1
        mnbr[2] = m + 1
        nnbr[3] = n
        mnbr[3] = m + 1
    else:
        # upper right
        nnbr[1] = n - 1
        mnbr[1] = m - 1
        nnbr[2] = n - 1
        mnbr[2] = m
        nnbr[3] = n
        mnbr[3] = m - 1    
    return nnbr,mnbr

def odd(num):
    if (num % 2) == 1:  
        return True
    else:  
        return False

def even(num):
    if (num % 2) == 0:  
        return True
    else:  
        return False

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

def binary_search(vals, val):    
    indx = np.searchsorted(vals, val)
    if indx<np.size(vals):
        if vals[indx] == val:
            return indx
    return None
