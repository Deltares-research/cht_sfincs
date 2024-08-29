# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 17:24:49 2022

@author: ormondt
"""
#import time
import numpy as np
from matplotlib import path
import math
# import matplotlib.pyplot as plt
from pyproj import Transformer

from cht_utils.misc_tools import interp2
from cht_bathymetry.bathymetry_database import bathymetry_database


class RegularGrid:
    def __init__(self,
                 x0,
                 y0,
                 dx,
                 dy,
                 nmax,
                 mmax,
                 rotation,
                 crs=None):

        self.x0 = x0
        self.y0 = y0
        self.dx = dx
        self.dy = dy
        self.nmax = nmax
        self.mmax = mmax
        self.rotation = rotation

        cosrot = math.cos(rotation*math.pi/180)
        sinrot = math.sin(rotation*math.pi/180)
        
        xx     = np.linspace(0.5*self.dx,
                                 self.mmax*self.dx - 0.5*self.dx,
                                 num=self.mmax)
        yy     = np.linspace(0.5*self.dy,
                                 self.nmax*self.dy - 0.5*self.dy,
                                 num=self.nmax)
            
        xg0, yg0 = np.meshgrid(xx, yy)
        xg = self.x0 + xg0*cosrot - yg0*sinrot
        yg = self.y0 + xg0*sinrot + yg0*cosrot

        self.crs  = crs
        self.xz   = xg
        self.yz   = yg
        self.zz   = None
        self.mask = None

    def get_bathymetry(self, bathymetry_sets, quiet=False):
        
        if not quiet:
            print("Getting bathymetry ...")

        # Initialize depth of depth to NaN
        self.zz = np.full(np.shape(self.xz), np.nan)

        # Prepare transformers
        bathymetry_transformers = []  
        for bathymetry in bathymetry_sets:
            bathymetry_transformers.append(Transformer.from_crs(self.crs,
                                                                bathymetry.crs,
                                                                always_xy=True))
        if self.crs.is_geographic:
            mean_lat =np.abs(np.mean(self.yz))
            dxm = self.dx*111111.0*np.cos(np.pi*mean_lat/180.0)
            dym = self.dy*111111.0
        else:
            dxm = self.dx
            dym = self.dy
            
        dxym = min(dxm, dym)    
        
        # Loop through bathymetry datasets
        for ibathy, bathymetry in enumerate(bathymetry_sets):

            if np.isnan(self.zz).any():

                xgb, ygb = bathymetry_transformers[ibathy].transform(self.xz, self.yz)

                if bathymetry.type == "source":
                
                    xmin = np.nanmin(np.nanmin(xgb))
                    xmax = np.nanmax(np.nanmax(xgb))
                    ymin = np.nanmin(np.nanmin(ygb))
                    ymax = np.nanmax(np.nanmax(ygb))
                    ddx  = 0.05*(xmax - xmin)
                    ddy  = 0.05*(ymax - ymin)
                    xl   = [xmin - ddx, xmax + ddx]
                    yl   = [ymin - ddy, ymax + ddy]
                
                    # Get DEM data (ddb format for now)
                    xb, yb, zb = bathymetry_database.get_data(bathymetry.name,
                                                              xl,
                                                              yl,
                                                              max_cell_size=dxym)

                    if zb is not np.nan:
                        zb[np.where(zb<bathymetry.zmin)] = np.nan
                        zb[np.where(zb>bathymetry.zmax)] = np.nan
                        if not np.isnan(zb).all():
                            zg1 = interp2(xb, yb, zb, xgb, ygb)
                            isn = np.where(np.isnan(self.zz))
                            self.zz[isn] = zg1[isn]
            
            elif bathymetry.type == "array":
                # Matrix provided, interpolate to subgrid mesh
                # TODO
#                            zg = interp2(bathymetry.x, bathymetry.y, bathymetry.z, xgb, ygb)
                pass

    def make_mask(self,
                  zmin=99999.0,
                  zmax=-99999.0,
                  include_polygons=None,
                  exclude_polygons=None,
                  open_boundary_polygons=None,
                  outflow_boundary_polygons=None,
                  quiet=False):

        if not quiet:
            print("Building regular mask ...")

        self.mask = np.zeros(np.shape(self.xz), dtype=int)

        if zmin>=zmax:
            # Do not include any points initially
            if not include_polygons:
                print("WARNING: Entire mask set to zeros! Please ensure zmax is greater than zmin, or provide include polygon(s) !")
                return
        else:
            if self.zz is not None:                
                # Set initial mask based on zmin and zmax
                iok = np.where((self.zz>=zmin) & (self.zz<=zmax))
                self.mask[iok] = 1
            else:
                print("WARNING: Entire mask set to zeros! No depth values found on grid.")
                        
        # Include polygons
        if include_polygons:
            for polygon in include_polygons:
                inpol = inpolygon(self.xz, self.yz, polygon.geometry)
                iok   = np.where((inpol) & (self.zz>=polygon.zmin) & (self.zz<=polygon.zmax))
                self.mask[iok] = 1

        # Exclude polygons
        if exclude_polygons:
            for polygon in exclude_polygons:
                inpol = inpolygon(self.xz, self.yz, polygon.geometry)
                iok   = np.where((inpol) & (self.zz>=polygon.zmin) & (self.zz<=polygon.zmax))
                self.mask[iok] = 0

        # Open boundary polygons
        if open_boundary_polygons:
            for polygon in open_boundary_polygons:
                inpol = inpolygon(self.xz, self.yz, polygon.geometry)
                # Only consider points that are:
                # 1) Inside the polygon
                # 2) Have a mask > 0
                # 3) z>=zmin
                # 4) z<=zmax
                iok   = np.where((inpol) & (self.mask>0) & (self.zz>=polygon.zmin) & (self.zz<=polygon.zmax))

                nok   = np.size(iok[0])

                for ic in range(nok):
                    
                    n = iok[0][ic]
                    m = iok[1][ic]

                    okay = False

                    # Check neighbors, cell must have at least one inactive neighbor

                    # Left
                    if m>0:
                        if self.mask[n, m - 1] == 0:
                            okay = True
                    else:
                        # Point at model boundary
                        okay = True
                        
                    # Below
                    if n>0:
                        if self.mask[n - 1, m] == 0:
                            okay = True
                    else:
                        # Point at model boundary
                        okay = True

                    # Right
                    if m<self.mmax - 1:
                        if self.mask[n, m + 1] == 0:
                            okay = True
                    else:
                        # Point at model boundary
                        okay = True

                    # Above
                    if n<self.nmax - 1:
                        if self.mask[n + 1, m] == 0:
                            okay = True
                    else:
                        # Point at model boundary
                        okay = True
                        
                    if okay:
                        self.mask[n, m] = 2

        # Outflow boundary polygons
        if outflow_boundary_polygons:
            for polygon in outflow_boundary_polygons:
                inpol = inpolygon(self.xz, self.yz, polygon.geometry)
                # Only consider points that are:
                # 1) Inside the polygon
                # 2) Have a mask > 0
                # 3) z>=zmin
                # 4) z<=zmax
                iok   = np.where((inpol) & (self.mask>0) & (self.zz>=polygon.zmin) & (self.zz<=polygon.zmax))

                nok   = np.size(iok[0])

                for ic in range(nok):
                    
                    n = iok[0][ic]
                    m = iok[1][ic]

                    okay = False

                    # Check neighbors, cell must have at least one inactive neighbor

                    # Left
                    if m>0:
                        if self.mask[n, m - 1] == 0:
                            okay = True
                    else:
                        # Point at model boundary
                        okay = True
                        
                    # Below
                    if n>0:
                        if self.mask[n - 1, m] == 0:
                            okay = True
                    else:
                        # Point at model boundary
                        okay = True

                    # Right
                    if m<self.mmax - 1:
                        if self.mask[n, m + 1] == 0:
                            okay = True
                    else:
                        # Point at model boundary
                        okay = True

                    # Above
                    if m<self.nmax - 1:
                        if self.mask[n + 1, m] == 0:
                            okay = True
                    else:
                        # Point at model boundary
                        okay = True
                        
                    if okay:
                        self.mask[n, m] = 3


    def load(self, file_name=None):
        pass

    def save(self, mskfile, depfile, indfile):
        
        iok = np.where(np.transpose(self.mask)>0)
        iok = (iok[1], iok[0])
        
        # Add 1 because indices in SFINCS start with 1, not 0
        ind = np.ravel_multi_index(iok, (self.nmax, self.mmax), order='F') + 1

        file = open(indfile, "wb")
        file.write(np.int32(np.size(ind)))
        file.write(np.int32(ind))
        file.close()
        
        mskv = self.mask[iok]
        file = open(mskfile, "wb")
        file.write(np.int8(mskv))
        file.close()

        if depfile:
            depv = self.zz[iok]
            file = open(depfile, "wb")
            file.write(np.float32(depv))
            file.close()
    

def inpolygon(xq, yq, p):
    shape = xq.shape
    xq = xq.reshape(-1)
    yq = yq.reshape(-1)
    q = [(xq[i], yq[i]) for i in range(xq.shape[0])]
    p = path.Path([(crds[0], crds[1]) for i, crds in enumerate(p.exterior.coords)])
    return p.contains_points(q).reshape(shape)

def binary_search(vals, val):    
    indx = np.searchsorted(vals, val)
    if indx<np.size(vals):
        if vals[indx] == val:
            return indx
    return None
