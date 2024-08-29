# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 16:25:23 2022

@author: ormondt
"""
import numpy as np
from pyproj import Transformer
#import matplotlib.pyplot as plt
from scipy import interpolate
#import time

#from cht_sfincs.quadtree import QuadtreeGrid
from cht_utils.misc_tools import interp2
from cht_bathymetry.bathymetry_database import bathymetry_database
    
class SubgridTableQuadtree:

    def __init__(self, version=0):
        # A subgrid table contains data for EACH cell, u and v point in the quadtree mesh,
        # regardless of the mask value!
        self.version = version

    def load(self, file_name):
        
        file = open(file_name, "rb")
        
        # File version        
        self.version      = np.fromfile(file, dtype="i4", count=1)[0]
        self.nr_cells     = np.fromfile(file, dtype="i4", count=1)[0]
        self.nr_uv_points = np.fromfile(file, dtype="i4", count=1)[0]
        self.nbins        = np.fromfile(file, dtype="i4", count=1)[0]
        self.z_zmin       = np.fromfile(file, dtype="f4", count=self.nr_cells)
        self.z_zmax       = np.fromfile(file, dtype="f4", count=self.nr_cells)
        self.z_zmean      = np.fromfile(file, dtype="f4", count=self.nr_cells)
        self.z_volmax     = np.fromfile(file, dtype="f4", count=self.nr_cells)
        self.z_depth      = np.zeros((self.nbins, self.nr_cells), dtype=float)
        for ibin in range(self.nbins):
            self.z_depth[ibin,:] = np.fromfile(file, dtype="f4", count=self.nr_cells)
        self.uv_zmin      = np.fromfile(file, dtype="f4", count=self.nr_uv_points)
        self.uv_zmax      = np.fromfile(file, dtype="f4", count=self.nr_uv_points)
        self.uv_hrep      = np.zeros((self.nbins, self.nr_uv_points), dtype=float)
        for ibin in range(self.nbins):
            self.uv_hrep[ibin,:] = np.fromfile(file, dtype="f4", count=self.nr_uv_points)
        self.uv_navg      = np.zeros((self.nbins, self.nr_uv_points), dtype=float)
        for ibin in range(self.nbins):
            self.uv_navg[ibin,:] = np.fromfile(file, dtype="f4", count=self.nr_uv_points)
        
        file.close()        

    def save(self, file_name):
        
        file = open(file_name, "wb")

        file.write(np.int32(self.version))
        file.write(np.int32(self.nr_cells))
        file.write(np.int32(self.nr_uv_points))
        file.write(np.int32(self.nbins))
        # Z
        file.write(np.float32(self.z_zmin))
        file.write(np.float32(self.z_zmax))
        file.write(np.float32(self.z_zmean))
        file.write(np.float32(self.z_volmax))
        for ibin in range(self.nbins):
            file.write(np.float32(np.squeeze(self.z_depth[ibin,:])))
        # UV    
        file.write(np.float32(self.uv_zmin))
        file.write(np.float32(self.uv_zmax))
        for ibin in range(self.nbins):
            file.write(np.float32(np.squeeze(self.uv_hrep[ibin,:])))
        for ibin in range(self.nbins):
            file.write(np.float32(np.squeeze(self.uv_navg[ibin,:])))
        
        file.close()
        
    def build(self,
              grid,
              bathymetry_sets,
              roughness_sets,
              file_name=None,
              mask=None,
              nr_bins=10,
              nr_subgrid_pixels=20,
              max_gradient=5.0,
              depth_factor=1.0,
              zmin=-99999.0,
              quiet=True):  

        nbins = nr_bins
        refi  = nr_subgrid_pixels
        z_minimum = zmin
        self.nbins    = nr_bins

        self.nr_cells = grid.nr_cells    

        # Prepare transformers
        bathymetry_transformers = []  
        for bathymetry in bathymetry_sets:
            bathymetry_transformers.append(Transformer.from_crs(grid.crs,
                                                                bathymetry.crs,
                                                                always_xy=True))
        roughness_transformers = []  
        for roughness in roughness_sets:
            if roughness.type == "source":
                roughness_transformers.append(Transformer.from_crs(grid.crs,
                                                                   roughness.crs,
                                                                   always_xy=True))
            else:
                roughness_transformers.append(None)

        nlev = grid.nr_refinement_levels

        # Z points        
        self.z_zmin   = np.empty(grid.nr_cells, dtype=float)
        self.z_zmax   = np.empty(grid.nr_cells, dtype=float)
        self.z_zmean  = np.empty(grid.nr_cells, dtype=float)
        self.z_volmax = np.empty(grid.nr_cells, dtype=float)
        self.z_depth  = np.empty((nbins, grid.nr_cells), dtype=float)

        # U/V points        
        # Need to count the number of uv points in order allocate arrays
        index_nu1 = np.zeros(self.nr_cells, dtype=int)
        index_nu2 = np.zeros(self.nr_cells, dtype=int)
        index_mu1 = np.zeros(self.nr_cells, dtype=int)
        index_mu2 = np.zeros(self.nr_cells, dtype=int)        
        nuv = 0
        for ip in range(self.nr_cells):
            if grid.mu1[ip]>=0:
                index_mu1[ip] = nuv
                nuv += 1
            if grid.mu2[ip]>=0:
                index_mu2[ip] = nuv
                nuv += 1
            if grid.nu1[ip]>=0:
                index_nu1[ip] = nuv
                nuv += 1
            if grid.nu2[ip]>=0:
                index_nu2[ip] = nuv
                nuv += 1
        self.uv_zmin = np.empty(nuv, dtype=float)
        self.uv_zmax = np.empty(nuv, dtype=float)
        self.uv_hrep = np.empty((nbins, nuv), dtype=float)
        self.uv_navg = np.empty((nbins, nuv), dtype=float)
        self.nr_uv_points = nuv
                
        cosrot = np.cos(grid.rotation*np.pi/180)
        sinrot = np.sin(grid.rotation*np.pi/180)
        nrmax  = 2000
        
        # Determine first indices and number of cells per refinement level
        ifirst = np.zeros(nlev, dtype=int)
        ilast  = np.zeros(nlev, dtype=int)
        nr_cells_per_level = np.zeros(nlev, dtype=int)
        ireflast = -1
        for ic in range(grid.nr_cells):
            if grid.level[ic]>ireflast:
                ifirst[grid.level[ic]] = ic
                ireflast = grid.level[ic]
        for ilev in range(nlev - 1):
            ilast[ilev] = ifirst[ilev + 1] - 1
        ilast[nlev - 1] = grid.nr_cells - 1
        for ilev in range(nlev):
            nr_cells_per_level[ilev] = ilast[ilev] - ifirst[ilev] + 1 

        # Loop through all levels
        for ilev in range(nlev):

            if not quiet:
                print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
                print("Processing level " + str(ilev + 1) + " of " + str(nlev) + " ...")
            
            # Make blocks off cells in this level only
            cell_indices_in_level = np.arange(ifirst[ilev], ilast[ilev] + 1, dtype=int)
            nr_cells_in_level = np.size(cell_indices_in_level)
            
            if nr_cells_in_level == 0:
                continue

            n0 = np.min(grid.n[ifirst[ilev]:ilast[ilev] + 1])
            n1 = np.max(grid.n[ifirst[ilev]:ilast[ilev] + 1]) # + 1 # add extra cell to compute u and v in the last row/column
            m0 = np.min(grid.m[ifirst[ilev]:ilast[ilev] + 1])
            m1 = np.max(grid.m[ifirst[ilev]:ilast[ilev] + 1]) # + 1 # add extra cell to compute u and v in the last row/column
            
            dx   = grid.dx/2**ilev      # cell size
            dy   = grid.dy/2**ilev      # cell size
            dxp  = dx/refi              # size of subgrid pixel
            dyp  = dy/refi              # size of subgrid pixel
            
            nrcb = int(np.floor(nrmax/refi))         # nr of regular cells in a block            
            nrbn = int(np.ceil((n1 - n0 + 1)/nrcb))  # nr of blocks in n direction
            nrbm = int(np.ceil((m1 - m0 + 1)/nrcb))  # nr of blocks in m direction

            if not quiet:
                print("Number of regular cells in a block : " + str(nrcb))
                print("Number of blocks in n direction    : " + str(nrbn))
                print("Number of blocks in m direction    : " + str(nrbm))
            
            if not quiet:
                print("Grid size of flux grid             : dx= " + str(dx) + ", dy= " + str(dy))
                print("Grid size of subgrid pixels        : dx= " + str(dxp) + ", dy= " + str(dyp))

            ## Loop through blocks
            ib = -1
            for ii in range(nrbm):
                for jj in range(nrbn):
                    
                    # Count
                    ib += 1
                    
                    bn0 = n0  + jj*nrcb               # Index of first n in block
                    bn1 = min(bn0 + nrcb - 1, n1) + 1 # Index of last n in block (cut off excess above, but add extra cell to compute u and v in the last row)
                    bm0 = m0  + ii*nrcb               # Index of first m in block
                    bm1 = min(bm0 + nrcb - 1, m1) + 1 # Index of last m in block (cut off excess to the right, but add extra cell to compute u and v in the last column)

                    if not quiet:
                        print("--------------------------------------------------------------")
                        print("Processing block " + str(ib + 1) + " of " + str(nrbn*nrbm) + " ...")

                    # Now build the pixel matrix
                    x00 = 0.5*dxp + bm0*refi*dyp
                    x01 = x00 + (bm1 - bm0 + 1)*refi*dxp
                    y00 = 0.5*dyp + bn0*refi*dyp
                    y01 = y00 + (bn1 - bn0 + 1)*refi*dyp
                    
                    x0 = np.arange(x00, x01, dxp)
                    y0 = np.arange(y00, y01, dyp)
                    xg0, yg0 = np.meshgrid(x0, y0)
                    # Rotate and translate
                    xg = grid.x0 + cosrot*xg0 - sinrot*yg0
                    yg = grid.y0 + sinrot*xg0 + cosrot*yg0                    

                    # Clear variables
                    del x0, y0, xg0, yg0
                    
                    # Initialize depth of subgrid at NaN
                    zg = np.full(np.shape(xg), np.nan)
                    
                    # Loop through bathymetry datasets
                    for ibathy, bathymetry in enumerate(bathymetry_sets):

                        # Check if there are NaNs left in this block 

                        if np.isnan(zg).any():

    
                            if bathymetry.type == "source":

                                xgb, ygb = bathymetry_transformers[ibathy].transform(xg, yg)

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
                                                                          max_cell_size=dxp)

                                if zb is not np.nan:
                                    zb[np.where(zb<bathymetry.zmin)] = np.nan
                                    zb[np.where(zb>bathymetry.zmax)] = np.nan
                                    if not np.isnan(zb).all():
                                        zg1 = interp2(xb, yb, zb, xgb, ygb)
                                        isn = np.where(np.isnan(zg))
                                        zg[isn] = zg1[isn]
                                
#                                    if not np.size(zb) == 1: #TL: size = 1 happens when zb = nan because of 'Tiles are outside of search range' in bathymetry_database.py > read_data_from_netcdf_tiles. Then following action doesn't work on a single float
#                                        zb[np.where(zb<bathymetry.zmin)] = np.nan
#                                        zb[np.where(zb>bathymetry.zmax)] = np.nan
#                                        
#                                    if not np.isnan(zb).all():
#                                        zg1 = interp2(xb, yb, zb, xgb, ygb)
#                                        isn = np.where(np.isnan(zg))
#                                        zg[isn] = zg1[isn]
                            
                        elif bathymetry.type == "array":
                            # Matrix provided, interpolate to subgrid mesh
#                            zg = interp2(bathymetry.x, bathymetry.y, bathymetry.z, xgb, ygb)
                            pass
                    
                    # Multiply zg with depth factor (had to use 0.9746 to get arrival
                    # times right in the Pacific)
                    zg = zg*depth_factor
                    
                    zg = np.maximum(zg, z_minimum)

                    # Manning's n values
                    
                    # Initialize roughness of subgrid at NaN
                    manning_grid = np.full(np.shape(xg), np.nan)

                    # Loop through roughness datasets
                    for irgh, roughness in enumerate(roughness_sets):

                        # Check if there are NaNs left in this block 

                        if np.isnan(manning_grid).any():
                        
                            if roughness.type == "source":

                                xgb, ygb = roughness_transformers[irgh].transform(xg, yg)

                                xmin = np.nanmin(np.nanmin(xgb))
                                xmax = np.nanmax(np.nanmax(xgb))
                                ymin = np.nanmin(np.nanmin(ygb))
                                ymax = np.nanmax(np.nanmax(ygb))
                                ddx  = 0.05*(xmax - xmin)
                                ddy  = 0.05*(ymax - ymin)
                                xl   = [xmin - ddx, xmax + ddx]
                                yl   = [ymin - ddy, ymax + ddy]
                            
                                # Get DEM data (ddb format for now)
                                xb, yb, zb = bathymetry_database.get_data(roughness.name,
                                                                          xl,
                                                                          yl,
                                                                          max_cell_size=dxp)

                                if zb is not np.nan:
                                    if not np.isnan(zb).all():
                                        zg1 = interp2(xb, yb, zb, xgb, ygb)
                                        isn = np.where(np.isnan(manning_grid))
                                        manning_grid[isn] = zg1[isn]
 
                            elif roughness.type == "constant":
                                isn = np.where(np.isnan(manning_grid))
#                                manning_grid = np.zeros(np.shape(zg))
#                                manning_grid[np.where(zg<=roughness.zlevel)] = roughness.roughness_deep
#                                manning_grid[np.where(zg>roughness.zlevel)] = roughness.roughness_shallow
                                manning_grid[(isn and np.where(zg<=roughness.zlevel))] = roughness.roughness_deep
                                manning_grid[(isn and np.where(zg>roughness.zlevel))] = roughness.roughness_shallow
                               
                            elif roughness.type == "array":
                                # Matrix provided, interpolate to subgrid mesh
                                # TODO
    #                            zg = interp2(roughness.x, roughness.y, roughness.z, xgb, ygb)
                                pass
                   
                    # clear temp variables
                    del xb, yb, zb
                    
                    # Now compute subgrid properties

                    # First we loop through all the possible cells in this block
                    index_cells_in_block = np.zeros(nrcb*nrcb, dtype=int)
                    # Loop through all cells in this level
                    nr_cells_in_block = 0
                    for ic in range(nr_cells_in_level):
                        indx = cell_indices_in_level[ic] # index of the whole quadtree
                        if grid.n[indx]>=bn0 and grid.n[indx]<bn1 and grid.m[indx]>=bm0 and grid.m[indx]<bm1:
                            # Cell falls inside block
                            index_cells_in_block[nr_cells_in_block] = indx
                            nr_cells_in_block += 1
                    index_cells_in_block = index_cells_in_block[0:nr_cells_in_block]

                    if not quiet:
                        print("Number of active cells in block    : " + str(nr_cells_in_block))

                    # Loop through all active cells in this block
                    for ic in range(nr_cells_in_block):
                        
                        indx = index_cells_in_block[ic]

                        # First the volumes in the cells
                        nn  = (grid.n[indx] - bn0) * refi
                        mm  = (grid.m[indx] - bm0) * refi
                        zgc = zg[nn : nn + refi, mm : mm + refi]

                        # Compute pixel size in metres
                        if grid.crs.is_geographic:
                            ygc = yg[nn : nn + refi, mm : mm + refi]
                            mean_lat =np.abs(np.mean(ygc))
                            dxpm = dxp*111111.0*np.cos(np.pi*mean_lat/180.0)
                            dypm = dyp*111111.0
                        else:
                            dxpm = dxp
                            dypm = dyp
                        
                        zv  = zgc.flatten()   
                        zvmin = -20.0
                        z, v, zmin, zmax, zmean = subgrid_v_table(zv, dxpm, dypm, nbins, zvmin, max_gradient)
                        self.z_zmin[indx]    = zmin
                        self.z_zmax[indx]    = zmax
                        self.z_zmean[indx]   = zmean
                        self.z_volmax[indx]  = v[-1]
                        self.z_depth[:,indx] = z[1:]
                        
                        # Now the U/V points
                        # First right
                        if grid.mu[indx] <= 0:
                            if grid.mu1[indx] >= 0:
                                nn  = (grid.n[indx] - bn0)*refi
                                mm  = (grid.m[indx] - bm0)*refi + int(0.5*refi)
                                zgu = zg[nn : nn + refi, mm : mm + refi]
                                zgu = np.transpose(zgu)
                                zv  = zgu.flatten()
                                manning = manning_grid[nn : nn + refi, mm : mm + refi]
                                manning = np.transpose(manning)
                                manning = manning.flatten()
                                zmin, zmax, hrep, navg, zz = subgrid_q_table(zv, manning, nbins)
                                iuv = index_mu1[indx]
                                self.uv_zmin[iuv]   = zmin
                                self.uv_zmax[iuv]   = zmax
                                self.uv_hrep[:,iuv] = hrep
                                self.uv_navg[:,iuv] = navg
                        else:        
                            if grid.mu1[indx] >= 0:
                                nn = (grid.n[indx] - bn0)*refi
                                mm = (grid.m[indx] - bm0)*refi + int(3*refi/4)
                                zgu = zg[nn : nn + int(refi/2), mm : mm + int(refi/2)]
                                zgu = np.transpose(zgu)
                                zv  = zgu.flatten()
                                manning = manning_grid[nn : nn + int(refi/2), mm : mm + int(refi/2)]
                                manning = np.transpose(manning)
                                manning = manning.flatten()
                                zmin, zmax, hrep, navg, zz = subgrid_q_table(zv, manning, nbins)
                                iuv = index_mu1[indx]
                                self.uv_zmin[iuv]   = zmin
                                self.uv_zmax[iuv]   = zmax
                                self.uv_hrep[:,iuv] = hrep
                                self.uv_navg[:,iuv] = navg
                            if grid.mu2[indx] >= 0:
                                nn = (grid.n[indx] - bn0)*refi + int(refi/2)
                                mm = (grid.m[indx] - bm0)*refi + int(3*refi/4)
                                zgu = zg[nn : nn + int(refi/2), mm : mm + int(refi/2)]
                                zgu = np.transpose(zgu)
                                zv  = zgu.flatten()
                                manning = manning_grid[nn : nn + int(refi/2), mm : mm + int(refi/2)]
                                manning = np.transpose(manning)
                                manning = manning.flatten()
                                zmin, zmax, hrep, navg, zz = subgrid_q_table(zv, manning, nbins)
                                iuv = index_mu2[indx]
                                self.uv_zmin[iuv]   = zmin
                                self.uv_zmax[iuv]   = zmax
                                self.uv_hrep[:,iuv] = hrep
                                self.uv_navg[:,iuv] = navg

                        if grid.nu[indx] <= 0:
                            if grid.nu1[indx] >= 0:
                                nn = (grid.n[indx] - bn0)*refi + int(0.5*refi)
                                mm = (grid.m[indx] - bm0)*refi
                                zgu = zg[nn : nn + refi, mm : mm + refi]
                                zv  = zgu.flatten()
                                manning = manning_grid[nn : nn + refi, mm : mm + refi]
                                manning = manning.flatten()
                                zmin, zmax, hrep, navg, zz = subgrid_q_table(zv, manning, nbins)
                                iuv = index_nu1[indx]
                                self.uv_zmin[iuv]   = zmin
                                self.uv_zmax[iuv]   = zmax
                                self.uv_hrep[:,iuv] = hrep
                                self.uv_navg[:,iuv] = navg                                
                        else:        
                            if grid.nu1[indx] >= 0:
                                nn = (grid.n[indx] - bn0)*refi + int(3*refi/4)
                                mm = (grid.m[indx] - bm0)*refi
                                zgu = zg[nn : nn + int(refi/2), mm : mm + int(refi/2)]
                                zv  = zgu.flatten()
                                manning = manning_grid[nn : nn + int(refi/2), mm : mm + int(refi/2)]
                                manning = manning.flatten()
                                zmin, zmax, hrep, navg, zz = subgrid_q_table(zv, manning, nbins)
                                iuv = index_nu1[indx]
                                self.uv_zmin[iuv]   = zmin
                                self.uv_zmax[iuv]   = zmax
                                self.uv_hrep[:,iuv] = hrep
                                self.uv_navg[:,iuv] = navg
                            if grid.nu2[indx] >= 0:
                                nn = (grid.n[indx] - bn0)*refi + int(3*refi/4)
                                mm = (grid.m[indx] - bm0)*refi + int(refi/2)
                                zgu = zg[nn : nn + int(refi/2), mm : mm + int(refi/2)]
                                zv  = zgu.flatten()
                                manning = manning_grid[nn : nn + int(refi/2), mm : mm + int(refi/2)]
                                manning = manning.flatten()
                                zmin, zmax, hrep, navg, zz = subgrid_q_table(zv, manning, nbins)
                                iuv = index_nu2[indx]
                                self.uv_zmin[iuv]   = zmin
                                self.uv_zmax[iuv]   = zmax
                                self.uv_hrep[:,iuv] = hrep
                                self.uv_navg[:,iuv] = navg

        if not quiet:
            print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        
        if file_name:
            self.save(file_name)



class SubgridTableRegular:

    def __init__(self, version=0):
        # A regular subgrid table contains only for cells with msk>0
        self.version = version

    def load(self, file_name):
        
        file = open(file_name, "rb")
        
        # File version        
        self.version      = np.fromfile(file, dtype="i4", count=1)[0]
        self.nr_cells     = np.fromfile(file, dtype="i4", count=1)[0]
        self.nr_uv_points = np.fromfile(file, dtype="i4", count=1)[0]
        self.nbins        = np.fromfile(file, dtype="i4", count=1)[0]
        self.z_zmin       = np.fromfile(file, dtype="f4", count=self.nr_cells)
        self.z_zmax       = np.fromfile(file, dtype="f4", count=self.nr_cells)
        self.z_zmean      = np.fromfile(file, dtype="f4", count=self.nr_cells)
        self.z_volmax     = np.fromfile(file, dtype="f4", count=self.nr_cells)
        self.z_depth      = np.zeros((self.nbins, self.nr_cells), dtype=float)
        for ibin in range(self.nbins):
            self.z_depth[ibin,:] = np.fromfile(file, dtype="f4", count=self.nr_cells)
        self.uv_zmin      = np.fromfile(file, dtype="f4", count=self.nr_uv_points)
        self.uv_zmax      = np.fromfile(file, dtype="f4", count=self.nr_uv_points)
        self.uv_hrep      = np.zeros((self.nbins, self.nr_uv_points), dtype=float)
        for ibin in range(self.nbins):
            self.uv_hrep[ibin,:] = np.fromfile(file, dtype="f4", count=self.nr_uv_points)
        self.uv_navg      = np.zeros((self.nbins, self.nr_uv_points), dtype=float)
        for ibin in range(self.nbins):
            self.uv_navg[ibin,:] = np.fromfile(file, dtype="f4", count=self.nr_uv_points)
        
        file.close()        

    def save(self, file_name, mask):

        iok = np.where(np.transpose(mask)>0)
        iok = (iok[1], iok[0])
        
        nmax = np.shape(self.z_zmin)[0]
        mmax = np.shape(self.z_zmin)[1]
        
        # Add 1 because indices in SFINCS start with 1, not 0
        ind = np.ravel_multi_index(iok, (nmax, mmax), order='F') + 1

        
        file = open(file_name, "wb")
        file.write(np.int32(np.size(ind))) # Nr of active points
        file.write(np.int32(1))            # min
        file.write(np.int32(self.nbins))

        # Z
        v = self.z_zmin[iok]
        print(np.shape(v))
        file.write(np.float32(v))
        v = self.z_zmax[iok]
        file.write(np.float32(v))
        v = self.z_volmax[iok]
        file.write(np.float32(v))
        for ibin in range(self.nbins):
            v = np.squeeze(self.z_depth[ibin,:,:])[iok]
            file.write(np.float32(v))

        # U
        v = self.u_zmin[iok]
        file.write(np.float32(v))
        v = self.u_zmax[iok]
        file.write(np.float32(v))
        dhdz = np.full(np.shape(v), 1.0)
        file.write(np.float32(dhdz)) # Not used in SFINCS anymore       
        for ibin in range(self.nbins):
            v = np.squeeze(self.u_hrep[ibin,:,:])[iok]
            file.write(np.float32(v))
        for ibin in range(self.nbins):
            v = np.squeeze(self.u_navg[ibin,:,:])[iok]
            file.write(np.float32(v))

        # V
        v = self.v_zmin[iok]
        file.write(np.float32(v))
        v = self.v_zmax[iok]
        file.write(np.float32(v))
        file.write(np.float32(dhdz)) # Not used in SFINCS anymore       
        for ibin in range(self.nbins):
            v = np.squeeze(self.v_hrep[ibin,:,:])[iok]
            file.write(np.float32(v))
        for ibin in range(self.nbins):
            v = np.squeeze(self.v_navg[ibin,:,:])[iok]
            file.write(np.float32(v))
        
        file.close()
        
    def build(self,
              grid,
              bathymetry_sets,
              roughness_sets,
              file_name=None,
              mask=None,
              nr_bins=10,
              nr_subgrid_pixels=20,
              max_gradient=5.0,
              depth_factor=1.0,
              zmin=-99999.0,
              quiet=True):  

        nbins = nr_bins
        refi  = nr_subgrid_pixels
        z_minimum = zmin
        self.nbins    = nr_bins

        # Prepare transformers
        bathymetry_transformers = []  
        for bathymetry in bathymetry_sets:
            bathymetry_transformers.append(Transformer.from_crs(grid.crs,
                                                                bathymetry.crs,
                                                                always_xy=True))
        roughness_transformers = []  
        for roughness in roughness_sets:
            if roughness.type == "source":
                roughness_transformers.append(Transformer.from_crs(grid.crs,
                                                                   roughness.crs,
                                                                   always_xy=True))
            else:
                roughness_transformers.append(None)

        grid_dim = (grid.nmax, grid.mmax)

        # Z points        
        self.z_zmin   = np.empty(grid_dim, dtype=float)
        self.z_zmax   = np.empty(grid_dim, dtype=float)
        self.z_zmean  = np.empty(grid_dim, dtype=float)
        self.z_volmax = np.empty(grid_dim, dtype=float)
        self.z_depth  = np.empty((nbins, grid.nmax, grid.mmax), dtype=float)

        # U points        
        self.u_zmin = np.empty(grid_dim, dtype=float)
        self.u_zmax = np.empty(grid_dim, dtype=float)
        self.u_hrep = np.empty((nbins, grid.nmax, grid.mmax), dtype=float)
        self.u_navg = np.empty((nbins, grid.nmax, grid.mmax), dtype=float)

        # V points        
        self.v_zmin = np.empty(grid_dim, dtype=float)
        self.v_zmax = np.empty(grid_dim, dtype=float)
        self.v_hrep = np.empty((nbins, grid.nmax, grid.mmax), dtype=float)
        self.v_navg = np.empty((nbins, grid.nmax, grid.mmax), dtype=float)
                
        cosrot = np.cos(grid.rotation*np.pi/180)
        sinrot = np.sin(grid.rotation*np.pi/180)
        nrmax  = 2000
        
        n0 = 0
        n1 = grid.nmax - 1 # + 1 # add extra cell to compute u and v in the last row/column
        m0 = 0
        m1 = grid.mmax - 1 # + 1 # add extra cell to compute u and v in the last row/column
        
        dx   = grid.dx      # cell size
        dy   = grid.dy      # cell size
        dxp  = dx/refi      # size of subgrid pixel
        dyp  = dy/refi      # size of subgrid pixel
        
        nrcb = int(np.floor(nrmax/refi))         # nr of regular cells in a block            
        nrbn = int(np.ceil((n1 - n0 + 1)/nrcb))  # nr of blocks in n direction
        nrbm = int(np.ceil((m1 - m0 + 1)/nrcb))  # nr of blocks in m direction

        if not quiet:
            print("Number of regular cells in a block : " + str(nrcb))
            print("Number of blocks in n direction    : " + str(nrbn))
            print("Number of blocks in m direction    : " + str(nrbm))
        
        if not quiet:
            print("Grid size of flux grid             : dx= " + str(dx) + ", dy= " + str(dy))
            print("Grid size of subgrid pixels        : dx= " + str(dxp) + ", dy= " + str(dyp))

        ## Loop through blocks
        ib = -1
        for ii in range(nrbm):
            for jj in range(nrbn):
                
                # Count
                ib += 1
                
                bn0 = n0  + jj*nrcb               # Index of first n in block
                bn1 = min(bn0 + nrcb - 1, n1) + 1 # Index of last n in block (cut off excess above, but add extra cell to compute u and v in the last row)
                bm0 = m0  + ii*nrcb               # Index of first m in block
                bm1 = min(bm0 + nrcb - 1, m1) + 1 # Index of last m in block (cut off excess to the right, but add extra cell to compute u and v in the last column)

                if not quiet:
                    print("--------------------------------------------------------------")
                    print("Processing block " + str(ib + 1) + " of " + str(nrbn*nrbm) + " ...")
                    
                # Now build the pixel matrix
                x00 = 0.5*dxp + bm0*refi*dyp
                x01 = x00 + (bm1 - bm0 + 1)*refi*dxp
                y00 = 0.5*dyp + bn0*refi*dyp
                y01 = y00 + (bn1 - bn0 + 1)*refi*dyp
                
                x0 = np.arange(x00, x01, dxp)
                y0 = np.arange(y00, y01, dyp)
                xg0, yg0 = np.meshgrid(x0, y0)
                # Rotate and translate
                xg = grid.x0 + cosrot*xg0 - sinrot*yg0
                yg = grid.y0 + sinrot*xg0 + cosrot*yg0                    

                # Clear variables
                del x0, y0, xg0, yg0
                
                # Initialize depth of subgrid at NaN
                zg = np.full(np.shape(xg), np.nan)
                
                # Loop through bathymetry datasets
                for ibathy, bathymetry in enumerate(bathymetry_sets):

                    # Check if there are NaNs left in this block 

                    if np.isnan(zg).any():


                        if bathymetry.type == "source":

                            xgb, ygb = bathymetry_transformers[ibathy].transform(xg, yg)

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
                                                                      max_cell_size=dxp)

                            if zb is not np.nan:
                                zb[np.where(zb<bathymetry.zmin)] = np.nan
                                zb[np.where(zb>bathymetry.zmax)] = np.nan
                                if not np.isnan(zb).all():
                                    zg1 = interp2(xb, yb, zb, xgb, ygb)
                                    isn = np.where(np.isnan(zg))
                                    zg[isn] = zg1[isn]
                            
#                                    if not np.size(zb) == 1: #TL: size = 1 happens when zb = nan because of 'Tiles are outside of search range' in bathymetry_database.py > read_data_from_netcdf_tiles. Then following action doesn't work on a single float
#                                        zb[np.where(zb<bathymetry.zmin)] = np.nan
#                                        zb[np.where(zb>bathymetry.zmax)] = np.nan
#                                        
#                                    if not np.isnan(zb).all():
#                                        zg1 = interp2(xb, yb, zb, xgb, ygb)
#                                        isn = np.where(np.isnan(zg))
#                                        zg[isn] = zg1[isn]
                        
                    elif bathymetry.type == "array":
                        # Matrix provided, interpolate to subgrid mesh
                        # TODO
#                            zg = interp2(bathymetry.x, bathymetry.y, bathymetry.z, xgb, ygb)
                        pass
                
                # Multiply zg with depth factor (had to use 0.9746 to get arrival
                # times right in the Pacific)
                zg = zg*depth_factor
                
                zg = np.maximum(zg, z_minimum)

                # Manning's n values
                
                # Initialize roughness of subgrid at NaN
                manning_grid = np.full(np.shape(xg), np.nan)

                # Loop through roughness datasets
                for irgh, roughness in enumerate(roughness_sets):

                    # Check if there are NaNs left in this block 

                    if np.isnan(manning_grid).any():
                    
                        if roughness.type == "source":

                            xgb, ygb = roughness_transformers[irgh].transform(xg, yg)

                            xmin = np.nanmin(np.nanmin(xgb))
                            xmax = np.nanmax(np.nanmax(xgb))
                            ymin = np.nanmin(np.nanmin(ygb))
                            ymax = np.nanmax(np.nanmax(ygb))
                            ddx  = 0.05*(xmax - xmin)
                            ddy  = 0.05*(ymax - ymin)
                            xl   = [xmin - ddx, xmax + ddx]
                            yl   = [ymin - ddy, ymax + ddy]
                        
                            # Get DEM data (ddb format for now)
                            xb, yb, zb = bathymetry_database.get_data(roughness.name,
                                                                      xl,
                                                                      yl,
                                                                      max_cell_size=dxp)

                            if zb is not np.nan:
                                if not np.isnan(zb).all():
                                    zg1 = interp2(xb, yb, zb, xgb, ygb)
                                    isn = np.where(np.isnan(manning_grid))
                                    manning_grid[isn] = zg1[isn]
 
                        elif roughness.type == "constant":
                            isn = np.where(np.isnan(manning_grid))
#                                manning_grid = np.zeros(np.shape(zg))
#                                manning_grid[np.where(zg<=roughness.zlevel)] = roughness.roughness_deep
#                                manning_grid[np.where(zg>roughness.zlevel)] = roughness.roughness_shallow
                            manning_grid[(isn and np.where(zg<=roughness.zlevel))] = roughness.roughness_deep
                            manning_grid[(isn and np.where(zg>roughness.zlevel))] = roughness.roughness_shallow
                           
                        elif roughness.type == "array":
                            # Matrix provided, interpolate to subgrid mesh
                            # TODO
#                            zg = interp2(roughness.x, roughness.y, roughness.z, xgb, ygb)
                            pass
               
                # clear temp variables
                del xb, yb, zb
                
                # Now compute subgrid properties

                # Loop through all active cells in this block
                for m in range(bm0, bm1):
                    for n in range(bn0, bn1):
                        
                        if grid.mask[n, m]<1:
                            # Not an active point
                            continue

                        # First the volumes in the cells
                        nn  = (n - bn0) * refi
                        mm  = (m - bm0) * refi
                        zgc = zg[nn : nn + refi, mm : mm + refi]
    
                        # Compute pixel size in metres
                        if grid.crs.is_geographic:
                            ygc = yg[nn : nn + refi, mm : mm + refi]
                            mean_lat =np.abs(np.mean(ygc))
                            dxpm = dxp*111111.0*np.cos(np.pi*mean_lat/180.0)
                            dypm = dyp*111111.0
                        else:
                            dxpm = dxp
                            dypm = dyp
                        
                        zv  = zgc.flatten()   
                        zvmin = -20.0
                        z, v, zmin, zmax, zmean = subgrid_v_table(zv, dxpm, dypm, nbins, zvmin, max_gradient)
                        self.z_zmin[n, m]    = zmin
                        self.z_zmax[n, m]    = zmax
                        self.z_zmean[n, m]   = zmean
                        self.z_volmax[n, m]  = v[-1]
                        self.z_depth[:, n, m] = z[1:]
                        
                        # Now the U/V points
                        # U
                        nn  = (n - bn0)*refi
                        mm  = (m - bm0)*refi + int(0.5*refi)
                        zgu = zg[nn : nn + refi, mm : mm + refi]
                        zgu = np.transpose(zgu)
                        zv  = zgu.flatten()
                        manning = manning_grid[nn : nn + refi, mm : mm + refi]
                        manning = np.transpose(manning)
                        manning = manning.flatten()
                        zmin, zmax, hrep, navg, zz = subgrid_q_table(zv, manning, nbins)
                        self.u_zmin[n, m]   = zmin
                        self.u_zmax[n, m]   = zmax
                        self.u_hrep[:, n, m] = hrep
                        self.u_navg[:, n, m] = navg

                        # V
                        nn = (n - bn0)*refi + int(0.5*refi)
                        mm = (m - bm0)*refi
                        zgu = zg[nn : nn + refi, mm : mm + refi]
                        zv  = zgu.flatten()
                        manning = manning_grid[nn : nn + refi, mm : mm + refi]
                        manning = manning.flatten()
                        zmin, zmax, hrep, navg, zz = subgrid_q_table(zv, manning, nbins)
                        self.v_zmin[n, m]    = zmin
                        self.v_zmax[n, m]    = zmax
                        self.v_hrep[:, n, m] = hrep
                        self.v_navg[:, n, m] = navg

        if not quiet:
            print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        
        if file_name:
            self.save(file_name, grid.mask)





# @njit
def subgrid_v_table(elevation, dx, dy, nbins, zvolmin, max_gradient):
    """
    map vector of elevation values into a hypsometric volume - depth relationship for one grid cell
    Parameters
    ----------
    elevation : np.ndarray (nr of pixels in one cell) containing subgrid elevation values for one grid cell [m]
    dx: float, x-directional cell size (typically not known at this level) [m]
    dy: float, y-directional cell size (typically not known at this level) [m]
    Return
    ------
    ele_sort : np.ndarray (1D flattened from elevation) with sorted and flattened elevation values
    volume : np.ndarray (1D flattened from elevation) containing volumes (lowest value zero) per sorted elevation value
    """

    def get_dzdh(z, V, a):
        # change in level per unit of volume (m/m)
        dz = np.diff(z)
        # change in volume (normalized to meters)
        dh = np.maximum(np.diff(V) / a, 0.001)
        return dz / dh

    # Cell area
    a = np.size(elevation)*dx*dy
    
    # Set minimum elevation to -20 (needed with single precision), and sort 
    ele_sort = np.sort(np.maximum(elevation, zvolmin).flatten())
    
    # Make sure each consecutive point is larger than previous
    for j in range(1, np.size(ele_sort)):
        if ele_sort[j]<=ele_sort[j - 1]:
            ele_sort[j] += 1.0e-6
        
    depth = ele_sort - ele_sort.min()

    volume = np.cumsum((np.diff(depth) * dx * dy) * np.arange(len(depth))[1:])
    # add trailing zero for first value
    volume = np.concatenate([np.array([0]), volume])
    
    # Resample volumes to discrete bins
    steps = np.arange(nbins + 1)/nbins
    V = steps*volume.max()
    dvol = volume.max()/nbins
    z = interpolate.interp1d(volume, ele_sort)(V)
    dzdh = get_dzdh(z, V, a)
    n = 0
    while ((dzdh.max() > max_gradient and not(np.isclose(dzdh.max(), max_gradient))) and n < nbins):
        # reshape until gradient is satisfactory
        idx = np.where(dzdh == dzdh.max())[0]
        z[idx + 1] = z[idx] + max_gradient*(dvol/a)
        dzdh = get_dzdh(z, V, a)
        n += 1
    return z, V, elevation.min(), z.max(), ele_sort.mean()

def subgrid_q_table(elevation, manning, nbins):
    """
    map vector of elevation values into a hypsometric hydraulic radius - depth relationship for one grid cell
    Parameters
    ----------
    elevation : np.ndarray (nr of pixels in one cell) containing subgrid elevation values for one grid cell [m]
    manning : np.ndarray (nr of pixels in one cell) containing subgrid manning roughness values for one grid cell [s m^(-1/3)]
    dx : float, x-directional cell size (typically not known at this level) [m]
    dy : float, y-directional cell size (typically not known at this level) [m]
    Returns
    -------
    ele_sort, R : np.ndarray of sorted elevation values, np.ndarray of sorted hydraulic radii that belong with depth
    """

    hrep = np.zeros(nbins)
    navg = np.zeros(nbins)
    zz   = np.zeros(nbins)

    n   = int(np.size(elevation)) # Nr of pixels in grid cell
    n05 = int(n/2)
 
    zmin_a      = np.min(elevation[0:n05])
    zmax_a      = np.max(elevation[0:n05])
    
    zmin_b      = np.min(elevation[n05:])
    zmax_b      = np.max(elevation[n05:])
    
    zmin = max(zmin_a, zmin_b)
    zmax = max(zmax_a, zmax_b)
    
    # Make sure zmax is a bit higher than zmin
    if zmax<zmin + 0.01:
       zmax += 0.01

    # Determine bin size
    dbin = (zmax - zmin)/nbins
     
    # Loop through bins
    for ibin in range(nbins):

        # Top of bin
        zbin = zmin + (ibin + 1)*dbin
        zz[ibin] = zbin
        
        ibelow = np.where(elevation<=zbin)                           # index of pixels below bin level
        h      = np.maximum(zbin - np.maximum(elevation, zmin), 0.0) # water depth in each pixel
        qi     = h**(5.0/3.0)/manning           # unit discharge in each pixel
        q      = np.sum(qi)/n                   # combined unit discharge for cell

        if not np.any(manning[ibelow]):
            print("NaNs found?!")
        navg[ibin] = manning[ibelow].mean()       # mean manning's n
        hrep[ibin] = (q*navg[ibin])**(3.0/5.0)    # conveyance depth
         
    return zmin, zmax, hrep, navg, zz       

def binary_search(vals, val):    
    indx = np.searchsorted(vals, val)
    if indx<np.size(vals):
        if vals[indx] == val:
            return indx
    return None
