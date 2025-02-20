# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 16:25:23 2022

@author: ormondt
"""
import numpy as np
from scipy import interpolate
import os
import xarray as xr
from multiprocessing.pool import ThreadPool
# from numba import njit

from cht_utils.misc_tools import interp2
# from cht_bathymetry.bathymetry_database import bathymetry_database
    
class SfincsSubgridTable:

    def __init__(self, model, version=0):
        # A subgrid table contains data for EACH cell, u and v point in the quadtree mesh,
        # regardless of the mask value!
        self.model = model
        self.version = version

    def read(self):

        # Check if file exists
        if not self.model.input.variables.sbgfile:
            return

        file_name = os.path.join(self.model.path, self.model.input.variables.sbgfile)
        if not os.path.isfile(file_name):
            print("File " + file_name + " does not exist!")
            return

        # Read from netcdf file with xarray
        self.ds = xr.open_dataset(file_name)
        self.ds.close() # Should this be closed ?

    def write(self, file_name=None):
        if not file_name:
            if not self.model.input.variables.sbgfile:
                return
            file_name = os.path.join(self.model.path, self.model.input.variables.sbgfile)

        # Write XArray dataset to netcdf file
        self.ds.to_netcdf(file_name)
        
    def build(self,
              bathymetry_sets,
              roughness_sets,
              manning_land=0.04,
              manning_water=0.020,
              manning_level=1.0,
              nr_bins=None,
              nr_levels=10,
              nr_subgrid_pixels=20,
              max_gradient=999.0,
              depth_factor=1.0,
              huthresh=0.01,
              zmin=-999999.0,
              zmax=999999.0,
              weight_option="min",
              file_name="",
              bathymetry_database=None,
              quiet=False,
              progress_bar=None):  

        # If filename is empty
        if not file_name:
            if self.model.input.variables.sbgfile:
                file_name = os.path.join(self.model.path, self.model.input.variables.sbgfile)
            else:
                file_name = os.path.join(self.model.path, "sfincs.sbg")
                self.model.input.variables.sbgfile = "sfincs.sbg"

        if nr_bins:
            nr_levels = nr_bins 

        grid = self.model.grid
        
        # Dimensions etc
        refi   = nr_subgrid_pixels
        npc    = grid.nr_cells    
        nr_ref_levs = grid.data.attrs["nr_levels"] # number of refinement levels
        cosrot = np.cos(grid.data.attrs["rotation"]*np.pi/180)
        sinrot = np.sin(grid.data.attrs["rotation"]*np.pi/180)
        nrmax  = 2000
        # nrmax  = 200
        zminimum = zmin
        zmaximum = zmax

        # Grid neighbors (subtract 1 from indices to get zero-based indices)
        level = grid.data["level"].values[:] - 1
        n     = grid.data["n"].values[:] - 1
        m     = grid.data["m"].values[:] - 1
        nu    = grid.data["nu"].values[:]
        nu1   = grid.data["nu1"].values[:] - 1
        nu2   = grid.data["nu2"].values[:] - 1
        mu    = grid.data["mu"].values[:]
        mu1   = grid.data["mu1"].values[:] - 1
        mu2   = grid.data["mu2"].values[:] - 1

        # U/V points 
        # Need to count the number of uv points in order allocate arrays (probably better to store this in the grid)
        if self.model.grid.type == "quadtree":   

            # Loop through cells to count number of velocity points
            npuv = 0
            for ip in range(npc):
                if mu1[ip]>=0:
                    npuv += 1
                if mu2[ip]>=0:
                    npuv += 1
                if nu1[ip]>=0:
                    npuv += 1
                if nu2[ip]>=0:
                    npuv += 1

            # Allocate some arrays with info about the uv points
            uv_index_z_nm = np.zeros(npuv, dtype=int)
            uv_index_z_nmu = np.zeros(npuv, dtype=int)
            uv_flags_dir = np.zeros(npuv, dtype=int)
            uv_flags_level = np.zeros(npuv, dtype=int)
            uv_flags_type = np.zeros(npuv, dtype=int)
            # Determine what type of uv point it is
            ip = -1
            for ic in range(npc):
                if mu[ic]<=0:
                    # Regular or coarser to the right
                    if mu1[ic]>=0:
                        ip += 1
                        uv_index_z_nm[ip] = ic
                        uv_index_z_nmu[ip] = mu1[ic]
                        uv_flags_dir[ip] = 0
                        uv_flags_level[ip] = level[ic]
                        uv_flags_type[ip] = mu[ic]
                else:
                    if mu1[ic]>=0:
                        ip += 1
                        uv_index_z_nm[ip] = ic
                        uv_index_z_nmu[ip] = mu1[ic]
                        uv_flags_dir[ip] = 0 # x
                        uv_flags_level[ip] = level[ic] + 1
                        uv_flags_type[ip] = mu[ic]
                    if mu2[ic]>=0:
                        ip += 1
                        uv_index_z_nm[ip] = ic
                        uv_index_z_nmu[ip] = mu2[ic]
                        uv_flags_dir[ip] = 0 # x
                        uv_flags_level[ip] = level[ic] + 1
                        uv_flags_type[ip] = mu[ic]
                if nu[ic]<=0:
                    # Regular or coarser above
                    if nu1[ic]>=0:
                        ip += 1
                        uv_index_z_nm[ip] = ic
                        uv_index_z_nmu[ip] = nu1[ic]
                        uv_flags_dir[ip] = 1
                        uv_flags_level[ip] = level[ic]
                        uv_flags_type[ip] = nu[ic]
                else:
                    if nu1[ic]>=0:
                        ip += 1
                        uv_index_z_nm[ip] = ic
                        uv_index_z_nmu[ip] = nu1[ic]
                        uv_flags_dir[ip] = 1 # y
                        uv_flags_level[ip] = level[ic] + 1
                        uv_flags_type[ip] = nu[ic]
                    if nu2[ic]>=0:
                        ip += 1
                        uv_index_z_nm[ip] = ic
                        uv_index_z_nmu[ip] = nu2[ic]
                        uv_flags_dir[ip] = 1
                        uv_flags_level[ip] = level[ic] + 1
                        uv_flags_type[ip] = nu[ic]       
                
        else:
            # For regular grids, only the points with mask>0 are stored
            index_nu1 = np.zeros(grid.nr_cells, dtype=int) - 1
            index_nu2 = np.zeros(grid.nr_cells, dtype=int) - 1
            index_mu1 = np.zeros(grid.nr_cells, dtype=int) - 1
            index_mu2 = np.zeros(grid.nr_cells, dtype=int) - 1
            index_nm  = np.zeros(grid.nr_cells, dtype=int) - 1
            npuv = 0
            npc = 0
            # Loop through all cells
            for ip in range(grid.nr_cells):
                # Check if this cell is active
                if grid.data["mask"].values[ip] > 0:
                    index_nm[ip] = npc
                    npc += 1
                    if mu1[ip]>=0:
                        if grid.data["mask"].values[mu1[ip]] > 0:
                            index_mu1[ip] = npuv
                            npuv += 1
                    if mu2[ip]>=0:
                        if grid.data["mask"].values[mu2[ip]] > 0:
                            index_mu2[ip] = npuv
                            npuv += 1
                    if nu1[ip]>=0:
                        if grid.data["mask"].values[nu1[ip]] > 0:
                            index_nu1[ip] = npuv
                            npuv += 1
                    if nu2[ip]>=0:
                        if grid.data["mask"].values[nu2[ip]] > 0:
                            index_nu2[ip] = npuv
                            npuv += 1


        # Create xarray dataset with empty arrays
        self.ds = xr.Dataset()
        self.ds.attrs["version"] = self.version
        self.ds["z_zmin"] = xr.DataArray(np.zeros(npc), dims=["np"])
        self.ds["z_zmax"] = xr.DataArray(np.zeros(npc), dims=["np"])
        self.ds["z_volmax"] = xr.DataArray(np.zeros(npc), dims=["np"])
        self.ds["z_level"] = xr.DataArray(np.zeros((npc, nr_levels)), dims=["np", "levels"])
        self.ds["uv_zmin"] = xr.DataArray(np.zeros(npuv), dims=["npuv"])
        self.ds["uv_zmax"] = xr.DataArray(np.zeros(npuv), dims=["npuv"])
        self.ds["uv_havg"] = xr.DataArray(np.zeros((npuv, nr_levels)), dims=["npuv", "levels"])
        self.ds["uv_nrep"] = xr.DataArray(np.zeros((npuv, nr_levels)), dims=["npuv", "levels"])
        self.ds["uv_pwet"] = xr.DataArray(np.zeros((npuv, nr_levels)), dims=["npuv", "levels"])
        self.ds["uv_ffit"] = xr.DataArray(np.zeros(npuv), dims=["npuv"])
        self.ds["uv_navg"] = xr.DataArray(np.zeros(npuv), dims=["npuv"])
        
        # Determine first indices and number of cells per refinement level
        ifirst = np.zeros(nr_ref_levs, dtype=int)
        ilast  = np.zeros(nr_ref_levs, dtype=int)
        nr_cells_per_level = np.zeros(nr_ref_levs, dtype=int)
        ireflast = -1
        for ic in range(npc):
            if level[ic]>ireflast:
                ifirst[level[ic]] = ic
                ireflast = level[ic]
        for ilev in range(nr_ref_levs - 1):
            ilast[ilev] = ifirst[ilev + 1] - 1
        ilast[nr_ref_levs - 1] = grid.nr_cells - 1
        for ilev in range(nr_ref_levs):
            nr_cells_per_level[ilev] = ilast[ilev] - ifirst[ilev] + 1 

        # Loop through all levels
        for ilev in range(nr_ref_levs):

            if not quiet:
                print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
                print("Processing level " + str(ilev + 1) + " of " + str(nr_ref_levs) + " ...")
            
            # Make blocks off cells in this level only
            cell_indices_in_level = np.arange(ifirst[ilev], ilast[ilev] + 1, dtype=int)
            nr_cells_in_level = np.size(cell_indices_in_level)
            
            if nr_cells_in_level == 0:
                continue

            n0 = np.min(n[ifirst[ilev]:ilast[ilev] + 1])
            n1 = np.max(n[ifirst[ilev]:ilast[ilev] + 1]) # + 1 # add extra cell to compute u and v in the last row/column
            m0 = np.min(m[ifirst[ilev]:ilast[ilev] + 1])
            m1 = np.max(m[ifirst[ilev]:ilast[ilev] + 1]) # + 1 # add extra cell to compute u and v in the last row/column
            
            dx   = grid.data.attrs["dx"]/2**ilev      # cell size
            dy   = grid.data.attrs["dy"]/2**ilev      # cell size
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
            ibt = 1
            if progress_bar:
                progress_bar.set_text("               Generating Sub-grid Tables (level " + str(ilev) + ") ...                ")
                progress_bar.set_maximum(nrbm * nrbn)

            # Start with the cell centres (do uv points next)    

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
                        print("Getting bathy/topo ...")

                    # Now build the pixel matrix
                    x00 = 0.5*dxp + bm0*refi*dxp
                    x01 = x00 + (bm1 - bm0 + 1)*refi*dxp
                    y00 = 0.5*dyp + bn0*refi*dyp
                    y01 = y00 + (bn1 - bn0 + 1)*refi*dyp
                    
                    x0 = np.arange(x00, x01, dxp)
                    y0 = np.arange(y00, y01, dyp)
                    xg0, yg0 = np.meshgrid(x0, y0)
                    # Rotate and translate
                    xg = grid.data.attrs["x0"] + cosrot*xg0 - sinrot*yg0
                    yg = grid.data.attrs["y0"] + sinrot*xg0 + cosrot*yg0                    

                    # Clear variables
                    del x0, y0, xg0, yg0
                    
                    # Get bathymetry on subgrid from bathymetry database
                    zg = bathymetry_database.get_bathymetry_on_grid(xg, yg,
                                                                    self.model.crs,
                                                                    bathymetry_sets,
                                                                    method="linear")
                                        
                    # Multiply zg with depth factor (had to use 0.9746 to get arrival
                    # times right in the Pacific)
                    zg = zg*depth_factor

                    # Set minimum depth                    
                    zg = np.maximum(zg, zminimum)
                    zg = np.minimum(zg, zmaximum)

                    # replace NaNs with 0.0
                    zg[np.isnan(zg)] = 0.0

                    # Now compute subgrid properties

                    # First we loop through all the possible cells in this block
                    index_cells_in_block = np.zeros(nrcb*nrcb, dtype=int)

                    # Loop through all cells in this level
                    nr_cells_in_block = 0
                    for ic in range(nr_cells_in_level):
                        indx = cell_indices_in_level[ic] # index of the whole quadtree
                        if n[indx]>=bn0 and n[indx]<bn1 and m[indx]>=bm0 and m[indx]<bm1:
                            # Cell falls inside block
                            index_cells_in_block[nr_cells_in_block] = indx
                            nr_cells_in_block += 1
                    index_cells_in_block = index_cells_in_block[0:nr_cells_in_block]

                    if not quiet:
                        print("Number of active cells in block    : " + str(nr_cells_in_block))

                    # # Loop through all active cells in this block
                    for ic in range(nr_cells_in_block):
                        
                        indx = index_cells_in_block[ic] # nm index

                        # Pixel indices for this cell
                        nn  = (n[indx] - bn0) * refi
                        mm  = (m[indx] - bm0) * refi
                        zgc = zg[nn : nn + refi, mm : mm + refi]

                        # Compute pixel size in metres
                        if self.model.crs.is_geographic:
                            # ygc = yg[nn : nn + refi, mm : mm + refi]
                            # mean_lat =np.abs(np.mean(ygc))
                            mean_lat = yg[0, 0]
                            dxpm = dxp*111111.0*np.cos(np.pi*mean_lat/180.0)
                            dypm = dyp*111111.0
                        else:
                            dxpm = dxp
                            dypm = dyp
                        
                        zv  = zgc.flatten()   
                        zvmin = -20.0
                        z, v, zmin, zmax, zmean = subgrid_v_table(zv, dxpm, dypm, nr_levels, zvmin, max_gradient)

                        # Check if this is an active point 
                        if indx > -1:
                            self.ds["z_zmin"][indx]    = zmin
                            self.ds["z_zmax"][indx]    = zmax
                            self.ds["z_volmax"][indx]  = v[-1]
                            self.ds["z_level"][indx,:] = z

            # Now do the u/v points
            # Loop through blocks
            ib = -1
            for ii in range(nrbm):
                for jj in range(nrbn):
                    
                    # Count
                    ib += 1
                    if not quiet:
                        print("--------------------------------------------------------------")
                        print("Processing U/V points in block " + str(ib + 1) + " of " + str(nrbn*nrbm) + " ...")
                    
                    bn0 = n0  + jj*nrcb               # Index of first n in block
                    bn1 = min(bn0 + nrcb - 1, n1) + 1 # Index of last n in block (cut off excess above, but add extra cell to compute u and v in the last row)
                    bm0 = m0  + ii*nrcb               # Index of first m in block
                    bm1 = min(bm0 + nrcb - 1, m1) + 1 # Index of last m in block (cut off excess to the right, but add extra cell to compute u and v in the last column)

                    # Now build the pixel matrix
                    x00 = 0.5*dxp + bm0*refi*dxp - 0.5*refi*dxp # start half a cell to the left
                    x01 = x00 + (bm1 - bm0 + 1)*refi*dxp
                    y00 = 0.5*dyp + bn0*refi*dyp - 0.5*refi*dyp # start half a cell below
                    y01 = y00 + (bn1 - bn0 + 1)*refi*dyp
                    
                    x0 = np.arange(x00, x01, dxp)
                    y0 = np.arange(y00, y01, dyp)
                    xg0, yg0 = np.meshgrid(x0, y0)
                    # Rotate and translate
                    xg = grid.data.attrs["x0"] + cosrot*xg0 - sinrot*yg0
                    yg = grid.data.attrs["y0"] + sinrot*xg0 + cosrot*yg0                    

                    # Clear variables
                    del x0, y0, xg0, yg0

                    if not quiet:
                        print("Getting bathy/topo ...")

                    # Get bathymetry on subgrid from bathymetry database
                    zg = bathymetry_database.get_bathymetry_on_grid(xg, yg,
                                                                    self.model.crs,
                                                                    bathymetry_sets)
                    
                    # Multiply zg with depth factor (had to use 0.9746 to get arrival
                    # times right in the Pacific)
                    zg = zg*depth_factor

                    # Set minimum depth                    
                    zg = np.maximum(zg, zminimum)
                    zg = np.minimum(zg, zmaximum)

                    # replace NaNs with 0.0
                    zg[np.isnan(zg)] = 0.0

                    # Manning's n values
                    
                    # Initialize roughness of subgrid at NaN
                    manning_grid = np.full(np.shape(xg), np.nan)

                    if roughness_sets: # this still needs to be implemented
                        manning_grid = bathymetry_database.get_bathymetry_on_grid(xg, yg,
                                                                        self.model.crs,
                                                                        roughness_sets)

                    # Fill in remaining NaNs with default values
                    isn = np.where(np.isnan(manning_grid))
                    try:
                        manning_grid[(isn and np.where(zg<=manning_level))] = manning_water
                    except:
                        pass
                    manning_grid[(isn and np.where(zg>manning_level))] = manning_land


                    # Loop through uv points an see if they are in this block (and at this level)
                    for ip in range(npuv):

                        # Check if this uv point is at the correct level
                        if uv_flags_level[ip] != ilev:
                            continue

                        # Check if this uv point is in this block
                        nm = uv_index_z_nm[ip]
                        nmu = uv_index_z_nmu[ip]

                        # Get the pixel indices for this uv point

                        # Now build the pixel matrix
                        if uv_flags_type[ip] <= 0:

                            # Normal point

                            if n[nm] < bn0 or n[nm] >= bn1 or m[nm] < bm0 or m[nm] >= bm1:
                                # Outside block
                                continue

                            if uv_flags_dir[ip] == 0:
                                # x
                                nn  = (n[nm] - bn0) * refi
                                mm  = (m[nm] - bm0) * refi + int(0.5*refi)
                            else:
                                # y
                                nn  = (n[nm] - bn0) * refi + int(0.5*refi)
                                mm  = (m[nm] - bm0) * refi

                        else: # uv_flags_type[ip] == 1

                            if n[nmu] < bn0 or n[nmu] >= bn1 or m[nmu] < bm0 or m[nmu] >= bm1:
                                # Outside block
                                continue

                            # Coarse to fine
                            if uv_flags_dir[ip] == 0:
                                # x
                                nn  = (n[nmu] - bn0) * refi
                                mm  = (m[nmu] - bm0) * refi - int(0.5*refi)
                            else:
                                # y
                                nn  = (n[nmu] - bn0) * refi - int(0.5*refi)
                                mm  = (m[nmu] - bm0) * refi

                        # Pixel Block actually starts half a grid cell to the left and below, so need to add 0.5*refi
                        nn += int(0.5*refi)
                        mm += int(0.5*refi)

                        # Pixel indices for this cell
                        zgu = zg[nn : nn + refi, mm : mm + refi]
                        manning = manning_grid[nn : nn + refi, mm : mm + refi]
 
                        if uv_flags_dir[ip] == 0:
                            zgu = np.transpose(zgu)
                            manning = np.transpose(manning)

                        zv  = zgu.flatten()
                        manning = manning.flatten()

                        z_zmin_nm = self.ds["z_zmin"].values[nm]
                        z_zmin_nmu = self.ds["z_zmin"].values[nmu]
                        zmin, zmax, havg, nrep, pwet, ffit, navg, zz = subgrid_q_table(zv,
                                                                                       manning,
                                                                                       nr_levels,
                                                                                       huthresh,
                                                                                       2,
                                                                                       z_zmin_nm,
                                                                                       z_zmin_nmu,
                                                                                       weight_option)

                        self.ds["uv_zmin"][ip]   = zmin
                        self.ds["uv_zmax"][ip]   = zmax
                        self.ds["uv_havg"][ip,:] = havg
                        self.ds["uv_nrep"][ip,:] = nrep
                        self.ds["uv_pwet"][ip,:] = pwet
                        self.ds["uv_ffit"][ip]   = ffit
                        self.ds["uv_navg"][ip]   = navg

                    if progress_bar:
                        progress_bar.set_value(ibt)
                        if progress_bar.was_canceled():
                            return
                        ibt += 1

        if not quiet:
            print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        
        if file_name:
            self.write(file_name)


# def process_cell(ic, ds, options):
#     # Loop through all active cells in this block

#     index_cells_in_block = options["index_cells_in_block"]
#     n = options["n"]
#     m = options["m"]
#     refi = options["refi"]
#     bn0 = options["bn0"]
#     bm0 = options["bm0"]
#     yg = options["yg"]
#     zg = options["zg"]
#     dxp = options["dxp"]
#     dyp = options["dyp"]
#     nr_levels = options["nr_levels"]
#     max_gradient = options["max_gradient"]
#     index_nm = options["index_nm"]
#     index_mu1 = options["index_mu1"]
#     index_mu2 = options["index_mu2"]
#     index_nu1 = options["index_nu1"]
#     index_nu2 = options["index_nu2"]
#     manning_grid = options["manning_grid"]
#     is_geographic = options["is_geographic"]
#     huthresh = options["huthresh"]
#     mu = options["mu"]
#     mu1 = options["mu1"]
#     mu2 = options["mu2"]
#     nu = options["nu"]
#     nu1 = options["nu1"]
#     nu2 = options["nu2"]


#     indx = index_cells_in_block[ic]

#     # First the volumes in the cells
#     nn  = (n[indx] - bn0) * refi
#     mm  = (m[indx] - bm0) * refi
#     zgc = zg[nn : nn + refi, mm : mm + refi]

#     # Compute pixel size in metres
#     if is_geographic:
#         ygc = yg[nn : nn + refi, mm : mm + refi]
#         mean_lat =np.abs(np.mean(ygc))
#         dxpm = dxp*111111.0*np.cos(np.pi*mean_lat/180.0)
#         dypm = dyp*111111.0
#     else:
#         dxpm = dxp
#         dypm = dyp
    
#     zv  = zgc.flatten()   
#     zvmin = -20.0
#     z, v, zmin, zmax, zmean = subgrid_v_table(zv, dxpm, dypm, nr_levels, zvmin, max_gradient)

#     # Check if this is an active point 
#     if index_nm[indx] > -1:
#         ds["z_zmin"][index_nm[indx]]    = zmin
#         ds["z_zmax"][index_nm[indx]]    = zmax
#         ds["z_volmax"][index_nm[indx]]  = v[-1]
#         ds["z_level"][index_nm[indx],:] = z
    
#     # Now the U/V points
#     # First right
#     if mu[indx] <= 0:
#         # Cell to the right is equal or larger
#         if mu1[indx] >= 0:
#             # And it exists
#             iuv = index_mu1[indx]
#             if iuv>=0:
#                 nn  = (n[indx] - bn0)*refi
#                 mm  = (m[indx] - bm0)*refi + int(0.5*refi)
#                 zgu = zg[nn : nn + refi, mm : mm + refi]
#                 zgu = np.transpose(zgu)
#                 zv  = zgu.flatten()
#                 manning = manning_grid[nn : nn + refi, mm : mm + refi]
#                 manning = np.transpose(manning)
#                 manning = manning.flatten()
#                 zmin, zmax, havg, nrep, pwet, ffit, navg, zz = subgrid_q_table(zv, manning, nr_levels, huthresh)
#                 ds["uv_zmin"][iuv]   = zmin
#                 ds["uv_zmax"][iuv]   = zmax
#                 ds["uv_havg"][iuv,:] = havg
#                 ds["uv_nrep"][iuv,:] = nrep
#                 ds["uv_pwet"][iuv,:] = pwet
#                 ds["uv_ffit"][iuv]   = ffit
#                 ds["uv_navg"][iuv]   = navg
#     else:
#         # Cell to the right is smaller
#         if mu1[indx] >= 0:
#             # And the bottom neighbor exists
#             iuv = index_mu1[indx]
#             if iuv>=0:
#                 nn = (n[indx] - bn0)*refi
#                 mm = (m[indx] - bm0)*refi + int(3*refi/4)
#                 zgu = zg[nn : nn + int(refi/2), mm : mm + int(refi/2)]
#                 zgu = np.transpose(zgu)
#                 zv  = zgu.flatten()
#                 manning = manning_grid[nn : nn + int(refi/2), mm : mm + int(refi/2)]
#                 manning = np.transpose(manning)
#                 manning = manning.flatten()
#                 zmin, zmax, havg, nrep, pwet, ffit, navg, zz = subgrid_q_table(zv, manning, nr_levels, huthresh)
#                 ds["uv_zmin"][iuv]   = zmin
#                 ds["uv_zmax"][iuv]   = zmax
#                 ds["uv_havg"][iuv,:] = havg
#                 ds["uv_nrep"][iuv,:] = nrep
#                 ds["uv_pwet"][iuv,:] = pwet
#                 ds["uv_ffit"][iuv]   = ffit
#                 ds["uv_navg"][iuv]   = navg
#         if mu2[indx] >= 0:
#             # And the top neighbor exists
#             iuv = index_mu2[indx]
#             if iuv>=0:
#                 nn = (n[indx] - bn0)*refi + int(refi/2)
#                 mm = (m[indx] - bm0)*refi + int(3*refi/4)
#                 zgu = zg[nn : nn + int(refi/2), mm : mm + int(refi/2)]
#                 zgu = np.transpose(zgu)
#                 zv  = zgu.flatten()
#                 manning = manning_grid[nn : nn + int(refi/2), mm : mm + int(refi/2)]
#                 manning = np.transpose(manning)
#                 manning = manning.flatten()
#                 zmin, zmax, havg, nrep, pwet, ffit, navg, zz = subgrid_q_table(zv, manning, nr_levels, huthresh)
#                 ds["uv_zmin"][iuv]   = zmin
#                 ds["uv_zmax"][iuv]   = zmax
#                 ds["uv_havg"][iuv,:] = havg
#                 ds["uv_nrep"][iuv,:] = nrep
#                 ds["uv_pwet"][iuv,:] = pwet
#                 ds["uv_ffit"][iuv]   = ffit
#                 ds["uv_navg"][iuv]   = navg

#     # Now above
#     if nu[indx] <= 0:
#         # Cell above is equal or larger
#         if nu1[indx] >= 0:
#             # And it exists
#             iuv = index_nu1[indx]
#             if iuv>=0:
#                 nn = (n[indx] - bn0)*refi + int(0.5*refi)
#                 mm = (m[indx] - bm0)*refi
#                 zgu = zg[nn : nn + refi, mm : mm + refi]
#                 zv  = zgu.flatten()
#                 manning = manning_grid[nn : nn + refi, mm : mm + refi]
#                 manning = manning.flatten()
#                 zmin, zmax, havg, nrep, pwet, ffit, navg, zz = subgrid_q_table(zv, manning, nr_levels, huthresh)
#                 ds["uv_zmin"][iuv]   = zmin
#                 ds["uv_zmax"][iuv]   = zmax
#                 ds["uv_havg"][iuv,:] = havg
#                 ds["uv_nrep"][iuv,:] = nrep
#                 ds["uv_pwet"][iuv,:] = pwet
#                 ds["uv_ffit"][iuv]   = ffit
#                 ds["uv_navg"][iuv]   = navg
#     else:
#         # Cell above is smaller
#         if nu1[indx] >= 0:
#             # And the left neighbor exists
#             iuv = index_nu1[indx]
#             if iuv>=0:
#                 nn = (n[indx] - bn0)*refi + int(3*refi/4)
#                 mm = (m[indx] - bm0)*refi
#                 zgu = zg[nn : nn + int(refi/2), mm : mm + int(refi/2)]
#                 zv  = zgu.flatten()
#                 manning = manning_grid[nn : nn + int(refi/2), mm : mm + int(refi/2)]
#                 manning = manning.flatten()
#                 zmin, zmax, havg, nrep, pwet, ffit, navg, zz = subgrid_q_table(zv, manning, nr_levels, huthresh)
#                 ds["uv_zmin"][iuv]   = zmin
#                 ds["uv_zmax"][iuv]   = zmax
#                 ds["uv_havg"][iuv,:] = havg
#                 ds["uv_nrep"][iuv,:] = nrep
#                 ds["uv_pwet"][iuv,:] = pwet
#                 ds["uv_ffit"][iuv]   = ffit
#                 ds["uv_navg"][iuv]   = navg
#         if nu2[indx] >= 0:
#             # And the right neighbor exists
#             iuv = index_nu2[indx]
#             if iuv>=0:
#                 nn = (n[indx] - bn0)*refi + int(3*refi/4)
#                 mm = (m[indx] - bm0)*refi + int(refi/2)
#                 zgu = zg[nn : nn + int(refi/2), mm : mm + int(refi/2)]
#                 zv  = zgu.flatten()
#                 manning = manning_grid[nn : nn + int(refi/2), mm : mm + int(refi/2)]
#                 manning = manning.flatten()
#                 zmin, zmax, havg, nrep, pwet, ffit, navg, zz = subgrid_q_table(zv, manning, nr_levels, huthresh)
#                 ds["uv_zmin"][iuv]   = zmin
#                 ds["uv_zmax"][iuv]   = zmax
#                 ds["uv_havg"][iuv,:] = havg
#                 ds["uv_nrep"][iuv,:] = nrep
#                 ds["uv_pwet"][iuv,:] = pwet
#                 ds["uv_ffit"][iuv]   = ffit
#                 ds["uv_navg"][iuv]   = navg



# @njit
def subgrid_v_table(elevation, dx, dy, nlevels, zvolmin, max_gradient):
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

    # Create ele_sort and limit to zvolmin (-20.0, needed with single precision) and zvolmax
    zvolmax = 100000.0
    ele_sort = np.minimum(np.maximum(elevation, zvolmin), zvolmax).flatten()
    # Add tiny random number to each elevation to avoid equal values
    ele_sort += 1.0e-6*np.random.rand(np.size(ele_sort)) - 0.5e-6
    # And sort
    ele_sort = np.sort(ele_sort)
        
    depth = ele_sort - ele_sort.min()

    volume = np.cumsum((np.diff(depth) * dx * dy) * np.arange(len(depth))[1:])
    # add trailing zero for first value
    volume = np.concatenate([np.array([0]), volume])
    
    # Resample volumes to discrete levels
    steps = np.arange(nlevels)/(nlevels - 1)
    V = steps*volume.max()
    dvol = volume.max()/(nlevels - 1)
    z = interpolate.interp1d(volume, ele_sort)(V)
    dzdh = get_dzdh(z, V, a)
    n = 0
    while ((dzdh.max() > max_gradient and not(np.isclose(dzdh.max(), max_gradient))) and n < nlevels):
        # reshape until gradient is satisfactory
        idx = np.where(dzdh == dzdh.max())[0]
        z[idx + 1] = z[idx] + max_gradient*(dvol/a)
        dzdh = get_dzdh(z, V, a)
        n += 1

    zmin = elevation.min()
    zmax = elevation.max()
    if zmax < zmin + 0.001:
        zmax = zmin + 0.001
    zmean = ele_sort.mean()

    return z, V, zmin, zmax, zmean

# @njit
def subgrid_q_table(
    elevation: np.ndarray,
    manning: np.ndarray,
    nlevels: int,
    huthresh: float,
    option: int = 2,
    z_zmin_a: float = -99999.0,
    z_zmin_b: float = -99999.0,
    weight_option: str = "min"    
):
    """
    map vector of elevation values into a hypsometric hydraulic radius - depth relationship for one u/v point
    Parameters
    ----------
    elevation : np.ndarray (nr of pixels in one cell) containing subgrid elevation values for one grid cell [m]
    manning : np.ndarray (nr of pixels in one cell) containing subgrid manning roughness values for one grid cell [s m^(-1/3)]
    nlevels : int, number of vertical levels [-]
    huthresh : float, threshold depth [m]
    option : int, option to use "old" or "new" method for computing conveyance depth at u/v points
    z_zmin_a : float, elevation of lowest pixel in neighboring cell A [m]
    z_zmin_b : float, elevation of lowest pixel in neighboring cell B [m]
    weight_option : str, weight of q between sides A and B ("min" or "mean")

    Returns
    -------
    zmin : float, minimum elevation [m]
    zmax : float, maximum elevation [m]
    havg : np.ndarray (nlevels) grid-average depth for vertical levels [m]
    nrep : np.ndarray (nlevels) representative roughness for vertical levels [m1/3/s] ?
    pwet : np.ndarray (nlevels) wet fraction for vertical levels [-] ?
    navg : float, grid-average Manning's n [m 1/3 / s]
    ffit : float, fitting coefficient [-]
    zz   : np.ndarray (nlevels) elevation of vertical levels [m]
    """
    # Initialize output arrays
    havg = np.zeros(nlevels)
    nrep = np.zeros(nlevels)
    pwet = np.zeros(nlevels)
    zz   = np.zeros(nlevels)

    n   = int(np.size(elevation)) # Nr of pixels in grid cell
    n05 = int(n / 2)              # Nr of pixels in half grid cell 

    # Sort elevation and manning values by side A and B
    dd_a      = elevation[0:n05]
    dd_b      = elevation[n05:]
    manning_a = manning[0:n05]
    manning_b = manning[n05:]

    # Ensure that pixels are at least as high as the minimum elevation in the neighbouring cells
    # This should always be the case, but there may be errors in the interpolation to the subgrid pixels
    dd_a      = np.maximum(dd_a, z_zmin_a)
    dd_b      = np.maximum(dd_b, z_zmin_b)

    # Determine min and max elevation
    zmin_a    = np.min(dd_a)
    zmax_a    = np.max(dd_a)    
    zmin_b    = np.min(dd_b)
    zmax_b    = np.max(dd_b)
    
    # Add huthresh to zmin
    zmin = max(zmin_a, zmin_b) + huthresh
    zmax = max(zmax_a, zmax_b)
    
    # Make sure zmax is at least 0.01 m higher than zmin
    zmax = max(zmax, zmin + 0.01)

    # Determine bin size
    dlevel = (zmax - zmin) / (nlevels - 1)

    # Option can be either 1 ("old") or 2 ("new")
    # Should never use option 1 !
    option = 2

    # Loop through levels
    for ibin in range(nlevels):

        # Top of bin
        zbin = zmin + ibin * dlevel
        zz[ibin] = zbin

        h = np.maximum(zbin - elevation, 0.0)  # water depth in each pixel

        # Side A
        h_a   = np.maximum(zbin - dd_a, 0.0)  # Depth of all pixels (but set min pixel height to zbot). Can be negative, but not zero (because zmin = zbot + huthresh, so there must be pixels below zb).
        q_a   = h_a**(5.0 / 3.0) / manning_a  # Determine 'flux' for each pixel
        q_a   = np.mean(q_a)                  # Grid-average flux through all the pixels
        h_a   = np.mean(h_a)                  # Grid-average depth through all the pixels
        
        # Side B
        h_b   = np.maximum(zbin - dd_b, 0.0)  # Depth of all pixels (but set min pixel height to zbot). Can be negative, but not zero (because zmin = zbot + huthresh, so there must be pixels below zb).
        q_b   = h_b**(5.0 / 3.0) / manning_b  # Determine 'flux' for each pixel
        q_b   = np.mean(q_b)                  # Grid-average flux through all the pixels
        h_b   = np.mean(h_b)                  # Grid-average depth through all the pixels

        # Compute q and h
        q_all = np.mean(h**(5.0 / 3.0) / manning)   # Determine grid average 'flux' for each pixel
        h_all = np.mean(h)                          # grid averaged depth of A and B combined
        q_min = np.minimum(q_a, q_b)
        h_min = np.minimum(h_a, h_b)

        if option == 1:
            # Use old 1 option (weighted average of q_ab and q_all) option (min at bottom bin, mean at top bin) 
            w     = (ibin) / (nlevels - 1)              # Weight (increase from 0 to 1 from bottom to top bin)
            q     = (1.0 - w) * q_min + w * q_all        # Weighted average of q_min and q_all
            hmean = h_all
            # Wet fraction
            pwet[ibin] = (zbin > elevation + huthresh).sum() / n

        elif option == 2:
            # Use newer 2 option (minimum of q_a an q_b, minimum of h_a and h_b increasing to h_all, using pwet for weighting) option
            # This is done by making sure that the wet fraction is 0.0 in the first level on the shallowest side (i.e. if ibin==0, pwet_a or pwet_b must be 0.0).
            # As a result, the weight w will be 0.0 in the first level on the shallowest side.

            pwet_a = (zbin > dd_a).sum() / (n / 2) 
            pwet_b = (zbin > dd_b).sum() / (n / 2)

            if ibin == 0:
                # Ensure that at bottom level, either pwet_a or pwet_b is 0.0   
                if pwet_a < pwet_b:
                    pwet_a = 0.0
                else:
                    pwet_b = 0.0
            elif ibin == nlevels - 1:
                # Ensure that at top level, both pwet_a and pwet_b are 1.0
                pwet_a = 1.0
                pwet_b = 1.0        

            if weight_option == "mean":
                # Weight increases linearly from 0 to 1 from bottom to top bin use percentage wet in sides A and B
                w     = 2 * np.minimum(pwet_a, pwet_b) / max(pwet_a + pwet_b, 1.0e-9)
                q     = (1.0 - w) * q_min + w * q_all        # Weighted average of q_min and q_all
                hmean = (1.0 - w) * h_min + w * h_all        # Weighted average of h_min and h_all

            else:
                # Take minimum of q_a and q_b
                if q_a < q_b:
                    q     = q_a
                    hmean = h_a
                else:
                    q     = q_b
                    hmean = h_b

            pwet[ibin] = 0.5 * (pwet_a + pwet_b)         # Combined pwet_a and pwet_b

        havg[ibin] = hmean                          # conveyance depth
        nrep[ibin] = hmean**(5.0 / 3.0) / q         # Representative n for qmean and hmean
    
    nrep_top = nrep[-1]    
    havg_top = havg[-1]

    ### Fitting for nrep above zmax

    # Determine nfit at zfit
    zfit  = zmax + zmax - zmin
    hfit  = havg_top + zmax - zmin                 # mean water depth in cell as computed in SFINCS (assuming linear relation between water level and water depth above zmax)

    # Compute q and navg
    if weight_option == "mean":
        # Use entire uv point 
        h     = np.maximum(zfit - elevation, 0.0)      # water depth in each pixel
        q     = np.mean(h**(5.0 / 3.0) / manning)      # combined unit discharge for cell
        navg  = np.mean(manning)

    else:
        # Use minimum of q_a and q_b
        if q_a < q_b:
            h     = np.maximum(zfit - dd_a, 0.0)         # water depth in each pixel
            q     = np.mean(h**(5.0 / 3.0) / manning_a)  # combined unit discharge for cell
            navg  = np.mean(manning_a)
        else:
            h     = np.maximum(zfit - dd_b, 0.0)
            q     = np.mean(h**(5.0 / 3.0) / manning_b)
            navg  = np.mean(manning_b)

    nfit = hfit**(5.0 / 3.0) / q

    # Actually apply fit on gn2 (this is what is used in sfincs)
    gnavg2 = 9.81 * navg**2
    gnavg_top2 = 9.81 * nrep_top**2

    if gnavg2 / gnavg_top2 > 0.99 and gnavg2 / gnavg_top2 < 1.01:
        # gnavg2 and gnavg_top2 are almost identical
        ffit = 0.0
    else:
        if navg > nrep_top:
            if nfit > navg:
                nfit = nrep_top + 0.9 * (navg - nrep_top)
            if nfit < nrep_top:
                nfit = nrep_top + 0.1 * (navg - nrep_top)
        else:
            if nfit < navg:
                nfit = nrep_top + 0.9 * (navg - nrep_top)
            if nfit > nrep_top:
                nfit = nrep_top + 0.1 * (navg - nrep_top)
        gnfit2 = 9.81 * nfit**2
        ffit = (((gnavg2 - gnavg_top2) / (gnavg2 - gnfit2)) - 1) / (zfit - zmax)
         
    return zmin, zmax, havg, nrep, pwet, ffit, navg, zz       
