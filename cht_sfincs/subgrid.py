# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 16:25:23 2022

@author: ormondt
"""
import numpy as np
from scipy import interpolate
import os
import xarray as xr

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

    def save(self, file_name=None):
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
              zmin=-99999.0,
              file_name="sfincs.sbg",
              bathymetry_database=None,
              quiet=False,
              progress_bar=None):  

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
        zminimum = zmin

        # Grid neighbors
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
            # For quadtree grids, all points are stored
            index_nu1 = np.zeros(npc, dtype=int)
            index_nu2 = np.zeros(npc, dtype=int)
            index_mu1 = np.zeros(npc, dtype=int)
            index_mu2 = np.zeros(npc, dtype=int)        
            index_nm  = np.zeros(grid.nr_cells, dtype=int)
            npuv = 0
            for ip in range(npc):
                index_nm[ip] = ip
                if mu1[ip]>=0:
                    index_mu1[ip] = npuv
                    npuv += 1
                if mu2[ip]>=0:
                    index_mu2[ip] = npuv
                    npuv += 1
                if nu1[ip]>=0:
                    index_nu1[ip] = npuv
                    npuv += 1
                if nu2[ip]>=0:
                    index_nu2[ip] = npuv
                    npuv += 1
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
            ib = -1
            ibt = 1
            if progress_bar:
                progress_bar.set_text("               Generating Sub-grid Tables (level " + str(ilev) + ") ...                ")
                progress_bar.set_maximum(nrbm * nrbn)

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
                    xg = grid.data.attrs["x0"] + cosrot*xg0 - sinrot*yg0
                    yg = grid.data.attrs["y0"] + sinrot*xg0 + cosrot*yg0                    

                    # Clear variables
                    del x0, y0, xg0, yg0
                    
                    # Get bathymetry on subgrid from bathymetry database
                    try: 
                        zg = bathymetry_database.get_bathymetry_on_grid(xg, yg,
                                                                        self.model.crs,
                                                                        bathymetry_sets)
                    except:
                        pass   
                    
                    # Multiply zg with depth factor (had to use 0.9746 to get arrival
                    # times right in the Pacific)
                    zg = zg*depth_factor

                    # Set minimum depth                    
                    zg = np.maximum(zg, zminimum)

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


                    # Parallel from here

                    # Loop through all active cells in this block
                    for ic in range(nr_cells_in_block):
                        
                        indx = index_cells_in_block[ic]

                        # First the volumes in the cells
                        nn  = (n[indx] - bn0) * refi
                        mm  = (m[indx] - bm0) * refi
                        zgc = zg[nn : nn + refi, mm : mm + refi]

                        # Compute pixel size in metres
                        if self.model.crs.is_geographic:
                            ygc = yg[nn : nn + refi, mm : mm + refi]
                            mean_lat =np.abs(np.mean(ygc))
                            dxpm = dxp*111111.0*np.cos(np.pi*mean_lat/180.0)
                            dypm = dyp*111111.0
                        else:
                            dxpm = dxp
                            dypm = dyp
                        
                        zv  = zgc.flatten()   
                        zvmin = -20.0
                        z, v, zmin, zmax, zmean = subgrid_v_table(zv, dxpm, dypm, nr_levels, zvmin, max_gradient)

                        # Check if this is an active point 
                        if index_nm[indx] > -1:
                            self.ds["z_zmin"][index_nm[indx]]    = zmin
                            self.ds["z_zmax"][index_nm[indx]]    = zmax
                            self.ds["z_volmax"][index_nm[indx]]  = v[-1]
                            self.ds["z_level"][index_nm[indx],:] = z
                        
                        # Now the U/V points
                        # First right
                        if mu[indx] <= 0:
                            if mu1[indx] >= 0:
                                nn  = (n[indx] - bn0)*refi
                                mm  = (m[indx] - bm0)*refi + int(0.5*refi)
                                zgu = zg[nn : nn + refi, mm : mm + refi]
                                zgu = np.transpose(zgu)
                                zv  = zgu.flatten()
                                manning = manning_grid[nn : nn + refi, mm : mm + refi]
                                manning = np.transpose(manning)
                                manning = manning.flatten()
                                iuv = index_mu1[indx]
                                if iuv>=0:
                                    zmin, zmax, havg, nrep, pwet, ffit, navg, zz = subgrid_q_table(zv, manning, nr_levels, huthresh)
                                    self.ds["uv_zmin"][iuv]   = zmin
                                    self.ds["uv_zmax"][iuv]   = zmax
                                    self.ds["uv_havg"][iuv,:] = havg
                                    self.ds["uv_nrep"][iuv,:] = nrep
                                    self.ds["uv_pwet"][iuv,:] = pwet
                                    self.ds["uv_ffit"][iuv]   = ffit
                                    self.ds["uv_navg"][iuv]   = navg
                        else:        
                            if mu1[indx] >= 0:
                                nn = (n[indx] - bn0)*refi
                                mm = (m[indx] - bm0)*refi + int(3*refi/4)
                                zgu = zg[nn : nn + int(refi/2), mm : mm + int(refi/2)]
                                zgu = np.transpose(zgu)
                                zv  = zgu.flatten()
                                manning = manning_grid[nn : nn + int(refi/2), mm : mm + int(refi/2)]
                                manning = np.transpose(manning)
                                manning = manning.flatten()
                                iuv = index_mu1[indx]
                                if iuv>=0:
                                    zmin, zmax, havg, nrep, pwet, ffit, navg, zz = subgrid_q_table(zv, manning, nr_levels, huthresh)
                                    self.ds["uv_zmin"][iuv]   = zmin
                                    self.ds["uv_zmax"][iuv]   = zmax
                                    self.ds["uv_havg"][iuv,:] = havg
                                    self.ds["uv_nrep"][iuv,:] = nrep
                                    self.ds["uv_pwet"][iuv,:] = pwet
                                    self.ds["uv_ffit"][iuv]   = ffit
                                    self.ds["uv_navg"][iuv]   = navg
                            if mu2[indx] >= 0:
                                nn = (n[indx] - bn0)*refi + int(refi/2)
                                mm = (m[indx] - bm0)*refi + int(3*refi/4)
                                zgu = zg[nn : nn + int(refi/2), mm : mm + int(refi/2)]
                                zgu = np.transpose(zgu)
                                zv  = zgu.flatten()
                                manning = manning_grid[nn : nn + int(refi/2), mm : mm + int(refi/2)]
                                manning = np.transpose(manning)
                                manning = manning.flatten()
                                iuv = index_mu2[indx]
                                if iuv>=0:
                                    zmin, zmax, havg, nrep, pwet, ffit, navg, zz = subgrid_q_table(zv, manning, nr_levels, huthresh)
                                    self.ds["uv_zmin"][iuv]   = zmin
                                    self.ds["uv_zmax"][iuv]   = zmax
                                    self.ds["uv_havg"][iuv,:] = havg
                                    self.ds["uv_nrep"][iuv,:] = nrep
                                    self.ds["uv_pwet"][iuv,:] = pwet
                                    self.ds["uv_ffit"][iuv]   = ffit
                                    self.ds["uv_navg"][iuv]   = navg

                        # Now above
                        if nu[indx] <= 0:
                            if nu1[indx] >= 0:
                                nn = (n[indx] - bn0)*refi + int(0.5*refi)
                                mm = (m[indx] - bm0)*refi
                                zgu = zg[nn : nn + refi, mm : mm + refi]
                                zv  = zgu.flatten()
                                manning = manning_grid[nn : nn + refi, mm : mm + refi]
                                manning = manning.flatten()
                                iuv = index_nu1[indx]
                                if iuv>=0:
                                    zmin, zmax, havg, nrep, pwet, ffit, navg, zz = subgrid_q_table(zv, manning, nr_levels, huthresh)
                                    self.ds["uv_zmin"][iuv]   = zmin
                                    self.ds["uv_zmax"][iuv]   = zmax
                                    self.ds["uv_havg"][iuv,:] = havg
                                    self.ds["uv_nrep"][iuv,:] = nrep
                                    self.ds["uv_pwet"][iuv,:] = pwet
                                    self.ds["uv_ffit"][iuv]   = ffit
                                    self.ds["uv_navg"][iuv]   = navg
                        else:        
                            if nu1[indx] >= 0:
                                nn = (n[indx] - bn0)*refi + int(3*refi/4)
                                mm = (m[indx] - bm0)*refi
                                zgu = zg[nn : nn + int(refi/2), mm : mm + int(refi/2)]
                                zv  = zgu.flatten()
                                manning = manning_grid[nn : nn + int(refi/2), mm : mm + int(refi/2)]
                                manning = manning.flatten()
                                iuv = index_nu1[indx]
                                if iuv>=0:
                                    zmin, zmax, havg, nrep, pwet, ffit, navg, zz = subgrid_q_table(zv, manning, nr_levels, huthresh)
                                    self.ds["uv_zmin"][iuv]   = zmin
                                    self.ds["uv_zmax"][iuv]   = zmax
                                    self.ds["uv_havg"][iuv,:] = havg
                                    self.ds["uv_nrep"][iuv,:] = nrep
                                    self.ds["uv_pwet"][iuv,:] = pwet
                                    self.ds["uv_ffit"][iuv]   = ffit
                                    self.ds["uv_navg"][iuv]   = navg
                            if nu2[indx] >= 0:
                                nn = (n[indx] - bn0)*refi + int(3*refi/4)
                                mm = (m[indx] - bm0)*refi + int(refi/2)
                                zgu = zg[nn : nn + int(refi/2), mm : mm + int(refi/2)]
                                zv  = zgu.flatten()
                                manning = manning_grid[nn : nn + int(refi/2), mm : mm + int(refi/2)]
                                manning = manning.flatten()
                                iuv = index_nu2[indx]
                                if iuv>=0:
                                    zmin, zmax, havg, nrep, pwet, ffit, navg, zz = subgrid_q_table(zv, manning, nr_levels, huthresh)
                                    self.ds["uv_zmin"][iuv]   = zmin
                                    self.ds["uv_zmax"][iuv]   = zmax
                                    self.ds["uv_havg"][iuv,:] = havg
                                    self.ds["uv_nrep"][iuv,:] = nrep
                                    self.ds["uv_pwet"][iuv,:] = pwet
                                    self.ds["uv_ffit"][iuv]   = ffit
                                    self.ds["uv_navg"][iuv]   = navg

                    if progress_bar:
                        progress_bar.set_value(ibt)
                        if progress_bar.was_canceled():
                            return
                        ibt += 1

        if not quiet:
            print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        
        if file_name:
            self.save(file_name)



# class SubgridTableRegular:

#     def __init__(self, model, version=0):
#         # A regular subgrid table contains only for cells with msk>0
#         self.model = model
#         self.version = version

#     def read(self):
#         file_name = os.path.join(self.model.path, self.model.input.variables.sbgfile)
#         self.load(file_name)

#     def load(self, file_name):
        
#         file = open(file_name, "rb")
        
#         # File version        
# #        self.version      = np.fromfile(file, dtype="i4", count=1)[0]
#         self.nr_cells     = np.fromfile(file, dtype="i4", count=1)[0]
#         self.nr_uv_points = np.fromfile(file, dtype="i4", count=1)[0]
#         self.nlevels        = np.fromfile(file, dtype="i4", count=1)[0]
#         self.z_zmin       = np.fromfile(file, dtype="f4", count=self.nr_cells)
#         self.z_zmax       = np.fromfile(file, dtype="f4", count=self.nr_cells)
#         self.z_zmean      = np.fromfile(file, dtype="f4", count=self.nr_cells)
#         self.z_volmax     = np.fromfile(file, dtype="f4", count=self.nr_cells)
#         self.z_depth      = np.zeros((self.nlevels, self.nr_cells), dtype=float)
#         for ibin in range(self.nbins):
#             self.z_depth[ibin,:] = np.fromfile(file, dtype="f4", count=self.nr_cells)
#         self.uv_zmin      = np.fromfile(file, dtype="f4", count=self.nr_uv_points)
#         self.uv_zmax      = np.fromfile(file, dtype="f4", count=self.nr_uv_points)
#         self.uv_hrep      = np.zeros((self.nbins, self.nr_uv_points), dtype=float)
#         for ibin in range(self.nbins):
#             self.uv_hrep[ibin,:] = np.fromfile(file, dtype="f4", count=self.nr_uv_points)
#         self.uv_navg      = np.zeros((self.nbins, self.nr_uv_points), dtype=float)
#         for ibin in range(self.nbins):
#             self.uv_navg[ibin,:] = np.fromfile(file, dtype="f4", count=self.nr_uv_points)
        
#         file.close()        

#     def save(self, file_name, mask):

#         iok = np.where(np.transpose(mask)>0)
#         iok = (iok[1], iok[0])
        
#         nmax = np.shape(self.z_zmin)[0]
#         mmax = np.shape(self.z_zmin)[1]
        
#         # Add 1 because indices in SFINCS start with 1, not 0
#         ind = np.ravel_multi_index(iok, (nmax, mmax), order='F') + 1

        
#         file = open(file_name, "wb")
#         file.write(np.int32(np.size(ind))) # Nr of active points
#         file.write(np.int32(1))            # min
#         file.write(np.int32(self.nbins))

#         # Z
#         v = self.z_zmin[iok]
#         print(np.shape(v))
#         file.write(np.float32(v))
#         v = self.z_zmax[iok]
#         file.write(np.float32(v))
#         v = self.z_volmax[iok]
#         file.write(np.float32(v))
#         for ibin in range(self.nbins):
#             v = np.squeeze(self.z_depth[ibin,:,:])[iok]
#             file.write(np.float32(v))

#         # U
#         v = self.u_zmin[iok]
#         file.write(np.float32(v))
#         v = self.u_zmax[iok]
#         file.write(np.float32(v))
#         dhdz = np.full(np.shape(v), 1.0)
#         file.write(np.float32(dhdz)) # Not used in SFINCS anymore       
#         for ibin in range(self.nbins):
#             v = np.squeeze(self.u_hrep[ibin,:,:])[iok]
#             file.write(np.float32(v))
#         for ibin in range(self.nbins):
#             v = np.squeeze(self.u_navg[ibin,:,:])[iok]
#             file.write(np.float32(v))

#         # V
#         v = self.v_zmin[iok]
#         file.write(np.float32(v))
#         v = self.v_zmax[iok]
#         file.write(np.float32(v))
#         file.write(np.float32(dhdz)) # Not used in SFINCS anymore       
#         for ibin in range(self.nbins):
#             v = np.squeeze(self.v_hrep[ibin,:,:])[iok]
#             file.write(np.float32(v))
#         for ibin in range(self.nbins):
#             v = np.squeeze(self.v_navg[ibin,:,:])[iok]
#             file.write(np.float32(v))
        
#         file.close()
        
#     def build(self,
#               grid,
#               bathymetry_sets,
#               roughness_sets,
#               file_name=None,
#               mask=None,
#               nr_bins=10,
#               nr_subgrid_pixels=20,
#               max_gradient=5.0,
#               depth_factor=1.0,
#               zmin=-99999.0,
#               huthresh=0.01,
#               quiet=True):  

#         nbins = nr_bins
#         refi  = nr_subgrid_pixels
#         z_minimum = zmin
#         self.nbins    = nr_bins

#         # Prepare transformers
#         bathymetry_transformers = []  
#         for bathymetry in bathymetry_sets:
#             bathymetry_transformers.append(Transformer.from_crs(grid.crs,
#                                                                 bathymetry.crs,
#                                                                 always_xy=True))
#         roughness_transformers = []  
#         for roughness in roughness_sets:
#             if roughness.type == "source":
#                 roughness_transformers.append(Transformer.from_crs(grid.crs,
#                                                                    roughness.crs,
#                                                                    always_xy=True))
#             else:
#                 roughness_transformers.append(None)

#         grid_dim = (grid.nmax, grid.mmax)

#         # Z points        
#         self.z_zmin   = np.empty(grid_dim, dtype=float)
#         self.z_zmax   = np.empty(grid_dim, dtype=float)
#         self.z_zmean  = np.empty(grid_dim, dtype=float)
#         self.z_volmax = np.empty(grid_dim, dtype=float)
#         self.z_depth  = np.empty((nbins, grid.nmax, grid.mmax), dtype=float)

#         # U points        
#         self.u_zmin = np.empty(grid_dim, dtype=float)
#         self.u_zmax = np.empty(grid_dim, dtype=float)
#         self.u_havg = np.empty((nbins, grid.nmax, grid.mmax), dtype=float)
#         self.u_nrep = np.empty((nbins, grid.nmax, grid.mmax), dtype=float)
#         self.u_pwet = np.empty((nbins, grid.nmax, grid.mmax), dtype=float)
#         self.u_fnfit = np.empty((grid.nmax, grid.mmax), dtype=float)
#         self.u_navg_w = np.empty((grid.nmax, grid.mmax), dtype=float)

#         # V points        
#         self.v_zmin = np.empty(grid_dim, dtype=float)
#         self.v_zmax = np.empty(grid_dim, dtype=float)
#         self.v_havg = np.empty((nbins, grid.nmax, grid.mmax), dtype=float)
#         self.v_nrep = np.empty((nbins, grid.nmax, grid.mmax), dtype=float)
#         self.v_pwet = np.empty((nbins, grid.nmax, grid.mmax), dtype=float)
#         self.v_fnfit = np.empty((grid.nmax, grid.mmax), dtype=float)
#         self.v_navg_w = np.empty((grid.nmax, grid.mmax), dtype=float)
                
#         cosrot = np.cos(grid.rotation*np.pi/180)
#         sinrot = np.sin(grid.rotation*np.pi/180)
#         nrmax  = 2000
        
#         n0 = 0
#         n1 = grid.nmax - 1 # + 1 # add extra cell to compute u and v in the last row/column
#         m0 = 0
#         m1 = grid.mmax - 1 # + 1 # add extra cell to compute u and v in the last row/column
        
#         dx   = grid.dx      # cell size
#         dy   = grid.dy      # cell size
#         dxp  = dx/refi      # size of subgrid pixel
#         dyp  = dy/refi      # size of subgrid pixel
        
#         nrcb = int(np.floor(nrmax/refi))         # nr of regular cells in a block            
#         nrbn = int(np.ceil((n1 - n0 + 1)/nrcb))  # nr of blocks in n direction
#         nrbm = int(np.ceil((m1 - m0 + 1)/nrcb))  # nr of blocks in m direction

#         if not quiet:
#             print("Number of regular cells in a block : " + str(nrcb))
#             print("Number of blocks in n direction    : " + str(nrbn))
#             print("Number of blocks in m direction    : " + str(nrbm))
        
#         if not quiet:
#             print("Grid size of flux grid             : dx= " + str(dx) + ", dy= " + str(dy))
#             print("Grid size of subgrid pixels        : dx= " + str(dxp) + ", dy= " + str(dyp))

#         ## Loop through blocks
#         ib = -1
#         for ii in range(nrbm):
#             for jj in range(nrbn):
                
#                 # Count
#                 ib += 1
                
#                 bn0 = n0  + jj*nrcb               # Index of first n in block
#                 bn1 = min(bn0 + nrcb - 1, n1) + 1 # Index of last n in block (cut off excess above, but add extra cell to compute u and v in the last row)
#                 bm0 = m0  + ii*nrcb               # Index of first m in block
#                 bm1 = min(bm0 + nrcb - 1, m1) + 1 # Index of last m in block (cut off excess to the right, but add extra cell to compute u and v in the last column)

#                 if not quiet:
#                     print("--------------------------------------------------------------")
#                     print("Processing block " + str(ib + 1) + " of " + str(nrbn*nrbm) + " ...")
                    
#                 # Now build the pixel matrix
#                 x00 = 0.5*dxp + bm0*refi*dyp
#                 x01 = x00 + (bm1 - bm0 + 1)*refi*dxp
#                 y00 = 0.5*dyp + bn0*refi*dyp
#                 y01 = y00 + (bn1 - bn0 + 1)*refi*dyp
                
#                 x0 = np.arange(x00, x01, dxp)
#                 y0 = np.arange(y00, y01, dyp)
#                 xg0, yg0 = np.meshgrid(x0, y0)
#                 # Rotate and translate
#                 xg = grid.x0 + cosrot*xg0 - sinrot*yg0
#                 yg = grid.y0 + sinrot*xg0 + cosrot*yg0                    

#                 # Clear variables
#                 del x0, y0, xg0, yg0
                
#                 # Initialize depth of subgrid at NaN
#                 zg = np.full(np.shape(xg), np.nan)
                
#                 # Loop through bathymetry datasets
#                 for ibathy, bathymetry in enumerate(bathymetry_sets):

#                     # Check if there are NaNs left in this block 

#                     if np.isnan(zg).any():


#                         if bathymetry.type == "source":

#                             xgb, ygb = bathymetry_transformers[ibathy].transform(xg, yg)

#                             xmin = np.nanmin(np.nanmin(xgb))
#                             xmax = np.nanmax(np.nanmax(xgb))
#                             ymin = np.nanmin(np.nanmin(ygb))
#                             ymax = np.nanmax(np.nanmax(ygb))
#                             ddx  = 0.05*(xmax - xmin)
#                             ddy  = 0.05*(ymax - ymin)
#                             xl   = [xmin - ddx, xmax + ddx]
#                             yl   = [ymin - ddy, ymax + ddy]
                        
#                             # Get DEM data (ddb format for now)
#                             xb, yb, zb = bathymetry_database.get_data(bathymetry.name,
#                                                                       xl,
#                                                                       yl,
#                                                                       max_cell_size=dxp)

#                             if zb is not np.nan:
#                                 zb[np.where(zb<bathymetry.zmin)] = np.nan
#                                 zb[np.where(zb>bathymetry.zmax)] = np.nan
#                                 if not np.isnan(zb).all():
#                                     zg1 = interp2(xb, yb, zb, xgb, ygb)
#                                     isn = np.where(np.isnan(zg))
#                                     zg[isn] = zg1[isn]
                            
# #                                    if not np.size(zb) == 1: #TL: size = 1 happens when zb = nan because of 'Tiles are outside of search range' in bathymetry_database.py > read_data_from_netcdf_tiles. Then following action doesn't work on a single float
# #                                        zb[np.where(zb<bathymetry.zmin)] = np.nan
# #                                        zb[np.where(zb>bathymetry.zmax)] = np.nan
# #                                        
# #                                    if not np.isnan(zb).all():
# #                                        zg1 = interp2(xb, yb, zb, xgb, ygb)
# #                                        isn = np.where(np.isnan(zg))
# #                                        zg[isn] = zg1[isn]
                        
#                     elif bathymetry.type == "array":
#                         # Matrix provided, interpolate to subgrid mesh
#                         # TODO
# #                            zg = interp2(bathymetry.x, bathymetry.y, bathymetry.z, xgb, ygb)
#                         pass
                
#                 # Multiply zg with depth factor (had to use 0.9746 to get arrival
#                 # times right in the Pacific)
#                 zg = zg*depth_factor
                
#                 zg = np.maximum(zg, z_minimum)

#                 # Manning's n values
                
#                 # Initialize roughness of subgrid at NaN
#                 manning_grid = np.full(np.shape(xg), np.nan)

#                 # Loop through roughness datasets
#                 for irgh, roughness in enumerate(roughness_sets):

#                     # Check if there are NaNs left in this block 

#                     if np.isnan(manning_grid).any():
                    
#                         if roughness.type == "source":

#                             xgb, ygb = roughness_transformers[irgh].transform(xg, yg)

#                             xmin = np.nanmin(np.nanmin(xgb))
#                             xmax = np.nanmax(np.nanmax(xgb))
#                             ymin = np.nanmin(np.nanmin(ygb))
#                             ymax = np.nanmax(np.nanmax(ygb))
#                             ddx  = 0.05*(xmax - xmin)
#                             ddy  = 0.05*(ymax - ymin)
#                             xl   = [xmin - ddx, xmax + ddx]
#                             yl   = [ymin - ddy, ymax + ddy]
                        
#                             # Get DEM data (ddb format for now)
#                             xb, yb, zb = bathymetry_database.get_data(roughness.name,
#                                                                       xl,
#                                                                       yl,
#                                                                       max_cell_size=dxp)

#                             if zb is not np.nan:
#                                 if not np.isnan(zb).all():
#                                     zg1 = interp2(xb, yb, zb, xgb, ygb)
#                                     isn = np.where(np.isnan(manning_grid))
#                                     manning_grid[isn] = zg1[isn]
 
#                         elif roughness.type == "constant":
#                             isn = np.where(np.isnan(manning_grid))
# #                                manning_grid = np.zeros(np.shape(zg))
# #                                manning_grid[np.where(zg<=roughness.zlevel)] = roughness.roughness_deep
# #                                manning_grid[np.where(zg>roughness.zlevel)] = roughness.roughness_shallow
#                             manning_grid[(isn and np.where(zg<=roughness.zlevel))] = roughness.roughness_deep
#                             manning_grid[(isn and np.where(zg>roughness.zlevel))] = roughness.roughness_shallow
                           
#                         elif roughness.type == "array":
#                             # Matrix provided, interpolate to subgrid mesh
#                             # TODO
# #                            zg = interp2(roughness.x, roughness.y, roughness.z, xgb, ygb)
#                             pass
               
#                 # clear temp variables
#                 del xb, yb, zb
                
#                 # Now compute subgrid properties

#                 # Loop through all active cells in this block
#                 for m in range(bm0, bm1):
#                     for n in range(bn0, bn1):
                        
#                         if grid.mask[n, m]<1:
#                             # Not an active point
#                             continue

#                         # First the volumes in the cells
#                         nn  = (n - bn0) * refi
#                         mm  = (m - bm0) * refi
#                         zgc = zg[nn : nn + refi, mm : mm + refi]
    
#                         # Compute pixel size in metres
#                         if grid.crs.is_geographic:
#                             ygc = yg[nn : nn + refi, mm : mm + refi]
#                             mean_lat =np.abs(np.mean(ygc))
#                             dxpm = dxp*111111.0*np.cos(np.pi*mean_lat/180.0)
#                             dypm = dyp*111111.0
#                         else:
#                             dxpm = dxp
#                             dypm = dyp
                        
#                         zv  = zgc.flatten()   
#                         zvmin = -20.0
#                         z, v, zmin, zmax, zmean = subgrid_v_table(zv, dxpm, dypm, nbins, zvmin, max_gradient)
#                         self.z_zmin[n, m]    = zmin
#                         self.z_zmax[n, m]    = zmax
#                         self.z_zmean[n, m]   = zmean
#                         self.z_volmax[n, m]  = v[-1]
#                         self.z_depth[:, n, m] = z[1:]
                        
#                         # Now the U/V points
#                         # U
#                         nn  = (n - bn0)*refi
#                         mm  = (m - bm0)*refi + int(0.5*refi)
#                         zgu = zg[nn : nn + refi, mm : mm + refi]
#                         zgu = np.transpose(zgu)
#                         zv  = zgu.flatten()
#                         manning = manning_grid[nn : nn + refi, mm : mm + refi]
#                         manning = np.transpose(manning)
#                         manning = manning.flatten()
#                         huthresh = 0.01
#                         zmin, zmax, havg, nrep, pwet, fnfit, navg_w, zz = subgrid_q_table(zv, manning, nbins, huthresh)
#                         self.u_zmin[n, m]   = zmin
#                         self.u_zmax[n, m]   = zmax
#                         self.u_havg[:, n, m] = havg
#                         self.u_nrep[:, n, m] = nrep
#                         self.u_pwet[:, n, m] = pwet
#                         self.u_fnfit[n, m] = fnfit
#                         self.u_navg_w[n, m] = navg_w

#                         # V
#                         nn = (n - bn0)*refi + int(0.5*refi)
#                         mm = (m - bm0)*refi
#                         zgu = zg[nn : nn + refi, mm : mm + refi]
#                         zv  = zgu.flatten()
#                         manning = manning_grid[nn : nn + refi, mm : mm + refi]
#                         manning = manning.flatten()
#                         zmin, zmax, havg, nrep, pwet, fnfit, navg_w, zz = subgrid_q_table(zv, manning, nbins, huthresh)
#                         self.v_zmin[n, m]    = zmin
#                         self.v_zmax[n, m]    = zmax
#                         self.v_havg[:, n, m] = havg
#                         self.v_navg[:, n, m] = nrep
#                         self.v_pwet[:, n, m] = pwet
#                         self.v_fnfit[n, m] = fnfit
#                         self.v_navg_w[n, m] = navg_w

#         if not quiet:
#             print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        
#         if file_name:
#             self.save(file_name, grid.mask)





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
    return z, V, elevation.min(), z.max(), ele_sort.mean()

def subgrid_q_table(elevation, manning, nlevels, huthresh):
    """
    map vector of elevation values into a hypsometric hydraulic radius - depth relationship for one u/v point
    Parameters
    ----------
    elevation : np.ndarray (nr of pixels in one cell) containing subgrid elevation values for one grid cell [m]
    manning : np.ndarray (nr of pixels in one cell) containing subgrid manning roughness values for one grid cell [s m^(-1/3)]
    nlevels : int, number of vertical levels [-]
    huthresh : float, threshold depth [m]
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

    havg = np.zeros(nlevels)
    nrep = np.zeros(nlevels)
    pwet = np.zeros(nlevels)
    zz   = np.zeros(nlevels)

    n   = int(np.size(elevation)) # Nr of pixels in grid cell
    n05 = int(n / 2)

    dd_a      = elevation[0:n05]
    dd_b      = elevation[n05:]
    manning_a = manning[0:n05]
    manning_b = manning[n05:]

    zmin_a    = np.min(dd_a)
    zmax_a    = np.max(dd_a)
    
    zmin_b    = np.min(dd_b)
    zmax_b    = np.max(dd_b)
    
    zmin = max(zmin_a, zmin_b) + huthresh
    zmax = max(zmax_a, zmax_b)
    
    # Make sure zmax is a bit higher than zmin
    if zmax < zmin + 0.001:
       zmax = max(zmax, zmin + 0.001)

    # Determine bin size
    dlevel = (zmax - zmin)/(nlevels - 1)

    # Option can be either 1 ("old") or 2 ("new")
    option = 2

    # Loop through levels
    for ibin in range(nlevels):

        # Top of bin
        zbin = zmin + ibin * dlevel
        zz[ibin] = zbin

        h = np.maximum(zbin - elevation, 0.0)  # water depth in each pixel


        pwet[ibin] = (zbin - elevation > -1.0e-6).sum() / n

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

        elif option == 2:
            # Use newer 2 option (minimum of q_a an q_b, minimum of h_a and h_b increasing to h_all, using pwet for weighting) option
            pwet_a = (zbin - dd_a > -1.0e-6).sum() / (n / 2) 
            pwet_b = (zbin - dd_b > -1.0e-6).sum() / (n / 2) 
            # Weight increases linearly from 0 to 1 from bottom to top bin use percentage wet in sides A and B
            w = 2 * np.minimum(pwet_a, pwet_b) / (pwet_a + pwet_b)
            q     = (1.0 - w) * q_min + w * q_all        # Weighted average of q_min and q_all
            hmean = (1.0 - w) * h_min + w * h_all        # Weighted average of h_min and h_all

        havg[ibin] = hmean                          # conveyance depth
        nrep[ibin] = hmean**(5.0 / 3.0) / q           # Representative n for qmean and hmean

    nrep_top = nrep[-1]    
    havg_top = havg[-1]

    ### Fitting for nrep above zmax

    # Determine nfit at zfit
    zfit  = zmax + zmax - zmin
    hfit  = havg_top + zmax - zmin                 # mean water depth in cell as computed in SFINCS (assuming linear relation between water level and water depth above zmax)

    # Compute q and navg
    h     = np.maximum(zfit - elevation, 0.0)      # water depth in each pixel
    q     = np.mean(h**(5.0 / 3.0) / manning)          # combined unit discharge for cell
    navg  = np.mean(manning)

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
