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

from cht_utils.misc_tools import interp2


class QuadtreeMask:
    def __init__(self, qtrgrid,
                 zmin=99999.0,
                 zmax=-99999.0,
                 include_polygons=None,
                 exclude_polygons=None,
                 open_boundary_polygons=None,
                 outflow_boundary_polygons=None,
                 quiet=True):

        if not quiet:
            print("Building quadtree mask ...")

        self.mask = np.zeros(qtrgrid.nr_cells, dtype=int)

        if zmin>=zmax:
            # Do not include any points initially
            if not include_polygons:
                print("WARNING: Entire mask set to zeros! Please ensure zmax is greater than zmin, or provide include polygon(s) !")
                return
        else:
            if qtrgrid.z is not None:                
                # Set initial mask based on zmin and zmax
                iok = np.where((qtrgrid.z>=zmin) & (qtrgrid.z<=zmax))
                self.mask[iok] = 1
            else:
                print("WARNING: Entire mask set to zeros! No depth values found on grid.")
                        
        # Include polygons
        if include_polygons:
            for polygon in include_polygons:
                inpol = inpolygon(qtrgrid.x, qtrgrid.y, polygon.geometry)
                iok   = np.where((inpol) & (qtrgrid.z>=polygon.zmin) & (qtrgrid.z<=polygon.zmax))
                self.mask[iok] = 1

        # Exclude polygons
        if exclude_polygons:
            for polygon in exclude_polygons:
                inpol = inpolygon(qtrgrid.x, qtrgrid.y, polygon.geometry)
                iok   = np.where((inpol) & (qtrgrid.z>=polygon.zmin) & (qtrgrid.z<=polygon.zmax))
                self.mask[iok] = 0

        # Open boundary polygons
        if open_boundary_polygons:
            for polygon in open_boundary_polygons:
                inpol = inpolygon(qtrgrid.x, qtrgrid.y, polygon.geometry)
                # Only consider points that are:
                # 1) Inside the polygon
                # 2) Have a mask > 0
                # 3) z>=zmin
                # 4) z<=zmax
                iok   = np.where((inpol) & (self.mask>0) & (qtrgrid.z>=polygon.zmin) & (qtrgrid.z<=polygon.zmax))
                for ic in iok[0]:

                    okay = False

                    # Check neighbors, cell must have at least one inactive neighbor

                    # Left
                    if qtrgrid.md[ic]<=0:
                        # Coarser or equal to the left
                        if qtrgrid.md1[ic]>=0:
                            # Cell has neighbor to the left
                            if self.mask[qtrgrid.md1[ic]]==0:
                                # And it's inactive
                                okay = True
                        else:
                            # No neighbor, so set mask = 2
                            okay = True
                    else:
                        # Finer to the left
                        if qtrgrid.md1[ic]>=0:
                            # Cell has neighbor to the left
                            if self.mask[qtrgrid.md1[ic]]==0:
                                # And it's inactive
                                okay = True
                        else:
                            # No neighbor, so set mask = 2
                            okay = True
                        if qtrgrid.md2[ic]>=0:
                            # Cell has neighbor to the left
                            if self.mask[qtrgrid.md2[ic]]==0:
                                # And it's inactive
                                okay = True
                        else:
                            # No neighbor, so set mask = 2
                            okay = True
                        
                    # Below
                    if qtrgrid.nd[ic]<=0:
                        # Coarser or equal below
                        if qtrgrid.nd1[ic]>=0:
                            # Cell has neighbor below
                            if self.mask[qtrgrid.nd1[ic]]==0:
                                # And it's inactive
                                okay = True
                        else:
                            # No neighbor, so set mask = 2
                            okay = True
                    else:
                        # Finer below
                        if qtrgrid.nd1[ic]>=0:
                            # Cell has neighbor below
                            if self.mask[qtrgrid.nd1[ic]]==0:
                                # And it's inactive
                                okay = True
                        else:
                            # No neighbor, so set mask = 2
                            okay = True
                        if qtrgrid.nd2[ic]>=0:
                            # Cell has neighbor below
                            if self.mask[qtrgrid.nd2[ic]]==0:
                                # And it's inactive
                                okay = True
                        else:
                            # No neighbor, so set mask = 2
                            okay = True

                    # Right
                    if qtrgrid.mu[ic]<=0:
                        # Coarser or equal to the right
                        if qtrgrid.mu1[ic]>=0:
                            # Cell has neighbor to the right
                            if self.mask[qtrgrid.mu1[ic]]==0:
                                # And it's inactive
                                okay = True
                        else:
                            # No neighbor, so set mask = 2
                            okay = True
                    else:
                        # Finer to the left
                        if qtrgrid.mu1[ic]>=0:
                            # Cell has neighbor to the right
                            if self.mask[qtrgrid.mu1[ic]]==0:
                                # And it's inactive
                                okay = True
                        else:
                            # No neighbor, so set mask = 2
                            okay = True
                        if qtrgrid.mu2[ic]>=0:
                            # Cell has neighbor to the right
                            if self.mask[qtrgrid.mu2[ic]]==0:
                                # And it's inactive
                                okay = True
                        else:
                            # No neighbor, so set mask = 2
                            okay = True

                    # Above
                    if qtrgrid.nu[ic]<=0:
                        # Coarser or equal above
                        if qtrgrid.nu1[ic]>=0:
                            # Cell has neighbor above
                            if self.mask[qtrgrid.nu1[ic]]==0:
                                # And it's inactive
                                okay = True
                        else:
                            # No neighbor, so set mask = 2
                            okay = True
                    else:
                        # Finer below
                        if qtrgrid.nu1[ic]>=0:
                            # Cell has neighbor above
                            if self.mask[qtrgrid.nu1[ic]]==0:
                                # And it's inactive
                                okay = True
                        else:
                            # No neighbor, so set mask = 2
                            okay = True
                        if qtrgrid.nu2[ic]>=0:
                            # Cell has neighbor above
                            if self.mask[qtrgrid.nu2[ic]]==0:
                                # And it's inactive
                                okay = True
                        else:
                            # No neighbor, so set mask = 2
                            okay = True
                        
                    if okay:
                        self.mask[ic] = 2

        # Outflow boundary polygons
        if outflow_boundary_polygons:
            for polygon in outflow_boundary_polygons:
                inpol = inpolygon(qtrgrid.x, qtrgrid.y, polygon.geometry)
                # Only consider points that are:
                # 1) Inside the polygon
                # 2) Have a mask > 0
                # 3) z>=zmin
                # 4) z<=zmax
                iok   = np.where((inpol) & (self.mask>0) & (qtrgrid.z>=polygon.zmin) & (qtrgrid.z<=polygon.zmax))
                for ic in iok[0]:

                    okay = False

                    # Check neighbors, cell must have at least one inactive neighbor

                    # Left
                    if qtrgrid.md[ic]<=0:
                        # Coarser or equal to the left
                        if qtrgrid.md1[ic]>=0:
                            # Cell has neighbor to the left
                            if self.mask[qtrgrid.md1[ic]]==0:
                                # And it's inactive
                                okay = True
                        else:
                            # No neighbor, so set mask = 3
                            okay = True
                    else:
                        # Finer to the left
                        if qtrgrid.md1[ic]>=0:
                            # Cell has neighbor to the left
                            if self.mask[qtrgrid.md1[ic]]==0:
                                # And it's inactive
                                okay = True
                        else:
                            # No neighbor, so set mask = 3
                            okay = True
                        if qtrgrid.md2[ic]>=0:
                            # Cell has neighbor to the left
                            if self.mask[qtrgrid.md2[ic]]==0:
                                # And it's inactive
                                okay = True
                        else:
                            # No neighbor, so set mask = 3
                            okay = True
                        
                    # Below
                    if qtrgrid.nd[ic]<=0:
                        # Coarser or equal below
                        if qtrgrid.nd1[ic]>=0:
                            # Cell has neighbor below
                            if self.mask[qtrgrid.nd1[ic]]==0:
                                # And it's inactive
                                okay = True
                        else:
                            # No neighbor, so set mask = 3
                            okay = True
                    else:
                        # Finer below
                        if qtrgrid.nd1[ic]>=0:
                            # Cell has neighbor below
                            if self.mask[qtrgrid.nd1[ic]]==0:
                                # And it's inactive
                                okay = True
                        else:
                            # No neighbor, so set mask = 3
                            okay = True
                        if qtrgrid.nd2[ic]>=0:
                            # Cell has neighbor below
                            if self.mask[qtrgrid.nd2[ic]]==0:
                                # And it's inactive
                                okay = True
                        else:
                            # No neighbor, so set mask = 3
                            okay = True

                    # Right
                    if qtrgrid.mu[ic]<=0:
                        # Coarser or equal to the right
                        if qtrgrid.mu1[ic]>=0:
                            # Cell has neighbor to the right
                            if self.mask[qtrgrid.mu1[ic]]==0:
                                # And it's inactive
                                okay = True
                        else:
                            # No neighbor, so set mask = 3
                            okay = True
                    else:
                        # Finer to the left
                        if qtrgrid.mu1[ic]>=0:
                            # Cell has neighbor to the right
                            if self.mask[qtrgrid.mu1[ic]]==0:
                                # And it's inactive
                                okay = True
                        else:
                            # No neighbor, so set mask = 3
                            okay = True
                        if qtrgrid.mu2[ic]>=0:
                            # Cell has neighbor to the right
                            if self.mask[qtrgrid.mu2[ic]]==0:
                                # And it's inactive
                                okay = True
                        else:
                            # No neighbor, so set mask = 3
                            okay = True

                    # Above
                    if qtrgrid.nu[ic]<=0:
                        # Coarser or equal above
                        if qtrgrid.nu1[ic]>=0:
                            # Cell has neighbor above
                            if self.mask[qtrgrid.nu1[ic]]==0:
                                # And it's inactive
                                okay = True
                        else:
                            # No neighbor, so set mask = 3
                            okay = True
                    else:
                        # Finer below
                        if qtrgrid.nu1[ic]>=0:
                            # Cell has neighbor above
                            if self.mask[qtrgrid.nu1[ic]]==0:
                                # And it's inactive
                                okay = True
                        else:
                            # No neighbor, so set mask = 3
                            okay = True
                        if qtrgrid.nu2[ic]>=0:
                            # Cell has neighbor above
                            if self.mask[qtrgrid.nu2[ic]]==0:
                                # And it's inactive
                                okay = True
                        else:
                            # No neighbor, so set mask = 3
                            okay = True
                        
                    if okay:
                        self.mask[ic] = 3

    def load(self, file_name=None):
        pass

    def save(self, file_name):
        file = open(file_name, "wb")
        file.write(np.int8(self.mask))
        file.close()
    
class QuadtreeGrid:

    def __init__(self, crs=None, regular_grid=None, mask=None):
        
        self.x0 = None
        self.y0 = None
        self.dx = None
        self.dy = None
        self.rotation = None
        self.nr_cells = 0
        self.nr_refinement_levels = 1
        self.crs = crs
        self.version = 0
        
        self.n   = None
        self.m   = None
        self.level = None
        self.x   = None
        self.y   = None
        self.z   = None

        self.mu  = None
        self.mu1 = None
        self.mu2 = None
        self.md  = None
        self.md1 = None
        self.md2 = None
        self.nu  = None
        self.nu1 = None
        self.nu2 = None
        self.nd  = None
        self.nd1 = None
        self.nd2 = None
        
        if regular_grid:
            # Convert regular grid to quadtree grid
            self.from_regular_grid(regular_grid, mask)
            

    def load(self, file_name):
        
        file = open(file_name, "rb")
        
        # File version        
        self.version = np.fromfile(file, dtype="i1", count=1)[0]

        # CRS
        epsg_code = np.fromfile(file, dtype="i4", count=1)[0]
        try:
            self.crs = CRS.from_epsg(epsg_code)
        except:
            print("EPSG code is " + str(epsg_code) + ". No matching CRS found.")
            pass

        # Number of cells
        self.nr_cells = np.fromfile(file, dtype="i4", count=1)[0]
        
        # Nr levels (should remove the -1 here, but then also in sfincs code)
        self.nr_refinement_levels = np.fromfile(file, dtype="i1", count=1)[0]
        
        # Grid stuff
        self.x0 = np.fromfile(file, dtype="f4", count=1)[0]
        self.y0 = np.fromfile(file, dtype="f4", count=1)[0]
        self.dx = np.fromfile(file, dtype="f4", count=1)[0]
        self.dy = np.fromfile(file, dtype="f4", count=1)[0]
        self.rotation = np.fromfile(file, dtype="f4", count=1)[0]
        
        # Levels
        self.level = np.fromfile(file, dtype="i1", count=self.nr_cells) - 1
        
        # N
        self.n = np.fromfile(file, dtype="i4", count=self.nr_cells) - 1
        
        # M
        self.m = np.fromfile(file, dtype="i4", count=self.nr_cells) - 1
        
        # Neighbors
        self.nu  = np.fromfile(file, dtype="i1", count=self.nr_cells)
        self.nu1 = np.fromfile(file, dtype="i4", count=self.nr_cells) - 1
        self.nu2 = np.fromfile(file, dtype="i4", count=self.nr_cells) - 1
        self.mu  = np.fromfile(file, dtype="i1", count=self.nr_cells)
        self.mu1 = np.fromfile(file, dtype="i4", count=self.nr_cells) - 1
        self.mu2 = np.fromfile(file, dtype="i4", count=self.nr_cells) - 1
        self.nd  = np.fromfile(file, dtype="i1", count=self.nr_cells)
        self.nd1 = np.fromfile(file, dtype="i4", count=self.nr_cells) - 1
        self.nd2 = np.fromfile(file, dtype="i4", count=self.nr_cells) - 1
        self.md  = np.fromfile(file, dtype="i1", count=self.nr_cells)
        self.md1 = np.fromfile(file, dtype="i4", count=self.nr_cells) - 1
        self.md2 = np.fromfile(file, dtype="i4", count=self.nr_cells) - 1
        self.z   = np.fromfile(file, dtype="f4", count=self.nr_cells)
        
        file.close()
        
        self.compute_cell_centre_coordinates()

        # First indices of levels
        ilast = -1
        self.level_index = []
        for j in range(self.nr_cells):
#            print(j)
            if self.level[j] > ilast:
                self.level_index.append(j)
                ilast = ilast + 1


    def save(self, file_name, version=0):
        
        file = open(file_name, "wb")
        
        # File version        
        file.write(np.int8(version))

        # CRS
        if self.crs:
            epsg_code = self.crs.to_epsg()
        else:
            epsg_code = 0
        file.write(np.int32(epsg_code))
        
        # Number of cells
        file.write(np.int32(self.nr_cells))
        
        # Nr levels (should remove the -1 here, but then also in sfincs code)
        file.write(np.int8(self.nr_refinement_levels))
        
        # Grid stuff
        file.write(np.float32(self.x0))
        file.write(np.float32(self.y0))
        file.write(np.float32(self.dx))
        file.write(np.float32(self.dy))
        file.write(np.float32(self.rotation))
        
        # Levels
        file.write(np.int8(self.level + 1))
        
        # N
        file.write(np.int32(self.n + 1))
        
        # M
        file.write(np.int32(self.m + 1))
        
        # Neighbors
        # NU
        file.write(np.int8(self.nu))
        file.write(np.int32(self.nu1 + 1))
        file.write(np.int32(self.nu2 + 1))
        # MU
        file.write(np.int8(self.mu))
        file.write(np.int32(self.mu1 + 1))
        file.write(np.int32(self.mu2 + 1))
        # ND
        file.write(np.int8(self.nd))
        file.write(np.int32(self.nd1 + 1))
        file.write(np.int32(self.nd2 + 1))
        # MD
        file.write(np.int8(self.md))
        file.write(np.int32(self.md1 + 1))
        file.write(np.int32(self.md2 + 1))
        # Z
        file.write(np.float32(self.z))
        
        file.close()
    
    def build(self,
                  x0,
                  y0,
                  nmax,
                  mmax,
                  dx,
                  dy,
                  rotation,
                  refinement_polygons=None):

        print("Building mesh ...")

        start = time.time()

        self.x0 = x0
        self.y0 = y0
        self.dx = dx
        self.dy = dy
        self.nmax = nmax
        self.mmax = mmax
        self.rotation = rotation

        cosrot = np.cos(rotation*np.pi/180)
        sinrot = np.sin(rotation*np.pi/180)
        
        refmax = 0
        if refinement_polygons:
            for pol in refinement_polygons:
                refmax = max(refmax, pol.refinement_level)
            
        # Number of refinement levels
        nlev = refmax + 1
        self.nr_refinement_levels = nlev
        self.ifirst = np.zeros(nlev, dtype=int)

        # Set refinement mask
        nmx       = []
        mmx       = []
        dxb       = []
        dyb       = []
        refmsk    = []
        inirefmsk = []
        isrefined = []
        
        # Loop through refinement levels to set some constants per level
        for ilev in range(nlev):
            nmx.append(nmax*2**(ilev))
            mmx.append(mmax*2**(ilev))
            dxb.append(dx/2**(ilev))
            dyb.append(dy/2**(ilev))
            refmsk.append(np.zeros((nmx[ilev], mmx[ilev]), dtype=int))
            inirefmsk.append(np.zeros((nmx[ilev], mmx[ilev]), dtype=int))
            isrefined.append(np.zeros((nmx[ilev], mmx[ilev]), dtype=int))
        
        inirefmsk[0] += 1

        # First set initial refinement levels based on polygons
        if refinement_polygons:
            for ilev in reversed(range(nlev)):
                # Loop through grid points
                xcor = np.zeros((4, nmx[ilev], mmx[ilev]))
                ycor = np.zeros((4, nmx[ilev], mmx[ilev]))
                for m in range(mmx[ilev]):
                    for n in range(nmx[ilev]):
                        # 4 corner points of this cell
                        xcor[0,n,m] = x0 + cosrot*((m    )*dxb[ilev]) - sinrot*((n    )*dyb[ilev])
                        ycor[0,n,m] = y0 + sinrot*((m    )*dxb[ilev]) + cosrot*((n    )*dyb[ilev])
                        xcor[1,n,m] = x0 + cosrot*((m + 1)*dxb[ilev]) - sinrot*((n    )*dyb[ilev])
                        ycor[1,n,m] = y0 + sinrot*((m + 1)*dxb[ilev]) + cosrot*((n    )*dyb[ilev])
                        xcor[2,n,m] = x0 + cosrot*((m + 1)*dxb[ilev]) - sinrot*((n + 1)*dyb[ilev])
                        ycor[2,n,m] = y0 + sinrot*((m + 1)*dxb[ilev]) + cosrot*((n + 1)*dyb[ilev])
                        xcor[3,n,m] = x0 + cosrot*((m    )*dxb[ilev]) - sinrot*((n + 1)*dyb[ilev])
                        ycor[3,n,m] = y0 + sinrot*((m    )*dxb[ilev]) + cosrot*((n + 1)*dyb[ilev])
    
                # Loop through polygons
                inp = np.empty((nmx[ilev], mmx[ilev])).astype(bool)*False
                for ipol, polygon in enumerate(refinement_polygons):
                    # Check if this refinement level ilev matches refinement levels of this polygon
                    if polygon.refinement_level == ilev:
                        for j in range(4):
                            inp0 = inpolygon(np.squeeze(xcor[j,:,:]),
                                             np.squeeze(ycor[j,:,:]),
                                             polygon.geometry)
                            inp[np.where(inp0)] = True
                            inirefmsk[ilev][np.where(inp)] = 1
            
        # Highest levels have now been set        
        
        # Loop through levels in reverse order to refine cells
        for ilev in reversed(range(nlev)):
            for m in range(mmx[ilev]):
                for n in range(nmx[ilev]):
                    if not isrefined[ilev][n, m]:
                        
                        # Two reasons to use this block
                        # 1) Neighbor is refined
                        # 2) Initial minimum level is ilev

                        iok = False
                          
                        if inirefmsk[ilev][n, m] == 1:
                            # This cell lies within a refinement polygon at this level
                            iok = True

                        else:

                            # Check for neighbors (only for coarser levels)
                            if ilev<nlev - 1:
                                # Left
                                if m>0:
                                    if isrefined[ilev][n, m - 1]:
                                        iok = True
                                # Right
                                if m<mmx[ilev] - 1:
                                    if isrefined[ilev][n, m + 1]:
                                        iok = True
                                # Top
                                if n>0:
                                    if isrefined[ilev][n - 1, m]:
                                        iok = True
                                # Bottom
                                if n<nmx[ilev] - 1:
                                    if isrefined[ilev][n + 1, m]:
                                        iok = True

                        if iok:
                            
                            # Should use this cell
                            refmsk[ilev][n, m] = 1

                            # Set lower level cells to refined so that we know in lower
                            # refinement levels that they should not be used
                            nn = n
                            mm = m
                            
                            for jlev in reversed(range(ilev)):
                                if odd(nn):
                                    nnu = int((nn + 1)/2 - 1)
                                else:
                                    nnu = int((nn)/2)
                                if odd(mm):
                                    mmu = int((mm + 1)/2 - 1)
                                else:
                                    mmu = int((mm)/2)
                                isrefined[jlev][nnu, mmu] = 1
                                nn = nnu
                                mm = mmu                        

                            # Also set 3 other blocks also to 1, unless already refined
                            [nnbr,mnbr] = get_neighbors_in_larger_cell(n, m)
                            for j in range(4):
                                if nnbr[j]>-1 and nnbr[j]<=nmx[ilev] - 1 and mnbr[j]>-1 and mnbr[j]<=mmx[ilev] - 1:
                                    if not isrefined[ilev][nnbr[j], mnbr[j]]:
                                        refmsk[ilev][nnbr[j], mnbr[j]] = 1


        # Count total number of cells
        nb = 0
        for ilev in range(nlev):
           for m in range(mmx[ilev]):
               for n in range(nmx[ilev]):
                    if refmsk[ilev][n, m]:
                        nb = nb + 1
                
        self.level = np.empty(nb, dtype=int)
        self.n     = np.empty(nb, dtype=int)
        self.m     = np.empty(nb, dtype=int)
        self.z     = np.zeros(nb, dtype=float)
        self.nr_cells = nb

        nb = 0
        last_lev = -1
        for ilev in range(nlev):
           for m in range(mmx[ilev]):
               for n in range(nmx[ilev]):
                    if refmsk[ilev][n, m]:
                        self.level[nb] = ilev
                        self.n[nb] = n
                        self.m[nb] = m
                        if ilev>last_lev:
                            self.ifirst[ilev] = nb
                            last_lev = ilev
                        nb = nb + 1

        self.find_neighbors()
        self.compute_cell_centre_coordinates()

        print("Number of cells : " + str(nb))
        print("Time elapsed : " + str(time.time() - start) + " s")
        

    def find_neighbors(self):
        
#        print("Finding neighbors ...")
#        start = time.time()
                
        # Initialize neighbor arrays
        # Set indices of neighbors to -1
        self.mu  = np.zeros(self.nr_cells, dtype=int)
        self.mu1 = np.zeros(self.nr_cells, dtype=int) - 1
        self.mu2 = np.zeros(self.nr_cells, dtype=int) - 1
        self.md  = np.zeros(self.nr_cells, dtype=int)
        self.md1 = np.zeros(self.nr_cells, dtype=int) - 1
        self.md2 = np.zeros(self.nr_cells, dtype=int) - 1
        self.nu  = np.zeros(self.nr_cells, dtype=int)
        self.nu1 = np.zeros(self.nr_cells, dtype=int) - 1
        self.nu2 = np.zeros(self.nr_cells, dtype=int) - 1
        self.nd  = np.zeros(self.nr_cells, dtype=int)
        self.nd1 = np.zeros(self.nr_cells, dtype=int) - 1
        self.nd2 = np.zeros(self.nr_cells, dtype=int) - 1
        
        # nuv = 0
        nmx = np.zeros(self.nr_refinement_levels, dtype=int)
        # Determine maximum n index for each level
        for ilev in range(self.nr_refinement_levels):
            ifirst = self.ifirst[ilev]
            # Now find index of last point in this level
            if ilev<self.nr_refinement_levels - 1:
                ilast = self.ifirst[ilev + 1] - 1
            else:
                ilast = self.nr_cells - 1
            ns  = self.n[ifirst:ilast + 1] # All the n indices in this level
            nmx[ilev] = ns.max()

        for ilev in range(self.nr_refinement_levels):

            # Find neighbors in same level

            # Index of first point in this level
            ifirst = self.ifirst[ilev]
            # Now find index of last point in this level
            if ilev<self.nr_refinement_levels - 1:
                ilast = self.ifirst[ilev + 1] - 1
            else:
                ilast = self.nr_cells - 1
            
            nr = ilast - ifirst + 1         # number of cells in this level
            ns  = self.n[ifirst:ilast + 1] # All the n indices in this level
            ms  = self.m[ifirst:ilast + 1] # All the m indices in this level
#            nmx = self.nmax*2**ilev        # nmax for this level 
#            nmx = ns.max() + 1
            nms = ms*(nmx[ilev] + 1) + ns               # nm indices for this level
            
            for ic in range(nr):
                
                ib = ifirst + ic
                
                # if ilev==0:
                #     if self.n[ib]==nmx-1:
                #         shite=1
                
                # Right
                nm  = (self.m[ib] + 1)*(nmx[ilev] + 1) + self.n[ib]
                j = binary_search(nms, nm)
                if j is not None:
                    indxn = j + ifirst # index of neighbor
                    self.mu[ib]     = 0
                    self.mu1[ib]    = indxn
                    self.md[indxn]  = 0
                    self.md1[indxn] = ib
                    
                # Above (make sure we don't look neighbor in column to the right)
                if self.n[ib] < nmx[ilev]:
                    nm  = self.m[ib]*(nmx[ilev] + 1) + self.n[ib] + 1
                    j = binary_search(nms, nm)
                    if j is not None:
                        indxn = j + ifirst # index of neighbor
                        self.nu[ib]     = 0
                        self.nu1[ib]    = indxn
                        self.nd[indxn]  = 0
                        self.nd1[indxn] = ib

            # Find neighbors in coarser level
            
            if ilev>0:
        
                # Index of first point in the coarser level
                ifirstc = self.ifirst[ilev - 1]
                # Now find index of last point in the coarser level
                ilastc = self.ifirst[ilev] - 1
                
                nsc  = self.n[ifirstc:ilastc + 1] # All the n indices in coarser level
                msc  = self.m[ifirstc:ilastc + 1] # All the m indices in coarser level
#                nmxc = self.nmax*2**(ilev - 1)    # nmax for coarserlevel 
                nmxc = nmx[ilev - 1]
                nmsc = msc*(nmxc + 1) + nsc             # nm indices for coarser level
                
                for ic in range(nr):
                    
                    ib = ifirst + ic
                    
                    # Only need to check if we haven't already found a neighbor at the same level

                    if self.mu1[ib]<0:                    
                        # Right
                        if odd(self.m[ib]):
                            if even(self.n[ib]):
                                # Finer cell is the lower one
                                nc  = int(self.n[ib]/2)
                                mc  = int((self.m[ib] + 1) / 2)
                                nmc = mc*(nmxc + 1) + nc
                                j = binary_search(nmsc, nmc)
                                if j is not None:
                                    indxn = j + ifirstc # index of neighbor
                                    self.mu[ib]     = -1
                                    self.mu1[ib]    = indxn
                                    self.md[indxn]  = 1
                                    self.md1[indxn] = ib
                            else:    
                                # Finer cell is the upper one
                                nc  = int((self.n[ib] - 1) / 2)
                                mc  = int((self.m[ib] + 1) / 2)
                                nmc = mc*(nmxc + 1) + nc
                                j = binary_search(nmsc, nmc)
                                if j is not None:
                                    indxn = j + ifirstc # index of neighbor
                                    self.mu[ib]     = -1
                                    self.mu1[ib]    = indxn
                                    self.md[indxn]  = 1
                                    self.md2[indxn] = ib
                
                    if self.nu1[ib]<0:    
                        # Above
                        if odd(self.n[ib]):
                            if even(self.m[ib]):
                                # Finer cell is the left one
                                nc  = int((self.n[ib] + 1) / 2)
                                if nc<nmxc:
                                    mc  = int(self.m[ib]/2)
                                    nmc = mc*(nmxc + 1) + nc
                                    j = binary_search(nmsc, nmc)
                                    if j is not None:
                                        indxn = j + ifirstc # index of neighbor
                                        self.nu[ib]     = -1
                                        self.nu1[ib]    = indxn
                                        self.nd[indxn]  = 1
                                        self.nd1[indxn] = ib
                            else:    
                                # Finer cell is the right one
                                nc  = int((self.n[ib] + 1) / 2)
                                if nc<nmxc:
                                    mc  = int((self.m[ib] - 1) / 2)
                                    nmc = mc*(nmxc + 1) + nc
                                    j = binary_search(nmsc, nmc)
                                    if j is not None:
                                        indxn = j + ifirstc # index of neighbor
                                        self.nu[ib]     = -1
                                        self.nu1[ib]    = indxn
                                        self.nd[indxn]  = 1
                                        self.nd2[indxn] = ib

            # Find neighbors in finer level
            if ilev<self.nr_refinement_levels - 1:
        
                # Index of first point in the finer level
                ifirstf = self.ifirst[ilev + 1]
                # Now find index of last point in the finer level
                if ilev<self.nr_refinement_levels - 2:                
                    ilastf = self.ifirst[ilev + 2] - 1
                else:
                    ilastf = self.nr_cells - 1
                
                nsf  = self.n[ifirstf:ilastf + 1] # All the n indices in finer level
                msf  = self.m[ifirstf:ilastf + 1] # All the m indices in finer level
#                nmxf = nsf.max() + 1
                nmxf = nmx[ilev + 1]
#                nmxf = self.nmax*2**(ilev + 1)    # nmax for finer level 
                nmsf = msf*(nmxf + 1) + nsf        # nm indices for finer level
                
                for ic in range(nr):
                    
                    ib = ifirst + ic

                    # Only need to check if we haven't already found a neighbor at the same level
                        
                    if self.mu1[ib]<0:                    
                        # Right
                        # Finer cell is the lower one
                        nf  = int(self.n[ib]*2)
                        mf  = int((self.m[ib] + 1)*2)
                        nmf = mf*(nmxf + 1) + nf
                        j = binary_search(nmsf, nmf)
                        if j is not None:
                            indxn = j + ifirstf # index of neighbor
                            self.mu[ib]     = 1
                            self.mu1[ib]    = indxn
                            self.md[indxn]  = -1
                            self.md1[indxn] = ib
                        # Finer cell is the upper one
                        nf  = int(self.n[ib]*2) + 1
                        mf  = int((self.m[ib] + 1)*2)
                        nmf = mf*(nmxf + 1) + nf
                        j = binary_search(nmsf, nmf)
                        if j is not None:
                            indxn = j + ifirstf # index of neighbor
                            self.mu[ib]     = 1
                            self.mu2[ib]    = indxn
                            self.md[indxn]  = -1
                            self.md1[indxn] = ib

                    if self.nu1[ib]<0:                
                        # Above
                        # Finer cell is the left one
                        nf  = int((self.n[ib] + 1)*2)
                        if nf<nmxf:
                            mf  = int(self.m[ib]*2)
                            nmf = mf*(nmxf + 1) + nf
                            j = binary_search(nmsf, nmf)
                            if j is not None:
                                indxn = j + ifirstf # index of neighbor
                                self.nu[ib]     = 1
                                self.nu1[ib]    = indxn
                                self.nd[indxn]  = -1
                                self.nd1[indxn] = ib
                        # Finer cell is the right one
                        nf  = int((self.n[ib] + 1)*2)
                        if nf<nmxf:
                            mf  = int(self.m[ib]*2) + 1
                            nmf = mf*(nmxf + 1) + nf
                            j = binary_search(nmsf, nmf)
                            if j is not None:
                                indxn = j + ifirstf # index of neighbor
                                self.nu[ib]     = 1
                                self.nu2[ib]    = indxn               
                                self.nd[indxn]  = -1
                                self.nd1[indxn] = ib

#        print("Time elapsed : " + str(time.time() - start) + " s")

    def find_nodes_and_links(self):
        
        # First add all four corners of each cell to nodes nm array
        # The nm array contains the nm indices at the finest level

        print("Finding nodes and links ...")
        start = time.time()
         
        # Inialize node_nm array        
        node_nm     = np.empty(self.nr_cells*4, dtype=int)
        node_n      = np.empty(self.nr_cells*4, dtype=int)
        node_m      = np.empty(self.nr_cells*4, dtype=int)

        # Number of nodes at finest levels in n direction
        # Add 1 as there is one more node than nr cells in n direction
        nmx  = self.nmax*2**(self.nr_refinement_levels - 1) + 1

        # Loop through cells and add all four corner points to nodes arrays
        for ic in range(self.nr_cells):
            ifac = 2**(self.nr_refinement_levels - self.level[ic] - 1)
            # Lower-left
            node_n[ic*4 + 0] = self.n[ic]*ifac
            node_m[ic*4 + 0] = self.m[ic]*ifac
            node_nm[ic*4 + 0] = node_m[ic*4 + 0]*nmx + node_n[ic*4 + 0]
            # Upper-left
            node_n[ic*4 + 1]  = (self.n[ic] + 1)*ifac
            node_m[ic*4 + 1]  = self.m[ic]*ifac
            node_nm[ic*4 + 1] = node_m[ic*4 + 1]*nmx + node_n[ic*4 + 1]
            # Lower-right
            node_n[ic*4 + 2]  = self.n[ic]*ifac
            node_m[ic*4 + 2]  = (self.m[ic] + 1)*ifac
            node_nm[ic*4 + 2] = node_m[ic*4 + 2]*nmx + node_n[ic*4 + 2]
            # Upper-right
            node_n[ic*4 + 3]  = (self.n[ic] + 1)*ifac
            node_m[ic*4 + 3]  = (self.m[ic] + 1)*ifac
            node_nm[ic*4 + 3] = node_m[ic*4 + 3]*nmx + node_n[ic*4 + 3]
                    
        # Now get rid of duplicates and sort by nm index
        node_nm, ind = np.unique(node_nm, return_index=True)
        node_n       = node_n[ind] 
        node_m       = node_m[ind] 

        self.nr_nodes = np.size(node_nm)

        # Compute x,y coordinates for each node        
        cosrot = np.cos(self.rotation*np.pi/180)
        sinrot = np.sin(self.rotation*np.pi/180)        
        # dx/dy of finest level            
        dx = self.dx*0.5**(self.nr_refinement_levels - 1)
        dy = self.dy*0.5**(self.nr_refinement_levels - 1)
        self.node_x = self.x0 + cosrot*(node_m * dx) - sinrot*(node_n * dy)
        self.node_y = self.y0 + sinrot*(node_m * dx) + cosrot*(node_n * dy)

        # Links

        links = np.empty((2, self.nr_cells*8), dtype=int)

        # Loop through cells to get the links
        ilnk = 0
        for ic in range(self.nr_cells):

            ifac = 2**(self.nr_refinement_levels - self.level[ic] - 1)
            
            # Left
            if self.md[ic] <= 0:
                # Same level or coarser or no neighbor to the left
                n0    = self.n[ic]*ifac
                m0    = self.m[ic]*ifac
                nm0   = m0*nmx + n0
                node0 = binary_search(node_nm, nm0)
                n1    = (self.n[ic] + 1)*ifac
                m1    = m0
                nm1    = m1*nmx + n1
                node1 = binary_search(node_nm, nm1)
                links[0, ilnk] = node0
                links[1, ilnk] = node1
                ilnk += 1
            else:
                # Finer
                if self.md1[ic] >= 0:
                    n0    = self.n[ic]*ifac
                    m0    = self.m[ic]*ifac
                    nm0   = m0*nmx + n0
                    node0 = binary_search(node_nm, nm0)
                    n1    = n0 + int(ifac/2)
                    m1    = m0
                    nm1   = m1*nmx + n1
                    node1 = binary_search(node_nm, nm1)
                    links[0, ilnk] = node0
                    links[1, ilnk] = node1
                    ilnk += 1
                if self.md2[ic] >= 0:
                    n0    = self.n[ic]*ifac + int(ifac/2)
                    m0    = self.m[ic]*ifac
                    nm0   = m0*nmx + n0
                    node0 = binary_search(node_nm, nm0)
                    n1    = (self.n[ic] + 1)*ifac
                    m1    = m0
                    nm1   = m1*nmx + n1
                    node1 = binary_search(node_nm, nm1)
                    links[0, ilnk] = node0
                    links[1, ilnk] = node1
                    ilnk += 1

            # Below
            if self.nd[ic] <= 0:
                # Same level or coarser or no neighbor below
                n0    = self.n[ic]*ifac
                m0    = self.m[ic]*ifac
                nm0   = m0*nmx + n0
                node0 = binary_search(node_nm, nm0)
                n1    = n0
                m1    = (self.m[ic] + 1)*ifac
                nm1    = m1*nmx + n1
                node1 = binary_search(node_nm, nm1)
                links[0, ilnk] = node0
                links[1, ilnk] = node1
                ilnk += 1
            else:
                # Finer
                if self.nd1[ic] >= 0:
                    n0    = self.n[ic]*ifac
                    m0    = self.m[ic]*ifac
                    nm0   = m0*nmx + n0
                    node0 = binary_search(node_nm, nm0)
                    n1    = n0
                    m1    = m0 + int(ifac/2)
                    nm1   = m1*nmx + n1
                    node1 = binary_search(node_nm, nm1)
                    links[0, ilnk] = node0
                    links[1, ilnk] = node1
                    ilnk += 1
                if self.nd2[ic] >= 0:
                    n0    = self.n[ic]*ifac
                    m0    = self.m[ic]*ifac + int(ifac/2)
                    nm0   = m0*nmx + n0
                    node0 = binary_search(node_nm, nm0)
                    n1    = n0
                    m1    = (self.m[ic] + 1)*ifac
                    nm1   = m1*nmx + n1
                    node1 = binary_search(node_nm, nm1)
                    links[0, ilnk] = node0
                    links[1, ilnk] = node1
                    ilnk += 1

            # Now do links for right and top side. This is only necessary when cells
            # do not have a neighbor to the right or above.
            
            # Right
            if self.mu[ic] <= 0:
                # Same level or coarser or no neighbor to the right
                if self.mu1[ic] < 0:
                    # No neighbor
                    n0    = self.n[ic]*ifac
                    m0    = (self.m[ic] + 1)*ifac
                    nm0   = m0*nmx + n0
                    node0 = binary_search(node_nm, nm0)
                    n1    = (self.n[ic] + 1)*ifac
                    m1    = m0
                    nm1   = m1*nmx + n1
                    node1 = binary_search(node_nm, nm1)
                    links[0, ilnk] = node0
                    links[1, ilnk] = node1
                    ilnk += 1
            else:
                if self.mu1[ic] < 0:
                    # No neighbor in first point
                    n0    = self.n[ic]*ifac
                    m0    = (self.m[ic] + 1)*ifac
                    nm0   = m0*nmx + n0
                    node0 = binary_search(node_nm, nm0)
                    n1    = self.n[ic]*ifac + int(ifac/2)
                    m1    = m0
                    nm1   = m1*nmx + n1
                    node1 = binary_search(node_nm, nm1)
                    links[0, ilnk] = node0
                    links[1, ilnk] = node1
                    ilnk += 1
                if self.mu2[ic] < 0:
                    # No neighbor in second point
                    n0    = self.n[ic]*ifac + int(ifac/2)
                    m0    = (self.m[ic] + 1)*ifac
                    nm0   = m0*nmx + n0
                    node0 = binary_search(node_nm, nm0)
                    n1    = (self.n[ic] + 1)*ifac
                    m1    = m0
                    nm1   = m1*nmx + n1
                    node1 = binary_search(node_nm, nm1)
                    links[0, ilnk] = node0
                    links[1, ilnk] = node1
                    ilnk += 1
                

            # Above
            if self.nu[ic] <= 0:
                # Same level or coarser or no neighbor to the right
                if self.nu1[ic] < 0:
                    # No neighbor
                    n0    = (self.n[ic] + 1)*ifac
                    m0    = self.m[ic]*ifac
                    nm0   = m0*nmx + n0
                    node0 = binary_search(node_nm, nm0)
                    n1    = n0
                    m1    = (self.m[ic] + 1)*ifac
                    nm1   = m1*nmx + n1
                    node1 = binary_search(node_nm, nm1)
                    links[0, ilnk] = node0
                    links[1, ilnk] = node1
                    ilnk += 1
            else:
                if self.nu1[ic] < 0:
                    # No neighbor in first point
                    n0    = (self.n[ic] + 1)*ifac
                    m0    = self.m[ic]*ifac
                    nm0   = m0*nmx + n0
                    node0 = binary_search(node_nm, nm0)
                    n1    = n0
                    m1    = self.m[ic]*ifac + int(ifac/2)
                    nm1   = m1*nmx + n1
                    node1 = binary_search(node_nm, nm1)
                    links[0, ilnk] = node0
                    links[1, ilnk] = node1
                    ilnk += 1
                if self.mu2[ic] < 0:
                    # No neighbor in second point
                    n0    = (self.n[ic] + 1)*ifac
                    m0    = self.m[ic]*ifac + int(ifac/2)
                    nm0   = m0*nmx + n0
                    node0 = binary_search(node_nm, nm0)
                    n1    = n0
                    m1    = (self.m[ic] + 1)*ifac
                    nm1   = m1*nmx + n1
                    node1 = binary_search(node_nm, nm1)
                    links[0, ilnk] = node0
                    links[1, ilnk] = node1
                    ilnk += 1
                    
        self.nr_links = ilnk
        self.links = links[:, 0:self.nr_links]
                
        print("Time elapsed : " + str(time.time() - start) + " s")

    def get_bathymetry(self, bathymetry_sets, quiet=True):
        
        from cht_bathymetry.bathymetry_database import bathymetry_database

        if not quiet:
            print("Getting bathymetry data ...")

        # Prepare transformers
        bathymetry_transformers = []  
        for bathymetry in bathymetry_sets:
            bathymetry_transformers.append(Transformer.from_crs(self.crs,
                                                                bathymetry.crs,
                                                                always_xy=True))

        nlev = self.nr_refinement_levels
        
        # Check if coordinates already exists
        if self.x is None:
            self.compute_cell_centre_coordinates()
        zg = np.full(self.nr_cells, np.nan)

                        
        # Loop through all levels
        for ilev in range(nlev):

            if not quiet:
                print("Processing bathymetry level " + str(ilev + 1) + " of " + str(nlev) + " ...")
            
            ifirst = self.ifirst[ilev]
            if ilev<nlev - 1:
                ilast = self.ifirst[ilev + 1]
            else:
                ilast = self.nr_cells - 1
            
            # Make blocks off cells in this level only
            cell_indices_in_level = np.arange(ifirst, ilast + 1, dtype=int)
            nr_cells_in_level = np.size(cell_indices_in_level)
                  
            xg  = self.x[cell_indices_in_level]
            yg  = self.y[cell_indices_in_level]
            zgl = np.full(nr_cells_in_level, np.nan)

            dx   = self.dx/2**ilev      # max cell size
            # Make sure dx is in metres
            if self.crs.is_geographic:            
                dx = 111111.0*dx*np.cos(np.max(np.abs(yg)))
            
            # Loop through bathymetry datasets
            for ibathy, bathymetry in enumerate(bathymetry_sets):                
                if bathymetry.type == "source":
                    if np.isnan(zg).any():
                        xgb, ygb = bathymetry_transformers[ibathy].transform(xg, yg)        
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
                                                                      max_cell_size=dx)
                            zb[np.where(zb<bathymetry.zmin)] = np.nan
                            zb[np.where(zb>bathymetry.zmax)] = np.nan
                            if not np.isnan(zb).all():
                                zg1 = interp2(xb, yb, zb, xgb, ygb)
                                isn = np.where(np.isnan(zgl))
                                zgl[isn] = zg1[isn]
                    
                elif bathymetry.type == "array":
                    # Matrix provided, interpolate to subgrid mesh
#                            zg = interp2(bathymetry.x, bathymetry.y, bathymetry.z, xgb, ygb)
                    pass
            zg[cell_indices_in_level] = zgl

        self.z = zg
            
    def compute_cell_centre_coordinates(self):
        cosrot = np.cos(self.rotation*np.pi/180)
        sinrot = np.sin(self.rotation*np.pi/180)                
        dx = self.dx/2**(self.level)
        dy = self.dy/2**(self.level)
        self.x = self.x0 + cosrot*((self.m + 0.5)*dx) - sinrot*((self.n + 0.5)*dy)
        self.y = self.y0 + sinrot*((self.m + 0.5)*dx) + cosrot*((self.n + 0.5)*dy)

    def compute_uv_coordinates(self):
        cosrot = np.cos(self.rotation*np.pi/180)
        sinrot = np.sin(self.rotation*np.pi/180)
        self.xuv = np.zeros(self.nr_cells*4)
        self.yuv = np.zeros(self.nr_cells*4)
        iuv = 0
        for ip in range(self.nr_cells):
            dx = self.dx/2**(self.level[ip])
            dy = self.dy/2**(self.level[ip])
            if self.mu[ip]<=0:
                if self.mu1[ip]>=0:
                    self.xuv[iuv] = self.x0 + cosrot*((self.m[ip] + 1.0)*dx) - sinrot*((self.n[ip] + 0.5)*dy)
                    self.yuv[iuv] = self.x0 + sinrot*((self.m[ip] + 1.0)*dx) + cosrot*((self.n[ip] + 0.5)*dy)
                    iuv += 1
            else:        
                if self.mu1[ip]>=0:
                    self.xuv[iuv] = self.x0 + cosrot*((self.m[ip] + 1.0)*dx) - sinrot*((self.n[ip] + 0.25)*dy)
                    self.yuv[iuv] = self.x0 + sinrot*((self.m[ip] + 1.0)*dx) + cosrot*((self.n[ip] + 0.25)*dy)
                    iuv += 1
                if self.mu2[ip]>=0:
                    self.xuv[iuv] = self.x0 + cosrot*((self.m[ip] + 1.0)*dx) - sinrot*((self.n[ip] + 0.75)*dy)
                    self.yuv[iuv] = self.x0 + sinrot*((self.m[ip] + 1.0)*dx) + cosrot*((self.n[ip] + 0.75)*dy)
                    iuv += 1
            if self.nu[ip]<=0:
                if self.nu1[ip]>=0:
                    self.xuv[iuv] = self.x0 + cosrot*((self.m[ip] + 0.5)*dx) - sinrot*((self.n[ip] + 1.0)*dy)
                    self.yuv[iuv] = self.x0 + sinrot*((self.m[ip] + 0.5)*dx) + cosrot*((self.n[ip] + 1.0)*dy)
                    iuv += 1
            else:        
                if self.nu1[ip]>=0:
                    self.xuv[iuv] = self.x0 + cosrot*((self.m[ip] + 0.25)*dx) - sinrot*((self.n[ip] + 1.0)*dy)
                    self.yuv[iuv] = self.x0 + sinrot*((self.m[ip] + 0.25)*dx) + cosrot*((self.n[ip] + 1.0)*dy)
                    iuv += 1
                if self.nu2[ip]>=0:
                    self.xuv[iuv] = self.x0 + cosrot*((self.m[ip] + 0.75)*dx) - sinrot*((self.n[ip] + 1.0)*dy)
                    self.yuv[iuv] = self.x0 + sinrot*((self.m[ip] + 0.75)*dx) + cosrot*((self.n[ip] + 1.0)*dy)
                    iuv += 1
        self.xuv = self.xuv[0:iuv]
        self.yuv = self.yuv[0:iuv]
        
    def to_snapwave(self, mask, file_name, index_file_name="snapwave.ind"):
        from snapwave_mesh import SnapWaveMesh
        network = SnapWaveMesh()
        network.from_quadtree(self, mask,
                              file_name=file_name,
                              index_file_name=index_file_name)
        return network

    def to_topojson(self):
        pass
    
    def from_regular_grid(regular_grid, mask):
        pass

    def plot(self, ax, color="k"):
        xlink = self.node_x[self.links]
        ylink = self.node_y[self.links]
        ax.plot(xlink, ylink, color)        

    def cut_inactive_cells(self, flow_mask, wave_mask=None):
        
        mask = flow_mask.mask*1
        if wave_mask is not None:
            mask = mask + wave_mask.mask
        indx = np.where(mask>0)
        flow_mask.mask = flow_mask.mask[indx]
        if wave_mask is not None:
            wave_mask.mask = wave_mask.mask[indx]        
            
        self.nr_cells = np.size(indx)
        self.n        = self.n[indx]
        self.m        = self.m[indx]
        self.level    = self.level[indx]
        
        if self.x is not None:
            self.x    = self.x[indx]
            self.y    = self.y[indx]        
        if self.z is not None:
            self.z    = self.z[indx]
    
        self.ifirst = np.zeros(self.nr_refinement_levels, dtype=int)
        last_lev = -1
        for ic in range(self.nr_cells):
            ilev = self.level[ic]
            if ilev>last_lev:
                # Found new level
                self.ifirst[ilev] = ic
                last_lev = ilev
    
        self.find_neighbors()


    def make_index_tiles(self, path, zoom_range=None, format=0):
        
        import math
        from cht_tiling.tiling import deg2num
        from cht_tiling.tiling import num2deg
        import cht_utils.fileops as fo
        
        npix = 256
        
        if not zoom_range:
            zoom_range = [0, 13]

        cosrot = math.cos(-self.rotation*math.pi/180)
        sinrot = math.sin(-self.rotation*math.pi/180)       

        # Compute lon/lat range
        xmin = np.amin(self.x) - 10*self.dx
        xmax = np.amax(self.x) + 10*self.dx
        ymin = np.amin(self.y) - 10*self.dy
        ymax = np.amax(self.y) + 10*self.dy
        transformer = Transformer.from_crs(self.crs,
                                            CRS.from_epsg(4326),
                                            always_xy=True)
        lon_min, lat_min = transformer.transform(xmin, ymin)
        lon_max, lat_max = transformer.transform(xmax, ymax)
        lon_range = [lon_min, lon_max]
        lat_range = [lat_min, lat_max]        
        
        transformer_a = Transformer.from_crs(CRS.from_epsg(4326),
                                                CRS.from_epsg(3857),
                                                always_xy=True)
        transformer_b = Transformer.from_crs(CRS.from_epsg(3857),
                                                self.crs,
                                                always_xy=True)
        
        i0_lev = []
        i1_lev = []
        nmax_lev = []
        mmax_lev = []
        nm_lev = []
        for level in range(self.nr_refinement_levels):
            i0 = self.level_index[level]
            if level<self.nr_refinement_levels - 1:
                i1 = self.level_index[level + 1]
            else:
                i1 = self.nr_cells   
            i0_lev.append(i0)    
            i1_lev.append(i1)    
            nmax_lev.append(np.amax(self.n[i0:i1]) + 1)
            mmax_lev.append(np.amax(self.m[i0:i1]) + 1)
            mm = self.m[i0:i1]
            nn = self.n[i0:i1]
            nm_lev.append(mm*nmax_lev[level] + nn)

        for izoom in range(zoom_range[0], zoom_range[1] + 1):
            
            print("Processing zoom level " + str(izoom))
        
            zoom_path = os.path.join(path, str(izoom))
        
            dxy = (40075016.686/npix) / 2 ** izoom
            xx = np.linspace(0.0, (npix - 1)*dxy, num=npix)
            yy = xx[:]
            xv, yv = np.meshgrid(xx, yy)
        
            ix0, iy0 = deg2num(lat_range[0], lon_range[0], izoom)
            ix1, iy1 = deg2num(lat_range[1], lon_range[1], izoom)
        
            for i in range(ix0, ix1 + 1):
            
                path_okay = False
                zoom_path_i = os.path.join(zoom_path, str(i))
            
                for j in range(iy0, iy1 + 1):
            
                    file_name = os.path.join(zoom_path_i, str(j) + ".dat")
            
                    # Compute lat/lon at ll corner of tile
                    lat, lon = num2deg(i, j, izoom)
            
                    # Convert to Global Mercator
                    xo, yo   = transformer_a.transform(lon,lat)
            
                    # Tile grid on local mercator
                    x = xv[:] + xo + 0.5*dxy
                    y = yv[:] + yo + 0.5*dxy
            
                    # Convert tile grid to crs of SFINCS model
                    x, y = transformer_b.transform(x, y)

                    # Now rotate around origin of SFINCS model
                    x00 = x - self.x0
                    y00 = y - self.y0
                    xg  = x00*cosrot - y00*sinrot
                    yg  = x00*sinrot + y00*cosrot

                    indx = np.full((npix, npix), -999, dtype=int)

                    for ilev in range(self.nr_refinement_levels):
                        nmax = nmax_lev[ilev]
                        mmax = mmax_lev[ilev]
                        i0   = i0_lev[ilev]
                        i1   = i1_lev[ilev]
                        dx   = self.dx/2**ilev
                        dy   = self.dy/2**ilev
                        iind = np.floor(xg/dx).astype(int)
                        jind = np.floor(yg/dy).astype(int)
                        # Now check whether this cell exists on this level
                        ind  = iind*nmax + jind
                        ind[iind<0]   = -999
                        ind[jind<0]   = -999
                        ind[iind>=mmax] = -999
                        ind[jind>=nmax] = -999

                        ingrid = np.isin(ind, nm_lev[ilev], assume_unique=False) # return boolean for each pixel that falls inside a grid cell
                        incell = np.where(ingrid)                                # tuple of arrays of pixel indices that fall in a cell

                        if incell[0].size>0:
                            # Now find the cell indices
                            try:
                                cell_indices = np.searchsorted(nm_lev[ilev], ind[incell[0], incell[1]]) + i0_lev[ilev]
                                indx[incell[0], incell[1]] = cell_indices
                            except:
                                pass

                    if np.any(indx>=0):                        
                        if not path_okay:
                            if not os.path.exists(zoom_path_i):
                                fo.mkdir(zoom_path_i)
                                path_okay = True
                                
                        # And write indices to file
                        fid = open(file_name, "wb")
                        fid.write(indx)
                        fid.close()


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
