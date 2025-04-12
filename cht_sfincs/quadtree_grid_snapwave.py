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

# from .to_xugrid import xug
import warnings

np.warnings = warnings

import geopandas as gpd
import pandas as pd

import datashader as ds
import datashader.transfer_functions as tf
from datashader.utils import export_image

from cht_utils.misc_tools import interp2


def snapwave_quadtree2mesh(qtr, file_name=None):
    # Steps:
    #
    # 4) Loop through all points and make cells for points where msk==1.
    #    The node indices in the cells will point to the indices of the entire 
    #    In a second temporary mask array mask2, determine which nodes are actually active (being part a cell)
    # 5) Count actual number of active nodes and cells, and allocate arrays
    # 6) Set node data and re-map indices
    #
    # STEP 1 - Read quadtree file
    #
    nr_cells = len(nr_cells = len(qtr.data.mesh2d_nFaces))

    mask2 = np.zeros(nr_cells, dtype=int) + 1

    ns     = qtr.data["n"].values[:]
    ms     = qtr.data["m"].values[:]
    nus    = qtr.data["nu"].values[:]
    nu1s   = qtr.data["nu1"].values[:]
    nu2s   = qtr.data["nu2"].values[:]
    mus    = qtr.data["mu"].values[:]
    mu1s   = qtr.data["mu1"].values[:]
    mu2s   = qtr.data["mu2"].values[:]
    mask   = qtr.data["snapwave_mask"].values[:]
    level  = qtr.data["level"].values[:]

    x0, y0 = qtr.face_coordinates()
    z0     = qtr.data["z"].values[:]

    # STEP 4 - Make faces
    faces = np.zeros((4, 4 * nr_cells), dtype=int) - 1
    nfaces = -1
    for ip in range(nr_cells):

        if mask[ip] == 0:
            continue

        n = ns[ip]
        m = ms[ip]
        nu = nus[ip]
        nu1 = nu1s[ip]
        nu2 = nu2s[ip]
        mu = mus[ip]
        mu1 = mu1s[ip]
        mu2 = mu2s[ip]

        # Using 0-based indexing here, so different than in Fortran 
        if odd(n):
            n_odd = False
        else:
            n_odd = True

        if odd(m):
            m_odd = False
        else:
            m_odd = True
        #
        # Turn off neighbors with msk==0
        #
        if mu1 > -1:
            if mask[mu1] == 0:
                mu1 = -1
        if mu2 > -1:
            if mask[mu2] == 0:
                mu2 = -1
        if nu1 > -1:
            if mask[nu1] == 0:
                nu1 = -1
        if nu2 > -1:
            if mask[nu2] == 0:
                nu2 = -1
        mnu = 0
        mnu1 = -1
        #
        # Find neighbor above-right
        #
        # Try going the right first
        #
        if mu == 0:
            # Same level right
            if mu1 > -1:
                if nus[mu1] == 0:
                    # Same level above right
                    if nu1s[mu1] > -1:
                        # It exists
                        if mask[nu1s[mu1]] > 0:
                            # And it's active
                            mnu = 0
                            mnu1 = nu1s[mu1]
                elif nus[mu1] == 1:
                    # Finer above-right
                    if nu1s[mu1] > -1:
                        # It exists
                        if mask[nu1s[mu1]] > 0:
                            # And it's active
                            mnu = 1
                            mnu1 = nu1s[mu1]
                elif nus[mu1] == -1:
                    # Coarser above-right
                    if nu1s[mu1] > -1:
                        # It exists
                        if mask[nu1s[mu1]] > 0:
                            # And it's active
                            mnu = -1
                            mnu1 = nu1s[mu1]
        elif mu == -1:
            # Coarser to the right
            if mu1 > -1:
                if nus[mu1] == 0:
                    # Same level above right
                    if nu1s[mu1] > -1:
                        # It exists
                        if mask[nu1s[mu1]] > 0:
                            # And it's active
                            mnu = -1
                            mnu1 = nu1s[mu1]
                elif nus[mu1] == 1:
                    # Finer above-right
                    if nu1s[mu1] > -1:
                        # It exists
                        if mask[nu1s[mu1]] > 0:
                            # And it's active
                            mnu = 0
                            mnu1 = nu1s[mu1]
                elif nus[mu1] == -1:
                    # Coarser above-right
                    if nu1s[mu1] > -1:
                        # It exists
                        if mask[nu1s[mu1]] > 0:
                            # And it's active
                            mnu = -2
                            mnu1 = nu1s[mu1]
        else:
            # Finer to the right
            if mu2 > -1:
                if nus[mu2] == 0:
                    # Same level above-right
                    if nu1s[mu2] > -1:
                        # It exists
                        mnu = 1
                        mnu1 = nu1s[mu2]
                elif nus[mu2] == 0:
                    # Finer above-right
                    if nu1s[mu2] > -1:
                        # It exists
                        mnu = 2  # twice as fine
                        mnu1 = nu1s[mu2]
                else:
                    # Finer above right
                    if nu1s[mu2] > -1:
                        # It exists
                        mnu = 0  # Same level
                        mnu1 = nu1s[mu2]
        #
        # Okay, found all the neighbors
        #
        # Now let's see what sort of cells we need
        #
        if mu == 0 and nu == 0 and mnu == 0:
            #
            # Type 1 - Most normal cell possible
            #
            if mu1 > -1 and nu1 > -1 and mnu1 > -1:
                nfaces += 1
                faces[0, nfaces] = ip
                faces[1, nfaces] = mu1
                faces[2, nfaces] = mnu1
                faces[3, nfaces] = nu1
                mask2[ip] = 1
                mask2[mu1] = 1
                mask2[mnu1] = 1
                mask2[nu1] = 1
        elif mu == 1 and nu == 0 and mnu == 0:
            #
            # Type 2
            #
            if mu1 > -1 and mu2 > -1:
                nfaces += 1
                faces[0, nfaces] = ip
                faces[1, nfaces] = mu1
                faces[2, nfaces] = mu2
                mask2[ip] = 1
                mask2[mu1] = 1
                mask2[mu2] = 1
            if mu2 > -1 and mnu1 > -1 and nu1 > -1:
                nfaces += 1
                faces[0, nfaces] = mu2
                faces[1, nfaces] = mnu1
                faces[2, nfaces] = nu1
                mask2[mu2] = 1
                mask2[mnu1] = 1
                mask2[nu1] = 1
            if mu2 > -1 and nu1 > -1:
                nfaces += 1
                faces[0, nfaces] = ip
                faces[1, nfaces] = mu2
                faces[2, nfaces] = nu1
                mask2[ip] = 1
                mask2[mu2] = 1
                mask2[nu1] = 1
        elif mu == 0 and nu == 0 and mnu == 1:
            #
            # Type 3
            #
            if mu1 > -1 and mnu1 > -1:
                nfaces += 1
                faces[0, nfaces] = ip
                faces[1, nfaces] = mu1
                faces[2, nfaces] = mnu1
                mask2[ip] = 1
                mask2[mu1] = 1
                mask2[mnu1] = 1
            if nu1 > -1 and mnu1 > -1:
                nfaces += 1
                faces[0, nfaces] = ip
                faces[1, nfaces] = mnu1
                faces[2, nfaces] = nu1
                mask2[ip] = 1
                mask2[mnu1] = 1
                mask2[nu1] = 1
        elif mu == 0 and nu == 1 and mnu == 0:
            #
            # Type 4
            #
            if mu1 > -1 and nu2 > -1:
                nfaces += 1
                faces[0, nfaces] = ip
                faces[1, nfaces] = mu1
                faces[2, nfaces] = nu2
                mask2[ip] = 1
                mask2[mu1] = 1
                mask2[nu2] = 1
            if mu1 > -1 and mnu1 > -1 and nu2 > -1:
                nfaces += 1
                faces[0, nfaces] = mu1
                faces[1, nfaces] = mnu1
                faces[2, nfaces] = nu2
                mask2[mu1] = 1
                mask2[mnu1] = 1
                mask2[nu2] = 1
            if nu1 > -1 and nu2 > -1:
                nfaces += 1
                faces[0, nfaces] = ip
                faces[1, nfaces] = nu1
                faces[2, nfaces] = nu2
                mask2[ip] = 1
                mask2[nu1] = 1
                mask2[nu2] = 1
        elif mu == 1 and nu == 0 and mnu == 1:
            #
            # Type 5
            #
            if mu1 > -1 and mu2 > -1:
                nfaces += 1
                faces[0, nfaces] = ip
                faces[1, nfaces] = mu1
                faces[2, nfaces] = mu2
                mask2[ip] = 1
                mask2[mu1] = 1
                mask2[mu2] = 1
            if mu2 > -1 and mnu1 > -1 and nu1 > -1:
                nfaces += 1
                faces[0, nfaces] = ip
                faces[1, nfaces] = mu2
                faces[2, nfaces] = mnu1
                faces[3, nfaces] = nu1
                mask2[ip] = 1
                mask2[mu2] = 1
                mask2[mnu1] = 1
                mask2[nu1] = 1
        elif mu == 0 and nu == 1 and mnu == 1:
            #
            # Type 6
            #
            if mu1 > -1 and mnu1 > -1 and nu2 > -1:
                nfaces += 1
                faces[0, nfaces] = ip
                faces[1, nfaces] = mu1
                faces[2, nfaces] = mnu1
                faces[3, nfaces] = nu2
                mask2[ip] = 1
                mask2[mu1] = 1
                mask2[mnu1] = 1
                mask2[nu2] = 1
            if nu1 > -1 and nu2 > -1:
                nfaces += 1
                faces[0, nfaces] = ip
                faces[1, nfaces] = nu2
                faces[2, nfaces] = nu1
                mask2[ip] = 1
                mask2[nu2] = 1
                mask2[nu1] = 1
        elif mu == 1 and nu == 1 and (mnu == 1 or mnu == 0):
            #
            # Type 7 and 8
            #
            if mu1 > -1 and mu2 > -1:
                nfaces += 1
                faces[0, nfaces] = ip
                faces[1, nfaces] = mu1
                faces[2, nfaces] = mu2
                mask2[ip] = 1
                mask2[mu1] = 1
                mask2[mu2] = 1
            if mu2 > -1 and nu2 > -1:
                nfaces += 1
                faces[0, nfaces] = ip
                faces[1, nfaces] = mu2
                faces[2, nfaces] = nu2
                mask2[ip] = 1
                mask2[mu2] = 1
                mask2[nu2] = 1
            if mu2 > -1 and nu2 > -1 and mnu1 > -1:
                nfaces += 1
                faces[0, nfaces] = mu2
                faces[1, nfaces] = mnu1
                faces[2, nfaces] = nu2
                mask2[mu2] = 1
                mask2[mnu1] = 1
                mask2[nu2] = 1
            if nu1 > -1 and nu2 > -1:
                nfaces += 1
                faces[0, nfaces] = ip
                faces[1, nfaces] = nu2
                faces[2, nfaces] = nu1
                mask2[ip] = 1
                mask2[nu2] = 1
                mask2[nu1] = 1
        elif mu == -1 and nu == 0 and n_odd:
            #
            # Type 9
            #
            if mu1 > -1 and nu1 > -1:
                nfaces += 1
                faces[0, nfaces] = ip
                faces[1, nfaces] = mu1
                faces[2, nfaces] = nu1
                mask2[ip] = 1
                mask2[mu1] = 1
                mask2[nu1] = 1
            #
        # %         elif (mu==-1 and nu==0 and mnu==0 and odd(buq.n[ip]))
        # %             % Type 9
        # %             if mu1>0 and nu1>0
        # %                 nfaces=nfaces+1;
        # %                 faces[0, nfaces] = ip;
        # %                 faces[1, nfaces] = mu1;
        # %                 faces[2, nfaces] = nu1;
        # %             end
        elif mu == -1 and nu == -1 and mnu == -1:
            #
            # Type 10
            #
            if mu1 > -1 and mnu1 > -1:
                nfaces += 1
                faces[0, nfaces] = ip
                faces[1, nfaces] = mu1
                faces[2, nfaces] = mnu1
                mask2[ip] = 1
                mask2[mu1] = 1
                mask2[mnu1] = 1
            if mnu1 > -1 and nu1 > -1:
                nfaces += 1
                faces[0, nfaces] = ip
                faces[1, nfaces] = mnu1
                faces[2, nfaces] = nu1
                mask2[ip] = 1
                mask2[mnu1] = 1
                mask2[nu1] = 1
            #
        elif mu == -1 and nu == -1 and mnu == 0:
            #
            # Type 11
            #
            if mu1 > -1 and mnu1 > -1:
                nfaces += 1
                faces[0, nfaces] = ip
                faces[1, nfaces] = mu1
                faces[2, nfaces] = mnu1
                mask2[ip] = 1
                mask2[mu1] = 1
                mask2[mnu1] = 1
            if mnu1 > -1 and nu1 > -1:
                nfaces += 1
                faces[0, nfaces] = ip
                faces[1, nfaces] = mnu1
                faces[2, nfaces] = nu1
                mask2[ip] = 1
                mask2[mnu1] = 1
                mask2[nu1] = 1
        elif mu == 0 and nu == -1 and mnu == -1 and not m_odd:
            #
            # Type 12
            #
            if mu1 > -1 and mnu1 > -1 and nu1 > -1:
                nfaces += 1
                faces[0, nfaces] = ip
                faces[1, nfaces] = mu1
                faces[2, nfaces] = mnu1
                faces[3, nfaces] = nu1
                mask2[ip] = 1
                mask2[mu1] = 1
                mask2[mnu1] = 1
                mask2[nu1] = 1
        elif mu == -1 and nu == 0 and mnu == -1 and not n_odd:
            #
            # Type 13
            #
            if mu1 > -1 and mnu1 > -1 and nu1 > -1:
                nfaces += 1
                faces[0, nfaces] = ip
                faces[1, nfaces] = mu1
                faces[2, nfaces] = mnu1
                faces[3, nfaces] = nu1
                mask2[ip] = 1
                mask2[mu1] = 1
                mask2[mnu1] = 1
                mask2[nu1] = 1
        elif mu == 0 and nu == -1 and m_odd:
            #
            # Type 14
            #
            if mu1 > -1 and nu1 > -1:
                nfaces += 1
                faces[0, nfaces] = ip
                faces[1, nfaces] = mu1
                faces[2, nfaces] = nu1
                mask2[ip] = 1
                mask2[mu1] = 1
                mask2[nu1] = 1
        elif mu == 1 and nu == -1 and mnu == 0:
            #
            # Type 15
            #
            if mu1 > -1 and mu2 > -1:
                nfaces += 1
                faces[0, nfaces] = ip
                faces[1, nfaces] = mu1
                faces[2, nfaces] = mu2
                mask2[ip] = 1
                mask2[mu1] = 1
                mask2[mu2] = 1
            if mu2 > -1 and mnu1 > -1 and nu1 > -1:
                nfaces += 1
                faces[0, nfaces] = mu2
                faces[1, nfaces] = mnu1
                faces[2, nfaces] = nu1
                mask2[mu2] = 1
                mask2[mnu1] = 1
                mask2[nu1] = 1
            if mu2 > -1 and nu1 > -1:
                nfaces += 1
                faces[0, nfaces] = ip
                faces[1, nfaces] = mu2
                faces[2, nfaces] = nu1
                mask2[ip] = 1
                mask2[mu2] = 1
                mask2[nu1] = 1
        elif mu == -1 and nu == -1 and mnu == -2:
            #
            # Type 16
            #
            if mu1 > -1 and mnu1 > -1:
                nfaces += 1
                faces[0, nfaces] = ip
                faces[1, nfaces] = mu1
                faces[2, nfaces] = mnu1
                mask2[ip] = 1
                mask2[mu1] = 1
                mask2[mnu1] = 1
            if mnu1 > -1 and nu1 > -1:
                nfaces += 1
                faces[0, nfaces] = ip
                faces[1, nfaces] = mnu1
                faces[2, nfaces] = nu1
                mask2[ip] = 1
                mask2[mnu1] = 1
                mask2[nu1] = 1
        elif mu == 0 and nu == 0 and mnu == -1:
            #
            # Type 16
            #
            if mu1 > -1 and nu1 > -1:
                nfaces += 1
                faces[0, nfaces] = ip
                faces[1, nfaces] = mu1
                faces[2, nfaces] = nu1
                mask2[ip] = 1
                mask2[mu1] = 1
                mask2[nu1] = 1
            if mu1 > -1 and nu1 > -1 and mnu1 > -1:
                nfaces += 1
                faces[0, nfaces] = mu1
                faces[1, nfaces] = mnu1
                faces[2, nfaces] = nu1
                mask2[mu1] = 1
                mask2[mnu1] = 1
                mask2[nu1] = 1
        elif mu == 0 and nu == -1 and mnu == 0:
            #
            # Type 17
            #
            if mu1 > -1 and mnu1 > -1:
                nfaces += 1
                faces[0, nfaces] = ip
                faces[1, nfaces] = mu1
                faces[2, nfaces] = mnu1
                mask2[ip] = 1
                mask2[mu1] = 1
                mask2[mnu1] = 1
            if mnu1 > -1 and nu1 > -1:
                nfaces += 1
                faces[0, nfaces] = ip
                faces[1, nfaces] = mnu1
                faces[2, nfaces] = nu1
                mask2[ip] = 1
                mask2[mnu1] = 1
                mask2[nu1] = 1
        elif mu == 1 and nu == 1 and mnu == 2:
            #
            # Type 17
            #
            if mu1 > -1 and mu2 > -1:
                nfaces += 1
                faces[0, nfaces] = ip
                faces[1, nfaces] = mu1
                faces[2, nfaces] = mu2
                mask2[ip] = 1
                mask2[mu1] = 1
                mask2[mu2] = 1
            if mu2 > -1 and mnu1 > -1:
                nfaces += 1
                faces[0, nfaces] = ip
                faces[1, nfaces] = mu2
                faces[2, nfaces] = mnu1
                mask2[ip] = 1
                mask2[mu2] = 1
                mask2[mnu1] = 1
            if nu2 > -1 and mnu1 > -1:
                nfaces += 1
                faces[0, nfaces] = ip
                faces[1, nfaces] = mnu1
                faces[2, nfaces] = nu2
                mask2[ip] = 1
                mask2[mnu1] = 1
                mask2[nu2] = 1
            if nu1 > -1 and nu2 > -1:
                nfaces += 1
                faces[0, nfaces] = ip
                faces[1, nfaces] = nu1
                faces[2, nfaces] = nu2
                mask2[ip] = 1
                mask2[nu1] = 1
                mask2[nu2] = 1
        elif mu == 1 and nu == 1 and mnu == 2:
            #
            # Type 18
            #
            if mu1 > -1 and mu2 > -1:
                nfaces += 1
                faces[0, nfaces] = ip
                faces[1, nfaces] = mu1
                faces[2, nfaces] = mu2
                mask2[ip] = 1
                mask2[mu1] = 1
                mask2[mu2] = 1
            if mu2 > -1 and mnu1 > -1:
                nfaces += 1
                faces[0, nfaces] = ip
                faces[1, nfaces] = mu2
                faces[2, nfaces] = mnu1
                mask2[ip] = 1
                mask2[mu2] = 1
                mask2[mnu1] = 1
            if nu2 > -1 and mnu1 > -1:
                nfaces += 1
                faces[0, nfaces] = ip
                faces[1, nfaces] = mnu1
                faces[2, nfaces] = nu2
                mask2[ip] = 1
                mask2[mnu1] = 1
                mask2[nu2] = 1
            if nu1 > -1 and nu2 > -1:
                nfaces += 1
                faces[0, nfaces] = ip
                faces[1, nfaces] = nu1
                faces[2, nfaces] = nu2
                mask2[ip] = 1
                mask2[nu1] = 1
                mask2[nu2] = 1
        elif mu == -1 and nu == 1 and mnu == 0:
            #
            # Type 19
            #
            if mu1 > -1 and nu2 > -1:
                nfaces += 1
                faces[0, nfaces] = ip
                faces[1, nfaces] = mu1
                faces[2, nfaces] = nu2
                mask2[ip] = 1
                mask2[mu1] = 1
                mask2[nu2] = 1
            if nu2 > -1 and mnu1 > -1 and mu1 > -1:
                nfaces += 1
                faces[0, nfaces] = mu1
                faces[1, nfaces] = mnu1
                faces[2, nfaces] = nu2
                mask2[mu1] = 1
                mask2[mnu1] = 1
                mask2[nu2] = 1
            if nu1 > -1 and nu2 > -1:
                nfaces += 1
                faces[0, nfaces] = ip
                faces[1, nfaces] = nu2
                faces[2, nfaces] = nu1
                mask2[ip] = 1
                mask2[nu2] = 1
                mask2[nu1] = 1
        elif mu == -1 and nu == 0 and mnu == 0 and not n_odd:
            #
            # Type 20
            #
            if mu1 > -1 and mnu1 > -1:
                nfaces += 1
                faces[0, nfaces] = ip
                faces[1, nfaces] = mu1
                faces[2, nfaces] = mnu1
                mask2[ip] = 1
                mask2[mu1] = 1
                mask2[mnu1] = 1
            if mnu1 > -1 and nu1 > -1:
                nfaces += 1
                faces[0, nfaces] = ip
                faces[1, nfaces] = mnu1
                faces[2, nfaces] = nu1
                mask2[ip] = 1
                mask2[mnu1] = 1
                mask2[nu1] = 1

    # STEP 5 - count number of active points and allocate arrays
    nac = 0
    for ip in range(nr_cells):
        if mask2[ip] == 1:
            nac += 1
    no_nodes = nac
    no_faces = nfaces + 1
    # Allocate nodes
    x = np.empty(no_nodes, dtype=float)
    y = np.empty(no_nodes, dtype=float)
    z = np.empty(no_nodes, dtype=float)
    msk = np.empty(no_nodes, dtype=int)
    face_nodes = np.zeros((4, no_faces), dtype=int) - 1
    index_snapwave_in_quadtree = np.zeros(nr_cells, dtype=int) - 1

    # STEP 6 - re-map and set values
    nac = 0
    for ip in range(nr_cells):
        if mask2[ip] > 0:
            # Re-map
            index_snapwave_in_quadtree[ip] = nac
            # Set node values
            x[nac] = x0[ip]
            y[nac] = y0[ip]
            z[nac] = z0[ip]
            msk[nac] = mask[ip]
            nac += 1

    # Loop through cells to re-maps the face nodes
    for iface in range(no_faces):
        for j in range(4):
            if faces[j, iface] > -1:
                ip0 = faces[j, iface]                  # index in full quadtree
                ip1 = index_snapwave_in_quadtree[ip0]  # index in reduced quadtree
                face_nodes[j, iface] = ip1             # set index to that of reduced mesh

    nodes = np.transpose(np.vstack((x, y)))
    faces = np.transpose(face_nodes)
    fill_value = -1

    xugrid = xu.Ugrid2d(nodes[:, 0], nodes[:, 1], fill_value, faces)

    ds = xu.UgridDataset(grids=xugrid)     

    da = xr.DataArray(
        data=z,
        dims=[xugrid.node_dimension],
    )

    uda = xu.UgridDataArray(da, xugrid)

    ds["z"] = uda

    if file_name is not None:
        ds.ugrid.to_netcdf(file_name)
    
    return ds



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
    if indx < np.size(vals):
        if vals[indx] == val:
            return indx
    return None
