# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 17:24:49 2022

@author: ormondt
"""
import time
import os
import numpy as np
from matplotlib import path
from pyproj import CRS, Transformer
import shapely

from shapely.geometry import Polygon
from shapely.prepared import prep

import xugrid as xu
import xarray as xr
import warnings
import geopandas as gpd
np.warnings = warnings

import pandas as pd

import datashader as ds
import datashader.transfer_functions as tf
from datashader.utils import export_image

from cht_utils.misc_tools import interp2
from .quadtree_builder import build_quadtree_xugrid, cut_inactive_cells

class SfincsGrid:
    def __init__(self, model):
        self.model = model
        self.data = None
        self.type = "regular"
        self.exterior = gpd.GeoDataFrame()
        self.datashader_dataframe = pd.DataFrame()

    def read(self, file_name=None):
        if file_name is None:
            if not self.model.input.variables.qtrfile: 
                self.model.input.variables.qtrfile = "sfincs.nc"
            file_name = os.path.join(self.model.path, self.model.input.variables.qtrfile)
        self.data = xu.load_dataset(file_name)

        self.type = "quadtree"

        self.get_exterior()

        crd_dict = self.data["crs"].attrs
        if "projected_crs_name" in crd_dict:
            self.model.crs = CRS(crd_dict["projected_crs_name"])
        elif "geographic_crs_name" in crd_dict:
            self.model.crs = CRS(crd_dict["geographic_crs_name"])
        else:
            print("Could not find CRS in quadtree netcdf file")

        self.data["crs"] = self.model.crs.to_epsg()
        self.data["crs"].attrs = self.model.crs.to_cf()

    def write(self, file_name=None, version=0):
        if file_name is None:
            if not self.model.input.variables.qtrfile: 
                self.model.input.variables.qtrfile = "sfincs.nc"
            file_name = os.path.join(self.model.path, self.model.input.variables.qtrfile)

        ds = self.data.ugrid.to_dataset()
        ds.attrs = self.data.attrs
        ds.to_netcdf(file_name)
        ds.close()

    def build(self,
              x0,
              y0,
              nmax,
              mmax,
              dx,
              dy,
              rotation,
              refinement_polygons=None,
              bathymetry_sets=None,
              bathymetry_database=None):

        print("Building mesh ...")

        # Always quadtree !
        self.type = "quadtree"

        # Clear mask datashader dataframe
        self.clear_datashader_dataframe()
        self.model.mask.clear_datashader_dataframe()

        self.data = build_quadtree_xugrid(
            x0, y0, nmax, mmax, dx, dy, rotation, self.model.crs,
            refinement_polygons=refinement_polygons,
            bathymetry_sets=bathymetry_sets,
            bathymetry_database=bathymetry_database
        )
        
        self.get_exterior()

    def cut_inactive_cells(self):
        # Clear datashader dataframes (new ones will be created when needed by map_overlay methods)
        self.clear_datashader_dataframe()
        self.model.mask.clear_datashader_dataframe()
        # Cut inactive cells
        self.data = cut_inactive_cells(self.data)
        self.get_exterior()

    def interpolate_bathymetry(self, x, y, z, method="linear"):
        """x, y, and z are numpy arrays with coordinates and bathymetry values"""
        xy = self.data.grid.face_coordinates
        # zz = np.full(self.nr_cells, np.nan)
        xz = xy[:, 0]
        yz = xy[:, 1]
        zz = interp2(x, y, z, xz, yz, method=method)
        ugrid2d = self.data.grid
        self.data["z"] = xu.UgridDataArray(xr.DataArray(data=zz, dims=[ugrid2d.face_dimension]), ugrid2d)

    def set_bathymetry(self, bathymetry_sets, bathymetry_database=None, zmin=-1.0e9, zmax=1.0e9, quiet=True):
        
        if bathymetry_database is None:
            print("Error! No bathymetry database provided!")
            return

        if not quiet:
            print("Getting bathymetry data ...")

        # Number of refinement levels
        nlev = self.data.attrs["nr_levels"]
        # Cell centre coordinates
        xy = self.data.grid.face_coordinates
        # Get number of cells
        nr_cells = len(xy)
        # Initialize bathymetry array
        zz = np.full(nr_cells, np.nan)
        # cell size of coarsest level
        dx = self.data.attrs["dx"]

        # Determine first indices and number of cells per refinement level
        # This is also done when the grid is built, but that information is not stored
        ifirst = np.zeros(nlev, dtype=int)
        ilast = np.zeros(nlev, dtype=int)
        level = self.data["level"].values[:] - 1 # 0-based
        for ilev in range(0, nlev):
            # Find index of first cell with this level
            ifirst[ilev] = np.where(level == ilev)[0][0]
            # Find index of last cell with this level
            if ilev < nlev - 1:
                ilast[ilev] = np.where(level == ilev + 1)[0][0] - 1
            else:
                ilast[ilev] = nr_cells - 1

        # Loop through all levels
        for ilev in range(nlev):

            if not quiet:
                print("Processing bathymetry level " + str(ilev + 1) + " of " + str(nlev) + " ...")

            # First and last cell indices in this level            
            i0 = ifirst[ilev]
            i1 = ilast[ilev]
            
            # Make blocks of cells in this level only
            cell_indices_in_level = np.arange(i0, i1 + 1, dtype=int)
                  
            xz  = xy[cell_indices_in_level, 0]
            yz  = xy[cell_indices_in_level, 1]
            dxmin = dx / 2**ilev

            # if self.data.grid.crs.is_geographic:
            if self.model.crs.is_geographic:
                dxmin = dxmin * 111000.0

            zgl = bathymetry_database.get_bathymetry_on_points(xz,
                                                               yz,
                                                               dxmin,
                                                               self.model.crs,
                                                               bathymetry_sets)
            
            # Limit zgl to zmin and zmax
            zgl = np.maximum(zgl, zmin)
            zgl = np.minimum(zgl, zmax)

            zz[cell_indices_in_level] = zgl

        ugrid2d = self.data.grid
        self.data["z"] = xu.UgridDataArray(xr.DataArray(data=zz, dims=[ugrid2d.face_dimension]), ugrid2d)

    def snap_to_grid(self, polyline, max_snap_distance=1.0):
        if len(polyline) == 0:
            return gpd.GeoDataFrame()
        geom_list = []
        for iline, line in polyline.iterrows():
            geom = line["geometry"]
            if geom.geom_type == 'LineString':
                geom_list.append(geom)
        gdf = gpd.GeoDataFrame({'geometry': geom_list})    
        print("Snapping to grid ...")
        snapped_uds, snapped_gdf = xu.snap_to_grid(gdf, self.data.grid, max_snap_distance=max_snap_distance)
        print("Snapping to grid done.")
        snapped_gdf = snapped_gdf.set_crs(self.model.crs)
        return snapped_gdf

    def face_coordinates(self):
        # if self.data is None:
        #     return None, None
        xy = self.data.grid.face_coordinates
        return xy[:, 0], xy[:,1]

    def get_exterior(self):
        try:
            indx = self.data.grid.edge_node_connectivity[self.data.grid.exterior_edges, :]
            x = self.data.grid.node_x[indx]
            y = self.data.grid.node_y[indx]
            # Make linestrings from numpy arrays x and y
            linestrings = [shapely.LineString(np.column_stack((x[i], y[i]))) for i in range(len(x))]
            # Merge linestrings
            merged = shapely.ops.linemerge(linestrings)
            # Merge polygons
            polygons = shapely.ops.polygonize(merged)
    #        polygons = shapely.simplify(polygons, self.dx)
            self.exterior = gpd.GeoDataFrame(geometry=list(polygons), crs=self.model.crs)
        except:
            self.exterior = gpd.GeoDataFrame()    

    def bounds(self, crs=None, buffer=0.0):
        """Returns list with bounds (lon1, lat1, lon2, lat2), with buffer (default 0.0) and in any CRS (default : same CRS as model)"""
        if crs is None:
            crs = self.crs
        # Convert exterior gdf to WGS 84
        lst = self.exterior.to_crs(crs=crs).total_bounds.tolist()
        dx = lst[2] - lst[0]
        dy = lst[3] - lst[1]
        lst[0] = lst[0] - buffer * dx
        lst[1] = lst[1] - buffer * dy
        lst[2] = lst[2] + buffer * dx
        lst[3] = lst[3] + buffer * dy
        return lst

    def map_overlay(self, file_name, xlim=None, ylim=None, color="black", width=800):

        if self.data is None:
            # No grid (yet)
            return False

        try:
            # Check if datashader dataframe is empty (maybe it was not made yet, or it was cleared)
            if self.datashader_dataframe.empty:
                self.get_datashader_dataframe()

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
            agg = cvs.line(self.datashader_dataframe,
                           x=['x1', 'x2'],
                           y=['y1', 'y2'],
                           axis=1)
            img = tf.shade(agg)
            path = os.path.dirname(file_name)
            if not path:
                path = os.getcwd()
            name = os.path.basename(file_name)
            name = os.path.splitext(name)[0]
            export_image(img, name, export_path=path)
            return True
        except Exception as e:
            return False

    def get_datashader_dataframe(self):
        """Creates a dataframe with line elements for datashader"""
        # Create a dataframe with line elements
        x1 = self.data.grid.edge_node_coordinates[:,0,0]
        x2 = self.data.grid.edge_node_coordinates[:,1,0]
        y1 = self.data.grid.edge_node_coordinates[:,0,1]
        y2 = self.data.grid.edge_node_coordinates[:,1,1]
        # Check if grid crosses the dateline
        cross_dateline = False
        if self.model.crs.is_geographic:
            if np.max(x1) > 180.0 or np.max(x2) > 180.0:
                cross_dateline = True
        transformer = Transformer.from_crs(self.model.crs,
                                            3857,
                                            always_xy=True)
        x1, y1 = transformer.transform(x1, y1)
        x2, y2 = transformer.transform(x2, y2)
        if cross_dateline:
            x1[x1 < 0] += 40075016.68557849
            x2[x2 < 0] += 40075016.68557849
        self.datashader_dataframe = pd.DataFrame(dict(x1=x1, y1=y1, x2=x2, y2=y2))

    def clear_datashader_dataframe(self):
        """Clears the datashader dataframe"""
        self.datashader_dataframe = pd.DataFrame() 

