# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 17:24:49 2022

@author: ormondt
"""
import os
import numpy as np
from pyproj import CRS, Transformer
import shapely
import rasterio
from rasterio.transform import from_origin
from rasterio.enums import Resampling
from rasterio.windows import Window
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

        # Check if "snapwave.upw" is in this folder. If so, delete it.
        snapwave_file = os.path.join(self.model.path, "snapwave.upw")
        if os.path.exists(snapwave_file):
            os.remove(snapwave_file)

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

    def set_uniform_bathymetry(self, zb):
        self.data["z"][:] = zb

    def set_bathymetry(self,
                       bathymetry_sets,
                       bathymetry_database=None,
                       zmin=-1.0e9,
                       zmax=1.0e9,
                       chunk_size=2000,
                       quiet=True):
        
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
            dxlev = dx / 2**ilev # cell size in this level (m or degrees if geographic)
            # dxmin is cell size in meters 
            if self.model.crs.is_geographic:
                dxmin = dxlev * 111000.0
            else:
                dxmin = dxlev

            # Perhaps we need to do this in chunks if the cells cover a large area.
            # We first determine the bounding box of all cells in this level.
            # If if is expected that the total number of cells that will be loaded
            # from the bathymetry database in x or y direction exceeds chunk_size,
            # we do it in chunks. It would be better to do all of this 
            # in cht_bathymetry!

            # Boundaries of all cells in this level
            x_min = np.min(xz) - dxlev
            x_max = np.max(xz) + dxlev
            y_min = np.min(yz) - dxlev
            y_max = np.max(yz) + dxlev
            # Create chunk boundaries
            x_chunks = np.arange(x_min, x_max, chunk_size * dxlev)
            y_chunks = np.arange(y_min, y_max, chunk_size * dxlev)

            if np.size(x_chunks) > 1 or np.size(y_chunks) > 1:

                # Looks like we need to do it in chunks. 

                if not quiet:
                    print(f"Processing in {len(x_chunks)} x {len(y_chunks)} chunks ...")

                zgl = np.full(len(xz), np.nan)
                # Loop through x and y chunks
                for ix in range(len(x_chunks)):
                    for iy in range(len(y_chunks)):
                        if not quiet:
                            print(f"Processing chunk {ix+1}, {iy+1} of {len(x_chunks)}, {len(y_chunks)} ...")

                        # Find points xz and yz in this chunk

                        if ix < len(x_chunks) - 1:
                            x_min_chunk = x_chunks[ix]
                            x_max_chunk = x_chunks[ix + 1]
                        else:
                            x_min_chunk = x_chunks[ix]
                            x_max_chunk = x_max

                        if iy < len(y_chunks) - 1:
                            y_min_chunk = y_chunks[iy]
                            y_max_chunk = y_chunks[iy + 1]
                        else:
                            y_min_chunk = y_chunks[iy]
                            y_max_chunk = y_max

                        in_chunk = np.where((xz >= x_min_chunk) & (xz < x_max_chunk) &
                                            (yz >= y_min_chunk) & (yz < y_max_chunk))[0]

                        if len(in_chunk) > 0:

                            xzc = xz[in_chunk]
                            yzc = yz[in_chunk]
                            zgc = bathymetry_database.get_bathymetry_on_points(xzc,
                                                                               yzc,
                                                                               dxmin,
                                                                               self.model.crs,
                                                                               bathymetry_sets)
                            zgl[in_chunk] = zgc

            else:                
                # No need for chuncking. Do it in one go.
                zgl = bathymetry_database.get_bathymetry_on_points(xz,
                                                                   yz,
                                                                   dxmin,
                                                                   self.model.crs,
                                                                   bathymetry_sets)

            # Limit zgl to zmin and zmax
            zgl = np.maximum(zgl, zmin)
            zgl = np.minimum(zgl, zmax)
            zz[cell_indices_in_level] = zgl
            
            # Limit zgl to zmin and zmax
            zgl = np.maximum(zgl, zmin)
            zgl = np.minimum(zgl, zmax)

            zz[cell_indices_in_level] = zgl

        ugrid2d = self.data.grid
        self.data["z"] = xu.UgridDataArray(xr.DataArray(data=zz, dims=[ugrid2d.face_dimension]), ugrid2d)

    def snap_to_grid(self, polyline):
        if len(polyline) == 0:
            return gpd.GeoDataFrame()
        # If geographic coordinates, set max_snap_distance to 0.1 degrees
        if self.model.crs.is_geographic:
            max_snap_distance = 1.0e-6
        else:
            max_snap_distance = 0.1

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
            crs = self.model.crs
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

    def get_indices_at_points(self, x, y):

        # x and y are 2D arrays of coordinates (x, y) in the same projection as the model
        # if x is a float, convert to 2D array
        if np.ndim(x) == 0:
            x = np.array([[x]])
        if np.ndim(y) == 0:
            y = np.array([[y]])

        x0 = self.data.attrs["x0"]
        y0 = self.data.attrs["y0"]
        dx = self.data.attrs["dx"]
        dy = self.data.attrs["dy"]
        nmax = self.data.attrs["nmax"]
        mmax = self.data.attrs["mmax"]
        rotation = self.data.attrs["rotation"]
        nr_refinement_levels = self.data.attrs["nr_levels"]

        nr_cells = len(self.data["level"])

        cosrot = np.cos(-rotation * np.pi / 180)
        sinrot = np.sin(-rotation * np.pi / 180)

        # Now rotate around origin of SFINCS model
        x00 = x - x0
        y00 = y - y0
        xg = x00 * cosrot - y00 * sinrot
        yg = x00 * sinrot + y00 * cosrot

        # Find index of first cell in each level
        if not hasattr(self, "ifirst"):
            ifirst = np.zeros(nr_refinement_levels, dtype=int)
            for ilev in range(0, nr_refinement_levels):
                # Find index of first cell with this level
                ifirst[ilev] = np.where(self.data["level"].to_numpy()[:] == ilev + 1)[0][0]
            self.ifirst = ifirst

        ifirst = self.ifirst    

        i0_lev = []
        i1_lev = []
        nmax_lev = []
        mmax_lev = []
        nm_lev = []

        for level in range(nr_refinement_levels):

            i0 = ifirst[level]
            if level < nr_refinement_levels - 1:
                i1 = ifirst[level + 1]
            else:
                i1 = nr_cells
            i0_lev.append(i0)
            i1_lev.append(i1)
            nmax_lev.append(np.amax(self.data["n"].to_numpy()[i0:i1]) + 1)
            mmax_lev.append(np.amax(self.data["m"].to_numpy()[i0:i1]) + 1)
            nn = self.data["n"].to_numpy()[i0:i1] - 1
            mm = self.data["m"].to_numpy()[i0:i1] - 1
            nm_lev.append(mm * nmax_lev[level] + nn)

        # Initialize index array
        indx = np.full(np.shape(x), -999, dtype=int)

        for ilev in range(nr_refinement_levels):
            nmax = nmax_lev[ilev]
            mmax = mmax_lev[ilev]
            i0 = i0_lev[ilev]
            i1 = i1_lev[ilev]
            dxr = dx / 2**ilev
            dyr = dy / 2**ilev
            iind = np.floor(xg / dxr).astype(int)
            jind = np.floor(yg / dyr).astype(int)
            # Now check whether this cell exists on this level
            ind = iind * nmax + jind
            ind[iind < 0] = -999
            ind[jind < 0] = -999
            ind[iind >= mmax] = -999
            ind[jind >= nmax] = -999

            ingrid = np.isin(
                ind, nm_lev[ilev], assume_unique=False
            )  # return boolean for each pixel that falls inside a grid cell
            incell = np.where(
                ingrid
            )  # tuple of arrays of pixel indices that fall in a cell

            if incell[0].size > 0:
                # Now find the cell indices
                try:
                    cell_indices = (
                        binary_search(nm_lev[ilev], ind[incell[0], incell[1]])
                        + i0_lev[ilev]
                    )
                    indx[incell[0], incell[1]] = cell_indices
                except Exception:
                    print("Error in binary search")
                    pass

        return indx

    def make_topobathy_cog(self,
                           filename,
                           bathymetry_sets,
                           bathymetry_database=None,
                           dx=10.0):
        
        """Make a COG file with topobathy. Now only works for projected coordinates. This always make the topobathy COG in the same projection as the model."""

        # Get the bounds of the grid
        bounds = self.bounds()

        x0 = bounds[0]
        y0 = bounds[1]
        x1 = bounds[2]
        y1 = bounds[3]

        # Round up and down to nearest dx
        x0 = x0 - (x0 % dx)
        x1 = x1 + (dx - x1 % dx)
        y0 = y0 - (y0 % dx)
        y1 = y1 + (dx - y1 % dx)

        xx = np.arange(x0, x1, dx) + 0.5 * dx
        yy = np.arange(y1, y0, -dx) - 0.5 * dx
        zz = np.empty((len(yy), len(xx),), dtype=np.float32)

        xx, yy = np.meshgrid(xx, yy)
        zz = bathymetry_database.get_bathymetry_on_points(xx,
                                                          yy,
                                                          dx,
                                                          self.model.crs,
                                                          bathymetry_sets)

        # And now to cog (use -999 as the nodata value)
        with rasterio.open(
            filename,
            "w",
            driver="COG",
            height=zz.shape[0],
            width=zz.shape[1],
            count=1,
            dtype=zz.dtype,
            crs=self.model.crs,
            transform=from_origin(x0, y1, dx, dx),
            nodata=-999.0,
        ) as dst:
            dst.write(zz, 1)

    def make_index_cog(self, filename, filename_topobathy):
    # def make_index_cog(self, filename, dx=10.0):
        """Make a COG file with indices of the quadtree grid cells."""

        # Read coordinates from topobathy file
        with rasterio.open(filename_topobathy) as src:
            # Get the bounds of the grid
            bounds = src.bounds
            dx = src.res[0]
            # Get the CRS of the grid
            self.model.crs = src.crs
            # Get the nodata value
            nodata = src.nodata
            # Get the transform of the grid
            transform = src.transform
            # Get the width and height of the grid
            width = src.width
            height = src.height

        # Now create numpy arrays with the coordinates of geotiff
        # Get the coordinates of the grid
        x0 = bounds.left
        y0 = bounds.bottom
        x1 = bounds.right
        y1 = bounds.top

        # # Round up and down to nearest dx
        # x0 = x0 - (x0 % dx)
        # x1 = x1 + (dx - x1 % dx)
        # y0 = y0 - (y0 % dx)
        # y1 = y1 + (dx - y1 % dx)

        xx = np.arange(x0, x1, dx) + 0.5 * dx
        yy = np.arange(y1, y0, -dx) - 0.5 * dx

        nodata = 2147483647

        # # # Get the bounds of the grid
        # # bounds = self.bounds()

        # x0 = bounds[0]
        # y0 = bounds[1]
        # x1 = bounds[2]
        # y1 = bounds[3]

        # # Round up and down to nearest dx
        # x0 = x0 - (x0 % dx)
        # x1 = x1 + (dx - x1 % dx)
        # y0 = y0 - (y0 % dx)
        # y1 = y1 + (dx - y1 % dx)

        xx = np.arange(x0, x1, dx) + 0.5 * dx
        yy = np.arange(y1, y0, -dx) - 0.5 * dx
        ii = np.empty((len(yy), len(xx),), dtype=np.uint32)

        # # Create empty ds
        # ds = xr.Dataset(
        #     {
        #         "index": (["y", "x"], ii),
        #     },
        #     coords={
        #         "x": xx,
        #         "y": yy,
        #     },
        # )
        # # Set no data value in ds
        # ds["index"].attrs["_FillValue"] = nodata

        # Go through refinement levels in grid
        xx, yy = np.meshgrid(xx, yy)
        indices = self.get_indices_at_points(xx, yy)
        indices[np.where(indices == -999)] = nodata

        # Fill the array with indices
        ii[:, :] = indices        

        # # Write first to netcdf
        # ds.to_netcdf("index.nc")

        # And now to cog (use -999 as the nodata value)
        with rasterio.open(
            filename,
            "w",
            driver="COG",
            height=height,
            width=width,
            count=1,
            dtype=ii.dtype,
            crs=self.model.crs,
            transform=transform,
            nodata=nodata,
            overview_resampling=Resampling.nearest,
        ) as dst:
            dst.write(ii, 1)


    def make_index_cog_chunked(self, filename, filename_topobathy, blocksize=1024):
        """
        Make a COG file with indices of the quadtree grid cells,
        processing in chunks so it works for very large rasters.
        """

        # Read metadata from topobathy file
        with rasterio.open(filename_topobathy) as src:
            dx = src.res[0]
            self.model.crs = src.crs
            transform = src.transform
            width = src.width
            height = src.height

        nodata = np.uint32(2147483647)

        # Prepare output COG
        profile = {
            "driver": "COG",
            "height": height,
            "width": width,
            "count": 1,
            "dtype": "uint32",
            "crs": self.model.crs,
            "transform": transform,
            "nodata": nodata,
            "overview_resampling": Resampling.nearest,
            "blockxsize": blocksize,
            "blockysize": blocksize,
            "tiled": True,
        }

        with rasterio.open(filename, "w", **profile) as dst:

            # Process block by block
            for row_off in range(0, height, blocksize):
                for col_off in range(0, width, blocksize):

                    win_width = min(blocksize, width - col_off)
                    win_height = min(blocksize, height - row_off)
                    window = Window(col_off, row_off, win_width, win_height)

                    # Build row/col indices for this block
                    rows = np.arange(row_off, row_off + win_height)
                    cols = np.arange(col_off, col_off + win_width)

                    # Compute x, y coordinates of pixel centers (meshgrid form)
                    x_coords = transform.c + (cols + 0.5) * transform.a
                    y_coords = transform.f + (rows + 0.5) * transform.e

                    xx, yy = np.meshgrid(x_coords, y_coords)

                    # Look up indices in flood model
                    indices = self.get_indices_at_points(xx, yy)
                    indices = indices.astype(np.int32, copy=False)

                    # Replace invalid values
                    indices[indices == -999] = nodata

                    # Write chunk into output
                    dst.write(indices, 1, window=window)

def binary_search(val_array, vals):
    indx = np.searchsorted(val_array, vals)  # ind is size of vals
    not_ok = np.where(indx == len(val_array))[
        0
    ]  # size of vals, points that are out of bounds
    indx[np.where(indx == len(val_array))[0]] = (
        0  # Set to zero to avoid out of bounds error
    )
    is_ok = np.where(val_array[indx] == vals)[0]  # size of vals
    indices = np.zeros(len(vals), dtype=int) - 1
    indices[is_ok] = indx[is_ok]
    indices[not_ok] = -1
    return indices
