"""SFINCS active-cell mask builder and visualisation utilities.

Provides the SfincsMask class for constructing the computational mask from
bathymetry thresholds and polygons, and for generating map overlay images via
Datashader.
"""

import os

# from .to_xugrid import xug
import warnings

import numpy as np
import shapely
import xarray as xr
import xugrid as xu

# import matplotlib.pyplot as plt
from pyproj import Transformer

np.warnings = warnings

import datashader as ds
import datashader.transfer_functions as tf
import geopandas as gpd
import pandas as pd
from datashader.utils import export_image


class SfincsMask:
    """SFINCS active-cell mask builder and visualisation utilities.

    Constructs and manages the integer mask array that defines which grid
    cells are active (1), open-boundary (2), outflow-boundary (3),
    downstream-river-boundary (5), or Neumann-boundary (6).

    Parameters
    ----------
    model : SFINCS
        The parent SFINCS model instance.
    """

    def __init__(self, model: "SFINCS") -> None:
        self.model = model
        # For plotting map overlay (This is the only data that is stored in the object! All other data is stored in the model.grid.data["mask"])
        self.datashader_dataframe = pd.DataFrame()

    def build(
        self,
        zmin: float = 99999.0,
        zmax: float = -99999.0,
        include_polygon=None,
        exclude_polygon=None,
        open_boundary_polygon=None,
        outflow_boundary_polygon=None,
        neumann_boundary_polygon=None,
        downstream_boundary_polygon=None,
        include_zmin: float = -99999.0,
        include_zmax: float = 99999.0,
        exclude_zmin: float = -99999.0,
        exclude_zmax: float = 99999.0,
        open_boundary_zmin: float = -99999.0,
        open_boundary_zmax: float = 99999.0,
        outflow_boundary_zmin: float = -99999.0,
        outflow_boundary_zmax: float = 99999.0,
        neumann_boundary_zmin: float = -99999.0,
        neumann_boundary_zmax: float = 99999.0,
        downstream_boundary_zmin: float = -99999.0,
        downstream_boundary_zmax: float = 99999.0,
        update_datashader_dataframe: bool = False,
        quiet: bool = True,
    ) -> None:
        """Build the SFINCS active-cell mask.

        Parameters
        ----------
        zmin : float, optional
            Minimum bed level (m) for active cells. Defaults to ``99999.0``.
        zmax : float, optional
            Maximum bed level (m) for active cells. Defaults to ``-99999.0``.
        include_polygon : geopandas.GeoDataFrame, optional
            Polygons of cells to force-include (set mask = 1).
        exclude_polygon : geopandas.GeoDataFrame, optional
            Polygons of cells to force-exclude (set mask = 0).
        open_boundary_polygon : geopandas.GeoDataFrame, optional
            Polygons defining open (water level) boundary cells (mask = 2).
        outflow_boundary_polygon : geopandas.GeoDataFrame, optional
            Polygons defining outflow boundary cells (mask = 3).
        neumann_boundary_polygon : geopandas.GeoDataFrame, optional
            Polygons defining Neumann boundary cells (mask = 6).
        downstream_boundary_polygon : geopandas.GeoDataFrame, optional
            Polygons defining downstream river boundary cells (mask = 5).
        include_zmin : float, optional
            Min bed level for include polygons. Defaults to ``-99999.0``.
        include_zmax : float, optional
            Max bed level for include polygons. Defaults to ``99999.0``.
        exclude_zmin : float, optional
            Min bed level for exclude polygons. Defaults to ``-99999.0``.
        exclude_zmax : float, optional
            Max bed level for exclude polygons. Defaults to ``99999.0``.
        open_boundary_zmin : float, optional
            Min bed level for open boundary polygons. Defaults to ``-99999.0``.
        open_boundary_zmax : float, optional
            Max bed level for open boundary polygons. Defaults to ``99999.0``.
        outflow_boundary_zmin : float, optional
            Min bed level for outflow boundary polygons. Defaults to ``-99999.0``.
        outflow_boundary_zmax : float, optional
            Max bed level for outflow boundary polygons. Defaults to ``99999.0``.
        neumann_boundary_zmin : float, optional
            Min bed level for Neumann boundary polygons. Defaults to ``-99999.0``.
        neumann_boundary_zmax : float, optional
            Max bed level for Neumann boundary polygons. Defaults to ``99999.0``.
        downstream_boundary_zmin : float, optional
            Min bed level for downstream boundary polygons. Defaults to ``-99999.0``.
        downstream_boundary_zmax : float, optional
            Max bed level for downstream boundary polygons. Defaults to ``99999.0``.
        update_datashader_dataframe : bool, optional
            Update the internal Datashader DataFrame after building.
            Defaults to ``False``.
        quiet : bool, optional
            Suppress progress messages. Defaults to ``True``.

        Returns
        -------
        None
        """
        if not quiet:
            print("Building mask ...")

        # Initialize mask
        nr_cells = self.model.grid.data.sizes["mesh2d_nFaces"]
        mask = np.zeros(nr_cells, dtype=np.int8)
        x, y = self.model.grid.face_coordinates()
        z = self.model.grid.data["z"].values[:]

        if zmin >= zmax:
            # Do not include any points initially
            if include_polygon is None:
                print(
                    "WARNING: Entire mask set to zeros! Please ensure zmax is greater than zmin, or provide include polygon(s) !"
                )
                return
        else:
            if z is not None:
                # Set initial mask based on zmin and zmax
                iok = np.where((z >= zmin) & (z <= zmax))
                mask[iok] = 1
            else:
                print(
                    "WARNING: Entire mask set to zeros! No depth values found on grid."
                )

        # Include polygons
        if include_polygon is not None:
            for ip, polygon in include_polygon.iterrows():
                inpol = inpolygon(x, y, polygon["geometry"])
                iok = np.where((inpol) & (z >= include_zmin) & (z <= include_zmax))
                mask[iok] = 1

        # Exclude polygons
        if exclude_polygon is not None:
            for ip, polygon in exclude_polygon.iterrows():
                inpol = inpolygon(x, y, polygon["geometry"])
                iok = np.where((inpol) & (z >= exclude_zmin) & (z <= exclude_zmax))
                mask[iok] = 0

        # Open boundary polygons
        if open_boundary_polygon is not None:
            self.set_boundary_mask(
                mask, open_boundary_polygon, open_boundary_zmin, open_boundary_zmax, 2
            )

        # Outflow boundary polygons
        if outflow_boundary_polygon is not None:
            self.set_boundary_mask(
                mask,
                outflow_boundary_polygon,
                outflow_boundary_zmin,
                outflow_boundary_zmax,
                3,
            )

        # Downstream river boundary polygons
        if downstream_boundary_polygon is not None:
            self.set_boundary_mask(
                mask,
                downstream_boundary_polygon,
                downstream_boundary_zmin,
                downstream_boundary_zmax,
                5,
            )

        # Neumann boundary polygons
        if neumann_boundary_polygon is not None:
            self.set_boundary_mask(
                mask,
                neumann_boundary_polygon,
                neumann_boundary_zmin,
                neumann_boundary_zmax,
                6,
            )

        # Now add the data arrays
        ugrid2d = self.model.grid.data.grid
        self.model.grid.data["mask"] = xu.UgridDataArray(
            xr.DataArray(data=mask, dims=[ugrid2d.face_dimension]), ugrid2d
        )

        if update_datashader_dataframe:
            # For use in DelftDashboard
            self.get_datashader_dataframe()

    # def read(self, file_name=None):
    #     pass

    # def write(self):
    #     mask = self.model.grid.data["mask"].values[:]
    #     file_name = os.path.join(self.model.path, self.model.input.variables.mskfile)
    #     file = open(file_name, "wb")
    #     file.write(np.int8(mask))
    #     file.close()

    def set_boundary_mask(
        self,
        mask: "np.ndarray",
        boundary_polygon: "gpd.GeoDataFrame",
        zmin: float,
        zmax: float,
        mask_value: int,
    ) -> None:
        """Set the mask value for cells inside a boundary polygon.

        Parameters
        ----------
        mask : numpy.ndarray
            Integer mask array to modify in-place.
        boundary_polygon : geopandas.GeoDataFrame
            GeoDataFrame containing the polygon(s) that define the boundary.
        zmin : float
            Minimum bed level (m) threshold.
        zmax : float
            Maximum bed level (m) threshold.
        mask_value : int
            Mask value to assign (e.g. 2 for open boundary, 3 for outflow).

        Returns
        -------
        None
        """
        x, y = self.model.grid.face_coordinates()
        z = self.model.grid.data["z"].values[:]

        # Indices are 1-based in SFINCS so subtract 1 for python 0-based indexing
        mu = self.model.grid.data["mu"].values[:]
        mu1 = self.model.grid.data["mu1"].values[:] - 1
        mu2 = self.model.grid.data["mu2"].values[:] - 1
        nu = self.model.grid.data["nu"].values[:]
        nu1 = self.model.grid.data["nu1"].values[:] - 1
        nu2 = self.model.grid.data["nu2"].values[:] - 1
        md = self.model.grid.data["md"].values[:]
        md1 = self.model.grid.data["md1"].values[:] - 1
        md2 = self.model.grid.data["md2"].values[:] - 1
        nd = self.model.grid.data["nd"].values[:]
        nd1 = self.model.grid.data["nd1"].values[:] - 1
        nd2 = self.model.grid.data["nd2"].values[:] - 1

        for ip, polygon in boundary_polygon.iterrows():
            inpol = inpolygon(x, y, polygon["geometry"])
            # Only consider points that are:
            # 1) Inside the polygon
            # 2) Have a mask > 0
            # 3) z>=zmin
            # 4) z<=zmax
            iok = np.where((inpol) & (mask > 0) & (z >= zmin) & (z <= zmax))
            for ic in iok[0]:
                okay = False
                # Check neighbors, cell must have at least one inactive neighbor
                # Left
                if md[ic] <= 0:
                    # Coarser or equal to the left
                    if md1[ic] >= 0:
                        # Cell has neighbor to the left
                        if mask[md1[ic]] == 0:
                            # And it's inactive
                            okay = True
                    else:
                        # No neighbor, so set mask = 2
                        okay = True
                else:
                    # Finer to the left
                    if md1[ic] >= 0:
                        # Cell has neighbor to the left
                        if mask[md1[ic]] == 0:
                            # And it's inactive
                            okay = True
                    else:
                        # No neighbor, so set mask = 2
                        okay = True
                    if md2[ic] >= 0:
                        # Cell has neighbor to the left
                        if mask[md2[ic]] == 0:
                            # And it's inactive
                            okay = True
                    else:
                        # No neighbor, so set mask = 2
                        okay = True

                # Below
                if nd[ic] <= 0:
                    # Coarser or equal below
                    if nd1[ic] >= 0:
                        # Cell has neighbor below
                        if mask[nd1[ic]] == 0:
                            # And it's inactive
                            okay = True
                    else:
                        # No neighbor, so set mask = 2
                        okay = True
                else:
                    # Finer below
                    if nd1[ic] >= 0:
                        # Cell has neighbor below
                        if mask[nd1[ic]] == 0:
                            # And it's inactive
                            okay = True
                    else:
                        # No neighbor, so set mask = 2
                        okay = True
                    if nd2[ic] >= 0:
                        # Cell has neighbor below
                        if mask[nd2[ic]] == 0:
                            # And it's inactive
                            okay = True
                    else:
                        # No neighbor, so set mask = 2
                        okay = True

                # Right
                if mu[ic] <= 0:
                    # Coarser or equal to the right
                    if mu1[ic] >= 0:
                        # Cell has neighbor to the right
                        if mask[mu1[ic]] == 0:
                            # And it's inactive
                            okay = True
                    else:
                        # No neighbor, so set mask = 2
                        okay = True
                else:
                    # Finer to the left
                    if mu1[ic] >= 0:
                        # Cell has neighbor to the right
                        if mask[mu1[ic]] == 0:
                            # And it's inactive
                            okay = True
                    else:
                        # No neighbor, so set mask = 2
                        okay = True
                    if mu2[ic] >= 0:
                        # Cell has neighbor to the right
                        if mask[mu2[ic]] == 0:
                            # And it's inactive
                            okay = True
                    else:
                        # No neighbor, so set mask = 2
                        okay = True

                # Above
                if nu[ic] <= 0:
                    # Coarser or equal above
                    if nu1[ic] >= 0:
                        # Cell has neighbor above
                        if mask[nu1[ic]] == 0:
                            # And it's inactive
                            okay = True
                    else:
                        # No neighbor, so set mask = 2
                        okay = True
                else:
                    # Finer below
                    if nu1[ic] >= 0:
                        # Cell has neighbor above
                        if mask[nu1[ic]] == 0:
                            # And it's inactive
                            okay = True
                    else:
                        # No neighbor, so set mask = 2
                        okay = True
                    if nu2[ic] >= 0:
                        # Cell has neighbor above
                        if mask[nu2[ic]] == 0:
                            # And it's inactive
                            okay = True
                    else:
                        # No neighbor, so set mask = 2
                        okay = True

                if okay:
                    mask[ic] = mask_value

    def to_gdf(self, option: str = "all"):
        """Return mask cells as a GeoDataFrame of point geometries.

        Parameters
        ----------
        option : str, optional
            Filter: ``"all"`` (mask > 0), ``"include"`` (mask == 1),
            ``"open"`` (mask == 2), ``"outflow"`` (mask == 3).
            Defaults to ``"all"``.

        Returns
        -------
        geopandas.GeoDataFrame
            Point geometries at cell centres for the selected mask values.
        """
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

    def read(self) -> None:
        """Read mask data (currently a no-op; mask is stored in the grid netcdf).

        Returns
        -------
        None
        """
        pass
        # # Read in index file, mask file and dep file
        # msk  = np.full([self.model.input.variables.nmax*self.model.input.variables.mmax], 0)
        # ind  = np.fromfile(self.model.input.variables.indexfile, dtype="i4")
        # npoints  = ind[0]
        # ind = np.squeeze(ind[1:]) - 1
        # mskv = np.fromfile(self.model.input.variables.mskfile, dtype="i1")
        # msk[ind] = mskv
        # dep = np.full([self.model.input.variables.nmax*self.model.input.variables.mmax], 0.0)
        # if self.model.input.variables.depfile:
        #     depv  = np.fromfile(self.model.input.variables.depfile, dtype="f4")
        #     dep[ind] = depv
        # self.model.grid.data["mask"].values[:] = msk
        # self.model.grid.data["z"].values[:] = dep

    def write(self) -> None:
        """Write mask data (currently a no-op; mask is stored in the grid netcdf).

        Returns
        -------
        None
        """
        pass
        # """Write msk, ind, and dep files"""
        # mskv = self.model.grid.data["mask"].values
        # ind  = np.where(mskv>0)
        # mskv = mskv[ind]
        # depv = self.model.grid.data["z"].values[ind]

        # # Add 1 because indices in SFINCS start with 1, not 0
        # ind = ind[0] + 1

        # # Write index file
        # self.model.input.variables.indexfile = "sfincs.ind"
        # file = open(self.model.input.variables.indexfile, "wb")
        # file.write(np.int32(np.size(ind)))
        # file.write(np.int32(ind))
        # file.close()

        # # Write mask file
        # self.model.input.variables.mskfile = "sfincs.msk"
        # file = open(self.model.input.variables.mskfile, "wb")
        # file.write(np.int8(mskv))
        # file.close()

        # # Write dep file
        # self.model.input.variables.depfile = "sfincs.dep"
        # file = open(self.model.input.variables.depfile, "wb")
        # file.write(np.float32(depv))
        # file.close()

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

    def has_open_boundaries(self) -> bool:
        """Return True if any cell has an open-boundary mask value (== 2).

        Returns
        -------
        bool
            ``True`` if at least one open-boundary cell exists.
        """
        mask = self.model.grid.data["mask"]
        if mask is None:
            return False
        if np.any(mask == 2):
            return True
        else:
            return False

    def get_datashader_dataframe(self) -> None:
        """Populate the internal Datashader DataFrame from the current mask.

        Returns
        -------
        None
        """
        # Coordinates of cell centers
        x = self.model.grid.data.grid.face_coordinates[:, 0]
        y = self.model.grid.data.grid.face_coordinates[:, 1]
        # Check if grid crosses the dateline
        cross_dateline = False
        if self.model.crs.is_geographic:
            if np.max(x) > 180.0:
                cross_dateline = True
        mask = self.model.grid.data["mask"].values[:]
        # Get rid of cells with mask = 0
        iok = np.where(mask > 0)
        x = x[iok]
        y = y[iok]
        mask = mask[iok]
        if np.size(x) == 0:
            # Set empty dataframe
            self.datashader_dataframe = pd.DataFrame()
            return
        # Transform all to 3857 (web mercator)
        transformer = Transformer.from_crs(self.model.crs, 3857, always_xy=True)
        x, y = transformer.transform(x, y)
        if cross_dateline:
            x[x < 0] += 40075016.68557849

        self.datashader_dataframe = pd.DataFrame(dict(x=x, y=y, mask=mask))

    def clear_datashader_dataframe(self) -> None:
        """Clear the internal Datashader DataFrame.

        Returns
        -------
        None
        """
        self.datashader_dataframe = pd.DataFrame()

    def map_overlay(
        self,
        file_name: str,
        xlim=None,
        ylim=None,
        active_color: str = "yellow",
        boundary_color: str = "red",
        downstream_color: str = "blue",
        neumann_color: str = "purple",
        outflow_color: str = "green",
        px: int = 2,
        width: int = 800,
    ) -> bool:
        """Render the SFINCS mask as a map overlay image using Datashader.

        Parameters
        ----------
        file_name : str
            Output image file path (without extension).
        xlim : list[float], optional
            Longitude extent ``[lon_min, lon_max]`` in geographic CRS.
        ylim : list[float], optional
            Latitude extent ``[lat_min, lat_max]`` in geographic CRS.
        active_color : str, optional
            Colour for active cells (mask == 1). Defaults to ``"yellow"``.
        boundary_color : str, optional
            Colour for open-boundary cells (mask == 2). Defaults to ``"red"``.
        downstream_color : str, optional
            Colour for downstream river boundary cells (mask == 5).
            Defaults to ``"blue"``.
        neumann_color : str, optional
            Colour for Neumann boundary cells (mask == 6). Defaults to
            ``"purple"``.
        outflow_color : str, optional
            Colour for outflow boundary cells (mask == 3). Defaults to
            ``"green"``.
        px : int, optional
            Pixel spread radius. Defaults to ``2``.
        width : int, optional
            Output image width in pixels. Defaults to ``800``.

        Returns
        -------
        bool
            ``True`` on success, ``False`` if the mask is empty or rendering
            fails.
        """

        if self.model.grid.data is None:
            # No mask points (yet)
            return False

        try:
            # Check if datashader dataframe is empty (maybe it was not made yet, or it was cleared)
            if self.datashader_dataframe.empty:
                self.get_datashader_dataframe()

            # If it is still empty (because there are no active cells), return False
            if self.datashader_dataframe.empty:
                return False

            transformer = Transformer.from_crs(4326, 3857, always_xy=True)
            xl0, yl0 = transformer.transform(xlim[0], ylim[0])
            xl1, yl1 = transformer.transform(xlim[1], ylim[1])
            if xl0 > xl1:
                xl1 += 40075016.68557849
            xlim = [xl0, xl1]
            ylim = [yl0, yl1]
            ratio = (ylim[1] - ylim[0]) / (xlim[1] - xlim[0])
            height = int(width * ratio)

            cvs = ds.Canvas(
                x_range=xlim, y_range=ylim, plot_height=height, plot_width=width
            )

            # Instead, we can create separate images for each mask and stack them
            dfact = self.datashader_dataframe[self.datashader_dataframe["mask"] == 1]
            dfbnd = self.datashader_dataframe[self.datashader_dataframe["mask"] == 2]
            dfout = self.datashader_dataframe[self.datashader_dataframe["mask"] == 3]
            dfneu = self.datashader_dataframe[self.datashader_dataframe["mask"] == 5]
            dfdwn = self.datashader_dataframe[self.datashader_dataframe["mask"] == 6]
            img_a = tf.shade(
                tf.spread(cvs.points(dfact, "x", "y", ds.any()), px=px),
                cmap=active_color,
            )
            img_b = tf.shade(
                tf.spread(cvs.points(dfbnd, "x", "y", ds.any()), px=px),
                cmap=boundary_color,
            )
            img_o = tf.shade(
                tf.spread(cvs.points(dfout, "x", "y", ds.any()), px=px),
                cmap=outflow_color,
            )
            img_n = tf.shade(
                tf.spread(cvs.points(dfneu, "x", "y", ds.any()), px=px),
                cmap=neumann_color,
            )
            img_d = tf.shade(
                tf.spread(cvs.points(dfdwn, "x", "y", ds.any()), px=px),
                cmap=downstream_color,
            )
            img = tf.stack(img_a, img_b, img_o, img_n, img_d)

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


def get_neighbors_in_larger_cell(n: int, m: int) -> tuple:
    """Return the neighbor (n, m) indices within the parent coarser cell.

    Parameters
    ----------
    n : int
        Row index (0-based) in the fine grid.
    m : int
        Column index (0-based) in the fine grid.

    Returns
    -------
    tuple[list[int], list[int]]
        ``(nnbr, mnbr)`` — four (n, m) neighbor index pairs; unused entries
        are ``-1``.
    """
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
    return nnbr, mnbr


def odd(num: int) -> bool:
    """Return ``True`` if *num* is odd.

    Parameters
    ----------
    num : int
        Integer to test.

    Returns
    -------
    bool
    """
    if (num % 2) == 1:
        return True
    else:
        return False


def even(num: int) -> bool:
    """Return ``True`` if *num* is even.

    Parameters
    ----------
    num : int
        Integer to test.

    Returns
    -------
    bool
    """
    if (num % 2) == 0:
        return True
    else:
        return False


# def inpolygon(xq, yq, p):
#     shape = xq.shape
#     xq = xq.reshape(-1)
#     yq = yq.reshape(-1)
#     q = [(xq[i], yq[i]) for i in range(xq.shape[0])]
#     p = path.Path([(crds[0], crds[1]) for i, crds in enumerate(p.exterior.coords)])
#     return p.contains_points(q).reshape(shape)


def inpolygon(xq: np.ndarray, yq: np.ndarray, poly) -> np.ndarray:
    """Test which query points lie inside a Shapely polygon.

    Parameters
    ----------
    xq : numpy.ndarray
        X-coordinates of the query points.
    yq : numpy.ndarray
        Y-coordinates of the query points.
    poly : shapely.geometry.base.BaseGeometry
        Polygon (or multipolygon) to test against.

    Returns
    -------
    numpy.ndarray
        Boolean array with the same shape as *xq*; ``True`` where the
        point is inside *poly*.
    """
    coords = np.column_stack((xq.ravel(), yq.ravel()))
    pts = shapely.points(coords)
    inside = shapely.contains(poly, pts)  # vectorized
    return inside.reshape(xq.shape)


def binary_search(vals: np.ndarray, val) -> int | None:
    """Binary search for *val* in a sorted array.

    Parameters
    ----------
    vals : numpy.ndarray
        Sorted 1-D array to search.
    val : scalar
        Value to search for.

    Returns
    -------
    int or None
        Index of *val* in *vals*, or ``None`` if not found.
    """
    indx = np.searchsorted(vals, val)
    if indx < np.size(vals):
        if vals[indx] == val:
            return indx
    return None
