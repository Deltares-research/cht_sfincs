# -*- coding: utf-8 -*-
"""

build_beira_model.py

Script to set up large-scale Beira model with Quadtree and subgrid functionality

Let's see if this works ...

Created on Sat Jun 18 09:03:08 2022

@author: ormondt
"""

import os
from pyproj import CRS

from cht_model_builder.model_builder import ModelBuilder
from cht_sfincs.sfincs import SFINCS
from cht_sfincs.quadtree import QuadtreeGrid, QuadtreeMask
from cht_sfincs.subgrid import SubgridTableQuadtree, SubgridTableRegular 
from cht_sfincs.regulargrid import RegularGrid
import cht_utils.fileops as fo
from cht_tiling.tiling import make_topobathy_tiles
from cht_bathymetry.bathymetry_database import bathymetry_database

class SfincsBuilder(ModelBuilder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def build(self,
              mskfile = "sfincs.msk",
              qtrfile = "sfincs.qtr",
              sbgfile = "sfincs.sbg",
              indfile = "sfincs.ind",
              depfile = "sfincs.dep",
              wavemskfile = "snapwave.msk",
              build_quadtree=True,
              build_subgrid=True,
              get_bathymetry=True,
              write_input=True,
              make_mask=True,
              make_tiles=True,
              quiet=False):
        
        crs = CRS(self.setup_config["coordinates"]["crs"])
        
        inpfile = os.path.join(self.model_path, "sfincs.inp")
        
        sf = SFINCS()
        
        sf.crs = crs
        sf.input.crs_name = self.setup_config["coordinates"]["crs"]

        if crs.is_geographic:
            sf.input.crs_type = "geographic"
            sf.input.crsgeo   = 1  
            sf.input.latitude = None
        else:    
            sf.input.crs_type = "projected"
            sf.input.crsgeo   = 0
            if "utm" in crs.name.lower():
               sf.input.crs_utmzone = crs.name[-3:]
            # Compute latitude   
        sf.input.crs_epsg = crs.to_epsg()
        
        # Set input parameters in sf.input
        for key in self.setup_config["input"]:
            setattr(sf.input, key, self.setup_config["input"][key])

        # # Copy other files to input folder
        # if os.path.exists(os.path.join(self.data_path, "sfincs.bnd")):
        #     fo.copy_file(os.path.join(self.data_path, "sfincs.bnd"),
        #                 self.model_path)
        #     sf.input.bndfile = "sfincs.bnd"
        # if os.path.exists(os.path.join(self.data_path, "sfincs.bca")):
        #     fo.copy_file(os.path.join(self.data_path, "sfincs.bca"),
        #                 self.model_path)
        #     sf.input.bcafile = "sfincs.bca"
            
        # Check if quadtree needs to be built (may also be regular grid)
        
        if self.setup_config["quadtree"]:
            # Quadtree not empty
            
            # This is the default (qtrfile=sfincs.qtr)
            # If the qtr file is None, a "traditional" regular grid is built        
            
            # The following are stored in the quadtree file and should not be in sfincs.inp
            sf.input.x0       = None
            sf.input.y0       = None
            sf.input.dx       = None
            sf.input.dy       = None
            sf.input.nmax     = None
            sf.input.mmax     = None
            sf.input.rotation = None
            
            # Initialize quadtree grid object
            qtr = QuadtreeGrid(crs=CRS(self.setup_config["coordinates"]["crs"]))
            
            if build_quadtree:
                # Build the mesh
                qtr.build(self.setup_config["coordinates"]["x0"],
                          self.setup_config["coordinates"]["y0"],
                          self.setup_config["coordinates"]["nmax"],
                          self.setup_config["coordinates"]["mmax"],
                          self.setup_config["coordinates"]["dx"],
                          self.setup_config["coordinates"]["dy"],
                          self.setup_config["coordinates"]["rotation"],
                          self.refinement_polygons)
    
            else:    
                # Load the mesh
                qtr.load(os.path.join(self.setup_config["path"], self.setup_config["qtr_file"]))
        
            if get_bathymetry:
                # Get bathy/topo
                qtr.get_bathymetry(self.bathymetry_list, quiet=False)
            
            # Make mask
            if make_mask or build_quadtree:

                flow_mask = QuadtreeMask(qtr,
                                         zmin=self.setup_config["mask"]["zmin"],
                                         zmax=self.setup_config["mask"]["zmax"],
                                         include_polygons=self.include_polygons,
                                         exclude_polygons=self.exclude_polygons,
                                         open_boundary_polygons=self.open_boundary_polygons,
                                         outflow_boundary_polygons=self.outflow_boundary_polygons)
 
                if self.setup_config["wave_mask"]:                 
                    wave_mask = QuadtreeMask(qtr,
                                             zmin=self.setup_config["wave_mask"]["zmin"],
                                             zmax=self.setup_config["wave_mask"]["zmax"],
                                             include_polygons=self.wave_include_polygons,
                                             exclude_polygons=self.wave_exclude_polygons)
                else:
                    wave_mask = None                
            
                # Remove cells with msk == 0
                qtr.cut_inactive_cells(flow_mask, wave_mask=wave_mask)
            
                # Save
                qtr.save(os.path.join(self.model_path, qtrfile))
                sf.input.qtrfile = qtrfile
                
                # Save flow mask
                flow_mask.save(os.path.join(self.model_path, mskfile))
                sf.input.mskfile = mskfile
                
                if self.setup_config["wave_mask"]:                 
                
                    # Save wave mask
                    wave_mask.save(os.path.join(self.model_path, wavemskfile))
                    sf.input.snapwave_mskfile = wavemskfile
                
        
            # And the subgrid table, for those who want things really fancy.
            if self.setup_config["subgrid"] and build_subgrid:
                sbg = SubgridTableQuadtree()
                sbg.build(qtr,
                          self.bathymetry_list,
                          self.roughness_list,
                          file_name=os.path.join(self.model_path, sbgfile),
                          nr_bins=self.setup_config["subgrid"]["nr_bins"],
                          max_gradient=self.setup_config["subgrid"]["max_gradient"],
                          nr_subgrid_pixels=self.setup_config["subgrid"]["nr_subgrid_pixels"],
                          zmin=self.setup_config["subgrid"]["zmin"],
                          quiet=quiet)    
                sf.input.sbgfile = sbgfile

        else:

            # Regular grid
            sf.input.x0       = self.setup_config["coordinates"]["x0"]
            sf.input.y0       = self.setup_config["coordinates"]["y0"]
            sf.input.dx       = self.setup_config["coordinates"]["dx"]
            sf.input.dy       = self.setup_config["coordinates"]["dy"]
            sf.input.nmax     = self.setup_config["coordinates"]["nmax"]
            sf.input.mmax     = self.setup_config["coordinates"]["mmax"]
            sf.input.rotation = self.setup_config["coordinates"]["rotation"]
            
            # Initialize regular grid
            # The regular grid class contains values for xz, yz, zz, and mask
            grd = RegularGrid(sf.input.x0,
                              sf.input.y0,
                              sf.input.dx,
                              sf.input.dy,
                              sf.input.nmax,
                              sf.input.mmax,
                              sf.input.rotation,
                              crs=CRS(self.setup_config["coordinates"]["crs"]))
                        
            # Get bathy/topo
            grd.get_bathymetry(self.bathymetry_list, quiet=False)
            
            # Make mask
            grd.make_mask(zmin=self.setup_config["mask"]["zmin"],
                          zmax=self.setup_config["mask"]["zmax"],
                          include_polygons=self.include_polygons,
                          exclude_polygons=self.exclude_polygons,
                          open_boundary_polygons=self.open_boundary_polygons,
                          outflow_boundary_polygons=self.outflow_boundary_polygons)
            
            # # Save msk, dep and ind files
            # if build_subgrid:
            #     # Do not include dep file for subgrid models
            #     depfile = None
            dpfile = os.path.join(self.model_path, depfile)

            grd.save(os.path.join(self.model_path, mskfile),
                     dpfile,
                     os.path.join(self.model_path, indfile))
            
            sf.input.mskfile   = mskfile
            sf.input.depfile   = depfile
            sf.input.indexfile = indfile

            # And the subgrid table, for those who want things really fancy.
            if self.setup_config["subgrid"] and build_subgrid:
                sbg = SubgridTableRegular()
                sbg.build(grd,
                          self.bathymetry_list,
                          self.roughness_list,
                          file_name=os.path.join(self.model_path, sbgfile),
                          nr_bins=self.setup_config["subgrid"]["nr_bins"],
                          max_gradient=self.setup_config["subgrid"]["max_gradient"],
                          nr_subgrid_pixels=self.setup_config["subgrid"]["nr_subgrid_pixels"],
                          zmin=self.setup_config["subgrid"]["zmin"],
                          quiet=quiet)    
                sf.input.sbgfile = sbgfile

            ### Tiles        
            if make_tiles and self.setup_config["tiling"]:
                zoom_range = []
                if self.setup_config["tiling"]["zmin"]>-99990.0 or self.setup_config["tiling"]["zmax"]<99990.0:
                    zoom_range = [self.setup_config["tiling"]["zoom_range_min"],
                                  self.setup_config["tiling"]["zoom_range_max"]]
                sf.make_index_tiles(os.path.join(self.tile_path, "indices"),
                                    zoom_range=zoom_range)
                lon_range, lat_range = sf.bounding_box(crs=CRS(4326))

                bathy_list = []
                for dataset in self.setup_config["bathymetry"]["dataset"]:
                    bathy_list.append(dataset["name"])
                make_topobathy_tiles(os.path.join(self.tile_path, "topobathy"),
                                     bathy_list,
                                     lon_range,
                                     lat_range,
                                     index_path=os.path.join(self.tile_path, "indices"),
                                     zoom_range=zoom_range,
                                     bathymetry_database_path="d:\\delftdashboard\\data\\bathymetry",
                                     quiet=False)

        if write_input:    
            sf.write_input_file(input_file=inpfile)

