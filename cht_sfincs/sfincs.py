# -*- coding: utf-8 -*-
"""
Created on Sat May 15 08:08:40 2021

@author: ormondt
"""
import os

from pyproj import CRS

from .input import SfincsInput

from .subgrid_v2 import SfincsSubgridTable
from .quadtree import SfincsGrid
from .mask import SfincsMask
from .initial_conditions import SfincsInitialConditions
from .boundary_conditions import SfincsBoundaryConditions
from .observation_points import SfincsObservationPoints
from .cross_sections import SfincsCrossSections
from .runup_gauges import SfincsRunupGauges
from .discharge_points import SfincsDischargePoints
from .thin_dams import SfincsThinDams
from .weirs import SfincsWeirs
from .wave_makers import SfincsWaveMakers
from .snapwave import SfincsSnapWave
from .output import SfincsOutput
# from .xmi import SfincsXmi

class SFINCS:    

    def __init__(self, root=None, crs=3395, mode="w", read_grid_data=True):
        # Use World Mercator as default CRS

        if not root:
            root = os.getcwd()

        self.exe_path                 = None
        self.path                     = root  
        self.input                    = SfincsInput(self)
        self.dll_path                 = None
        # if crs is an integer, assume it is an EPSG code
        if isinstance(crs, int):
            crs = CRS.from_epsg(crs)
        self.crs                      = crs
        # self.grid_type                = "regular"
        self.bathy_type               = "regular"
        self.grid                     = SfincsGrid(self)
        self.mask                     = SfincsMask(self)
        self.subgrid                  = SfincsSubgridTable(self)
        self.initial_conditions       = SfincsInitialConditions(self)
        self.boundary_conditions      = SfincsBoundaryConditions(self)
        self.observation_points       = SfincsObservationPoints(self)
        self.wave_makers              = SfincsWaveMakers(self)
        self.snapwave                 = SfincsSnapWave(self)
        self.cross_sections           = SfincsCrossSections(self)
        self.runup_gauges             = SfincsRunupGauges(self)
        self.discharge_points         = SfincsDischargePoints(self)
        self.thin_dams                = SfincsThinDams(self)
        self.weirs                    = SfincsWeirs(self)
        self.output                   = SfincsOutput(self)
        # self.xmi                      = SfincsXmi(self)
        # self.meteo_forcing            = None
        
        if mode == "r":
            self.input.read()
            if self.input.variables.epsg is not None:
                self.crs = CRS.from_epsg(self.input.variables.epsg)
            self.read_attribute_files(read_grid_data=read_grid_data)

    def read(self):
        # Reads sfincs.inp and attribute files
        self.input.read()
        if self.input.variables.epsg is not None:
            self.crs = CRS.from_epsg(self.input.variables.epsg)
        self.read_attribute_files()

    def write(self):
        # Writes sfincs.inp and attribute files
        self.input.write()
        self.write_attribute_files()

    def read_attribute_files(self, read_grid_data=True):
        
        self.grid = SfincsGrid(self)

        if self.input.variables.qtrfile:
            self.grid.type = "quadtree"
        else:
            self.grid.type = "regular"

        if read_grid_data: 

            if self.grid.type == "regular":

                self.grid.build(self.input.variables.x0,
                                self.input.variables.y0,
                                self.input.variables.nmax,
                                self.input.variables.mmax,
                                self.input.variables.dx,
                                self.input.variables.dy,
                                self.input.variables.rotation)

                # Read in mask, index and dep file (for quadtree the mask is stored in the quadtree file)
                self.mask.read()
                
            else:  
                # This reads in quadtree netcdf file. In case of index and mask file, it will generate the quadtree grid and save the file.
                # The grid object contains coordinates, neighbor indices, mask, snapwave mask and bed level.
                self.grid.read()

            # Sub-grid tables
            if self.bathy_type == "subgrid":
                self.subgrid.read()

        # Initial conditions (reads ini file)
        self.initial_conditions.read()

        # Boundary conditions (reads bnd and bzs file)
        self.boundary_conditions.read()

        # Observation points
        self.observation_points.read()

        # Cross sections
        self.cross_sections.read()

        # Runup gauges
        self.runup_gauges.read()

        # Thin dams
        self.thin_dams.read()

        # Weirs
        self.weirs.read()

        # Sources and sinks (reads src and dis file)
        self.discharge_points.read()

        # Infiltration
        # self.infiltration.read()

        # SnapWave (reads SnapWave boundary conditions (all the rest is already stored in the grid))
        self.snapwave.read()

        # Wave makers
        self.wave_makers.read()

    def write_attribute_files(self):
        """Writes all attribute files"""

        if self.grid.type == "regular":
            self.mask.write()
        else:    
            self.grid.write()

        # Boundary conditions
        self.boundary_conditions.write()
        # Observation points
        self.observation_points.write()
        # Cross sections
        self.cross_sections.write()
        # Runup gauges
        self.runup_gauges.write()
        # Thin dams
        self.thin_dams.write()
        # Weirs
        self.thin_dams.write()
        # Sources and sinks
        self.discharge_points.write()
        # # Infiltration
        # self.infiltration.write()
        # SnapWave
        self.snapwave.write()
        # Wave makers
        self.wave_makers.write()

    def write_batch_file(self):
        fid = open(os.path.join(self.path, "run.bat"), "w")
        fid.write("set HDF5_USE_FILE_LOCKING=FALSE\n")
        fid.write(self.exe_path + "\\" + "sfincs.exe")
        fid.close()

    def clear_spatial_attributes(self):
        # Clear all spatial data
        self.grid                 = SfincsGrid(self)
        self.mask                 = SfincsMask(self)
        self.subgrid              = SfincsSubgridTable(self)
        self.boundary_conditions  = SfincsBoundaryConditions(self)
        self.observation_points   = SfincsObservationPoints(self)
        self.cross_sections       = SfincsCrossSections(self)
        self.runup_gauges         = SfincsRunupGauges(self)
        self.thin_dams            = SfincsThinDams(self)
        self.weirs                = SfincsWeirs(self)
        self.wave_makers          = SfincsWaveMakers(self)
        self.snapwave             = SfincsSnapWave(self)

    def check_times(self):

        # Check that the boundary and other forcing fully covers the simulation time
        messages = []
        okay = True

        # Boundary conditions
        ok, message = self.boundary_conditions.check_times()
        if not ok:
            okay = False
            messages.append(message)
        
        # SnapWave boundary conditions
        ok, message = self.snapwave.boundary_conditions.check_times()
        if not ok:
            okay = False
            messages.append(message)

        # Discharges
        ok, message = self.discharge_points.check_times()
        if not ok:
            okay = False
            messages.append(message)

        return okay, messages
