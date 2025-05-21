# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 16:25:23 2022

@author: ormondt
"""
import os
import xarray as xr

from cht_sfincs.subgrid_quadtree_builder import build_subgrid_table_quadtree


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
        self.ds = xr.load_dataset(file_name)

    def write(self, file_name=None):
        if not file_name:
            if not self.model.input.variables.sbgfile:
                return
            file_name = os.path.join(
                self.model.path, self.model.input.variables.sbgfile
            )

        # Write XArray dataset to netcdf file
        self.ds.to_netcdf(file_name)

    def build(
        self,
        bathymetry_sets,
        roughness_sets,
        roughness_type="manning",
        manning_land=0.04,
        manning_water=0.020,
        manning_level=1.0,
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
        progress_bar=None,
    ):

        # If filename is empty
        if not file_name:
            if self.model.input.variables.sbgfile:
                file_name = os.path.join(
                    self.model.path, self.model.input.variables.sbgfile
                )
            else:
                file_name = os.path.join(self.model.path, "sfincs.sbg")
                self.model.input.variables.sbgfile = "sfincs.sbg"

        self.ds = build_subgrid_table_quadtree(
            self.model.grid.data,
            bathymetry_sets,
            roughness_sets,
            manning_land=manning_land,
            manning_water=manning_water,
            manning_level=manning_level,
            nr_levels=nr_levels,
            nr_subgrid_pixels=nr_subgrid_pixels,
            max_gradient=max_gradient,
            depth_factor=depth_factor,
            huthresh=huthresh,
            zmin=zmin,
            zmax=zmax,
            weight_option=weight_option,
            roughness_type=roughness_type,
            bathymetry_database=bathymetry_database,
            quiet=quiet,
            progress_bar=progress_bar,
            logger=None,
        )

        if file_name:
            self.write(file_name)
