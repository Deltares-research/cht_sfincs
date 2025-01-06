# -*- coding: utf-8 -*-
"""
Created on Sat Jun 18 09:03:08 2022
@author: ormondt
"""
import os
import geopandas as gpd
import shapely
import pandas as pd
import pathlib as pl

from xmipy import XmiWrapper
from cht_utils.misc_tools import interp2

class SfincsXmi(XmiWrapper):
    def __init__(self, sf, dll_path):
        # if dll_path is a string, convert it to a pathlib.Path object
        if isinstance(dll_path, str):
            dll_path = pl.Path(dll_path)
        super().__init__(dll_path, working_directory=sf.path)
        self.model = sf

    def get_domain(self):
        xy = self.model.grid.data.grid.face_coordinates
        self.xz = xy[:, 0]
        self.yz = xy[:, 1]
        self.zs = self.get_value_ptr("zs")
        self.zb = self.get_value_ptr("zb")
        self.zbini = self.zb[:].copy()

    def read(self):
        pass

    def write(self):
        pass

    def set_bed_level(self,
                      x=None,
                      y=None,
                      z=None,
                      update_water_level=False):
        """x, y, z are the coordinates of the new bed level (numpy arrays), that will be interpolated to the grid"""

        if x is None or y is None or z is None:
            # Assume that z 
            return

        # New bed level z 
        zb = interp2(x, y, z, self.xz, self.yz)

        # Difference w.r.t. previous time step
        dzb = zb - self.zb

        # Set new bed level
        self.zb[:] = zb

        if update_water_level:
            self.zs += dzb

        self.update_zbuv()    

    def set_bed_level_change(self,
                             x=None,
                             y=None,
                             dz=None,
                             update_water_level=False):
        """x, y, dz are the coordinates of the bed level change (w.r.t. initial bed level) (numpy arrays), that will be interpolated to the grid. Can be used for dynamic faulting or landslides"""

        # if x is None or y is None or z is None:
        #     # Assume that z 
        #     return

        # New bed level change dzb 
        dzb = interp2(x, y, dz, self.xz, self.yz)

        # Difference w.r.t. previous time step
        # Make a copy of self.zb
        zb0 = self.zb[:].copy()

        # Set new bed level
        self.zb[:] = self.zbini + dzb

        # Difference w.r.t. previous time step 
        dzt = self.zb[:] - zb0

        self.update_zbuv()    

        if update_water_level:
            self.zs += dzt

    def update_zbuv(self):
        self._execute_function(self.lib.update_zbuv)

    def update_water_level(self, t):
        pass

    def run_timestep(self):
        self.update()
        return self.get_current_time()
