# -*- coding: utf-8 -*-
"""
Created on Sat Jun 18 09:03:08 2022
@author: ormondt
"""
import os
import numpy as np
import xarray as xr
from pyproj import Transformer
import rioxarray

from cht_utils.misc_tools import interp2

class SfincsInitialConditions:

    def __init__(self, sf):
        self.model  = sf
        self.format = "netcdf"
        self.data = xr.Dataset()

    def interpolate(self, ds, var_name):
        # Interpolate the data to the mesh
        # ds is an xarray dataset
        # Get CRS from dataset
        if self.model.grid.data is None:
            self.model.grid.read()
        xy = self.model.grid.data.grid.face_coordinates
        # Transfrom model coordinates to dataset coordinates
        transformer = Transformer.from_proj(self.model.crs, ds.rio.crs)
        xd, yd = transformer.transform(xy[:, 0], xy[:, 1])
        if ds.rio.crs.is_geographic:
            # Shift model to make sure that the coordinates are in the range 0-360
            if xd.min() < 0.0:
                xd = xd + 360.0
            # Shift dataset to make sure that the coordinates are in the range 0-360
            if ds["x"].min() < 0.0:
                ds["x"] = ds["x"] + 360.0  
        # xd = xr.DataArray(xyd[:, 0], dims='np') + 360.0
        # yd = xr.DataArray(xyd[:, 1], dims='np')
        # Create empty xarray dataset
        self.data = xr.Dataset()
        # Interpolate. Replace NaN values with 0.0.
        # self.data["zs"] = ds[var_name].interp(x=xd, y=yd).fillna(0.0)
        self.data["x"] = xy[:, 0]
        self.data["y"] = xy[:, 1]
        dz = interp2(ds["x"].values, ds["y"].values, ds[var_name].values, xd, yd)
        dz[np.isnan(dz)] = 0.0
        self.data["zs"] = xr.DataArray(dz, dims=["np"]).fillna(0.0)
#        self.data["zs"].values = interp2(ds["x"].values, ds["y"].values, ds[var_name].values, xd, yd)
        self.data.rio.write_crs(self.model.crs, inplace=True)


    def read(self):
        # Read in initial conditions from netcdf file
        if not self.model.input.variables.ncinifile:
            # No initial conditions file specified
            return
        # file_name = os.path.join(self.model.path, self.model.input.variables.ncinifile)
        # # Read as netcdf file
        # self.data = xr.open_dataset(file_name)

    def write(self):
        # Write initial conditions to netcdf file
        if not self.model.input.variables.ncinifile:
            # No initial conditions file specified
            return
        file_name = os.path.join(self.model.path, self.model.input.variables.ncinifile)
        # Write as netcdf file
        self.data.to_netcdf(file_name)

    def clear(self):
        self.data = None
