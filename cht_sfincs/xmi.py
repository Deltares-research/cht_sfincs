# -*- coding: utf-8 -*-
"""
Created on Sat Jun 18 09:03:08 2022
@author: ormondt
"""
import pathlib as pl
from scipy.interpolate import RegularGridInterpolator
import numpy as np
from ctypes import (
    CDLL,
    POINTER,
    byref,
    c_char,
    c_char_p,
    c_double,
    c_int,
    c_void_p,
    create_string_buffer,
)

from xmipy import XmiWrapper

class SfincsXmi(XmiWrapper):
    def __init__(self, dll_path, working_directory):
        # if dll_path is a string, convert it to a pathlib.Path object
        if isinstance(dll_path, str):
            dll_path = pl.Path(dll_path)
        super().__init__(dll_path, working_directory)
        self.rtc_collection = RTCCollection(self)

    def get_domain(self):
        self.get_xz_yz()
        self.get_zs()
        self.get_zb()
        self.get_qext()
        self.zbini = self.zb[:].copy()

    def reset_qext(self):
        """Reset the external fluxes to zero"""
        self.qext[:] = 0.0

    def read(self):
        pass

    def write(self):
        pass

    def find_cell(self, x, y):
        """Find the index of the cell that contains the point (x, y)"""
        indx = self.get_sfincs_cell_index(x, y)
        return indx
    
    def find_cell_area(self, index):
        """Find the area of the cell with the given index"""
        # Get the area of the cell with the given index
        area = self.get_sfincs_cell_area(index)
        return area

    def get_xz_yz(self):
        """Get the bed level"""
        self.xz = self.get_value_ptr("z_xz")
        self.yz = self.get_value_ptr("z_yz")

    def get_zb(self):
        """Get the bed level"""
        self.zb = self.get_value_ptr("zb")

    def get_zs(self):
        """Get the water level at the current time step"""
        self.zs = self.get_value_ptr("zs")

    def get_qext(self):
        """Get qext"""
        self.qext = self.get_value_ptr("qext")

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

        # Update uv points in SFINCS
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

    def get_sfincs_cell_index(self, x, y):
        indx = c_int(0)
        self._execute_function(self.lib.get_sfincs_cell_index, byref(c_double(x)), byref(c_double(y)), byref(indx))
        # Index is 1-based in sfincs, so we need to subtract 1 to get the 0-based index
        return indx.value - 1

    def get_sfincs_cell_area(self, index):
        area = c_double(0.0)
        self._execute_function(self.lib.get_sfincs_cell_area, byref(c_int(index + 1)), byref(area))
        return area.value

    def get_cell_coordinates(self):
        self.xz = xy[:, 0]
        self.yz = xy[:, 1]


    def update_water_level(self, t):
        pass

    def run_timestep(self):
        self.update()
        return self.get_current_time()

class RTCCollection:
    """Collection of RTC objects"""
    def __init__(self, sfx):
        self.sfx = sfx

    def initialize(self, rtc_dict_list):    
        self.rtc_list = []
        # Loop through all RTC objects and create them
        for rtc_dict in rtc_dict_list:
            name = rtc_dict["name"]
            type = rtc_dict["type"]
            p1 = rtc_dict.get("p1", (0.0, 0.0))
            p2 = rtc_dict.get("p2", None) # p2 is optional, if not given, we assume a single point source
            elevation = rtc_dict.get("elevation", 0.0)
            width = rtc_dict.get("width", 0.0)
            alpha = rtc_dict.get("alpha", 0.5)
            zmin = rtc_dict.get("zmin", 0.0)
            qpump = rtc_dict.get("qpump", 0.0)
            dzpump = rtc_dict.get("dzpump", 0.02)
            zmin = rtc_dict.get("zmin", 0.0)
            t_relax = rtc_dict.get("t_relax", 10.0)
            # Create the RTC object
            rtc = RTC(self.sfx, name, type, p1, p2, elevation, width, alpha, qpump, dzpump, zmin, t_relax)
            self.rtc_list.append(rtc)

    def compute_fluxes(self, dt=1.0e6):

        # First set qext to zero
        self.sfx.reset_qext()

        # Now get water levels
        self.sfx.get_zs()

        # Loop through all RTC objects and compute the fluxes
        for rtc in self.rtc_list:
            # Compute the flux for this RTC object
            q = rtc.compute_flux()

            # Apply relaxation factor if t_relax > 0.0
            if rtc.t_relax > 0.0:
                # Compute the relaxation factor
                relaxation_factor = min(dt / rtc.t_relax, 1.0)
                # Compute the flux based on the relaxation factor
                # q0 = q
                q = (1.0 - relaxation_factor) * rtc.q + relaxation_factor * q
                # if rtc.type == "pump":
                #     print(f"RTC {rtc.name}: q0 = {q0:.4f} m3/s, q = {q:.4f}, rtc.q = {rtc.q:.4f}, relaxation_factor = {relaxation_factor:.2f}")

            # Fluxes in qext are in m/s, so we need to convert m3/s to m/s by dividing by the area
            # Determine the direction of the flux
            if q > 0.0:
                # If the flux is positive, we are discharging from the first point to the second point
                self.sfx.qext[rtc.index1] = -q / rtc.area1
                if rtc.index2 is not None:
                    # If p2 is given, we assume a two point source
                    # Set the flux for the second point
                    self.sfx.qext[rtc.index2] = q / rtc.area2
            else:
                # If the flux is negative, we are discharging from the second point to the first point
                self.sfx.qext[rtc.index1] = q / rtc.area1
                if rtc.index2 is not None:
                    # If p2 is given, we assume a two point source
                    self.sfx.qext[rtc.index2] = -q / rtc.area2
            rtc.q = q        

    def set_fluxes(self, qlist):
        """Set the fluxes for the RTC objects"""
        # q is a list of fluxes for each RTC object
        self.sfx.reset_qext()
        # Loop through all RTC objects and set the fluxes
        for irtc, rtc in enumerate(self.rtc_list):
            # Set the flux for this RTC object
            # Fluxes in qext are in m/s, so we need to convert m3/s to m/s by dividing by the area
            # Determine the direction of the flux
            if qlist[irtc] > 0.0:
                # If the flux is positive, we are discharging from the first point to the second point
                self.sfx.qext[rtc.index1] = -qlist[irtc] / rtc.area1
                self.sfx.qext[rtc.index2] = qlist[irtc]  / rtc.area2
            else:
                # If the flux is negative, we are discharging from the second point to the first point
                self.sfx.qext[rtc.index1] = qlist[irtc] / rtc.area1
                self.sfx.qext[rtc.index2] = -qlist[irtc] / rtc.area2

    def get_zs(self):
        """Get the water levels for all RTC objects. Returns a list of tuples with the water levels for each RTC object."""
        self.sfx.get_zs() # Update water levels in sfx
        zs = []
        for rtc in self.rtc_list:
            # Get the water levels for this RTC object
            z1 = self.sfx.zs[rtc.index1]
            z2 = self.sfx.zs[rtc.index2]
            zs.append((z1, z2))
        return zs

class RTC:
    def __init__(self, sfx, name, type, p1, p2,
                 elevation, width, alpha, qpump, dzpump, zmin, t_relax):
        """"""
        self.sfx = sfx
        self.name = name
        self.type = type
        self.index1 = self.sfx.find_cell(p1[0], p1[1])
        self.area1  = self.sfx.find_cell_area(self.index1)        
        if p2 is None:
            # If p2 is not given, we assume a single point source
            self.index2 = None
            self.area2  = None
        else:        
            self.index2 = self.sfx.find_cell(p2[0], p2[1])
            self.area2  = self.sfx.find_cell_area(self.index2)
        self.zb1    = self.sfx.zb[self.index1]    
        self.qmax   = 1.0e6
        self.qh     = [] # Q-H curve?
        self.qpump  = 0.0
        self.elevation = elevation
        self.width  = width
        self.alpha  = alpha
        self.qpump  = qpump
        self.dzpump = dzpump
        self.zmin   = zmin
        self.q      = 0.0
        self.t_relax = t_relax # Relaxation time (s)

    def compute_flux(self):

        # Get the water levels at the two points
        # zs = self.get_value_ptr("zs")
        z1 = self.sfx.zs[self.index1]
        if self.index2 is None:
            # If p2 is not given, we assume a single point source
            z2 = None
        else:
            z2 = self.sfx.zs[self.index2]

        if self.type == "weir":
            q = self.compute_flux_weir(z1, z2)
        if self.type == "pump":
            # water level z2 not used for pump
            q = self.compute_flux_pump(z1)

        return q    

    def compute_flux_weir(self, z1, z2):

        if z1 < self.elevation and z2 < self.elevation:
            # No flow, both levels below the weir
            return 0.0
        else:
            if z1 > z2:
                zup = z1
                zdown = z2
                direction = 1
            else:
                zup = z2
                zdown = z1
                direction = -1

            # Compute the flux based on very simple weir equation
            zmx = max(zdown, self.elevation) # max of zdown and elevation
            q = direction * self.alpha * self.width * (zup - self.elevation) * (zup - zmx) ** (2 / 3)
            # print(f"zup: {zup}, zdown: {zdown}, q: {q}")

            return q
        
    def compute_flux_pump(self, z1):
        """Compute the flux for a pump"""
        if z1 < self.zmin or z1 < self.zb1 + self.dzpump:
            # No flow, level below the minimum
            return 0.0
        else:
            # Compute the flux based on very simple pump equation. Let pumping gradually increase with water level.
            # fac = min((z1 - self.zmin) / self.dzpump, 1.0)
            fac = 1.0
            return fac * self.qpump  

def interp2(x0, y0, z0, x1, y1, method="linear"):

    # meanx = np.mean(x0)
    # meany = np.mean(y0)
    # x0 -= meanx
    # y0 -= meany
    # x1 -= meanx
    # y1 -= meany

    f = RegularGridInterpolator(
        (y0, x0), z0, bounds_error=False, fill_value=np.nan, method=method
    )
    # reshape x1 and y1
    if x1.ndim > 1:
        sz = x1.shape
        x1 = x1.reshape(sz[0] * sz[1])
        y1 = y1.reshape(sz[0] * sz[1])
        # interpolate
        z1 = f((y1, x1)).reshape(sz)
    else:
        z1 = f((y1, x1))

    return z1
