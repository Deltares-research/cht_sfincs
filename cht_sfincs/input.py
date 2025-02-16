# -*- coding: utf-8 -*-
"""
Created on Sat May 15 08:08:40 2021

@author: ormondt
"""
import os
import datetime
import copy
# from pyproj import CRS

class Variables:
    def __init__(self):
        self.mmax = 0
        self.nmax = 0
        self.dx = 0.1
        self.dy = 0.1
        self.x0 = 0.0
        self.y0 = 0.0
        self.rotation = 0.0
        self.latitude = 0.0
        self.qtrfile = None
        tnow = datetime.datetime.now().replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        self.tref = tnow
        self.tstart = tnow
        self.tstop = tnow + datetime.timedelta(days=1)
        self.tspinup = 3600.0
        self.t0out = None
        self.dtmapout = 3600.0
        self.dthisout = 600.0
        self.dtrstout = 0.0
        self.dtmaxout = 0.0
        self.trstout = -999.0
        self.dtwnd = 1800.0
        self.alpha = 0.5
        self.theta = 1.0
        self.viscosity = False
        self.nuvisc = 0.01
        self.huthresh = 0.01
        self.manning = 0.04
        self.manning_land = 0.04
        self.manning_sea = 0.02
        self.rgh_lev_land = 0.0
        self.zsini = 0.0
        self.qinf = 0.0
        self.igperiod = 120.0
        self.rhoa = 1.25
        self.rhow = 1024.0
        self.dtmax = 999.0
        self.maxlev = 999.0
        self.bndtype = 1
        self.advection = False
        self.baro = False
        self.pavbnd = 101200.0
        self.gapres = 101200.0
        self.advlim = 9999.9
        self.stopdepth = 1000.0
        self.crsgeo = False
        self.epsg = None
        self.wiggle_suppression = False
        self.storevel = False
        self.storemeteo = False
        self.store_dynamic_bed_level = False
        self.utmzone = None

        self.sbgfile = None
        self.depfile = None
        self.mskfile = None
        self.indexfile = None
        self.cstfile = None
        self.bndfile = None
        self.bzsfile = None
        self.bzifile = None
        self.bwvfile = None
        self.bhsfile = None
        #        self.bhifile=None
        #        self.bstfile=None
        self.btpfile = None
        self.bwdfile = None
        self.bdsfile = None
        self.bcafile = None
        self.corfile = None
        self.srcfile = None
        self.disfile = None
        self.inifile = None
        self.ncinifile = None
        self.spwfile = None
        self.amufile = None
        self.amvfile = None
        self.ampfile = None
        self.amprfile = None
        self.wndfile = None
        self.precipfile = None
        self.obsfile = None
        self.crsfile = None
        self.thdfile = None
        self.manningfile = None
        self.scsfile = None
        self.rstfile = None
        self.wfpfile = None
        self.whifile = None
        self.wtifile = None
        self.wstfile = None
        self.wvmfile = None

        self.dtwave               = 1800.0
        self.snapwave             = False
        self.snapwave_gamma       = 0.8
        self.snapwave_dtheta      = 15.0
        self.snapwave_hmin        = 0.1
        self.snapwave_fw0         = 0.01
        self.snapwave_crit        = 0.01
        self.snapwave_igwaves     = True
        self.snapwave_nrsweeps    = 1
        self.storefw              = True

        self.inputformat = "bin"
        self.outputformat = "net"

        self.cdnrb = 3
        self.cdwnd = [0.0, 28.0, 50.0]
        self.cdval = [0.001, 0.0025, 0.0025]


class SfincsInput:
    def __init__(self, sf):
        self.model = sf
        self.variables = Variables()

    def read(self):
        # Reads sfincs.inp

        input_file = os.path.join(self.model.path, "sfincs.inp")

        fid = open(input_file, "r")
        lines = fid.readlines()
        fid.close()
        for line in lines:
            strings = line.split("=")
            if len(strings) == 1:
                # Empty line
                continue
            name = strings[0].strip()
            val = strings[1].strip()
            try:
                # First try to convert to int
                val = int(val)
            except ValueError:
                try:
                    # Now try to convert to float
                    val = float(val)
                except:
                    pass
            if name == "tref":
                try:
                    val = datetime.datetime.strptime(val.rstrip(), "%Y%m%d %H%M%S")
                except:
                    val = None
            if name == "tstart":
                try:
                    val = datetime.datetime.strptime(val.rstrip(), "%Y%m%d %H%M%S")
                except:
                    val = None
            if name == "tstop":
                try:
                    val = datetime.datetime.strptime(val.rstrip(), "%Y%m%d %H%M%S")
                except:
                    val = None

            if hasattr(self.variables, name):
                if type(getattr(self.variables, name)) is bool:
                    if val == 0:
                        val = False       
                    elif val == 1:
                        val = True
                    elif val[0].lower() == "t" or val[0].lower() == "y":
                        val = True
                    elif val[0].lower() == "f" or val[0].lower() == "n":
                        val = False
                    else:
                        # Use default value
                        val = getattr(self.variables, name)

            setattr(self.variables, name, val)
        
        # if self.variables.qtrfile:
        #     self.model.grid_type = "quadtree"
        # else:
        #     self.model.grid_type = "regular"

        if self.variables.sbgfile:
            self.model.bathy_type = "subgrid"
        else:
            self.model.bathy_type = "regular"

        if isinstance(self.variables.cdwnd, str):
            self.variables.cdwnd = [float(x) for x in self.variables.cdwnd.split()]
        if isinstance(self.variables.cdval, str):
            self.variables.cdval = [float(x) for x in self.variables.cdval.split()]

    def write(self):
        # Write sfincs.inp
        input_file = os.path.join(self.model.path, "sfincs.inp")

        # Make some adjustments
        variables = copy.copy(self.variables)

        if self.model.crs.is_geographic:
            variables.crsgeo = 1
            variables.latitude = None

        if len(self.model.boundary_conditions.gdf) == 0:
            variables.bndfile = None
            variables.bzsfile = None

        if len(self.model.observation_points.gdf) == 0:
            variables.obsfile = None

        if self.model.grid.type == "quadtree":
            # Get rid of grid stuff
            variables.x0               = None
            variables.y0               = None
            variables.dx               = None
            variables.dy               = None
            variables.nmax             = None
            variables.mmax             = None
            variables.rotation         = None
            variables.depfile          = None # Depth is stored in the qtr file
        else:
            variables.qtrfile          = None    

        if not self.model.input.variables.snapwave:
            # Get rid of SnapWave stuff
            variables.snapwave             = None
            variables.dtwave               = None
            variables.snapwave_gamma       = None
            variables.snapwave_dtheta      = None
            variables.snapwave_hmin        = None
            variables.snapwave_fw0         = None
            variables.bwvfile              = None
            variables.bhsfile              = None
            variables.btpfile              = None
            variables.bwdfile              = None
            variables.bdsfile              = None
            variables.snapwave_crit        = None
            variables.snapwave_igwaves     = None
            variables.snapwave_nrsweeps    = None
            variables.storefw              = None
            variables.wvmfile = None
        else:    
            if len(self.model.wave_makers.gdf) == 0:
                variables.wvmfile = None

        if variables.sbgfile is not None:
            variables.manning              = None
            variables.manning_land         = None
            variables.manning_sea          = None
            variables.rgh_lev_land         = None

        if not variables.store_dynamic_bed_level:
            variables.store_dynamic_bed_level = None

        fid = open(input_file, "w")
        for key, value in variables.__dict__.items():
            if value is not None:
                if type(value) == "float":
                    string = f"{key.ljust(20)} = {float(value)}\n"
                elif type(value) == "int":
                    string = f"{key.ljust(20)} = {int(value)}\n"
                elif isinstance(value, bool):
                    if value:
                        string = f"{key.ljust(20)} = {int(1)}\n"
                    else:    
                        string = f"{key.ljust(20)} = {int(0)}\n"
                elif type(value) == list:
                    valstr = ""
                    for v in value:
                        valstr += str(v) + " "
                    string = f"{key.ljust(20)} = {valstr}\n"
                elif isinstance(value, datetime.date):
                    dstr = value.strftime("%Y%m%d %H%M%S")
                    string = f"{key.ljust(20)} = {dstr}\n"
                else:
                    string = f"{key.ljust(20)} = {value}\n"
                fid.write(string)
        fid.close()
