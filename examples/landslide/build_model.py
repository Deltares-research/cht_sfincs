import numpy as np
import datetime
import os

from cht_sfincs import SFINCS

sim_path = os.getcwd()

# Create a regular grid of points in the x-y plane with dx and dy is 10.0 m
x = np.arange(0, 1001, 10)
y = np.arange(0, 1001, 10)
xx, yy = np.meshgrid(x, y)
# Elevation z on a 1:10 slope going from 20 m to -20 m
zzini = np.maximum(20.0 - 0.1 * xx, -20.0)

# Define the landslide
x0lnd = 100.0
y0lnd = 500.0
vlnd  = 10.0 # forward motion of landslide
rlnd  = 50.0 # radius of landslide
hlnd  = 10.0 # height of landslide

# Build the model
sf = SFINCS()
x0 = 0.0
y0 = 0.0
nmax = 100
mmax = 100
dx = 10.0
dy = 10.0
rot = 0.0
sf.grid.build(x0, y0, nmax, mmax, dx, dy, rot)
sf.grid.interpolate_bathymetry(x, y, zzini)
sf.mask.build(zmin=-99999.0, zmax=99999.0) # Make whole domain active
sf.grid.write()
sf.input.variables.tref   = datetime.datetime(2020, 1, 1, 0,  0, 0)
sf.input.variables.tstart = datetime.datetime(2020, 1, 1, 0,  0, 0)
sf.input.variables.tstop  = datetime.datetime(2020, 1, 1, 0,  2, 0)
sf.input.variables.dtmapout = 1.0
sf.input.variables.advection = 1
sf.input.variables.store_dynamic_bed_level = True
sf.input.write()
