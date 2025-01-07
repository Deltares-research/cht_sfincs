import numpy as np

from cht_sfincs import SFINCS
from cht_sfincs.xmi import SfincsXmi

def landslide(x0, y0, r, h, xx, yy):
    # Landslide is shaped as a gaussian hill
    return h * np.exp(-((xx - x0)**2 + (yy - y0)**2) / r**2)

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

zlnd = landslide(x0lnd, y0lnd, rlnd, hlnd, xx, yy)

# Initial bed level (flat slope with a gaussian hill)
zz0  = zzini + zlnd

# Read the sfincs model
sf = SFINCS(mode="r", root="c:/work/projects/sfincs/landslides")

# Alternatively, the model can be created with the following code:
# sf = SFINCS()
# x0 = 0.0
# y0 = 0.0
# nmax = 100
# mmax = 100
# dx = 10.0
# dy = 10.0
# rot = 0.0
# sf.grid.build(x0, y0, nmax, mmax, dx, dy, rot)
# sf.grid.interpolate_bathymetry(x, y, zz0)
# sf.mask.build(zmin=-99999.0, zmax=99999.0) # Make whole domain active
# sf.grid.write()
# sf.input.variables.tref   = datetime.datetime(2020, 1, 1, 0,  0, 0)
# sf.input.variables.tstart = datetime.datetime(2020, 1, 1, 0,  0, 0)
# sf.input.variables.tstop  = datetime.datetime(2020, 1, 1, 0,  2, 0)
# sf.input.variables.dtmapout = 1.0
# sf.input.variables.advection = 1
# sf.input.variables.store_dynamic_bed_level = True
# sf.input.write()

# Create the SFINCS XMI object
dll_path = "c:/work/checkouts/git/sfincs/source/sfincs_dll/x64/Release/sfincs_dll.dll"
sfx = SfincsXmi(sf, dll_path)
sfx.initialize() # Reads input and initializes domain
sfx.get_domain() # Get domain coords and bed levels

# Change initial bed level in SFINCS
sfx.set_bed_level(x=x,
                  y=y,
                  z=zz0,
                  update_water_level=False)

# Set the initial time
t = 0.0
# Get the end time
end_time = sfx.get_end_time()

while t < end_time:

    # Calculate the landslide position
    xlnd = x0lnd + vlnd * t           # x position of landslide
    xlnd = np.minimum(xlnd, 400.0)    # stop at x = 400.0
    ylnd = y0lnd                      # y position of landslide
    h    = hlnd                       # height of landslide

    # Calculate the landslide (gaussian hill)
    zlnd = landslide(xlnd, ylnd, rlnd, h, xx, yy)

    # Add the landslide to the initial elevation
    zz = zzini + zlnd

    # Change bed level in SFINCS
    sfx.set_bed_level(x=x,
                      y=y,
                      z=zz,
                      update_water_level=True)

    # Run a time step (returns new time)
    t = sfx.run_timestep()

sfx.finalize()
