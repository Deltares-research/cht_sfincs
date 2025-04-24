import numpy as np
import os

from cht_sfincs.xmi import SfincsXmi

def landslide(x0, y0, r, h):
    # Landslide is shaped as a gaussian hill
    x = np.arange(-300.0, 300.0, 10.0)
    y = np.arange(-300.0, 300.0, 10.0)
    xx, yy = np.meshgrid(x, y)
    z = h * np.exp(-(xx**2 + yy**2) / r**2)
    return x + x0, y + y0, z

dll_path = "c:/work/checkouts/git/sfincs/source/sfincs_dll/x64/Release/sfincs_dll.dll"
sim_path = os.getcwd()

# Define the landslide
x0lnd = 100.0
y0lnd = 500.0
vlnd  = 10.0 # forward motion of landslide
rlnd  = 50.0 # radius of landslide
hlnd  = 10.0 # height of landslide

# Create the SFINCS XMI object
sfx = SfincsXmi(dll_path, sim_path)

sfx.initialize() # Reads input and initializes domain
sfx.get_domain() # Get domain coords and bed levels

zb0 = sfx.zb.copy() # Get initial bed level (without landslide)

# Compute initial landslide mesh
xlnd0, ylnd0, zlnd = landslide(x0lnd, y0lnd, rlnd, hlnd)

# Change initial bed level in SFINCS, but do not update water level
sfx.set_bed_level(x=xlnd0,
                  y=ylnd0,
                  z=zlnd,
                  zb0=zb0,
                  update_water_level=False)

# Set the initial time
t = 0.0

# Get the end time
end_time = sfx.get_end_time()

while t < end_time:

    # Calculate the new landslide position (by just moving it forward)

    if t<30.0:

        xlnd = xlnd0 + vlnd * t           # x position of landslide
        ylnd = ylnd0                      # y position does not change

        # Change bed level in SFINCS
        sfx.set_bed_level(x=xlnd,
                          y=ylnd,
                          z=zlnd,
                          zb0=zb0,
                          update_water_level=True)

    # Run a time step (returns new time)
    t = sfx.run_timestep()

sfx.finalize()
