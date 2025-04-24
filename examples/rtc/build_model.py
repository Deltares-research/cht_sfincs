import datetime

from cht_sfincs import SFINCS

# Build rectangular model, with four thin dams dividing the basin into four sectors
# and one discharge point in the northwest corner. The model is 100x100 m, with a grid spacing of 5 m.
#
# _____________________
# |         |         |
# |         |         |
# |         |         |
# |---------|---------|
# |         |         |
# |         |         |
# |         |         |
# _____________________


sf = SFINCS()
lx = 100.0
ly = 100.0
x0 = 0.0
y0 = 0.0
dx = 5.0
dy = 5.0
nmax = int(lx / dx)
mmax = int(ly / dy)
rot = 0.0
sf.grid.build(x0, y0, nmax, mmax, dx, dy, rot)
sf.grid.set_uniform_bathymetry(-2.0)
sf.mask.build(zmin=-99999.0, zmax=99999.0) # Make whole domain active
sf.grid.write()

# Add some thin dams (divide basin into 4 sectors)
sf.thin_dams.add_xy([50.0, 50.0], [0.0, 100.0])
sf.thin_dams.add_xy([0.0, 100.0], [50.0, 50.0])
sf.input.variables.thdfile = "sfincs.thd"
sf.thin_dams.write()

# Add on discharge point in northwest corner
sf.discharge_points.add_point(5.0, 95.0, "P1", q=1.0)
sf.discharge_points.write()

# Add observation points
sf.observation_points.add_point(25.0, 75.0, "NW")
sf.observation_points.add_point(75.0, 75.0, "NE")
sf.observation_points.add_point(25.0, 25.0, "SW")
sf.observation_points.add_point(75.0, 25.0, "SE")
sf.input.variables.obsfile = "sfincs.obs"
sf.observation_points.write()

# Run for three hours
sf.input.variables.tref   = datetime.datetime(2020, 1, 1, 0, 0, 0)
sf.input.variables.tstart = datetime.datetime(2020, 1, 1, 0, 0, 0)
sf.input.variables.tstop  = datetime.datetime(2020, 1, 1, 3, 0, 0)
sf.input.variables.dtmapout = 60.0
sf.input.variables.dthisout = 60.0
sf.input.variables.advection = 0
sf.input.variables.useqext = 1
sf.input.write()
