import numpy as np
import os

from cht_sfincs.xmi import SfincsXmi


# Set the sfincs dll path
dll_path = "c:/work/checkouts/git/sfincs/source/sfincs_dll/x64/Release/sfincs_dll.dll"
sim_path = os.getcwd()

# Make list with RTC objects (note that these are not directly added to the SFINCS model!)
# There are three weirs and one pump in the model. Note that the thin dams were added before. Probably better to also add them here.
rtc_list = []
rtc_list.append({"name": "weir1", "type": "weir", "p1": (25.0, 55.0), "p2": (25.0, 45.0), "elevation": 0.2, "alpha": 0.5, "width": 20.0, "t_relax": 60.0})
rtc_list.append({"name": "weir2", "type": "weir", "p1": (45.0, 25.0), "p2": (55.0, 25.0), "elevation": 0.1, "alpha": 0.5, "width": 10.0, "t_relax": 60.0})
rtc_list.append({"name": "weir3", "type": "weir", "p1": (75.0, 45.0), "p2": (75.0, 55.0), "elevation": 0.0, "alpha": 0.5, "width": 5.0, "t_relax": 60.0})
# Add pump that starts pumping water out of NE quadrant at rate of 1 m3/s if water level exceeds 0.2 m
rtc_list.append({"name": "pump1", "type": "pump", "p1": (75.0, 75.0), "qpump": 1.0, "zmin": 0.2, "t_relax": 300.0})

# Update RTC fluxes every 10 seconds
dt_rtc = 10.0  

# Create the SFINCS XMI object
sfx = SfincsXmi(dll_path, sim_path)

# Now start the simulation! sfx.initialize() and sfx.get_domain() MUST bed called before initializing the rtc_collection!

# Reads input and initializes domain
sfx.initialize() 
sfx.get_domain() # Get domain coords and bed levels (not actually necessary for this test)
sfx.rtc_collection.initialize(rtc_list)

# Set the times
end_time = sfx.get_end_time()
ntimes = int(end_time / dt_rtc) 
times = np.linspace(0.0, end_time, ntimes + 1, dtype=float)

# Loopt through XMI coupling times
for t in times[1:]:

    # Update fluxes through RTC objects
    sfx.rtc_collection.compute_fluxes(dt=dt_rtc)

    # If the fluxes are computed by another XMI model (let's call it d3dx), we can do something like this:
    # zsrtc = sfx.rtc_collection.get_zs() # Get water levels from RTC objects. This return a list of tuples with the water levels for each RTC object.
    # Somehow set the water levels in d3dx (or whatever model is used to compute the fluxes)
    # d3dx.update_until(t)    
    # # Somehow get list of fluxes (qlist) from d3dx (or whatever model is used to compute the fluxes)
    # e.g. qlist = [10.0, -5.0, 20.0, 0.0] # Example list of fluxes for each RTC object
    # sfx.rtc_collection.set_fluxes(qlist)

    sfx.update_until(t)

sfx.finalize()
