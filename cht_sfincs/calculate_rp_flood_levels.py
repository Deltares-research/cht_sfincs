import os
import xugrid as xu
import xarray as xr
import numpy as np
import pandas as pd
from numpy import matlib

def calculate_rp_flood_levels(sim_paths, frequencies, flood_map_rp, result_path):
    """Calculate flood risk maps from a set of (currently) SFINCS water level outputs using linear interpolation.

    It would be nice to make it more widely applicable and move the loading of the SFINCS results to self.postprocess_sfincs().

    Generates return period water level maps in netcdf format

    """

    zs_maps = []
    map_file = os.path.join(sim_paths[0], "sfincs_map.nc")
    dsin = xu.open_dataset(map_file)
    zb = dsin["zb"].load()
    dsin.close()

    for simulation_path in sim_paths:
        # read zsmax data from overland sfincs model
        map_file = os.path.join(simulation_path, "sfincs_map.nc")
        dsin = xr.open_dataset(map_file)
        zsmax = dsin["zsmax"].max("timemax").load()
        dsin.close()
        zs_maps.append(zsmax)

    # Create RP flood maps

    # 1a: make a table of all water levels and associated frequencies
    zs = xr.concat(zs_maps, pd.Index(frequencies, name="frequency"))
    # Get the indices of columns with all NaN values
    nan_cells = np.where(np.all(np.isnan(zs), axis=0))[0]
    # fill nan values with minimum bed levels in each grid cell, np.interp cannot ignore nan values
    zs = xr.where(np.isnan(zs), np.tile(zb, (zs.shape[0], 1)), zs)
    # Get table of frequencies
    freq = np.tile(frequencies, (zs.shape[1], 1)).transpose()

    # 1b: sort water levels in descending order and include the frequencies in the sorting process
    # (i.e. each h-value should be linked to the same p-values as in step 1a)
    sort_index = zs.argsort(axis=0)
    sorted_prob = np.flipud(np.take_along_axis(freq, sort_index, axis=0))
    sorted_zs = np.flipud(np.take_along_axis(zs.values, sort_index, axis=0))

    # 1c: Compute exceedance probabilities of water depths
    # Method: accumulate probabilities from top to bottom
    prob_exceed = np.cumsum(sorted_prob, axis=0)

    # 1d: Compute return periods of water depths
    # Method: simply take the inverse of the exceedance probability (1/Pex)
    rp_zs = 1.0 / prob_exceed

    # For each return period (T) of interest do the following:
    # For each grid cell do the following:
    # Use the table from step [1d] as a “lookup-table” to derive the T-year water depth. Use a 1-d interpolation technique:
    # h(T) = interp1 (log(T*), h*, log(T))
    # in which t* and h* are the values from the table and T is the return period (T) of interest
    # The resulting T-year water depths for all grids combined form the T-year hazard map
    rp_da = xr.DataArray(rp_zs, dims=zs.dims)

    h = matlib.repmat(
        np.copy(zb), len(flood_map_rp), 1
    )  # if not flooded (i.e. not in valid_cells) revert to bed_level, read from SFINCS results so it is the minimum bed level in a grid cell

    for jj in range(np.shape(h)[1]):  # looping over all non-masked cells.
        # linear interpolation for all return periods to evaluate
        h[:, jj] = np.interp(
            np.log10(flood_map_rp),
            np.log10(rp_da[::-1, jj]),
            sorted_zs[::-1, jj],
            left=0,
        )

    # Re-fill locations that had nan water level for all simulations with nans
    h[:, nan_cells] = np.full(h[:, nan_cells].shape, np.nan)

    # If a cell has the same water-level as the bed elevation it should be dry (turn to nan)
    diff = h - np.tile(zb, (h.shape[0], 1))
    dry = (
        diff < 10e-10
    )  # here we use a small number instead of zero for rounding errors
    h[dry] = np.nan

    for ii, rp in enumerate(flood_map_rp):
        print(f"Writing {result_path}/RP_{rp:05d}.nc")
        zs_rp_single = xr.DataArray(
            data=h[ii, :], attrs={"units": "meters"}
        ).unstack()
        zs_rp_single = zs_rp_single.to_dataset(name="risk_map")
        fn_rp = os.path.join(result_path, f"RP_{rp:05d}.nc")
        zs_rp_single.to_netcdf(fn_rp)
