from typing import Union
import xarray as xr
import numpy as np
import rioxarray
import rasterio
from pathlib import Path
import matplotlib.pyplot as plt
from pyproj import Transformer
from rasterio.warp import calculate_default_transform, reproject, Resampling
# from rasterio.enums import Resampling as RioResampling
from rasterio.windows import from_bounds


class FloodMap:
    def __init__(self):
    # def __init__(self, topobathy_file: Union[str, Path], index_file: Union[str, Path]):
        """
        Initialize the FloodMap class with topobathy data, and indices.

        Parameters:
        ----------
        topobathy_file : Union[str, Path]
            Topobathy data file (COG).
        index_file : Union[str, Path]
            Indices data file (COG).
        """
        # self.topobathy_file = topobathy_file
        # self.index_file = index_file
        # self.zb = rasterio.open(self.topobathy_file)
        # self.indices = rasterio.open(self.index_file)
        self.topobathy_file = None
        self.index_file = None
        self.zb = None
        self.indices = None
        self.zbmin = 0.0
        self.zbmax = 99999.9
        self.hmin = 0.1
        self.max_pixel_size = 0.0
        self.data_array_name = "water_depth"
        self.cmap = "jet"
        self.cmin = None
        self.cmax = None
        self.ds = xr.Dataset()

    def set_topobathy_file(self, topobathy_file: Union[str, Path]) -> None:
        """
        Set the topobathy file.

        Parameters:
        ----------
        topobathy_file : Union[str, Path]
            Topobathy data file (COG).
        """
        self.topobathy_file = topobathy_file
        self.zb = rasterio.open(self.topobathy_file)

    def set_index_file(self, index_file: Union[str, Path]) -> None:
        """
        Set the index file.

        Parameters:
        ----------
        index_file : Union[str, Path]
            Indices data file (COG).
        """
        self.index_file = index_file
        self.indices = rasterio.open(self.index_file)

    def set_water_level(self, zs: Union[float, np.ndarray]) -> None:
        """
        Set the water level data.

        Parameters:
        ----------
        zs : np.ndarray
            A 1D numpy array containing water level data for each index.
        """            
        self.zs = zs

    def make(self,
        max_pixel_size: float = 0.0,
        bbox=None,
    ) -> xr.Dataset:
        """
        Generate a flood map geotiff (COG) or netCDF file from water level data, topobathy data, and indices.

        Parameters:
        ----------
        output_file : str
            Path to the output file. The file extension determines the format (e.g., ".tif" for GeoTIFF, ".nc" for netCDF).
        zbmin : float, optional
            Minimum allowable topobathy value. Values below this will be masked. Default is 0.0.
        zbmax : float, optional
            Maximum allowable topobathy value. Values above this will be masked. Default is 99999.9.
        hmin : float, optional
            Minimum allowable water depth. Values below this will be masked. Default is 0.1.
        max_pixel_size : float, optional
            Maximum pixel size for the appropriate overview level. If 0.0, no overviews are used. Default is 0.0.
        data_array_name : str, optional
            Name of the data array in the output dataset. Default is "water_depth".

        Returns:
        -------
        xr.Dataset
            An xarray Dataset containing the computed flood map.

        Notes:
        -----
        - The function reads and processes topobathy and indices data, applies masks based on the provided thresholds,
        and computes water depth.
        - The output file can be saved as a GeoTIFF or netCDF file depending on the file extension.
        """

        # First get the overview level (assuming zb is a string or path)
        overview_level = 0

        if max_pixel_size > 0.0:
            overview_level = get_appropriate_overview_level(self.zb, max_pixel_size)

        # Read the data at the specified overview level
        if overview_level == 0:
            zb = rioxarray.open_rasterio(self.zb)
        else:
            zb = rioxarray.open_rasterio(self.zb, overview_level=overview_level)
        # Remove band dimension if it is 1, and squeeze the array to 2D
        if "band" in zb.dims and zb.sizes["band"] == 1:
            zb = zb.squeeze(dim="band", drop=True)
        if bbox is not None:
            zb = zb.rio.clip_box(
                minx=bbox[0], miny=bbox[1], maxx=bbox[2], maxy=bbox[3]
            )

        # Read the data at the specified overview level
        if overview_level == 0:
            indices = rioxarray.open_rasterio(self.indices)
        else:
            # Read the data at the specified overview level
            indices = rioxarray.open_rasterio(self.indices, overview_level=overview_level)
        # Remove band dimension if it is 1, and squeeze the array to 2D
        if "band" in indices.dims and indices.sizes["band"] == 1:
            indices = indices.squeeze(dim="band", drop=True)
        if bbox is not None:
            indices = indices.rio.clip_box(
                minx=bbox[0], miny=bbox[1], maxx=bbox[2], maxy=bbox[3]
            )

        # Get the no_data value from the indices array
        nan_val_indices = indices.attrs["_FillValue"]
        # Set the no_data mask
        no_data_mask = indices == nan_val_indices
        # Turn indices into numpy array and set no_data values to 0
        indices = np.squeeze(indices.values[:])
        indices[np.where(indices == nan_val_indices)] = 0

        # Compute water depth
        if isinstance(self.zs, float):
            # If zs is a float, use constant water level
            h = np.full(zb.shape, self.zs) - zb.values[:]
        else:
            h = self.zs[indices] - zb.values[:]
        # Set water depth to NaN where indices are no data
        h[no_data_mask] = np.nan
        # Set water depth to NaN where it is less than hmin
        h[h < self.hmin] = np.nan
        # Set water depth to NaN where zb is less than zbmin
        h[zb.values[:] < self.zbmin] = np.nan
        # Set water depth to NaN where zb is greater than zbmax
        h[zb.values[:] > self.zbmax] = np.nan

        # Turn h into a DataArray with the same dimensions as zb
        self.ds = xr.Dataset()
        self.ds[self.data_array_name] = xr.DataArray(h, dims=["y", "x"], coords={"y": zb.y, "x": zb.x})
        # Use same spatial_ref as zb
        self.ds[self.data_array_name] = self.ds[self.data_array_name].rio.write_crs(zb.rio.crs, inplace=True)

    def write(
        self,
        output_file: Union[str, Path] = "",
    ) -> None:
        """
        Write the flood map to a file.

        Parameters:
        ----------
        output_file : Union[str, Path]
            Path to the output file. The file extension determines the format (e.g., ".tif" for GeoTIFF, ".nc" for netCDF).

        Returns:
        -------
        None

        Notes:
        -----
        - If the output file is a NetCDF file (".nc"), the dataset is written directly without applying a colormap.
        - If the output file is a GeoTIFF (".tif") and a colormap is provided, the data is normalized and the colormap is applied.
        - If no colormap is provided for a GeoTIFF, the raw data is written as a binary raster.
        """
        if output_file.endswith(".nc"):
            # Write to netcdf
            self.ds.to_netcdf(output_file)

        elif output_file.endswith(".tif"):
            # Write to geotiff
            if self.cmap is not None:

                # Get RBG data array
                rgb_da = get_rgb_data_array(
                    self.ds[self.data_array_name], cmap=self.cmap, cmin=self.cmin, cmax=self.cmax
                )

                # # Load and squeeze to 2D if needed
                # da = self.ds[self.data_array_name].squeeze()

                # if cmin is None:
                #     cmin = da.min()
                # if cmax is None:
                #     cmax = da.max()
                # # Ensure cmin and cmax are not equal to avoid division by zero
                # if cmin == cmax:
                #     cmin = cmax - 1.0
                #     cmax = cmax + 1.0

                # # Normalize to [0, 1]
                # normed = (da - cmin) / (cmax - cmin)

                # # Get colormap
                # cmap = plt.get_cmap(cmap)

                # # Apply colormap (returns RGBA)
                # rgba = cmap(normed)

                # # Convert to 8-bit RGB and drop alpha
                # rgb = (rgba[:, :, :3] * 255).astype("uint8")

                # # Convert to DataArray with 'band' dimension
                # rgb_da = xr.DataArray(
                #     np.moveaxis(rgb, -1, 0),  # shape: (3, height, width)
                #     dims=("band", "y", "x"),
                #     coords={"band": [1, 2, 3], "y": da.y, "x": da.x},
                #     attrs=da.attrs,
                # )

                # # Add CRS and transform
                # rgb_da.rio.write_crs(da.rio.crs, inplace=True)
                # rgb_da.rio.write_transform(da.rio.transform(), inplace=True)

                # Write to file
                rgb_da.rio.to_raster(
                    output_file,
                    driver="COG",
                    compress="deflate",  # or "lzw"
                    blocksize=512,       # optional tuning
                    overview_resampling="nearest",  # controls how overviews are built
                )

            else:
                # Just write binary data
                self.ds[self.data_array_name].rio.to_raster(
                    output_file,
                    driver="COG",
                    compress="deflate",  # or "lzw"
                    blocksize=512,       # optional tuning
                    overview_resampling="nearest",  # controls how overviews are built
                )

    def map_overlay(self, file_name, xlim=None, ylim=None, width=800):
        """
        Create a map overlay of the flood map using the specified colormap and save it to a png file. The CRS is 3857.
        """

        if self.ds is None:
            return  
        
        try:

            # Get the bounds of the data
            lon_min = xlim[0]
            lat_min = ylim[0]
            lon_max = xlim[1]
            lat_max = ylim[1]

            # Get the bounds of the data in EPSG:3857
            transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
            x_min, y_min = transformer.transform(lon_min, lat_min)
            x_max, y_max = transformer.transform(lon_max, lat_max)

            # Get required pixel size
            dxy = (x_max - x_min) / width

            # Get bbox in local crs from xlim and ylim
            bbox = reproject_bbox(
                lon_min, lat_min, lon_max, lat_max,
                crs_src="EPSG:4326",
                crs_dst=self.zb.crs,
                buffer=0.05
            )

            self.make(max_pixel_size=dxy, bbox=bbox) 
                    
            rgb_da = get_rgb_data_array(
                self.ds[self.data_array_name], cmap=self.cmap, cmin=self.cmin, cmax=self.cmax)
            
            # Now reproject to EPSG:3857 and create a png file
            rgb_3857 = rgb_da.rio.reproject("EPSG:3857", resampling=Resampling.bilinear, nodata=0) # can also use nearest

            # Apply padding
            rgb_3857 = rgb_3857.rio.pad_box(minx=x_min, miny=y_min, maxx=x_max, maxy=y_max, constant_values=0)

            # Final clip to exact bbox
            rgb_crop = rgb_3857.rio.clip_box(minx=x_min, miny=y_min, maxx=x_max, maxy=y_max)

            # Convert to numpy array and transpose to (y, x, band)
            rgba = rgb_crop.transpose("y", "x", "band").values.astype("uint8")

            plt.imsave(file_name, rgba)
            
            return True

        except Exception as e:
            return False



def get_appropriate_overview_level(
    src: rasterio.io.DatasetReader, max_pixel_size: float
) -> int:
    """
    Given a rasterio dataset `src` and a desired `max_pixel_size`,
    determine the appropriate overview level (zoom level) that fits
    the maximum resolution allowed by `max_pixel_size`.

    Parameters:
    src (rasterio.io.DatasetReader): The rasterio dataset reader object.
    max_pixel_size (float): The maximum pixel size for the resolution.

    Returns:
    int: The appropriate overview level.
    """
    # Get the original resolution (pixel size) in terms of x and y
    original_resolution = src.res  # Tuple of (x_resolution, y_resolution)
    if src.crs.is_geographic:
        original_resolution = (
            original_resolution[0] * 111000,
            original_resolution[1] * 111000,
        )  # Convert to meters
    # Get the overviews for the dataset
    overview_levels = src.overviews(
        1
    )  # Overview levels for the first band (if multi-band, you can adjust this)

    # If there are no overviews, return 0 (native resolution)
    if not overview_levels:
        return 0

    # Calculate the resolution for each overview by multiplying the original resolution by the overview factor
    resolutions = [
        (original_resolution[0] * factor, original_resolution[1] * factor)
        for factor in overview_levels
    ]

    # Find the highest overview level that is smaller than or equal to the max_pixel_size
    selected_overview = 0
    for i, (x_res, y_res) in enumerate(resolutions):
        if x_res <= max_pixel_size and y_res <= max_pixel_size:
            selected_overview = i
        else:
            break

    return selected_overview

def get_rgb_data_array(
    da: xr.DataArray, cmap: str, cmin: float = None, cmax: float = None
) -> xr.DataArray:
    """
    Convert a DataArray to RGB using a colormap.

    Parameters:
    ----------
    da : xr.DataArray
        The input DataArray to be converted.
    cmap : str
        The colormap to use (e.g., 'viridis').
    cmin : float, optional
        Minimum value for normalization. If None, the minimum value of the data is used.
    cmax : float, optional
        Maximum value for normalization. If None, the maximum value of the data is used.

    Returns:
    -------
    xr.DataArray
        The RGB DataArray.
    """
    # Normalize the data
    if cmin is None:
        cmin = da.min()
    if cmax is None:
        cmax = da.max()

    # Ensure cmin and cmax are not equal to avoid division by zero
    if cmin == cmax:
        cmin = cmax - 1.0
        cmax = cmax + 1.0

    # Normalize to [0, 1]
    normed = (da - cmin) / (cmax - cmin)

    # Get colormap
    cmap = plt.get_cmap(cmap)

    # Apply colormap (returns RGBA)
    rgba = cmap(normed)

    # Convert to 8-bit RGB and drop alpha
    rgba = (rgba[:, :, :] * 255).astype("uint8")

    # Convert to DataArray with 'band' dimension
    rgb_da = xr.DataArray(
        np.moveaxis(rgba, -1, 0),  # shape: (3, height, width)
        dims=("band", "y", "x"),
        coords={"band": [0, 1, 2, 3], "y": da.y, "x": da.x},
        attrs=da.attrs,
    )

    rgb_da.rio.write_crs(da.rio.crs, inplace=True)

    return rgb_da

def reproject_bbox(xmin, ymin, xmax, ymax, crs_src, crs_dst, buffer=0.0):

    transformer = Transformer.from_crs(crs_src, crs_dst, always_xy=True)

    # Buffer the bounding box
    dx = (xmax - xmin) * buffer
    dy = (ymax - ymin) * buffer
    xmin -= dx
    xmax += dx
    ymin -= dy
    ymax += dy

    # Transform all four corners
    x0, y0 = transformer.transform(xmin, ymin)
    x1, y1 = transformer.transform(xmax, ymin)
    x2, y2 = transformer.transform(xmax, ymax)
    x3, y3 = transformer.transform(xmin, ymax)

    # New bounding box
    xs = [x0, x1, x2, x3]
    ys = [y0, y1, y2, y3]

    return min(xs), min(ys), max(xs), max(ys)
