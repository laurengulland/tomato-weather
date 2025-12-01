# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "affine==2.4.0",
#     "aiohttp==3.13.2",
#     "dash==3.3.0",
#     "dash-html-components==2.0.0",
#     "dash-leaflet==1.1.3",
#     "geopandas==1.1.1",
#     "ipython==9.7.0",
#     "mapclassify==2.10.0",
#     "marimo",
#     "matplotlib==3.10.7",
#     "numpy==2.2.6",
#     "pandas==2.3.3",
#     "plotly==6.5.0",
#     "pyproj==3.7.2",
#     "rasterio==1.4.3",
#     "rioxarray==0.20.0",
#     "shapely==2.1.2",
#     "xarray[complete]==2025.11.0",
#     "xvec==0.5.2",
#     "zarr==3.1.5",
# ]
# ///

import marimo

__generated_with = "0.18.1"
app = marimo.App(
    css_file="/usr/local/_marimo/custom.css",
    auto_download=["html"],
)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # NOAA HRRR 48 hour forecasts - interactive notebook
    Utilizing the zarr dataset provided by dynamical.org.

    Dataset documentation: https://dynamical.org/catalog/noaa-hrrr-forecast-48-hour/
    """)
    return


@app.cell
def _():
    # Even if xarray is installed in molab:
    # Make sure to explicitly install xarray[complete] !
    return


@app.cell
def _(mo):
    display_timezone = mo.ui.dropdown(["UTC","America/Chicago"], value="America/Chicago")

    mo.md(
        f"""
        Display timezone: {display_timezone}
        """
    )
    return (display_timezone,)


@app.cell
def _(display_timezone, ds, make_time_str, mo):
    init_time_options = [make_time_str(t, display_timezone.value) for t in ds.init_time.values[-20:]]
    init_time_dropdown = mo.ui.dropdown(init_time_options, value="2025-11-29 00:00")

    mo.md(
        f"""
        Init time input: {init_time_dropdown}
        """
    )
    return (init_time_dropdown,)


@app.cell
def _(mo):

    lead_time_slider = mo.ui.slider(start=1, stop=48, step=1, value=7, show_value=True)

    mo.md(
        f"""
        Lead time input (hrs): {lead_time_slider}
        """
    )
    return (lead_time_slider,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## TODO: location picker
    defaults, plus custom any lat/long input?

    See this marimo example for getting input: https://eoda-dev.github.io/py-openlayers/marimo/getting-started.html
    """)
    return


@app.cell
def _():
    riverwoods_x = -87.90219091178325
    riverwoods_y = 42.17245575311014
    return riverwoods_x, riverwoods_y


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## TODO: zoom level slider
    add zoom level slider to affect +/- around the point location to clip to -- but add a little extra and then clip down after reprojection so it's a box again.
    """)
    return


@app.cell
def _():
    zoom_level = 0.75 # degrees around center to add
    return (zoom_level,)


@app.cell
def _():
    # class ZoomLevels(Enum):
    #     Point = 0
    #     City = 2


    # zoom_level = mo.ui.dropdown(, value=1, show_value=True)

    # mo.md(
    #     f"""
    #     Zoom Level: {zoom_level}
    #     """
    # )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Plotting HRRR directly (og coords)
    """)
    return


@app.cell
def _(xr):
    DATASET_LINK = "https://data.dynamical.org/noaa/hrrr/forecast-48-hour/latest.zarr?email=laurengulland@gmail.com"

    ds = xr.open_zarr(
        DATASET_LINK,
        chunks=None
    )
    return DATASET_LINK, ds


@app.cell
def _(
    display_timezone,
    ds,
    init_time_dropdown,
    lead_time_slider,
    make_time_str,
    mo,
    pd,
    plt,
    riverwoods_x,
    riverwoods_y,
    zoom_level,
):
    utc_init_time = pd.Timestamp(init_time_dropdown.value).tz_localize(display_timezone.value).tz_convert("UTC").tz_localize(None)

    clip_ds = ds.sel(init_time=utc_init_time, lead_time=pd.Timedelta(hours=lead_time_slider.value), method="nearest").rio.clip_box(riverwoods_x-zoom_level, riverwoods_y-zoom_level, riverwoods_x+zoom_level, riverwoods_y+zoom_level, crs="EPSG:4326")["precipitation_surface"]


    fig, ax = plt.subplots()
    clip_ds.plot(ax=ax)
    valid_time_str = make_time_str(clip_ds.valid_time.values, tzinfo=display_timezone.value)
    ax.set_title(f"Total precip in Riverwoods at {valid_time_str}")
    mo.mpl.interactive(fig)
    return clip_ds, fig


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Solving the issue with reprojecting original ds to WGS84 turning it to nan
    """)
    return


@app.cell
def _(clip_ds, display_timezone, make_plot, make_time_str, mo, np):
    # Intermediate reprojection (tldr? Conic -> Ellipsoid?)
    ds_conic = clip_ds.rio.reproject("EPSG:5070", nodata=np.nan)
    # Reprojection to WGS84 (tldr? Ellipsoid to different ellipsoid)
    ds_wgs84 = ds_conic.rio.reproject("EPSG:4326", nodata=np.nan)

    title_str = f"Total precip in Riverwoods at {make_time_str(ds_wgs84.valid_time.values, tzinfo=display_timezone.value)}"

    fig_2 = make_plot(ds_wgs84, title_str)
    mo.mpl.interactive(fig_2)
    return (ds_wgs84,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Attempting to get osm features that I can aggregate over
    https://osmnx.readthedocs.io/en/stable/user-reference.html#module-osmnx.features

    https://github.com/gboeing/osmnx-examples/blob/main/notebooks/00-osmnx-features-demo.ipynb

    https://github.com/gboeing/osmnx-examples/blob/aefc513ed6d9425641bada8d61c2b0f32124a2f2/notebooks/16-download-osm-geospatial-features.ipynb

    https://www.openstreetmap.org/relation/122942#map=11/42.0916/-87.8199

    https://wiki.openstreetmap.org/wiki/Tag:boundary%3Dadministrative#Country_specific_values_%E2%80%8B%E2%80%8Bof_the_key_admin_level=*
    """)
    return


@app.cell
def _():
    # bbox = () # left, bottom, right, top, as lat/long (EPSG:4326)
    # chicago_geoms = ox.features_from_bbox(bbox, tags={"admin_level": "8"})
    # chicago_geoms
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Attempting to convert ds to geodataframe that's plottable
    """)
    return


@app.cell
def _(ds_wgs84, riverwoods_x, riverwoods_y, zoom_level):
    ds_wgs84_clip = ds_wgs84.rio.clip_box(riverwoods_x-zoom_level*.9, riverwoods_y-zoom_level*.9, riverwoods_x+zoom_level*.9, riverwoods_y+zoom_level*.9)
    return (ds_wgs84_clip,)


@app.cell
def _(gpd, test_df):
    gpd.points_from_xy(test_df.x, test_df.y)
    return


@app.cell
def _(ds_wgs84_clip, gpd):
    test_df = ds_wgs84_clip.to_dataframe().reset_index().drop(columns=["ingested_forecast_length", "expected_forecast_length", "spatial_ref", "lead_time"])
    gdf = gpd.GeoDataFrame(test_df, geometry=gpd.points_from_xy(test_df.x, test_df.y)).set_crs("EPSG:4326")
    gdf
    return gdf, test_df


@app.cell
def _(Polygon, ds_wgs84_clip, gpd, np, test_df):
    test_df_polygons = test_df.copy()
    # Get the coordinate arrays (assumes they are 1D and sorted)
    lats = ds_wgs84_clip['y'].values
    lons = ds_wgs84_clip['x'].values

    # 2. Calculate Half-Resolution / Cell Size
    # Calculate the difference between adjacent coordinates
    # We use numpy.diff and append the last value to keep the array the same size
    # This works for uniform grids, which is standard for rasters.
    lon_diff = np.diff(lons)
    lat_diff = np.diff(lats)

    # Calculate half-step for both dimensions
    half_lon_step = np.mean(np.abs(lon_diff)) / 2.0
    half_lat_step = np.mean(np.abs(lat_diff)) / 2.0

    # 3. Calculate Corner Coordinates
    # Apply the half-step offset to the center coordinates (lat, lon) to get the boundaries
    # The arrays are ordered from min to max, so 'min' is the starting boundary.
    test_df_polygons['lon_min'] = test_df_polygons['x'] - half_lon_step
    test_df_polygons['lon_max'] = test_df_polygons['x'] + half_lon_step
    test_df_polygons['lat_min'] = test_df_polygons['y'] - half_lat_step
    test_df_polygons['lat_max'] = test_df_polygons['y'] + half_lat_step

    # 4. Create Shapely Polygons
    # Create the Polygon object for each row/pixel.
    # Vertices must be defined in order: (min_lon, min_lat), (max_lon, min_lat), 
    # (max_lon, max_lat), (min_lon, max_lat), and back to the start.
    test_df_polygons['geometry'] = test_df_polygons.apply(
        lambda row: Polygon([
            (row['lon_min'], row['lat_min']),
            (row['lon_max'], row['lat_min']),
            (row['lon_max'], row['lat_max']),
            (row['lon_min'], row['lat_max'])
        ]), 
        axis=1
    )

    # 5. Create GeoDataFrame
    # 'value' is your original raster data, 'geometry' is the new polygon column
    gdf_polygons = gpd.GeoDataFrame(
        test_df_polygons[['x', 'y', 'precipitation_surface', 'geometry']], 
        crs="EPSG:4326"
    )

    # Drop the center coordinates if you only need the geometry
    gdf_polygons = gdf_polygons.drop(columns=['x', 'y'])

    print(gdf_polygons.iloc[0].geometry)
    return (gdf_polygons,)


@app.cell
def _(gdf):
    print(gdf.iloc[0].geometry)
    return


@app.cell
def _(gdf):
    gdf.explore("precipitation_surface")
    return


@app.cell
def _(gdf_polygons):
    map = gdf_polygons.explore("precipitation_surface",style_kwds={"stroke":False})
    map
    return (map,)


@app.cell
def _(map):
    type(map)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Attempting to plot with Canvas/datashader
    See this suuuuuper helpful link: https://martinfleischmann.net/sds/raster_data/hands_on.html
    """)
    return


@app.cell(disabled=True, hide_code=True)
def _(datashader):
    canvas = datashader.Canvas(plot_width=600, plot_height=600)
    canvas.raster()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Attempting to add map to it
    """)
    return


@app.cell
def _(go, mo, riverwoods_x, riverwoods_y):

    map_fig = go.Figure(go.Scattermap(
            lat=[riverwoods_y],
            lon=[riverwoods_x],
            mode='markers',
            marker=go.scattermap.Marker(
                size=14, color="pink"
            ),
            text=['Riverwoods'],
        ))

    map_fig.update_layout(
        hovermode='closest',
        map=dict(
            bearing=0,
            center=go.layout.map.Center(
                lat=riverwoods_y,
                lon=riverwoods_x
            ),
            pitch=0,
            zoom=10
        )
    )

    mo.ui.plotly(map_fig)
    # map_fig.show()
    return


@app.cell
def _(fig):
    dir(fig)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Helpers
    """)
    return


@app.cell
def _():
    import pandas as pd
    import rioxarray
    from IPython.display import HTML
    from matplotlib.animation import FuncAnimation
    import matplotlib.pyplot as plt
    import plotly.graph_objects as go
    from pyproj import Transformer
    import marimo as mo
    import xarray as xr
    import zarr
    import datetime
    import rasterio
    from rasterio import crs
    from rasterio.vrt import WarpedVRT
    import numpy as np
    from pyproj import CRS
    from affine import Affine
    import geopandas as gpd
    import xvec
    from shapely.geometry import Polygon
    return Affine, CRS, Polygon, go, gpd, mo, np, pd, plt, rasterio, xr


@app.cell
def _(pd):
    def make_time_str(t, tzinfo=None):
        ts = pd.Timestamp(t)
        if tzinfo is not None:
            ts = ts.tz_localize("UTC").tz_convert(tzinfo)
        return ts.strftime("%Y-%m-%d %H:%M")
    return (make_time_str,)


@app.cell
def _(plt):
    def make_plot(ds, title_str):

        fig, ax = plt.subplots()
        ds.plot(ax=ax)
        ax.set_title(title_str)
        return fig
    return (make_plot,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Attempting to Warp HRRR ds to WGS84 (unsuccessful so far)
    """)
    return


@app.cell(disabled=True, hide_code=True)
def _(DATASET_LINK, ds, rasterio):
    with rasterio.open(DATASET_LINK) as src:
        print(src.crs)
        # with WarpedVRT(src,resampling=1,src_crs=src.crs,crs=crs.CRS.from_epsg("EPSG:4326")) as vrt:
        #         print('Destination CRS:' +str(vrt.crs))
        #         ds = rioxarray.open_rasterio(vrt).chunk(ds_original_projection.chunks).to_dataset(name='hrrr')
    ds
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Stuff that didn't work to transform to wgs84
    """)
    return


@app.cell(disabled=True, hide_code=True)
def _(clip_ds, correct_crs):
    rewritten_clip_ds = clip_ds.rio.write_crs(correct_crs)
    rewritten_clip_ds.rio.crs
    return


@app.cell(hide_code=True)
def _(CRS):
    # 1. Define the correct LCC PROJ string based on your WKT
    # Key parameters:
    # +proj=lcc (Lambert Conformal Conic)
    # +lat_1=38.5 +lat_2=38.5 (Standard Parallels)
    # +lat_0=38.5 (Latitude of Origin)
    # +lon_0=-97.5 (Central Meridian)
    # +a=6371229 +b=6371229 (Sphere Radius)
    # +units=m (Units are Metres)
    proj4_string = "+proj=lcc +lat_1=38.5 +lat_2=38.5 +lat_0=38.5 +lon_0=-97.5 +a=6371229 +b=6371229 +units=m +no_defs"

    # 2. Create the CRS object from the PROJ string
    correct_crs = CRS.from_proj4(proj4_string)
    return (correct_crs,)


@app.cell(disabled=True, hide_code=True)
def _(clip_ds):
    clip_ds.rio.crs
    return


@app.cell(disabled=True, hide_code=True)
def _(clip_ds):
    clip_ds.rio.bounds()
    return


@app.cell(disabled=True, hide_code=True)
def _(Affine, clip_ds, np):
    # 1. Get the resolution (pixel size)
    x_res = clip_ds.x[1].item() - clip_ds.x[0].item()
    y_res = clip_ds.y[1].item() - clip_ds.y[0].item() # Note: y-resolution is often negative!

    # 2. Get the origin (top-left corner of the *top-left pixel*)
    # The coordinates usually represent the center of the pixel, so we adjust by half a resolution.
    x_min = clip_ds.x[0].item() - (x_res / 2)
    y_max = clip_ds.y[0].item() + (y_res / 2) # For y, we use the max coordinate (the top edge)

    # 3. Create the Affine Transform:
    # Affine(a, b, c, d, e, f) where:
    # a = x-resolution
    # b = 0 (no rotation)
    # c = x_min (top-left x-coordinate)
    # d = 0 (no rotation)
    # e = y-resolution (often negative)
    # f = y_max (top-left y-coordinate)

    transform = Affine(x_res, 0.0, x_min, 0.0, -abs(y_res), y_max)

    # 4. Write the calculated transform and then reproject
    affine_clip_ds = clip_ds.rio.write_transform(transform)

    print(affine_clip_ds.rio.transform()) # Verify the transform was written

    # 5. Reproject again
    affine_wgs84 = affine_clip_ds.rio.reproject("EPSG:4326", nodata=np.nan)

    affine_wgs84
    return


if __name__ == "__main__":
    app.run()
