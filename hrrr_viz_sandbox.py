# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "affine==2.4.0",
#     "aiohttp==3.13.2",
#     "dash==3.3.0",
#     "dash-html-components==2.0.0",
#     "dash-leaflet==1.1.3",
#     "folium==0.20.0",
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

with app.setup:
    # Initialization code that runs before all other cells
    pass


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Interactive weather raster exploration

    Fetching data from the zarr datasets provided by dynamical.org!
    See: https://dynamical.org/catalog
    """)
    return


@app.cell
def _():
    # Even if xarray is installed in molab:
    # Make sure to explicitly install xarray[complete] !
    return


@app.cell
def _(mo):
    dataset_name_to_dataset_path = {
        "NOAA HRRR": "noaa/hrrr/forecast-48-hour",
        # "ECMWF IFS ENS": "ecmwf/ifs-ens/forecast-15-day-0-25-degree",
        # "NOAA GEFS": "noaa/gefs/forecast-35-day",
        "NOAA GFS": "noaa/gfs/forecast",
    }

    dataset_name = mo.ui.dropdown(
        dataset_name_to_dataset_path.keys(), value="NOAA HRRR"
    )

    mo.md(
        f"""
        Weather model: {dataset_name}
        """
    )
    return dataset_name, dataset_name_to_dataset_path


@app.cell
def _(mo):
    display_timezone = mo.ui.dropdown(
        ["UTC", "America/Chicago"], value="America/Chicago"
    )

    mo.md(
        f"""
        Display timezone: {display_timezone}
        """
    )
    return (display_timezone,)


@app.cell
def _(display_timezone, ds, make_time_str, mo):
    init_time_options = [
        make_time_str(t, display_timezone.value) for t in ds.init_time.values[-30:]
    ]

    # default_value = "2025-11-29 06:00" if dataset_name.value == "NOAA HRRR" else "2025-11-29 00:00"
    init_time_dropdown = mo.ui.dropdown(
        init_time_options,
        value=make_time_str("2025-11-29 00:00", display_timezone.value),
    )

    mo.md(
        f"""
        Init time input: {init_time_dropdown}
        """
    )
    return (init_time_dropdown,)


@app.cell
def _(mo):
    lead_time_slider = mo.ui.slider(start=1, stop=48, step=1, value=18, show_value=True)

    mo.md(
        f"""
        Lead time input (hrs): {lead_time_slider}
        """
    )
    return (lead_time_slider,)


@app.cell
def _():
    """
    ## TODO: location picker
    defaults, plus custom any lat/long input?

    See this marimo example for getting input: https://eoda-dev.github.io/py-openlayers/marimo/getting-started.html
    """

    riverwoods_x = -87.90219091178325
    riverwoods_y = 42.17245575311014
    return riverwoods_x, riverwoods_y


@app.cell
def _():
    """
    ## TODO: zoom level slider
    add zoom level slider to affect +/- around the point location to clip to -- but add a little extra and then clip down after reprojection so it's a box again.
    """

    zoom_level = 0.75  # degrees around center to add
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


@app.cell
def _(mo):
    var_human_to_technical_name = {
        "Surface Precipitation": "precipitation_surface",
        "Temperature": "temperature_2m",
    }

    var_human_name = mo.ui.dropdown(
        var_human_to_technical_name.keys(), value="Surface Precipitation"
    )

    mo.md(
        f"""
        Forecast variable: {var_human_name}
        """
    )
    return var_human_name, var_human_to_technical_name


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Plotting the raw data, in original coordinates
    """)
    return


@app.cell
def _(dataset_name, dataset_name_to_dataset_path, xr):
    dataset_url = f"https://data.dynamical.org/{dataset_name_to_dataset_path[dataset_name.value]}/latest.zarr?email=laurengulland@gmail.com"

    ds = xr.open_zarr(dataset_url, chunks=None)
    return (ds,)


@app.cell
def _(
    display_timezone,
    ds,
    init_time_dropdown,
    lead_time_slider,
    make_plot,
    make_time_str,
    mo,
    pd,
    riverwoods_x,
    riverwoods_y,
    var_human_name,
    var_human_to_technical_name,
    zoom_level,
):
    utc_init_time = (
        pd.Timestamp(init_time_dropdown.value)
        .tz_localize(display_timezone.value)
        .tz_convert("UTC")
        .tz_localize(None)
    )
    var_name = var_human_to_technical_name[var_human_name.value]

    clip_ds = ds.sel(
        init_time=utc_init_time,
        lead_time=pd.Timedelta(hours=lead_time_slider.value),
        method="nearest",
    ).rio.clip_box(
        riverwoods_x - zoom_level,
        riverwoods_y - zoom_level,
        riverwoods_x + zoom_level,
        riverwoods_y + zoom_level,
        crs="EPSG:4326",
    )[var_name]

    _title_str = f"{var_human_name.value} in Riverwoods at {make_time_str(clip_ds.valid_time.values, tzinfo=display_timezone.value)}"

    _fig = make_plot(clip_ds, _title_str)
    mo.mpl.interactive(_fig)
    return clip_ds, var_name


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Optional step for HRRR: reproject from their weirdo crs
    """)
    return


@app.cell
def _(
    clip_ds,
    dataset_name,
    display_timezone,
    make_plot,
    make_time_str,
    mo,
    np,
    var_human_name,
):
    if dataset_name.value == "NOAA HRRR":
        # Intermediate reprojection (tldr? Conic -> Ellipsoid?)
        ds_conic = clip_ds.rio.reproject("EPSG:5070", nodata=np.nan)
        # Reprojection to WGS84 (tldr? Ellipsoid to different ellipsoid)
        ds_wgs84 = ds_conic.rio.reproject("EPSG:4326", nodata=np.nan)
    else:
        ds_wgs84 = clip_ds.rio.reproject("EPSG:4326", nodata=np.nan)

    _title_str = f"{var_human_name.value} in Riverwoods at {make_time_str(clip_ds.valid_time.values, tzinfo=display_timezone.value)}"

    _fig = make_plot(ds_wgs84, _title_str)
    mo.mpl.interactive(_fig)
    return (ds_wgs84,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Convert to geodataframe and plot on a map
    """)
    return


@app.cell
def _(ds_wgs84, riverwoods_x, riverwoods_y, zoom_level):
    ## Clip down to a rectangular area again now that it's in WGS84
    ds_wgs84_clip = ds_wgs84.rio.clip_box(
        riverwoods_x - zoom_level * 0.9,
        riverwoods_y - zoom_level * 0.9,
        riverwoods_x + zoom_level * 0.9,
        riverwoods_y + zoom_level * 0.9,
    )
    return (ds_wgs84_clip,)


@app.cell
def _(Polygon, ds_wgs84_clip, gpd, np, var_name):
    ## Make geodataframe from dataset, with the pixel geometries as polygons

    ## Make df -- will later turn this into a gdf once we create polygon geoms
    df_polygons = (
        ds_wgs84_clip.to_dataframe()
        .reset_index()
        .drop(
            columns=[
                "ingested_forecast_length",
                "expected_forecast_length",
                "spatial_ref",
                "lead_time",
            ]
        )
    )

    ## Get basic info to create polygons with
    # Get the coordinate arrays (assumes they are 1D and sorted)
    lons = ds_wgs84_clip["x"].values
    lats = ds_wgs84_clip["y"].values
    # Grab resolution directly off the diffs
    lon_diff = np.diff(lons)
    lat_diff = np.diff(lats)
    # Calculate half-step for both dimensions
    half_lon_step = np.mean(np.abs(lon_diff)) / 2.0
    half_lat_step = np.mean(np.abs(lat_diff)) / 2.0
    # Create min/max per pixel, to be turned into Polygon boundaries
    df_polygons["lon_min"] = df_polygons["x"] - half_lon_step
    df_polygons["lon_max"] = df_polygons["x"] + half_lon_step
    df_polygons["lat_min"] = df_polygons["y"] - half_lat_step
    df_polygons["lat_max"] = df_polygons["y"] + half_lat_step

    ## Create Polygon geoms
    df_polygons["geometry"] = df_polygons.apply(
        lambda row: Polygon(
            [
                # vertices, counterclockwise
                (row["lon_min"], row["lat_min"]),
                (row["lon_max"], row["lat_min"]),
                (row["lon_max"], row["lat_max"]),
                (row["lon_min"], row["lat_max"]),
                (row["lon_min"], row["lat_min"]),
            ]
        ),
        axis=1,
    )

    ## Use newly created geometry column to convert to GeoDataFrame
    gdf_polygons = gpd.GeoDataFrame(
        df_polygons[["x", "y", var_name, "geometry"]], crs="EPSG:4326"
    )
    return (gdf_polygons,)


@app.cell
def _(folium, gdf_polygons, riverwoods_x, riverwoods_y, var_name):
    # TODO: improve cmap by making 0 always white on temp graph?
    cmap = "coolwarm" if var_name == "temperature_2m" else None

    map = gdf_polygons.explore(
        var_name, style_kwds={"stroke": False, "fill_opacity": 0.6}, cmap=cmap
    )
    folium.Marker(location=(riverwoods_y, riverwoods_x), popup="Riverwoods, IL").add_to(
        map
    )
    map
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
    import folium

    return Polygon, folium, gpd, mo, np, pd, plt, xr


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


if __name__ == "__main__":
    app.run()
