import ee
import os
import requests
import numpy as np
import rasterio
from rasterio.transform import from_origin

def initialize_ee():
    try:
        ee.Initialize()
        print("Earth Engine initialized successfully.")
        return True
    except Exception as e:
        print(f"Error initializing Earth Engine: {e}")
        return False

def download_india_layer(image, name, scale=1000):
    """
    Download a single layer for all of India at a specific scale (in meters).
    Using 1000m (1km) for national-scale prototype efficiency.
    """
    print(f"Starting download for: {name} at {scale}m resolution...")
    india = ee.FeatureCollection('USDOS/LSIB_SIMPLE/2017').filter(ee.Filter.eq('country_na', 'India')).geometry()
    
    # Get download URL for the GeoTIFF
    url = image.getDownloadURL({
        'scale': scale,
        'crs': 'EPSG:4326',
        'region': india,
        'format': 'GEO_TIFF'
    })
    
    response = requests.get(url)
    output_path = f"/home/vaibhavjain/Desktop/IS/data/{name}.tif"
    with open(output_path, 'wb') as f:
        f.write(response.content)
    print(f"Saved {name} to {output_path}")

def run_acquisition():
    if not initialize_ee(): return

    india_geom = ee.FeatureCollection('USDOS/LSIB_SIMPLE/2017').filter(ee.Filter.eq('country_na', 'India')).geometry()
    
    # 1. VIIRS Nightlights (2023 Median)
    viirs = ee.ImageCollection('NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG') \
                .filterDate('2023-01-01', '2023-12-31') \
                .select('avg_rad').median().clip(india_geom)
    
    # 2. SRTM Elevation (30m source, resampled to 1km for national map)
    dem = ee.Image('USGS/SRTMGL1_003').clip(india_geom)
    
    # 3. Sentinel-2 NDVI (2023 Cloud-free Median)
    s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
           .filterDate('2023-01-01', '2023-12-31') \
           .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20)) \
           .median()
    ndvi = s2.normalizedDifference(['B8', 'B4']).rename('NDVI').clip(india_geom)

    # Execute Downloads
    download_india_layer(viirs, "viirs_nightlights")
    download_india_layer(dem, "srtm_dem")
    download_india_layer(ndvi, "s2_ndvi")

if __name__ == "__main__":
    run_acquisition()
