"""
generate_synthetic_data.py
Generates realistic synthetic raster data (GeoTIFF) for India
to allow the analysis pipeline to run without Google Earth Engine.

India bounding box (approx): lat 8-37, lon 68-97
At 1km resolution (0.009 degrees ≈ 1km), that's roughly:
  rows = (37-8) / 0.009 ≈ 3222
  cols = (97-68) / 0.009 ≈ 3222
We'll use ~500x500 for quicker demo at ~60km resolution.
"""

import numpy as np
import rasterio
from rasterio.transform import from_bounds
from rasterio.crs import CRS
import os

ROWS, COLS = 300, 300  # ~10km resolution grid covering India
INDIA_BBOX = (68.0, 8.0, 97.5, 37.5)   # (west, south, east, north)
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
os.makedirs(DATA_DIR, exist_ok=True)

np.random.seed(42)

def make_transform():
    west, south, east, north = INDIA_BBOX
    return from_bounds(west, south, east, north, COLS, ROWS)

def write_tif(data: np.ndarray, filename: str, nodata=-9999.0):
    transform = make_transform()
    crs = CRS.from_epsg(4326)
    path = os.path.join(DATA_DIR, filename)
    with rasterio.open(
        path, 'w',
        driver='GTiff',
        height=ROWS, width=COLS,
        count=1,
        dtype=data.dtype,
        crs=crs,
        transform=transform,
        nodata=nodata
    ) as dst:
        dst.write(data.astype(data.dtype), 1)
    print(f"  Saved → {path}  ({ROWS}×{COLS})")

def generate_viirs():
    """
    Nighttime lights (nW/cm²/sr). India pattern:
    - Bright cluster around major cities (Delhi, Mumbai, Kolkata, Chennai, Hyderabad, Bangalore)
    - Dim in Himalayan region (north), Thar Desert (northwest), forested NE
    """
    print("Generating VIIRS nighttime lights...")
    # Base very low background
    data = np.random.exponential(0.3, (ROWS, COLS)).astype(np.float32)

    def add_city(lat, lon, intensity, sigma_r=8, sigma_c=8):
        west, south, east, north = INDIA_BBOX
        r = int((north - lat) / (north - south) * ROWS)
        c = int((lon - west) / (east - west) * COLS)
        yr, xr = np.ogrid[:ROWS, :COLS]
        blob = intensity * np.exp(-((yr - r)**2 / (2*sigma_r**2) + (xr - c)**2 / (2*sigma_c**2)))
        return blob

    # Major metro areas
    cities = [
        (28.61, 77.21, 150, 14, 14),   # Delhi
        (19.08, 72.88, 130, 12, 12),   # Mumbai
        (22.57, 88.36, 110, 12, 12),   # Kolkata
        (13.08, 80.27, 100, 10, 10),   # Chennai
        (17.38, 78.49, 100, 10, 10),   # Hyderabad
        (12.97, 77.59, 100, 10, 10),   # Bangalore
        (23.02, 72.57, 80, 9, 9),       # Ahmedabad
        (18.52, 73.86, 75, 9, 9),       # Pune
        (26.85, 80.95, 70, 8, 8),       # Lucknow
        (21.14, 79.09, 60, 7, 7),       # Nagpur
        (25.59, 85.14, 55, 7, 7),       # Patna
        (26.92, 75.78, 60, 7, 7),       # Jaipur
        (30.73, 76.78, 55, 7, 7),       # Chandigarh
        (10.52, 76.21, 50, 7, 7),       # Kochi
        (11.66, 78.15, 45, 6, 6),       # Salem
    ]
    for lat, lon, intensity, sr, sc in cities:
        data += add_city(lat, lon, intensity, sr, sc).astype(np.float32)

    # Northern Himalayas: very dark
    west, south, east, north_b = INDIA_BBOX
    himalayas_top = int((north_b - 32) / (north_b - south) * ROWS)
    data[:himalayas_top, :] *= 0.1

    # Thar Desert (NW): sparse lights
    thar_c_start = int((68 - west) / (east - west) * COLS)
    thar_c_end   = int((74 - west) / (east - west) * COLS)
    thar_r_start = int((north_b - 30) / (north_b - south) * ROWS)
    thar_r_end   = int((north_b - 24) / (north_b - south) * ROWS)
    data[thar_r_start:thar_r_end, thar_c_start:thar_c_end] *= 0.15

    # NE India (Assam/Meghalaya forest): dim
    ne_c_start = int((91 - west) / (east - west) * COLS)
    ne_r_start = int((north_b - 28) / (north_b - south) * ROWS)
    ne_r_end   = int((north_b - 20) / (north_b - south) * ROWS)
    data[ne_r_start:ne_r_end, ne_c_start:] *= 0.25

    data = np.clip(data, 0, 300)
    write_tif(data, "viirs_nightlights.tif")

def generate_dem():
    """
    Elevation (metres). Highlights:
    - Himalayas in north (high)
    - Western/Eastern Ghats on coasts
    - Deccan Plateau (moderate 400-700m)
    - Coastal plains and Indo-Gangetic plain (low)
    """
    print("Generating SRTM DEM...")
    west, south, east, north = INDIA_BBOX
    data = np.zeros((ROWS, COLS), dtype=np.float32)

    lat_arr = np.linspace(north, south, ROWS)
    lon_arr = np.linspace(west, east, COLS)
    lon_grid, lat_grid = np.meshgrid(lon_arr, lat_arr)

    # Himalayas: 28N-37N, steep rise
    himalayas = np.where(lat_grid > 30, (lat_grid - 30) * 400, 0)
    himalayan_noise = np.random.normal(0, 200, (ROWS, COLS))
    data += himalayas + himalayan_noise * (np.clip(lat_grid - 28, 0, 9) / 9)

    # Deccan Plateau: 15N-22N, 73E-82E
    deccan = np.where((lat_grid > 14) & (lat_grid < 22) & (lon_grid > 73) & (lon_grid < 82),
                      np.random.normal(550, 80, (ROWS, COLS)), 0)
    data += deccan

    # Western Ghats: lon 73-76, lat 8-21
    wghats = np.where((lon_grid > 73) & (lon_grid < 76) & (lat_grid > 8) & (lat_grid < 21),
                      np.random.normal(900, 150, (ROWS, COLS)), 0)
    data += wghats

    # Eastern Ghats: lon 79-82, lat 12-20
    eghats = np.where((lon_grid > 79) & (lon_grid < 82) & (lat_grid > 11) & (lat_grid < 20),
                      np.random.normal(600, 100, (ROWS, COLS)), 0)
    data += eghats

    # Indo-Gangetic Plain: lat 24-29, very flat low
    igp = np.where((lat_grid > 24) & (lat_grid < 29) & (lon_grid > 75) & (lon_grid < 90),
                   np.random.normal(80, 20, (ROWS, COLS)), 0)
    data += igp

    # General noise
    data += np.random.normal(0, 30, (ROWS, COLS))
    data = np.clip(data, 0, 8848)
    write_tif(data, "srtm_dem.tif")

def generate_ndvi():
    """
    NDVI [-1, 1]. Highlights:
    - NE India / Andaman: dense forest (high NDVI 0.7-0.9)
    - Western Ghats: high (0.6-0.8)
    - Thar Desert: very low (0.05-0.15)
    - Crop areas (Gangetic plain): moderate-high (0.4-0.6)
    - Urban scrub: low-moderate
    """
    print("Generating Sentinel-2 NDVI...")
    west, south, east, north = INDIA_BBOX
    data = np.random.uniform(0.2, 0.4, (ROWS, COLS)).astype(np.float32)

    lat_arr = np.linspace(north, south, ROWS)
    lon_arr = np.linspace(west, east, COLS)
    lon_grid, lat_grid = np.meshgrid(lon_arr, lat_arr)

    # NE forests
    data = np.where((lon_grid > 91) & (lat_grid > 22) & (lat_grid < 29),
                    np.random.uniform(0.65, 0.88, (ROWS, COLS)), data).astype(np.float32)

    # Western Ghats
    data = np.where((lon_grid > 73) & (lon_grid < 76) & (lat_grid > 8) & (lat_grid < 21),
                    np.random.uniform(0.58, 0.82, (ROWS, COLS)), data).astype(np.float32)

    # Thar Desert (low)
    data = np.where((lon_grid > 68) & (lon_grid < 74) & (lat_grid > 24) & (lat_grid < 31),
                    np.random.uniform(0.03, 0.12, (ROWS, COLS)), data).astype(np.float32)

    # Gangetic plain crops
    data = np.where((lat_grid > 24) & (lat_grid < 29) & (lon_grid > 75) & (lon_grid < 90),
                    np.random.uniform(0.38, 0.62, (ROWS, COLS)), data).astype(np.float32)

    # Himalayas (sparse/snow → low NDVI above 34N)
    data = np.where(lat_grid > 34,
                    np.random.uniform(0.01, 0.18, (ROWS, COLS)), data).astype(np.float32)

    data = np.clip(data, -1, 1)
    write_tif(data, "s2_ndvi.tif")

if __name__ == "__main__":
    print(f"\n=== Generating Synthetic Raster Data for India ===")
    print(f"Grid: {ROWS}×{COLS} pixels | Bounding box: {INDIA_BBOX}\n")
    generate_viirs()
    generate_dem()
    generate_ndvi()
    print("\n✓ All synthetic rasters saved to:", DATA_DIR)
