"""
analysis_engine.py
Loads raster data and computes the Perceptual Anomaly Susceptibility Index (PASI).
"""

import os
import numpy as np
import scipy.ndimage as ndimage
import pandas as pd
import rasterio


# ── Path configuration ──────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")


# ── Core analysis functions ──────────────────────────────────────────────────

def calculate_slope_and_roughness(dem_array, res):
    """Calculate slope (degrees) and roughness (local std-dev) from a DEM."""
    dx, dy = res
    px, py = np.gradient(dem_array, dx, dy)
    slope = np.sqrt(px**2 + py**2)
    slope_deg = np.rad2deg(np.arctan(slope))
    roughness = ndimage.generic_filter(dem_array.astype(float), np.std, size=3)
    return slope_deg, roughness


def darkness_index(viirs_array, threshold=1.0):
    """
    Darkness Index [0, 1].
    threshold (nW/cm²/sr): radiance below this is considered 'dark'.
    1.0 = total darkness; 0.0 = bright urban area.
    """
    darkness = np.where(
        viirs_array < threshold,
        1.0,
        threshold / (viirs_array + 1e-6)
    )
    return np.clip(darkness, 0, 1).astype(np.float32)


def acoustic_proxy_model(slope_array, ndvi_array):
    """
    Acoustic Susceptibility Proxy [0, 1].
    High relief + low NDVI → reflective surfaces → echoes / infrasound potential.
    """
    norm_slope = (slope_array - np.nanmin(slope_array)) / (
        np.nanmax(slope_array) - np.nanmin(slope_array) + 1e-6
    )
    # Cliffs & rocky terrain (low NDVI, high slope) score highest
    acoustic_potential = norm_slope * (1.0 - np.clip(ndvi_array, 0, 1))
    return acoustic_potential.astype(np.float32)


def terrain_enclosure_index(dem_array):
    """
    Terrain Enclosure Index [0, 1].
    Regions below local mean elevation (valleys, ravines) → poor sky visibility.
    """
    local_mean = ndimage.uniform_filter(dem_array.astype(float), size=21)
    enclosure = np.where(
        dem_array < local_mean,
        (local_mean - dem_array) / (local_mean + 1e-6),
        0
    )
    return np.clip(enclosure, 0, 1).astype(np.float32)


def calculate_anomaly_susceptibility(dark_idx, enclosure_idx, acoustic_idx, visibility_idx, weights=None):
    """Combine individual indices into the final PASI score [0, 1]."""
    if weights is None:
        weights = {
            'darkness':   0.30,
            'enclosure':  0.30,
            'acoustics':  0.20,
            'visibility': 0.20,
        }
    final_score = (
        weights['darkness']   * dark_idx +
        weights['enclosure']  * enclosure_idx +
        weights['acoustics']  * acoustic_idx +
        weights['visibility'] * visibility_idx
    )
    return final_score.astype(np.float32)


# ── I/O helpers ──────────────────────────────────────────────────────────────

def load_raster(path):
    with rasterio.open(path) as src:
        data = src.read(1).astype(np.float32)
        transform = src.transform
    return data, transform


def get_coords(data, transform):
    """Return flat lat/lon arrays for every pixel."""
    rows, cols = np.indices(data.shape)
    lon, lat = rasterio.transform.xy(transform, rows.flatten(), cols.flatten())
    return np.array(lat, dtype=np.float32), np.array(lon, dtype=np.float32)


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run_full_analysis(data_dir=None, sample_frac=1.0, score_threshold=0.0):
    if data_dir is None:
        data_dir = DATA_DIR

    print(f"\n=== Perceptual Anomaly Susceptibility Analysis ===")
    print(f"Data directory : {data_dir}")

    # 1. Load rasters
    try:
        viirs, transform = load_raster(os.path.join(data_dir, "viirs_nightlights.tif"))
        dem,   _         = load_raster(os.path.join(data_dir, "srtm_dem.tif"))
        ndvi,  _         = load_raster(os.path.join(data_dir, "s2_ndvi.tif"))
        print(f"Loaded rasters  : VIIRS {viirs.shape}, DEM {dem.shape}, NDVI {ndvi.shape}")
    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}")
        print("Run  scripts/generate_synthetic_data.py  first to create the raster files.")
        return

    # Ensure identical shapes (resample to VIIRS grid if needed)
    target_shape = viirs.shape
    def match_shape(arr):
        if arr.shape != target_shape:
            zoom = (target_shape[0] / arr.shape[0], target_shape[1] / arr.shape[1])
            arr = ndimage.zoom(arr, zoom, order=1)
        return arr.astype(np.float32)

    dem  = match_shape(dem)
    ndvi = match_shape(ndvi)

    # 2. Compute individual indices
    print("\n--- Computing indices ---")
    d_idx = darkness_index(viirs)
    print(f"  Darkness index   : min={d_idx.min():.3f}  max={d_idx.max():.3f}  mean={d_idx.mean():.3f}")

    e_idx = terrain_enclosure_index(dem)
    print(f"  Enclosure index  : min={e_idx.min():.3f}  max={e_idx.max():.3f}  mean={e_idx.mean():.3f}")

    pixel_size_m = 1000   # synthetic data ≈ 10 km/pixel; use 1000 m as slope scale
    slope, roughness = calculate_slope_and_roughness(dem, (pixel_size_m, pixel_size_m))
    a_idx = acoustic_proxy_model(slope, ndvi)
    print(f"  Acoustic index   : min={a_idx.min():.3f}  max={a_idx.max():.3f}  mean={a_idx.mean():.3f}")

    v_idx = 1.0 - np.clip(ndvi, 0, 1)
    print(f"  Visibility index : min={v_idx.min():.3f}  max={v_idx.max():.3f}  mean={v_idx.mean():.3f}")

    # 3. Susceptibility composite
    susceptibility = calculate_anomaly_susceptibility(d_idx, e_idx, a_idx, v_idx)
    print(f"\n  PASI score       : min={susceptibility.min():.3f}  max={susceptibility.max():.3f}  mean={susceptibility.mean():.3f}")

    # 4. Export to CSV
    lat_arr, lon_arr = get_coords(viirs, transform)

    df = pd.DataFrame({
        'lat':       lat_arr,
        'lon':       lon_arr,
        'score':     susceptibility.flatten(),
        'darkness':  d_idx.flatten(),
        'enclosure': e_idx.flatten(),
        'acoustics': a_idx.flatten(),
    })

    df_filtered = df[df['score'] > score_threshold].sample(
        frac=sample_frac, random_state=42
    )

    output_path = os.path.join(data_dir, "susceptibility_results.csv")
    df_filtered.to_csv(output_path, index=False)
    print(f"\n✓ Results saved → {output_path}  ({len(df_filtered):,} rows)")

    # 5. Top-10 hotspots summary
    top10 = df.nlargest(10, 'score')[['lat', 'lon', 'score', 'darkness', 'enclosure', 'acoustics']]
    print("\n--- Top 10 High-Susceptibility Pixels ---")
    print(top10.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    return df_filtered


if __name__ == "__main__":
    run_full_analysis()
