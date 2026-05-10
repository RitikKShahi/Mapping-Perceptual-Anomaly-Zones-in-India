"""
score_haunted_sites.py
Extracts PASI sub-index scores at each haunted site coordinate
from the generated rasters. Saves haunted_sites_scores.csv.
"""

import os, sys
import numpy as np
import pandas as pd
import rasterio
from rasterio.sample import sample_gen
import scipy.ndimage as ndimage

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
sys.path.insert(0, os.path.join(BASE_DIR, "scripts"))
from analysis_engine import (
    darkness_index,
    terrain_enclosure_index,
    acoustic_proxy_model,
    calculate_slope_and_roughness,
    calculate_anomaly_susceptibility,
    load_raster,
)

HAUNTED_SITES = [
    {"name": "Bhangarh Fort",       "lat": 27.0965, "lon": 76.2861,
     "state": "Rajasthan",
     "factors": "Isolated valley, steep terrain, very low nighttime illumination"},
    {"name": "Kuldhara Village",     "lat": 26.8732, "lon": 70.7850,
     "state": "Rajasthan",
     "factors": "Extreme Thar Desert isolation, high wind resonance corridors"},
    {"name": "Dumas Beach",          "lat": 21.1114, "lon": 72.7118,
     "state": "Gujarat",
     "factors": "Coastal acoustic reflection, sea‑mist, low light"},
    {"name": "Shaniwar Wada",        "lat": 18.5192, "lon": 73.8553,
     "state": "Maharashtra",
     "factors": "Complex stone structure acoustics, historical enclosure"},
    {"name": "Dow Hill, Kurseong",   "lat": 26.8812, "lon": 88.2778,
     "state": "West Bengal",
     "factors": "Frequent fog, dense forest (high NDVI), steep Himalayan terrain"},
    {"name": "Lambi Dehar Mines",    "lat": 30.4367, "lon": 78.0263,
     "state": "Uttarakhand",
     "factors": "Deep mountain enclosure, infrasound‑prone geological features"},
    {"name": "GP Block, Meerut",     "lat": 28.9815, "lon": 77.7265,
     "state": "Uttar Pradesh",
     "factors": "Urban acoustic isolation, historical structural decay"},
]

def pixel_for_coord(transform, lat, lon, shape):
    """Return (row, col) for a lat/lon given a rasterio affine transform."""
    col, row = ~transform * (lon, lat)
    row, col = int(np.clip(round(row), 0, shape[0]-1)), int(np.clip(round(col), 0, shape[1]-1))
    return row, col

def main():
    viirs, tf  = load_raster(os.path.join(DATA_DIR, "viirs_nightlights.tif"))
    dem,   _   = load_raster(os.path.join(DATA_DIR, "srtm_dem.tif"))
    ndvi,  _   = load_raster(os.path.join(DATA_DIR, "s2_ndvi.tif"))

    # Build full index arrays (same as analysis_engine)
    d_all = darkness_index(viirs)
    e_all = terrain_enclosure_index(dem)
    slope, _ = calculate_slope_and_roughness(dem, (1000, 1000))
    a_all = acoustic_proxy_model(slope, ndvi)
    v_all = (1.0 - np.clip(ndvi, 0, 1)).astype(np.float32)
    pasi_all = calculate_anomaly_susceptibility(d_all, e_all, a_all, v_all)

    records = []
    for site in HAUNTED_SITES:
        r, c = pixel_for_coord(tf, site["lat"], site["lon"], viirs.shape)
        # 3×3 neighbourhood mean to smooth pixel noise
        r0, r1 = max(0,r-1), min(viirs.shape[0], r+2)
        c0, c1 = max(0,c-1), min(viirs.shape[1], c+2)
        dark  = float(d_all [r0:r1, c0:c1].mean())
        encl  = float(e_all [r0:r1, c0:c1].mean())
        acou  = float(a_all [r0:r1, c0:c1].mean())
        vis   = float(v_all [r0:r1, c0:c1].mean())
        pasi  = float(pasi_all[r0:r1, c0:c1].mean())
        records.append({
            "name":        site["name"],
            "state":       site["state"],
            "lat":         site["lat"],
            "lon":         site["lon"],
            "pasi":        round(pasi, 4),
            "darkness":    round(dark, 4),
            "enclosure":   round(encl, 4),
            "acoustics":   round(acou, 4),
            "visibility":  round(vis,  4),
            "factors":     site["factors"],
        })

    df = pd.DataFrame(records).sort_values("pasi", ascending=False).reset_index(drop=True)
    df.index += 1   # rank from 1
    out = os.path.join(DATA_DIR, "haunted_sites_scores.csv")
    df.to_csv(out, index_label="rank")
    print("\n=== PASI Scores at Haunted / Validation Sites ===\n")
    print(df[["name","state","pasi","darkness","enclosure","acoustics","visibility"]].to_string())
    print(f"\nSaved → {out}")

if __name__ == "__main__":
    main()
