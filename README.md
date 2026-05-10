# Mapping Perceptual Anomaly Zones in India

This project identifies and maps regions in India where environmental factors—such as low illumination, complex terrain, and weather—are likely to lead to perceptual anomalies interpreted as paranormal experiences.

## Core Idea
By combining geospatial datasets like nighttime lights, elevation, and climate patterns, we identify "High Susceptibility Zones" for perceptual anomalies (e.g., infrasound-induced unease, visual pareidolia in dark/foggy terrain).

## Features
- **Interactive Map:** Explore anomaly susceptibility scores across India.
- **Adjustable Weights:** Customize the influence of Darkness, Terrain, and Climate on the final score.
- **Regional Analysis:** Click any point on the map to get a detailed breakdown of environmental factors.

## Datasets
1.  **VIIRS Nighttime Lights:** Low-light zones.
2.  **SRTM 30m DEM:** Terrain relief and enclosure.
3.  **ERA5 Weather:** Fog, humidity, and wind patterns.
4.  **Sentinel-2 NDVI:** Vegetation density and visual occlusion.

## Installation
1.  Clone the repository.
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3.  (Optional) Authenticate Google Earth Engine if you wish to run the data acquisition script:
    ```bash
    earthengine authenticate
    ```

## Usage
### 1. Data Acquisition
Run the script to fetch datasets (requires GEE credentials):
```bash
python scripts/data_acquisition.py
```

### 2. Analysis
Generate the susceptibility index:
```bash
python scripts/analysis_engine.py
```

### 3. ML Analytics
Run the machine learning pipeline (clustering, feature importance, PCA, spatial autocorrelation):
```bash
python scripts/ml_analysis.py
```
This generates `data/ml_clusters.csv`, `data/ml_feature_importance.csv`, and `data/ml_summary.json`.

### 4. Dashboard
Launch the interactive web dashboard:
```bash
streamlit run web/app.py
```

## Project Structure
```
/
├── data/           # Raw and processed datasets
├── scripts/        # Data ingestion and analysis scripts
├── web/            # Streamlit dashboard
├── requirements.txt
└── README.md
```
