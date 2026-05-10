"""
ml_analysis.py
Machine Learning & Advanced Analytics for Perceptual Anomaly Zones.

Pipeline:
  1. K-Means clustering  → discover anomaly archetypes
  2. Random Forest        → feature importance analysis
  3. PCA                  → 2-component projection for visualization
  4. Spatial Autocorrelation (Moran's I) → detect geographic clustering
  5. Export results        → CSV / JSON for the dashboard
"""

import os, json
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from scipy.spatial import cKDTree

# ── Path configuration ──────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

FEATURES = ["darkness", "enclosure", "acoustics", "visibility"]


# ═══════════════════════════════════════════════════════════════════════════
# 1. K-MEANS CLUSTERING
# ═══════════════════════════════════════════════════════════════════════════

def find_optimal_k(X_scaled, k_range=range(2, 9)):
    """Elbow method — pick k that maximises silhouette score."""
    best_k, best_sil = 2, -1
    print("\n--- Elbow / Silhouette search ---")
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = km.fit_predict(X_scaled)
        sil = silhouette_score(X_scaled, labels, sample_size=min(5000, len(X_scaled)))
        print(f"  k={k}  silhouette={sil:.4f}")
        if sil > best_sil:
            best_k, best_sil = k, sil
    print(f"  → Best k = {best_k}  (silhouette = {best_sil:.4f})")
    return best_k, best_sil


def run_kmeans(X_scaled, k):
    km = KMeans(n_clusters=k, n_init=20, random_state=42)
    labels = km.fit_predict(X_scaled)
    return labels, km


# ═══════════════════════════════════════════════════════════════════════════
# 2. RANDOM FOREST FEATURE IMPORTANCE
# ═══════════════════════════════════════════════════════════════════════════

def feature_importance_analysis(X_scaled, labels):
    """Train RF on cluster labels and return per-feature importance."""
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=12, random_state=42, n_jobs=-1
    )
    rf.fit(X_scaled, labels)
    importances = rf.feature_importances_
    accuracy = rf.score(X_scaled, labels)
    print(f"\n--- Random Forest (train accuracy: {accuracy:.4f}) ---")
    for feat, imp in zip(FEATURES, importances):
        print(f"  {feat:>12s} : {imp:.4f}")
    return importances, accuracy


# ═══════════════════════════════════════════════════════════════════════════
# 3. PCA
# ═══════════════════════════════════════════════════════════════════════════

def run_pca(X_scaled):
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X_scaled)
    print(f"\n--- PCA ---")
    print(f"  Explained variance : {pca.explained_variance_ratio_}")
    print(f"  Total captured     : {pca.explained_variance_ratio_.sum():.4f}")
    return coords, pca


# ═══════════════════════════════════════════════════════════════════════════
# 4. SPATIAL AUTOCORRELATION  (Moran's I — pure Python/scipy)
# ═══════════════════════════════════════════════════════════════════════════

def morans_i(lats, lons, values, k_neighbours=8):
    """
    Compute Moran's I using a k-nearest-neighbours spatial weight matrix.
    Returns I, expected I, z-score, and p-value (normal approximation).
    """
    n = len(values)
    coords = np.column_stack([lats, lons])
    tree = cKDTree(coords)
    _, indices = tree.query(coords, k=k_neighbours + 1)  # +1 because self is included
    indices = indices[:, 1:]  # drop self

    z = values - values.mean()
    s2 = np.mean(z ** 2)

    # Compute numerator: sum of z_i * weighted z_j
    W = 0.0
    numerator = 0.0
    for i in range(n):
        for j_idx in indices[i]:
            W += 1.0
            numerator += z[i] * z[j_idx]

    I = (n / W) * (numerator / (n * s2 + 1e-12))
    E_I = -1.0 / (n - 1)

    # Variance under normality assumption (simplified)
    S1 = 2.0 * W  # each wij = 1, S1 = 2 * sum(w^2)
    S2 = n * (2 * k_neighbours) ** 2  # rough approx
    k_val = np.mean(z ** 4) / (s2 ** 2 + 1e-12)
    var_I = ((n * ((n**2 - 3*n + 3)*S1 - n*S2 + 3*W**2)
              - k_val * (n*(n-1)*S1 - 2*n*S2 + 6*W**2))
             / ((n-1)*(n-2)*(n-3)*W**2 + 1e-12)) - E_I**2

    z_score = (I - E_I) / (np.sqrt(abs(var_I)) + 1e-12)

    # Two-tailed p-value from normal approximation
    from scipy.stats import norm
    p_value = 2.0 * (1.0 - norm.cdf(abs(z_score)))

    print(f"\n--- Moran's I (k={k_neighbours} neighbours) ---")
    print(f"  I          = {I:.6f}")
    print(f"  E[I]       = {E_I:.6f}")
    print(f"  z-score    = {z_score:.4f}")
    print(f"  p-value    = {p_value:.6f}")
    sig = "YES (p < 0.05)" if p_value < 0.05 else "NO (p ≥ 0.05)"
    print(f"  Significant? {sig}")
    return {"I": round(float(I), 6),
            "E_I": round(float(E_I), 6),
            "z_score": round(float(z_score), 4),
            "p_value": round(float(p_value), 6),
            "significant": bool(p_value < 0.05)}


# ═══════════════════════════════════════════════════════════════════════════
# 5. CLUSTER PROFILING
# ═══════════════════════════════════════════════════════════════════════════

ARCHETYPE_NAMES = {
    0: "Dark Wilderness",
    1: "Enclosed Valley",
    2: "Acoustic Canyon",
    3: "Misty Lowland",
    4: "Shadow Plateau",
    5: "Twilight Basin",
    6: "Echo Ridge",
    7: "Fog Corridor",
}


def profile_clusters(df, labels, k):
    """Generate per-cluster mean profiles and assign archetype names."""
    df_copy = df.copy()
    df_copy["cluster"] = labels
    profiles = []
    for c in range(k):
        subset = df_copy[df_copy["cluster"] == c]
        profile = {
            "cluster": c,
            "name": ARCHETYPE_NAMES.get(c, f"Cluster {c}"),
            "count": int(len(subset)),
            "mean_score": round(float(subset["score"].mean()), 4),
            "mean_darkness": round(float(subset["darkness"].mean()), 4),
            "mean_enclosure": round(float(subset["enclosure"].mean()), 4),
            "mean_acoustics": round(float(subset["acoustics"].mean()), 4),
            "mean_visibility": round(float(subset.get("visibility", 1 - subset.get("ndvi", 0.5)).mean()), 4) if "visibility" in subset.columns else 0.0,
        }

        # Assign archetype name based on dominant feature
        means = [profile["mean_darkness"], profile["mean_enclosure"],
                 profile["mean_acoustics"]]
        dominant = FEATURES[:3][np.argmax(means)]
        profile["dominant_feature"] = dominant
        profiles.append(profile)

    return profiles


# ═══════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print(" ML & Advanced Analytics Pipeline")
    print("=" * 60)

    # ── Load data ────────────────────────────────────────────────────────
    csv_path = os.path.join(DATA_DIR, "susceptibility_results.csv")
    if not os.path.exists(csv_path):
        print(f"[ERROR] {csv_path} not found. Run analysis_engine.py first.")
        return

    df = pd.read_csv(csv_path)
    print(f"\nLoaded {len(df):,} rows from susceptibility_results.csv")

    # Add visibility column if missing (1 - ndvi proxy; the analysis engine
    # doesn't always export it, but score already accounts for it)
    if "visibility" not in df.columns:
        # Derive from score using the known PASI weights:
        # score ≈ 0.3*dark + 0.3*encl + 0.2*acou + 0.2*vis
        # → vis ≈ (score - 0.3*dark - 0.3*encl - 0.2*acou) / 0.2
        df["visibility"] = np.clip(
            (df["score"] - 0.3 * df["darkness"] - 0.3 * df["enclosure"] - 0.2 * df["acoustics"]) / 0.2,
            0, 1
        )

    # ── Feature matrix ──────────────────────────────────────────────────
    X = df[FEATURES].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ── 1. K-Means ──────────────────────────────────────────────────────
    best_k, best_sil = find_optimal_k(X_scaled)
    labels, km_model = run_kmeans(X_scaled, best_k)
    df["cluster"] = labels

    # ── 2. Random Forest ────────────────────────────────────────────────
    importances, rf_accuracy = feature_importance_analysis(X_scaled, labels)

    # ── 3. PCA ──────────────────────────────────────────────────────────
    pca_coords, pca_model = run_pca(X_scaled)
    df["pc1"] = pca_coords[:, 0]
    df["pc2"] = pca_coords[:, 1]

    # ── 4. Moran's I ────────────────────────────────────────────────────
    sample_size = min(3000, len(df))
    sample_idx = np.random.RandomState(42).choice(len(df), sample_size, replace=False)
    mi_result = morans_i(
        df["lat"].values[sample_idx],
        df["lon"].values[sample_idx],
        df["score"].values[sample_idx],
        k_neighbours=8,
    )

    # ── 5. Cluster profiles ─────────────────────────────────────────────
    profiles = profile_clusters(df, labels, best_k)
    print("\n--- Cluster Profiles ---")
    for p in profiles:
        print(f"  Cluster {p['cluster']} ({p['name']:20s}): "
              f"n={p['count']:>5,}  mean_PASI={p['mean_score']:.4f}  "
              f"dominant={p['dominant_feature']}")

    # ── Export results ──────────────────────────────────────────────────
    # (a) Clusters CSV
    clusters_path = os.path.join(DATA_DIR, "ml_clusters.csv")
    df.to_csv(clusters_path, index=False)
    print(f"\n✓ Clusters  → {clusters_path}")

    # (b) Feature importance CSV
    imp_df = pd.DataFrame({
        "feature": FEATURES,
        "importance": importances,
    }).sort_values("importance", ascending=False)
    imp_path = os.path.join(DATA_DIR, "ml_feature_importance.csv")
    imp_df.to_csv(imp_path, index=False)
    print(f"✓ Feature importance → {imp_path}")

    # (c) Summary JSON
    summary = {
        "n_clusters": best_k,
        "silhouette_score": round(best_sil, 4),
        "rf_accuracy": round(rf_accuracy, 4),
        "pca_explained_variance": [round(float(v), 4) for v in pca_model.explained_variance_ratio_],
        "morans_i": mi_result,
        "cluster_profiles": profiles,
    }
    summary_path = os.path.join(DATA_DIR, "ml_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"✓ Summary   → {summary_path}")

    print("\n" + "=" * 60)
    print(" ML Pipeline Complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
