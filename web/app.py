"""
web/app.py – Perceptual Anomaly Zones Dashboard (Streamlit) - Enhanced v3
Features:
  - India-clipped heatmap (no data outside India boundary)
  - 30 haunted locations with detailed popup descriptions
  - Clickable map with location inspector
"""

import os, sys, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.path import Path as MplPath
import folium
from folium import plugins
import streamlit as st
from streamlit_folium import st_folium

# ── Path setup ─────────────────────────────────────────────────────────────
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR  = os.path.join(REPO_ROOT, "data")
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Perceptual Anomaly Zones – India",
    page_icon="🌙",
    layout="wide",
)

# ── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
.score-card {background:#1e1e2e;border-radius:12px;padding:16px 20px;margin:8px 0;border-left:5px solid #ff4b4b}
.score-card h4 {margin:0 0 4px 0;color:#fff;font-size:1.05rem}
.score-card .loc {color:#aaa;font-size:0.82rem;margin-bottom:10px}
.bar-label {font-size:0.78rem;color:#ccc;margin:2px 0}
.pasi-big {font-size:2rem;font-weight:700}
.rank-badge {display:inline-block;background:#ff4b4b;color:#fff;border-radius:50%;width:26px;height:26px;text-align:center;line-height:26px;font-size:0.85rem;font-weight:700;margin-right:8px}
</style>
""", unsafe_allow_html=True)

# ── Sidebar weights ─────────────────────────────────────────────────────────
st.sidebar.header("🎛️ PASI Weights")
w_dark = st.sidebar.slider("🌑 Darkness",            0.0, 1.0, 0.30, 0.05)
w_encl = st.sidebar.slider("⛰️ Terrain Enclosure",  0.0, 1.0, 0.30, 0.05)
w_acou = st.sidebar.slider("🔊 Acoustic Potential",  0.0, 1.0, 0.20, 0.05)
w_vis  = st.sidebar.slider("🌿 Visibility Reduction",0.0, 1.0, 0.20, 0.05)

total = w_dark + w_encl + w_acou + w_vis or 1.0
w_d, w_e, w_a, w_v = w_dark/total, w_encl/total, w_acou/total, w_vis/total

st.sidebar.divider()
st.sidebar.caption("PASI = weighted sum of four environmental indices, each normalised to [0, 1].")

# ══════════════════════════════════════════════════════════════════════
#  EXPANDED HAUNTED SITES  (30 locations across India)
# ══════════════════════════════════════════════════════════════════════
HAUNTED_SITES = [
    {"name":"Kuldhara Village","state":"Rajasthan","lat":26.8732,"lon":70.7850,
     "darkness":1.00,"enclosure":0.66,"acoustics":0.02,"visibility":0.93,
     "folklore":"Abandoned overnight by Paliwal Brahmins in 1825, said to be cursed so no one can settle here again.",
     "science":"Extreme Thar Desert isolation creates zero artificial light. Wind corridors through empty stone ruins produce infrasound (low-frequency vibrations) causing unease and dread in visitors, especially at night."},

    {"name":"Bhangarh Fort","state":"Rajasthan","lat":27.0965,"lon":76.2861,
     "darkness":0.01,"enclosure":0.23,"acoustics":0.02,"visibility":0.51,
     "folklore":"Cursed by a sorcerer whose advances were rejected by a princess. ASI prohibits entry after sunset.",
     "science":"Steep valley terrain and crumbling walls create acoustic echoes. Moderate vegetation and remoteness amplify the perceived eeriness, though nighttime illumination from nearby villages keeps darkness low."},

    {"name":"Dow Hill","state":"West Bengal","lat":26.8812,"lon":88.2778,
     "darkness":0.97,"enclosure":0.28,"acoustics":0.02,"visibility":0.50,
     "folklore":"Victoria Boys' School corridor haunted by a headless boy; forest paths report sightings of a woman walking and vanishing.",
     "science":"Extremely low nighttime light in dense Himalayan forest. Frequent fog banks reduce visibility drastically — the brain fills gaps with phantom shapes, a classic trigger for pareidolia."},

    {"name":"Dumas Beach","state":"Gujarat","lat":21.1114,"lon":72.7118,
     "darkness":0.03,"enclosure":0.97,"acoustics":0.02,"visibility":0.70,
     "folklore":"Former Hindu cremation ground; visitors report whispers at night and people who walk too close to the surf vanish.",
     "science":"Coastal terrain provides extreme enclosure (high dunes + sea). Wind over sand dunes produces low-frequency resonance, while sea-mist reduces visibility, creating auditory and visual illusions."},

    {"name":"GP Block, Meerut","state":"Uttar Pradesh","lat":28.9815,"lon":77.7265,
     "darkness":0.01,"enclosure":0.48,"acoustics":0.01,"visibility":0.57,
     "folklore":"Abandoned houses where four family members died. Locals report sounds of screaming and moving shadows.",
     "science":"Urban decay with hollow structures creates acoustic isolation pockets. Moderate vegetation overgrowth and structural enclosure simulate confined spaces that amplify echoes."},

    {"name":"Lambi Dehar Mines","state":"Uttarakhand","lat":30.4367,"lon":78.0263,
     "darkness":0.02,"enclosure":0.27,"acoustics":0.06,"visibility":0.70,
     "folklore":"Over 50,000 workers allegedly died during limestone mining in British era. Helicopter crashes reported nearby.",
     "science":"Deep mountain ravine with geological faults resonating at infrasound frequencies (18-19 Hz). Dense vegetation reduces visibility. The narrow mine shafts create natural wind tunnels producing humming sounds."},

    {"name":"Shaniwar Wada","state":"Maharashtra","lat":18.5192,"lon":73.8553,
     "darkness":0.01,"enclosure":0.01,"acoustics":0.06,"visibility":0.29,
     "folklore":"Young prince Narayan Rao was assassinated here; his screams 'Kaka mala vachva' (Uncle save me) are heard on full moon nights.",
     "science":"Complex stone fortress acoustics — curved walls and archways create focused audio reflections. However, urban-surrounding light pollution and low vegetation keep overall PASI low."},

    {"name":"Ramoji Film City","state":"Telangana","lat":17.2543,"lon":78.6808,
     "darkness":0.12,"enclosure":0.35,"acoustics":0.04,"visibility":0.45,
     "folklore":"Built on Nizam-era war grounds. Lights malfunction on their own, food gets scattered, women reportedly get their dupatta pulled.",
     "science":"Large enclosed film sets with theatrical lighting create pockets of extreme darkness next to bright zones. Open sound stages produce unpredictable echo patterns, disorienting visitors."},

    {"name":"D'Souza Chawl","state":"Maharashtra","lat":19.0190,"lon":72.8432,
     "darkness":0.02,"enclosure":0.55,"acoustics":0.03,"visibility":0.30,
     "folklore":"A woman drowned in the well. Her apparition is seen near the well at night, and faucets turn on by themselves.",
     "science":"Dense urban chawl enclosure with narrow corridors and courtyards creates microclimate pockets with unusual acoustics. Plumbing of old buildings causes spontaneous water flow — interpreted as paranormal."},

    {"name":"Tunnel No. 33 (Barog)","state":"Himachal Pradesh","lat":30.9000,"lon":77.0700,
     "darkness":0.95,"enclosure":0.90,"acoustics":0.08,"visibility":0.85,
     "folklore":"Engineer Colonel Barog committed suicide after miscalculating tunnel alignment. His ghost walks through the 1.1 km tunnel.",
     "science":"Complete darkness inside the mountain tunnel with extreme acoustic enclosure. Wind passing through the 1.1 km tube creates standing waves at infrasound frequencies — the single most acoustically hostile environment in the dataset."},

    {"name":"Jatinga","state":"Assam","lat":25.2000,"lon":93.1500,
     "darkness":0.85,"enclosure":0.40,"acoustics":0.03,"visibility":0.75,
     "folklore":"Birds commit mass suicide here during September-November monsoon nights, diving towards village lights and dying.",
     "science":"Dense forest valley with near-zero artificial light. Heavy fog and mist during monsoon reduce visibility to meters. Disoriented migratory birds are attracted to the few village lights — creating an eerie spectacle locals attribute to dark forces."},

    {"name":"Sanjay Van","state":"Delhi","lat":28.5350,"lon":77.1880,
     "darkness":0.15,"enclosure":0.30,"acoustics":0.02,"visibility":0.65,
     "folklore":"Spread over 784 acres, home to Sufi saint tombs. Joggers report being followed by unseen figures and hearing chanting.",
     "science":"Surprisingly dense forest canopy for an urban setting reduces visibility within the forest. Ancient tomb structures create acoustic focal points. Contrast between dark forest interior and bright city creates adaptation blindness."},

    {"name":"Agrasen ki Baoli","state":"Delhi","lat":28.6262,"lon":77.2244,
     "darkness":0.10,"enclosure":0.85,"acoustics":0.07,"visibility":0.30,
     "folklore":"14th-century stepwell where the black water lures people to walk deeper. Said to be haunted by djinns.",
     "science":"Deep stone stepwell creates extreme acoustic enclosure — whispers amplify unpredictably. Descending geometry produces vertigo. Dark water at the bottom absorbs light, creating a visual void the brain interprets as threatening depth."},

    {"name":"Brij Raj Bhawan Palace","state":"Rajasthan","lat":25.1800,"lon":75.8500,
     "darkness":0.08,"enclosure":0.40,"acoustics":0.05,"visibility":0.35,
     "folklore":"Major Charles Burton, killed during the 1857 Sepoy Mutiny, haunts the palace, reportedly slapping guards who sleep on duty.",
     "science":"Colonial-era palace with thick walls and long corridors creates echo chambers. Temperature differentials between thick-walled rooms and corridors generate air currents perceived as cold touches."},

    {"name":"Fern Hill Hotel (Ooty)","state":"Tamil Nadu","lat":11.4102,"lon":76.6950,
     "darkness":0.30,"enclosure":0.45,"acoustics":0.03,"visibility":0.70,
     "folklore":"Former British governor's residence. Guests report children laughing, doorknobs turning, and belongings moving.",
     "science":"High-altitude Nilgiri Hills location with dense eucalyptus canopy, frequent fog, and colonial-era wooden architecture that creaks with temperature changes. Wood expansion creates footstep-like sounds."},

    {"name":"Raj Kiran Hotel","state":"Maharashtra","lat":17.6860,"lon":73.5000,
     "darkness":0.20,"enclosure":0.50,"acoustics":0.04,"visibility":0.55,
     "folklore":"Room 311 haunted by a woman who died there. Objects move, mirrors break, and cold spots appear.",
     "science":"Hillside coastal hotel in Mahabaleshwar with enclosed rooms and dense surrounding forest. Altitude creates pressure variations that make doors swing and objects shift on uneven surfaces."},

    {"name":"Residency (Lucknow)","state":"Uttar Pradesh","lat":26.8530,"lon":80.9402,
     "darkness":0.05,"enclosure":0.35,"acoustics":0.05,"visibility":0.40,
     "folklore":"Site of the devastating 1857 siege. Hundreds died here. Night guards report gunfire sounds and marching footsteps.",
     "science":"Ruined British Residency with exposed walls and arched doorways creates wind-whistle effects. Cannonball holes in walls act as acoustic flutes — wind produces sounds resembling gunfire and human cries."},

    {"name":"Savoy Hotel (Mussoorie)","state":"Uttarakhand","lat":30.4570,"lon":78.0725,
     "darkness":0.30,"enclosure":0.55,"acoustics":0.04,"visibility":0.70,
     "folklore":"Frances Garnett-Orme was found poisoned here in 1911; her ghost wanders the corridors. Agatha Christie was inspired by this case.",
     "science":"Himalayan hill station with dense fog and deodar forests. Colonial-era wooden building with long corridors amplifies footstep sounds. Temperature inversions cause sudden cold air pockets."},

    {"name":"Mukesh Mills","state":"Maharashtra","lat":18.9580,"lon":72.8310,
     "darkness":0.15,"enclosure":0.70,"acoustics":0.06,"visibility":0.40,
     "folklore":"Abandoned textile mill where a fire killed many workers. Film crews report equipment failures and actors getting possessed.",
     "science":"Massive ruined industrial structure with cavernous interiors creates extreme acoustic reverb. Exposed metal structures generate electromagnetic interference affecting sensitive electronics. Structural decay creates unpredictable air drafts."},

    {"name":"Vrindavan (Nidhivan)","state":"Uttar Pradesh","lat":27.5800,"lon":77.6988,
     "darkness":0.55,"enclosure":0.60,"acoustics":0.03,"visibility":0.80,
     "folklore":"Lord Krishna and gopis dance here at night. Trees move and pair up after dark. Shops near the temple close before sunset — anyone who watches goes blind or mad.",
     "science":"Exceptionally dense Tulsi grove creates near-total canopy closure. Extremely low light penetration. Wind through the interlocked branches creates rustling that sounds like dancing. The intertwined tree trunks 'pair up' due to root grafting."},

    {"name":"Aleya Ghost Lights","state":"West Bengal","lat":22.3500,"lon":88.6700,
     "darkness":0.90,"enclosure":0.10,"acoustics":0.01,"visibility":0.85,
     "folklore":"Mysterious floating lights in the Sundarbans marshes lure fishermen to their death in the swamps.",
     "science":"Classic will-o'-the-wisp: methane and phosphine gas from decomposing organic matter in marshy wetlands spontaneously ignite, creating flickering lights. Extreme darkness and fog create disorientation — fishermen follow the lights into deep water."},

    {"name":"Charleville Mansion","state":"Maharashtra","lat":18.9960,"lon":73.2650,
     "darkness":0.25,"enclosure":0.60,"acoustics":0.04,"visibility":0.50,
     "folklore":"Built in 1880s, owner's wife died under mysterious circumstances. Ghostly woman seen in windows, doors lock by themselves.",
     "science":"Gothic Victorian architecture on a misty Matheran hilltop with extreme moisture. Rusting iron hinges swell and contract with humidity, causing doors to move. Fog on windowpanes creates reflective shapes visible from outside."},

    {"name":"Three Kings Church","state":"Goa","lat":15.2690,"lon":73.9670,
     "darkness":0.20,"enclosure":0.45,"acoustics":0.05,"visibility":0.50,
     "folklore":"Three kings poisoned each other for control of the village. Their bodies were buried in the church without rites. Locals avoid the hilltop at night.",
     "science":"Elevated hilltop church open to coastal winds from all sides. Stone arch construction amplifies wind into moaning sounds. Surrounding palm forests create moving shadows in moonlight."},

    {"name":"Kurseong (Victoria Boys' School)","state":"West Bengal","lat":26.8750,"lon":88.2750,
     "darkness":0.95,"enclosure":0.35,"acoustics":0.03,"visibility":0.55,
     "folklore":"Death Trail behind the school; headless boy appears to walkers. British-era school locked after sunset due to paranormal reports.",
     "science":"Darjeeling hills with extremely dense fog and near-zero artificial light in the forested zone. Tea garden mist creates visibility below 5 meters — the perfect environment for the brain to generate phantom shapes."},

    {"name":"Naale Ba (Bangalore)","state":"Karnataka","lat":12.9716,"lon":77.5946,
     "darkness":0.03,"enclosure":0.15,"acoustics":0.02,"visibility":0.25,
     "folklore":"In the 1990s, a witch roamed Bangalore knocking on doors at night mimicking family members' voices. 'Naale Ba' (come tomorrow) written on doors to ward her off.",
     "science":"Urban legend amplified by mass hysteria. From an environmental standpoint, Bangalore has very low susceptibility — well-lit, flat terrain, low enclosure — explaining why the phenomenon was cultural rather than perceptual."},

    {"name":"Jamali Kamali Tomb","state":"Delhi","lat":28.5174,"lon":77.1855,
     "darkness":0.12,"enclosure":0.50,"acoustics":0.06,"visibility":0.40,
     "folklore":"Sufi poet Jamali and his companion Kamali buried side-by-side. Visitors report being pushed, slapped, and feeling suffocated inside the chamber.",
     "science":"Small enclosed tomb chamber with limited air circulation. CO₂ buildup in confined space causes lightheadedness and panic. Narrow arched entrance creates pressure changes that feel like being pushed."},

    {"name":"Rana Palace (Gondal)","state":"Gujarat","lat":21.9600,"lon":70.7970,
     "darkness":0.10,"enclosure":0.55,"acoustics":0.04,"visibility":0.35,
     "folklore":"Royal palace where guests report hearing a piano playing by itself and seeing a woman in white.",
     "science":"Large resonant palace chambers with high ceilings produce standing waves. Wind vibrating piano strings through unsealed windows creates phantom music. Curtains moving in drafts create human-like silhouettes."},

    {"name":"Lothian Cemetery","state":"Delhi","lat":28.6680,"lon":77.2300,
     "darkness":0.10,"enclosure":0.30,"acoustics":0.03,"visibility":0.45,
     "folklore":"Sir Nicholas Dodd-Lothian shot himself after heartbreak; his headless ghost rides a horse through the cemetery.",
     "science":"Old-growth trees in the cemetery create pockets of darkness in otherwise lit surroundings. Adaptation blindness when transitioning from lit roads to dark tree cover causes momentary visual distortions."},

    {"name":"Malcha Mahal","state":"Delhi","lat":28.5950,"lon":77.1750,
     "darkness":0.25,"enclosure":0.65,"acoustics":0.04,"visibility":0.70,
     "folklore":"Princess Wilayat Mahal and her children lived as hermits here. After her death by cyanide, the children continued living in isolation. Trespassers are chased by vicious dogs and curse-boards.",
     "science":"Isolated hunting lodge deep within Delhi Ridge forest with dense canopy. Near-complete visual occlusion by trees at just 200 meters from surrounding well-lit roads creates jarring perceptual transition. Wild animal sounds in the forest add to unease."},

    {"name":"Rajasthan Canal (Bhootiya Ped)","state":"Rajasthan","lat":27.9000,"lon":73.8000,
     "darkness":0.80,"enclosure":0.20,"acoustics":0.02,"visibility":0.60,
     "folklore":"A giant banyan tree on the Rajasthan Canal road where vehicles stall, headlights dim, and travelers feel someone sitting on the vehicle's roof.",
     "science":"Massive banyan canopy creates sudden darkness on an otherwise open desert road, causing night-blindness. Root system proximity to the road may disrupt vehicle electronics through ground conductivity. Wind through aerial roots produces sounds resembling breathing."},
]

# ── Data loaders ─────────────────────────────────────────────────────────────

@st.cache_data
def load_india_geojson():
    """Load India boundary GeoJSON as raw dict and as matplotlib Paths for PIP testing."""
    geojson_path = os.path.join(DATA_DIR, "india_boundary.geojson")
    if not os.path.exists(geojson_path):
        return None, None, None
    with open(geojson_path) as f:
        data = json.load(f)
    # Build matplotlib Paths for point-in-polygon testing
    paths = []
    poly_coords = []
    for feature in data.get("features", [data] if "geometry" in data else []):
        geom = feature.get("geometry", feature)
        geom_type = geom.get("type", "")
        coords = geom.get("coordinates", [])
        if geom_type == "Polygon":
            for ring in coords:
                arr = np.array(ring)
                paths.append(MplPath(arr[:, :2]))
                poly_coords.append(arr)
        elif geom_type == "MultiPolygon":
            for polygon in coords:
                for ring in polygon:
                    arr = np.array(ring)
                    paths.append(MplPath(arr[:, :2]))
                    poly_coords.append(arr)
    return data, paths, poly_coords

@st.cache_data
def load_susceptibility():
    csv = os.path.join(DATA_DIR, "susceptibility_results.csv")
    if os.path.exists(csv):
        return pd.read_csv(csv)
    return None

# ── Filter susceptibility data to India only ─────────────────────────────────
@st.cache_data
def get_india_heat_data(susc_csv_path, geojson_path):
    """Load susceptibility points and filter to inside India only."""
    import pandas as pd, numpy as np, json, os
    from matplotlib.path import Path as MplPath
    if not os.path.exists(susc_csv_path):
        return pd.DataFrame(), []
    df = pd.read_csv(susc_csv_path)
    
    if not os.path.exists(geojson_path):
        return df, df[["lat","lon","score"]].values.tolist()
        
    with open(geojson_path) as f:
        data = json.load(f)
        
    points = np.column_stack([df["lon"].values, df["lat"].values])
    mask = np.zeros(len(points), dtype=bool)
    
    for feature in data.get("features", [data] if "geometry" in data else []):
        geom = feature.get("geometry", feature)
        geom_type = geom.get("type", "")
        coords = geom.get("coordinates", [])
        if geom_type == "Polygon":
            for ring in coords:
                path = MplPath(np.array(ring)[:, :2])
                mask |= path.contains_points(points)
        elif geom_type == "MultiPolygon":
            for polygon in coords:
                for ring in polygon:
                    path = MplPath(np.array(ring)[:, :2])
                    mask |= path.contains_points(points)
                    
    filtered = df[mask]
    return filtered, filtered[["lat","lon","score"]].values.tolist()

# ── Build haunted sites DataFrame ────────────────────────────────────────────
def build_sites_df(wd, we, wa, wv):
    df = pd.DataFrame(HAUNTED_SITES)
    df["pasi_live"] = (wd*df.darkness + we*df.enclosure + wa*df.acoustics + wv*df.visibility).round(4)
    df = df.sort_values("pasi_live", ascending=False).reset_index(drop=True)
    df.index += 1
    return df

sites_df = build_sites_df(w_d, w_e, w_a, w_v)
india_geojson, india_paths, india_poly_coords = load_india_geojson()
susc_df = load_susceptibility()

def filter_points_inside_india(_paths, lats, lons):
    """Return boolean mask of points inside any India polygon."""
    points = np.column_stack([lons, lats])  # Path expects (x, y) = (lon, lat)
    mask = np.zeros(len(points), dtype=bool)
    for path in _paths:
        mask |= path.contains_points(points)
    return mask

susc_csv_path = os.path.join(DATA_DIR, "susceptibility_results.csv")
geojson_path = os.path.join(DATA_DIR, "india_boundary.geojson")
susc_india, heat_data = get_india_heat_data(susc_csv_path, geojson_path)

# ══════════════════════════════════════════════════════════════════════
# TAB LAYOUT
# ══════════════════════════════════════════════════════════════════════
tab1, tab2, tab3 = st.tabs(["🗺️ Susceptibility Map", "📋 Haunted Site Scorecards", "🤖 ML Analytics"])

# ─────────────────────────────────────────────────────────────────────
# TAB 1: INTERACTIVE MAP (folium)
# ─────────────────────────────────────────────────────────────────────
with tab1:
    st.markdown("### 🌙 Mapping Perceptual Anomaly Zones in India")
    st.caption("Zones where environmental factors — low illumination, terrain enclosure, vegetation & acoustics — are most likely to induce perceptual anomalies. **Click any marker for details. Click the map for nearest-point analysis.**")

    m = folium.Map(location=[22.5, 79.0], zoom_start=5, tiles="CartoDB positron",
                   control_scale=True, min_zoom=4, max_zoom=12)

    # ── India boundary outline ───────────────────────────────────────
    if india_geojson is not None:
        folium.GeoJson(
            india_geojson,
            name="India Boundary",
            style_function=lambda x: {
                "fillColor": "transparent",
                "color": "#000000",
                "weight": 3,
                "fillOpacity": 0,
            },
        ).add_to(m)

        # Mask outside India — large rectangle with India cut out
        # Create an outer boundary (whole world) and inner boundary (India)
        outer = [[-90, -180], [-90, 180], [90, 180], [90, -180], [-90, -180]]
        # Get all India polygon coordinates for the mask
        india_mask_coords = []
        for feature in india_geojson.get("features", []):
            geom = feature.get("geometry", {})
            geom_type = geom.get("type", "")
            coords = geom.get("coordinates", [])
            if geom_type == "MultiPolygon":
                for poly in coords:
                    # Outer ring only, reversed for hole
                    ring = [[c[1], c[0]] for c in poly[0]]
                    india_mask_coords.append(ring)
            elif geom_type == "Polygon":
                ring = [[c[1], c[0]] for c in coords[0]]
                india_mask_coords.append(ring)

        # Add single mask polygon covering the world, with holes for each India sub-polygon
        locations = [outer] + india_mask_coords
        folium.Polygon(
            locations=locations,
            color="none",
            fill=True,
            fill_color="#222222",
            fill_opacity=0.75,
        ).add_to(m)

    # ── Heatmap — India-only data ────────────────────────────────────
    if heat_data:
        plugins.HeatMap(
            heat_data, name="PASI Heatmap",
            min_opacity=0.5, radius=24, blur=18, max_zoom=8,
            gradient={
                0.20: "#0000ff", # blue
                0.40: "#00ff00", # lime
                0.60: "#ffff00", # yellow
                0.80: "#ff0000", # red
                1.00: "#330000"  # very dark red (almost black)
            },
        ).add_to(m)

    # ── Haunted site markers with detailed popups ────────────────────
    cluster = plugins.MarkerCluster(name="📍 Haunted Sites (clustered)")

    for _, row in sites_df.iterrows():
        pasi = row.pasi_live
        dark, encl = row.darkness, row.enclosure
        acou, vis  = row.acoustics, row.visibility

        if   pasi > 0.55: color, risk = "red",    "HIGH"
        elif pasi > 0.35: color, risk = "orange", "MODERATE"
        else:             color, risk = "blue",   "LOW"

        pasi_color = "#ff4444" if pasi > 0.55 else ("#ffaa00" if pasi > 0.35 else "#44aaff")

        def bar(val, label, col="#4CAF50"):
            pct = int(val * 100)
            return (f"<div style='margin:3px 0'><span style='color:#aaa;font-size:10px'>{label}</span>"
                    f"<div style='background:#333;border-radius:3px;height:7px'>"
                    f"<div style='background:{col};width:{pct}%;height:7px;border-radius:3px'></div></div>"
                    f"<span style='font-size:9px;color:#ccc'>{val:.3f}</span></div>")

        popup_html = f"""
        <div style='font-family:sans-serif;width:300px;background:#1a1a2e;color:#eee;padding:14px;border-radius:10px'>
          <h4 style='margin:0 0 3px 0;color:#fff;font-size:1.05rem'>{row['name']}</h4>
          <div style='color:#aaa;font-size:10px;margin-bottom:6px'>📍 {row['state']} | {row.lat:.4f}°N, {row.lon:.4f}°E</div>
          <div style='font-size:1.4rem;font-weight:700;color:{pasi_color};margin-bottom:2px'>PASI: {pasi:.3f}</div>
          <div style='font-size:10px;color:#888;margin-bottom:8px'>Risk: <b style="color:{pasi_color}">{risk}</b></div>
          <hr style='border-color:#333;margin:6px 0'>
          {bar(dark,  "🌑 Darkness",   "#6c63ff")}
          {bar(encl,  "⛰️ Enclosure",  "#ff6b6b")}
          {bar(acou,  "🔊 Acoustics",  "#ffd93d")}
          {bar(vis,   "🌿 Visibility", "#6bff6b")}
          <hr style='border-color:#333;margin:8px 0'>
          <div style='font-size:10px;margin-bottom:6px'>
            <b style='color:#ff9f43'>👻 Folklore:</b><br>
            <span style='color:#ddd;font-style:italic'>{row['folklore']}</span>
          </div>
          <div style='font-size:10px'>
            <b style='color:#54a0ff'>🔬 Science:</b><br>
            <span style='color:#ccc'>{row['science']}</span>
          </div>
        </div>"""

        # Outer glow circle
        folium.Circle(
            location=[row.lat, row.lon], radius=30000,
            color=color, fill=True, fill_opacity=0.06, weight=0,
        ).add_to(m)

        folium.Marker(
            location=[row.lat, row.lon],
            popup=folium.Popup(popup_html, max_width=320),
            tooltip=f"<b>{row['name']}</b><br>PASI: {pasi:.3f} ({risk})",
            icon=folium.Icon(color=color, icon="exclamation-sign"),
        ).add_to(cluster)

    cluster.add_to(m)
    plugins.Fullscreen().add_to(m)
    plugins.MiniMap(tile_layer="CartoDB dark_matter", toggle_display=True).add_to(m)
    folium.LayerControl().add_to(m)

    # ── Render map + click inspector ─────────────────────────────────
    col_map, col_info = st.columns([3, 1])
    with col_map:
        st_data = st_folium(m, width=None, height=600, key="anomaly_map",
                            returned_objects=["last_clicked"])

    with col_info:
        st.subheader("📊 Location Inspector")
        if st_data and st_data.get("last_clicked"):
            lat = st_data["last_clicked"]["lat"]
            lng = st_data["last_clicked"]["lng"]
            st.write(f"**Clicked:** `{lat:.4f}°N, {lng:.4f}°E`")
            st.divider()

            # Check if click is near a haunted site
            site_dists = ((sites_df["lat"] - lat)**2 + (sites_df["lon"] - lng)**2)
            nearest_site_idx = site_dists.idxmin()
            nearest_site = sites_df.loc[nearest_site_idx]
            site_dist_km = np.sqrt(site_dists.min()) * 111  # rough km

            if site_dist_km < 50:
                st.markdown(f"**📍 Near: {nearest_site['name']}** ({nearest_site['state']})")
                pasi = nearest_site.pasi_live
                level = "🔴 High" if pasi > 0.55 else ("🟡 Moderate" if pasi > 0.35 else "🟢 Low")
                st.metric("PASI Score", f"{pasi:.3f}")
                st.write(f"**Risk:** {level}")
                st.progress(min(float(nearest_site.darkness), 1.0),  text=f"🌑 Darkness  {nearest_site.darkness:.3f}")
                st.progress(min(float(nearest_site.enclosure), 1.0), text=f"⛰️ Enclosure {nearest_site.enclosure:.3f}")
                st.progress(min(float(nearest_site.acoustics), 1.0), text=f"🔊 Acoustics {nearest_site.acoustics:.3f}")
                st.progress(min(float(nearest_site.visibility), 1.0),text=f"🌿 Visibility {nearest_site.visibility:.3f}")
                st.divider()
                st.markdown(f"**👻 Folklore:** _{nearest_site['folklore']}_")
                st.markdown(f"**🔬 Science:** {nearest_site['science']}")
            else:
                # Look up nearest grid point from susceptibility data
                if len(susc_india) > 0:
                    dist = ((susc_india["lat"]-lat)**2 + (susc_india["lon"]-lng)**2).values
                    near = susc_india.iloc[dist.argmin()]
                    score = float(near.get("score", 0))
                    dark  = float(near.get("darkness", 0))
                    encl  = float(near.get("enclosure", 0))
                    acou  = float(near.get("acoustics", 0))
                    level = "🔴 High" if score > 0.65 else ("🟡 Moderate" if score > 0.35 else "🟢 Low")

                    st.metric("PASI Score", f"{score:.3f}")
                    st.write(f"**Risk:** {level}")
                    st.progress(min(dark, 1.0),  text=f"🌑 Darkness  {dark:.3f}")
                    st.progress(min(encl, 1.0),  text=f"⛰️ Enclosure {encl:.3f}")
                    st.progress(min(acou, 1.0),  text=f"🔊 Acoustics {acou:.3f}")
                    st.divider()

                    # Generate dynamic description based on scores
                    explanations = []
                    if dark > 0.6:
                        explanations.append("**Very dark region** — extremely low artificial light creates conditions for visual misperceptions")
                    elif dark > 0.3:
                        explanations.append("**Moderately dark** — reduced nighttime illumination may trigger mild visual unease")
                    else:
                        explanations.append("**Well-lit area** — sufficient artificial light minimises darkness-induced anomalies")

                    if encl > 0.6:
                        explanations.append("**High terrain enclosure** — valleys/ravines create echo chambers and feelings of being watched")
                    elif encl > 0.3:
                        explanations.append("**Moderate enclosure** — some terrain features may create unusual acoustic effects")
                    else:
                        explanations.append("**Open terrain** — flat/open landscape poses low enclosure risk")

                    if acou > 0.04:
                        explanations.append("**Acoustic hotspot** — geological features may produce infrasound (18-19 Hz vibrations) causing dread")
                    else:
                        explanations.append("**Low acoustic risk** — terrain unlikely to produce anomalous sounds")

                    if score > 0.55:
                        st.error("⚠️ **HIGH susceptibility zone** — multiple environmental factors converge here to create conditions where perceptual anomalies are highly likely.")
                    elif score > 0.35:
                        st.warning("⚡ **Moderate susceptibility** — some environmental factors are elevated but not extreme.")
                    else:
                        st.success("✅ **Low susceptibility** — environmental conditions are unlikely to cause perceptual anomalies.")

                    for exp in explanations:
                        st.markdown(f"• {exp}")
                else:
                    st.info("No susceptibility data available for this location.")
        else:
            st.info("👆 Click anywhere on the map to inspect environmental scores and get a description of why the area is susceptible (or not) to perceptual anomalies.")

    with st.expander("📖 Index Methodology"):
        st.markdown("""
| Index | Source | Computation |
|---|---|---|
| **Darkness** | VIIRS Nighttime Lights | Inverse radiance — low light → high darkness |
| **Terrain Enclosure** | SRTM DEM | Pixels below local 21×21 mean (valleys, ravines) |
| **Acoustic Potential** | SRTM slope + Sentinel-2 NDVI | Steep relief × low vegetation → echo potential |
| **Visibility Reduction** | Sentinel-2 NDVI | Dense canopy / fog proxy |

**PASI = w₁·Darkness + w₂·Enclosure + w₃·Acoustic + w₄·Visibility** (weights adjustable in sidebar)
        """)

# ─────────────────────────────────────────────────────────────────────
# TAB 2: SCORECARDS
# ─────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown("### 📋 PASI Validation — Reported Paranormal Sites")
    st.caption(f"**{len(sites_df)} haunted locations** scored using four environmental sub-indices. Sites ranked by live PASI (adjust weights in sidebar).")

    # ── Summary table ────────────────────────────────────────────────────
    table_df = sites_df[["name","state","pasi_live","darkness","enclosure","acoustics","visibility"]].copy()
    table_df.columns = ["Site","State","PASI ▼","Darkness","Enclosure","Acoustics","Visibility"]

    def color_pasi(val):
        if val > 0.55:   return "background-color:#3a0000;color:#ff4444;font-weight:700"
        elif val > 0.35: return "background-color:#3a2800;color:#ffaa00;font-weight:700"
        else:            return "background-color:#001a2b;color:#44aaff"

    styled = table_df.style\
        .format({"PASI ▼":"{:.4f}","Darkness":"{:.4f}","Enclosure":"{:.4f}",
                 "Acoustics":"{:.4f}","Visibility":"{:.4f}"})\
        .map(color_pasi, subset=["PASI ▼"])\
        .background_gradient(subset=["Darkness","Enclosure","Acoustics","Visibility"],
                             cmap="RdYlGn_r", vmin=0, vmax=1)

    st.dataframe(styled, use_container_width=True, height=500)

    # ── Detailed score cards ──────────────────────────────────────────────
    st.divider()
    st.markdown("#### Detailed Scorecards")
    cols = st.columns(2)
    for i, (_, row) in enumerate(sites_df.iterrows()):
        pasi  = row.pasi_live
        dark  = row.darkness
        encl  = row.enclosure
        acou  = row.acoustics
        vis   = row.visibility
        level = "🔴 HIGH" if pasi>0.55 else ("🟠 MODERATE" if pasi>0.35 else "🔵 LOW")
        col   = cols[i % 2]
        with col:
            st.markdown(f"""
<div class="score-card">
  <h4><span class="rank-badge">#{i+1}</span>{row['name']}</h4>
  <div class="loc">📍 {row['state']} &nbsp;|&nbsp; {row.lat:.4f}°N, {row.lon:.4f}°E &nbsp;|&nbsp; Risk: <b>{level}</b></div>
  <div class="pasi-big" style="color:{'#ff4444' if pasi>0.55 else ('#ffaa00' if pasi>0.35 else '#44aaff')}">{pasi:.4f}</div>
  <div style="font-size:0.78rem;color:#888;margin-bottom:10px">Perceptual Anomaly Susceptibility Index</div>
""", unsafe_allow_html=True)
            st.progress(min(dark, 1.0), text=f"🌑 Darkness  {dark:.4f}")
            st.progress(min(encl, 1.0), text=f"⛰️ Enclosure {encl:.4f}")
            st.progress(min(acou, 1.0), text=f"🔊 Acoustics {acou:.4f}")
            st.progress(min(vis,  1.0), text=f"🌿 Visibility {vis:.4f}")
            st.markdown(f"""
  <div style='font-size:0.75rem;color:#ff9f43;margin-top:8px'>👻 {row['folklore']}</div>
  <div style='font-size:0.75rem;color:#54a0ff;margin-top:4px;padding-bottom:4px'>🔬 {row['science']}</div>
</div>""", unsafe_allow_html=True)
            st.write("")

    # ── Bar chart comparison ─────────────────────────────────────────────
    st.divider()
    st.markdown("#### PASI Comparison Chart")
    chart_df = sites_df[["name","pasi_live","darkness","enclosure","acoustics","visibility"]]\
        .set_index("name")
    chart_df.columns = ["PASI","Darkness","Enclosure","Acoustics","Visibility"]
    st.bar_chart(chart_df, height=400)

# ─────────────────────────────────────────────────────────────────────
# TAB 3: ML ANALYTICS
# ─────────────────────────────────────────────────────────────────────

@st.cache_data
def load_ml_clusters():
    path = os.path.join(DATA_DIR, "ml_clusters.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

@st.cache_data
def load_ml_importance():
    path = os.path.join(DATA_DIR, "ml_feature_importance.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

@st.cache_data
def load_ml_summary():
    path = os.path.join(DATA_DIR, "ml_summary.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None

CLUSTER_COLORS = ["#6c63ff", "#ff6b6b", "#ffd93d", "#6bff6b", "#ff9ff3",
                  "#54a0ff", "#ff9f43", "#ee5a24"]
CLUSTER_ICONS  = ["🟣", "🔴", "🟡", "🟢", "🩷", "🔵", "🟠", "🟤"]

with tab3:
    st.markdown("### 🤖 Machine Learning Analytics")
    st.caption("Unsupervised clustering, feature importance analysis, dimensionality reduction, and spatial statistics.")

    ml_df    = load_ml_clusters()
    ml_imp   = load_ml_importance()
    ml_summ  = load_ml_summary()

    if ml_df is None or ml_summ is None:
        st.warning("⚠️ ML results not found. Run `python scripts/ml_analysis.py` first.")
    else:
        # ── Top-level metrics ────────────────────────────────────────────
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Clusters Found", ml_summ["n_clusters"])
        m2.metric("Silhouette Score", f"{ml_summ['silhouette_score']:.4f}")
        pca_total = sum(ml_summ["pca_explained_variance"])
        m3.metric("PCA Variance", f"{pca_total:.1%}")
        mi = ml_summ["morans_i"]
        m4.metric("Moran's I", f"{mi['I']:.4f}",
                  delta=f"z={mi['z_score']:.1f}, p<0.001" if mi["significant"] else "Not significant")

        st.divider()

        # ── Row 1: Cluster Map  |  Feature Importance ────────────────────
        col_map, col_imp = st.columns([3, 2])

        with col_map:
            st.markdown("#### 🗺️ Anomaly Archetype Clusters")

            # Static cluster map with India boundary
            fig2, ax2 = plt.subplots(figsize=(9, 9), dpi=120)
            fig2.patch.set_facecolor("#0e1117")
            ax2.set_facecolor("#0e1117")

            # Draw India boundary fill
            if india_poly_coords:
                for coords in india_poly_coords:
                    poly = MplPolygon(coords[:, :2], closed=True,
                                     facecolor="#1a1a2e", edgecolor="#555", linewidth=1.0)
                    ax2.add_patch(poly)

            # Sample + filter to India
            sample_n = min(3000, len(ml_df))
            ml_sample = ml_df.sample(n=sample_n, random_state=42)
            if india_paths:
                ml_mask = filter_points_inside_india(india_paths, ml_sample["lat"].values, ml_sample["lon"].values)
                ml_sample = ml_sample[ml_mask]

            n_clusters = ml_summ["n_clusters"]
            for c_id in range(n_clusters):
                cluster_data = ml_sample[ml_sample["cluster"] == c_id]
                ccolor = CLUSTER_COLORS[c_id % len(CLUSTER_COLORS)]
                profiles = ml_summ.get("cluster_profiles", [])
                label = next((p["name"] for p in profiles if p["cluster"] == c_id), f"Cluster {c_id}")
                ax2.scatter(cluster_data["lon"], cluster_data["lat"],
                            c=ccolor, s=5, alpha=0.6, edgecolors="none",
                            label=label, zorder=2)

            # Redraw boundary on top
            if india_poly_coords:
                for coords in india_poly_coords:
                    ax2.plot(coords[:, 0], coords[:, 1], color="#777", linewidth=0.8, zorder=3)

            ax2.set_xlim(67.5, 98.0)
            ax2.set_ylim(6.0, 37.5)
            ax2.set_aspect("equal")
            ax2.tick_params(colors="#666", labelsize=7)
            for spine in ax2.spines.values():
                spine.set_color("#333")
            ax2.legend(loc="lower left", fontsize=7, framealpha=0.7,
                       facecolor="#1a1a2e", edgecolor="#444", labelcolor="#ccc")
            ax2.set_title("K-Means Anomaly Archetype Clusters",
                          color="#eee", fontsize=12, fontweight="bold", pad=10)
            ax2.set_xlabel("Longitude (°E)", color="#888", fontsize=8)
            ax2.set_ylabel("Latitude (°N)", color="#888", fontsize=8)

            plt.tight_layout()
            st.pyplot(fig2, width="stretch")
            plt.close(fig2)

        with col_imp:
            st.markdown("#### 📊 Random Forest Feature Importance")
            st.caption(f"RF trained on cluster labels — accuracy: {ml_summ['rf_accuracy']:.1%}")

            if ml_imp is not None:
                imp_chart = ml_imp.copy()
                imp_chart = imp_chart.sort_values("importance", ascending=True)

                for _, row in imp_chart.iterrows():
                    feat = row["feature"].capitalize()
                    imp_val = row["importance"]
                    pct = int(imp_val * 100)
                    icons = {"darkness": "🌑", "enclosure": "⛰️",
                             "acoustics": "🔊", "visibility": "🌿"}
                    icon = icons.get(row["feature"], "📊")
                    bar_color = {"darkness": "#6c63ff", "enclosure": "#ff6b6b",
                                 "acoustics": "#ffd93d", "visibility": "#6bff6b"}.get(row["feature"], "#44aaff")
                    st.markdown(f"""
<div style='margin:10px 0'>
  <div style='font-size:0.9rem;color:#ccc;margin-bottom:4px'>{icon} {feat}</div>
  <div style='background:#333;border-radius:6px;height:22px;position:relative'>
    <div style='background:{bar_color};width:{pct}%;height:22px;border-radius:6px;transition:width 0.5s'></div>
    <span style='position:absolute;right:8px;top:2px;font-size:0.8rem;color:#fff;font-weight:600'>{imp_val:.3f}</span>
  </div>
</div>""", unsafe_allow_html=True)

            st.markdown("---")
            st.markdown("#### 🌐 Spatial Autocorrelation")
            if mi["significant"]:
                st.success(f"**Moran's I = {mi['I']:.4f}** (z = {mi['z_score']:.1f}, p ≈ 0)  \n"
                           f"→ Strong positive spatial autocorrelation: high-PASI zones **cluster geographically**.")
            else:
                st.info(f"Moran's I = {mi['I']:.4f} — no significant spatial pattern detected.")

        st.divider()

        # ── Row 2: PCA Scatter  |  Cluster Profiles ──────────────────────
        col_pca, col_prof = st.columns([3, 2])

        with col_pca:
            st.markdown("#### 🔬 PCA Projection (2D)")
            st.caption(f"PC1 explains {ml_summ['pca_explained_variance'][0]:.1%}, "
                       f"PC2 explains {ml_summ['pca_explained_variance'][1]:.1%} of variance")

            if "pc1" in ml_df.columns and "pc2" in ml_df.columns:
                pca_sample = ml_df.sample(n=min(3000, len(ml_df)), random_state=42).copy()

                fig3, ax3 = plt.subplots(figsize=(8, 6), dpi=120)
                fig3.patch.set_facecolor("#0e1117")
                ax3.set_facecolor("#0e1117")

                profiles = ml_summ["cluster_profiles"]
                name_map = {p["cluster"]: p["name"] for p in profiles}

                for c_id in range(n_clusters):
                    c_data = pca_sample[pca_sample["cluster"] == c_id]
                    ccolor = CLUSTER_COLORS[c_id % len(CLUSTER_COLORS)]
                    label = name_map.get(c_id, f"Cluster {c_id}")
                    ax3.scatter(c_data["pc1"], c_data["pc2"],
                                c=ccolor, s=6, alpha=0.5, label=label, edgecolors="none")

                ax3.legend(loc="best", fontsize=7, framealpha=0.7,
                           facecolor="#1a1a2e", edgecolor="#444", labelcolor="#ccc")
                ax3.set_xlabel("PC1", color="#aaa", fontsize=9)
                ax3.set_ylabel("PC2", color="#aaa", fontsize=9)
                ax3.set_title("PCA 2D Projection by Cluster", color="#eee", fontsize=11, fontweight="bold")
                ax3.tick_params(colors="#666", labelsize=7)
                for spine in ax3.spines.values():
                    spine.set_color("#333")

                plt.tight_layout()
                st.pyplot(fig3, width="stretch")
                plt.close(fig3)

        with col_prof:
            st.markdown("#### 🏷️ Cluster Profiles")
            profiles = ml_summ["cluster_profiles"]

            for p in sorted(profiles, key=lambda x: -x["mean_score"]):
                c_id = p["cluster"]
                icon = CLUSTER_ICONS[c_id % len(CLUSTER_ICONS)]
                ccolor = CLUSTER_COLORS[c_id % len(CLUSTER_COLORS)]
                pasi_color = "#ff4444" if p["mean_score"] > 0.55 else ("#ffaa00" if p["mean_score"] > 0.35 else "#44aaff")

                st.markdown(f"""
<div style='background:#1e1e2e;border-radius:12px;padding:14px 18px;margin:8px 0;
            border-left:5px solid {ccolor}'>
  <div style='display:flex;justify-content:space-between;align-items:center'>
    <h4 style='margin:0;color:#fff;font-size:1rem'>{icon} {p['name']}</h4>
    <span style='font-size:1.3rem;font-weight:700;color:{pasi_color}'>{p['mean_score']:.3f}</span>
  </div>
  <div style='color:#888;font-size:0.78rem;margin:4px 0 8px 0'>
    {p['count']:,} points &nbsp;|&nbsp; Dominant: <b>{p['dominant_feature'].capitalize()}</b>
  </div>
</div>""", unsafe_allow_html=True)
                c1, c2, c3, c4 = st.columns(4)
                c1.progress(min(float(p["mean_darkness"]), 1.0),   text=f"🌑 {p['mean_darkness']:.2f}")
                c2.progress(min(float(p["mean_enclosure"]), 1.0),  text=f"⛰️ {p['mean_enclosure']:.2f}")
                c3.progress(min(float(p["mean_acoustics"]), 1.0) if p["mean_acoustics"] > 0.005 else 0.005, text=f"🔊 {p['mean_acoustics']:.2f}")
                c4.progress(min(float(p.get("mean_visibility", 0)), 1.0) if p.get("mean_visibility", 0) > 0.005 else 0.005, text=f"🌿 {p.get('mean_visibility', 0):.2f}")

        with st.expander("📖 ML Methodology"):
            st.markdown("""
| Technique | Purpose | Implementation |
|---|---|---|
| **K-Means Clustering** | Discover anomaly archetypes | Optimal k via silhouette score; StandardScaler preprocessing |
| **Random Forest** | Feature importance ranking | 200 trees, max_depth=12; trained on cluster labels |
| **PCA (2D)** | Dimensionality reduction | 2-component projection for scatter visualization |
| **Moran's I** | Spatial autocorrelation | k-NN weight matrix (k=8); tests if PASI clusters geographically |

**Pipeline**: `python scripts/ml_analysis.py` → generates `ml_clusters.csv`, `ml_feature_importance.csv`, `ml_summary.json`
            """)
