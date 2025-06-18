import streamlit as st
import pandas as pd
import numpy as np
import folium
from folium.plugins import HeatMap, MarkerCluster
from streamlit_folium import folium_static
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier


st.set_page_config(page_title="Bengaluru Flood Dashboard", layout="wide")


def set_background(image_file):
    import base64
    with open(image_file, "rb") as image:
        encoded = base64.b64encode(image.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: linear-gradient(rgba(0, 0, 0, 0.47), rgba(0, 0, 0, 0.47)),
                        url("data:image/jpg;base64,{encoded}");
            background-size: cover;
            background-attachment: fixed;
            background-position: center;
            color: black;
        }}
        .block-style {{
            background-color: rgba(255, 255, 255, 0.6);
            padding: 1.5rem;
            border-radius: 0.75rem;
            margin-top: 1rem;
            margin-bottom: 1rem;
            color: black;
        }}
        h1, h2, h3, h4, h5, h6, .stMetricValue, .stMetricLabel {{
            color:white!important;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

set_background("background.jpg")


@st.cache_data
def load_data():
    return pd.read_csv("bengaluru-182.csv")

df = load_data()

features = ["rainfall", "elevation", "distance_to_river"]
target = "severity_code"
severity_map = {0: "Low", 1: "Medium", 2: "High"}
severity_color = {"Low": "green", "Medium": "orange", "High": "red"}

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(df[features], df[target])


st.markdown("""
<div style='background-color: #0066CC; padding: 1rem; border-radius: 0.5rem;'>
    <h1 style='color: white;'>Bengaluru Flood Risk Dashboard</h1>
</div>
""", unsafe_allow_html=True)


st.sidebar.header("Area Selection")
area_list = sorted(df["area"].unique())
selected_area = st.sidebar.selectbox("Choose Area", ["-- Select Area --"] + area_list)


st.subheader(" Flood Severity Map")

if selected_area != "-- Select Area --":
    area_group = df[df["area"] == selected_area]
    center_lat = area_group["latitude"].mean()
    center_lon = area_group["longitude"].mean()
    zoom_lvl = 13
else:
    center_lat, center_lon, zoom_lvl = 12.9716, 77.5946, 11

base_map = folium.Map(location=[center_lat, center_lon], zoom_start=zoom_lvl, control_scale=True)

layer_toggle = st.sidebar.multiselect("Enable Map Layers", ["Cluster View", "Heatmap Overlay"])

if "Cluster View" in layer_toggle:
    marker_cluster = MarkerCluster().add_to(base_map)


if selected_area != "-- Select Area --":
    folium.Marker(
        location=[center_lat, center_lon],
        popup=f"📍 {selected_area}",
        icon=folium.Icon(color="blue", icon="tag")
    ).add_to(base_map)

    folium.Circle(
        location=[center_lat, center_lon],
        radius=800,
        color='blue',
        fill=True,
        fill_opacity=0.1
    ).add_to(base_map)

for _, row in df.iterrows():
    sev = row["severity"].capitalize()
    color = severity_color.get(sev, "gray")
    marker = folium.CircleMarker(
        location=[row["latitude"], row["longitude"]],
        radius=5,
        popup=f"{row['area']} - {sev}",
        color=color,
        fill=True,
        fill_color=color,
        fill_opacity=0.6,
    )
    if "Cluster View" in layer_toggle:
        marker.add_to(marker_cluster)
    else:
        marker.add_to(base_map)

# Heatmap
if "Heatmap Overlay" in layer_toggle:
    heat_data = [[row["latitude"], row["longitude"], row["rainfall"]] for _, row in df.iterrows()]
    HeatMap(heat_data).add_to(base_map)

folium_static(base_map)

if selected_area != "-- Select Area --":
    st.markdown('<div class="block-style">', unsafe_allow_html=True)
    st.subheader(f" AI Prediction for {selected_area}")
    rain = area_group["rainfall"].mean()
    elev = area_group["elevation"].mean()
    dist = area_group["distance_to_river"].mean()
    pred = model.predict([[rain, elev, dist]])[0]
    sev = severity_map[pred]
    proba = model.predict_proba([[rain, elev, dist]])[0][pred]

    st.success(f" Predicted Severity: `{sev}` (Confidence: {proba*100:.1f}%)")
    st.metric(" Avg Rainfall (mm)", f"{rain:.2f}")
    st.metric(" Elevation (m)", f"{elev:.2f}")
    st.metric(" Distance to River (km)", f"{dist:.2f}")

    st.markdown("####  Suggested Precautions")
    if sev == "Low":
        st.info(" Low Risk: Stay aware, monitor local news during heavy rainfall.")
    elif sev == "Medium":
        st.warning(" Moderate Risk: Avoid low-lying areas, prepare for possible flooding.")
    elif sev == "High":
        st.error(" High Risk: Avoid travel, move valuables to higher ground, stay alert for evacuation alerts.")

    st.markdown(" [Check BBMP Alerts](https://varunamitra.karnataka.gov.in/Default/Index?service=FloodForecast)")
    st.markdown('</div>', unsafe_allow_html=True)


st.markdown('<div class="block-style">', unsafe_allow_html=True)
st.markdown("###  Predict Severity for Custom Conditions")
with st.form("custom_form"):
    col1, col2, col3 = st.columns(3)
    with col1:
        user_rain = st.slider("Rainfall (mm)", 0.0, 500.0, 100.0)
    with col2:
        user_elev = st.slider("Elevation (m)", 0.0, 300.0, 100.0)
    with col3:
        user_dist = st.slider("Distance to River (km)", 0.0, 10.0, 5.0)
    submitted = st.form_submit_button("🔍 Predict")

if submitted:
    pred = model.predict([[user_rain, user_elev, user_dist]])[0]
    proba = model.predict_proba([[user_rain, user_elev, user_dist]])[0][pred]
    sev = severity_map[pred]
    st.success(f" Predicted Severity: `{sev}`")
    st.progress(int(proba * 100), text=f"Confidence: {proba*100:.1f}%")
st.markdown('</div>', unsafe_allow_html=True)


st.markdown("###  Area-wise Rainfall and Severity Charts")
col1, col2 = st.columns(2)
with col1:
    rain_avg = df.groupby("area")["rainfall"].mean().sort_values()
    st.bar_chart(rain_avg)
with col2:
    severity_count = df.groupby("area")["severity_code"].mean().map(severity_map).value_counts()
    fig, ax = plt.subplots()
    sns.barplot(x=severity_count.index, y=severity_count.values, ax=ax)
    ax.set_title("Severity Distribution by Area")
    ax.set_ylabel("Number of Areas")
    st.pyplot(fig)


st.markdown("###  Real-time Rainfall (Windy Overlay)")
windy_html = """
<iframe width="100%" height="400" src="https://embed.windy.com/embed2.html?lat=12.9716&lon=77.5946&detailLat=12.9716&detailLon=77.5946&width=650&height=450&zoom=10&level=surface&overlay=rain&menu=true&message=true&marker=true&calendar=24&pressure=true&type=map&location=coordinates&detail=true&metricWind=default&metricTemp=default&radarRange=-1"
frameborder="0"></iframe>
"""
st.components.v1.html(windy_html, height=400)
