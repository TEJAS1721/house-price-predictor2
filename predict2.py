import math
import requests
import numpy as np
import pandas as pd
import streamlit as st
from geopy.geocoders import ArcGIS
import folium
from streamlit_folium import st_folium

# ----------------------------------------------------
# 1. PAGE SETUP (MUST BE RIGHT AFTER IMPORTS)
# ----------------------------------------------------
st.set_page_config(page_title="Real-Time House Price Predictor", page_icon="📍", layout="wide")
st.title("📍 Real-Time House Price Predictor")

st.markdown("""
    <style>
    div[data-testid="stForm"] {
        border: none !important;
        padding: 0 !important;
    }
    div[data-testid="column"] {
        display: flex;
        align-items: flex-end;
    }
    </style>
""", unsafe_allow_html=True)

# ----------------------------------------------------
# 2. CITY ANCHORS & HELPER FUNCTIONS
# ----------------------------------------------------
CITY_HUBS = {
    # Tier 1 Metros
    "Bengaluru": {"lat": 12.9716, "lng": 77.5946, "base_rate": 16000, "tier": 1},
    "Mumbai": {"lat": 19.0657, "lng": 72.8686, "base_rate": 35000, "tier": 1},
    "Delhi NCR": {"lat": 28.6315, "lng": 77.2167, "base_rate": 22000, "tier": 1},
    "Hyderabad": {"lat": 17.4435, "lng": 78.3772, "base_rate": 11000, "tier": 1},
    "Chennai": {"lat": 13.0604, "lng": 80.2496, "base_rate": 12000, "tier": 1},
    "Pune": {"lat": 18.5204, "lng": 73.8567, "base_rate": 10000, "tier": 1},
    "Kolkata": {"lat": 22.5726, "lng": 88.3639, "base_rate": 9500, "tier": 1},

    # Tier 2 Cities
    "Mysuru": {"lat": 12.2958, "lng": 76.6394, "base_rate": 5500, "tier": 2},
    "Mangaluru": {"lat": 12.9141, "lng": 74.8560, "base_rate": 5000, "tier": 2},
    "Hubballi": {"lat": 15.3647, "lng": 75.1240, "base_rate": 4200, "tier": 2},
    "Coimbatore": {"lat": 11.0168, "lng": 76.9558, "base_rate": 5500, "tier": 2},
    "Kochi": {"lat": 9.9312, "lng": 76.2673, "base_rate": 6500, "tier": 2},
    "Visakhapatnam": {"lat": 17.6868, "lng": 83.2185, "base_rate": 5500, "tier": 2},
    "Jaipur": {"lat": 26.9124, "lng": 75.7873, "base_rate": 6000, "tier": 2},
}


def calculate_distance_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2 + 
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2)
    return R * (2 * math.atan2(math.sqrt(a), math.sqrt(1 - a)))


def estimate_location_details(lat, lng):
    min_dist = float('inf')
    nearest_hub_name = None
    nearest_hub = None

    for hub, coords in CITY_HUBS.items():
        dist = calculate_distance_km(lat, lng, coords['lat'], coords['lng'])
        if dist < min_dist:
            min_dist = dist
            nearest_hub_name = hub
            nearest_hub = coords

    if min_dist <= 35:
        tier = nearest_hub['tier']
        decay_rate = 0.04
        calculated_rate = nearest_hub['base_rate'] * math.exp(-decay_rate * min_dist)
        base_rate = max(2800, calculated_rate)
        market_label = f"Tier-{tier} Area near {nearest_hub_name} ({round(min_dist, 1)} km)"
    else:
        tier = 3
        base_rate = 2200
        market_label = f"Tier-3 District / Town ({round(min_dist, 1)} km from {nearest_hub_name})"

    spatial_jitter = 1.0 + (((int(lat * 10000) ^ int(lng * 10000)) % 16) - 8) / 100.0
    final_sqft_price = int(base_rate * spatial_jitter)

    return final_sqft_price, market_label, tier


def get_circle_points(lat, lng, radius_meters=1800, num_points=64):
    points = []
    lat_rad = math.radians(lat)
    for i in range(num_points):
        angle = 2 * math.pi * i / num_points
        dy = radius_meters * math.sin(angle)
        dx = radius_meters * math.cos(angle)
        
        point_lat = lat + (dy / 111000.0)
        point_lng = lng + (dx / (111000.0 * math.cos(lat_rad)))
        points.append([point_lat, point_lng])
    return points


# ----------------------------------------------------
# 3. REAL POI RETRIEVAL VIA OVERPASS BOUNDING BOX
# ----------------------------------------------------
@st.cache_resource
def get_geolocator():
    return ArcGIS(timeout=10)

geolocator = get_geolocator()


@st.cache_data(show_spinner=False)
def fetch_real_nearby_pois(address, lat, lng):
    """Fetches real registered school names using Overpass bounding box query."""
    real_schools = []
    real_transport = []
    seen_names = set()

    # Bounding Box (~5km radius)
    delta_lat = 0.045
    delta_lng = 0.045
    s, w, n, e = lat - delta_lat, lng - delta_lng, lat + delta_lat, lng + delta_lat

    # 1. Direct Overpass Query
    try:
        overpass_url = "https://overpass-api.de/api/interpreter"
        overpass_query = f"""
        [out:json][timeout:15];
        (
          node["amenity"="school"]({s},{w},{n},{e});
          way["amenity"="school"]({s},{w},{n},{e});
          relation["amenity"="school"]({s},{w},{n},{e});
          node["amenity"="college"]({s},{w},{n},{e});
          way["amenity"="college"]({s},{w},{n},{e});
          node["building"="school"]({s},{w},{n},{e});
        );
        out center 20;
        """
        headers = {"User-Agent": "HousePricePredictor/3.0"}
        response = requests.post(overpass_url, data={'data': overpass_query}, headers=headers, timeout=8)
        
        if response.status_code == 200:
            data = response.json()
            elements = data.get("elements", [])

            for elem in elements:
                tags = elem.get("tags", {})
                s_name = tags.get("name") or tags.get("name:en") or tags.get("official_name")

                if s_name:
                    s_name = s_name.strip()
                    if s_name.lower() not in seen_names:
                        if "center" in elem:
                            s_lat, s_lng = elem["center"]["lat"], elem["center"]["lon"]
                        elif "lat" in elem and "lon" in elem:
                            s_lat, s_lng = elem["lat"], elem["lon"]
                        else:
                            continue

                        dist = calculate_distance_km(lat, lng, s_lat, s_lng)
                        seen_names.add(s_name.lower())
                        real_schools.append({
                            "name": s_name,
                            "lat": s_lat,
                            "lng": s_lng,
                            "dist": round(dist, 2)
                        })
    except Exception:
        pass

    # 2. ArcGIS Category Fallback for Schools
    try:
        geo = ArcGIS(timeout=10)
        clean_loc = address.split(',')[0].strip()
        candidates = geo.geocode(f"School in {clean_loc}", exactly_one=False, max_results=10) or []
        
        for item in candidates:
            s_name = item.address.split(',')[0].strip()
            if any(kw in s_name.lower() for kw in ["school", "academy", "vidyalaya", "high", "convent", "public"]):
                s_lat, s_lng = item.latitude, item.longitude
                dist = calculate_distance_km(lat, lng, s_lat, s_lng)

                if s_name.lower() not in seen_names and dist <= 8.0:
                    seen_names.add(s_name.lower())
                    real_schools.append({
                        "name": s_name,
                        "lat": s_lat,
                        "lng": s_lng,
                        "dist": round(dist, 2)
                    })
    except Exception:
        pass

    # 3. Real Transport Stations
    try:
        geo = ArcGIS(timeout=10)
        clean_loc = address.split(',')[0].strip()
        t_candidates = geo.geocode(f"Station in {clean_loc}", exactly_one=False, max_results=5) or []
        seen_t = set()
        
        for item in t_candidates:
            t_name = item.address.split(',')[0].strip()
            dist = calculate_distance_km(lat, lng, item.latitude, item.longitude)
            if t_name.lower() not in seen_t and dist <= 10.0:
                seen_t.add(t_name.lower())
                real_transport.append({
                    "name": t_name,
                    "lat": item.latitude,
                    "lng": item.longitude,
                    "dist": round(dist, 2)
                })
    except Exception:
        pass

    real_schools = sorted(real_schools, key=lambda x: x['dist'])
    real_transport = sorted(real_transport, key=lambda x: x['dist'])

    return real_schools, real_transport


@st.cache_data(show_spinner=False)
def get_location_data(address):
    if not address or len(address.strip()) < 2:
        return None

    raw_query = address.strip()
    if raw_query.isdigit() and len(raw_query) != 6:
        return None

    query = f"{raw_query}, India" if "india" not in raw_query.lower() else raw_query

    try:
        location = geolocator.geocode(query, out_fields="*")
        
        if location and location.latitude and location.longitude:
            raw_data = getattr(location, 'raw', {})
            attributes = raw_data.get('attributes', {}) or raw_data.get('feature', {}).get('attributes', {})
            
            score = attributes.get('Score', 100)
            addr_type = attributes.get('Addr_type', '') or attributes.get('Type', '')

            forbidden_types = ['StreetNameGroup', 'POI', 'Intersection', 'Transit', 'Bus Stop']
            
            if score < 70 or addr_type in forbidden_types:
                return None

            lat, lng = location.latitude, location.longitude
            schools, transports = fetch_real_nearby_pois(location.address, lat, lng)

            return {
                "lat": lat,
                "lng": lng,
                "address": location.address,
                "schools": schools,
                "transports": transports,
                "public_transport_count": max(len(transports), 1),
                "school_count": max(len(schools), 1)
            }
    except Exception:
        pass

    return None


# ----------------------------------------------------
# 4. STREAMLIT USER INTERFACE
# ----------------------------------------------------
if "user_role" not in st.session_state:
    st.session_state.user_role = None

st.subheader("1. Property Location")

search_container, _ = st.columns([2, 3])

with search_container:
    with st.form(key="search_form", border=False):
        c_input, c_btn = st.columns([3, 1], gap="small")
        with c_input:
            location_input = st.text_input(
                "Property Location", 
                placeholder="City, Locality, or Pincode...",
                label_visibility="collapsed"
            )
        with c_btn:
            search_submitted = st.form_submit_button("🔍 Search", use_container_width=True)

if location_input.strip():
    spatial_data = get_location_data(location_input)
    
    if spatial_data:
        st.markdown(
            """
            <style>
            div[data-baseweb="input"] {
                border: 2px solid #28a745 !important;
                border-radius: 8px !important;
                background-color: #f0fff4 !important;
            }
            div[data-baseweb="input"] input {
                color: #155724 !important;
            }
            </style>
            """,
            unsafe_allow_html=True
        )

        base_rate_est, market_label, loc_tier = estimate_location_details(
            spatial_data['lat'], spatial_data['lng']
        )
        
        loc_col, map_col = st.columns([1, 1])

        with loc_col:
            st.markdown(
                f"""
                <div style="background-color: #d4edda; color: #155724; padding: 16px; border-radius: 8px; border: 1px solid #c3e6cb; margin-bottom: 15px;">
                    ✅ <strong>{spatial_data['address']}</strong><br><br>
                    📍 <strong>Classification:</strong> Tier {loc_tier} ({market_label})<br>
                    🚌 <strong>Transport Hubs Found:</strong> {len(spatial_data['transports'])}<br>
                    🏫 <strong>Schools Found:</strong> {len(spatial_data['schools'])}
                </div>
                """, 
                unsafe_allow_html=True
            )

        with map_col:
            lat = spatial_data['lat']
            lng = spatial_data['lng']

            # Satellite Base Map
            m = folium.Map(
                location=[lat, lng],
                zoom_start=13,
                tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
                attr="Esri World Imagery"
            )

            # Outer Boundary Polygon Mask
            world_bounds = [[90, -180], [90, 180], [-90, 180], [-90, -180]]
            hole_cutout = get_circle_points(lat, lng, radius_meters=1800)

            folium.Polygon(
                locations=[world_bounds, hole_cutout],
                color="#000000",
                weight=1,
                fill=True,
                fill_color="#111111",
                fill_opacity=0.6,
                tooltip="Outside Focus Area"
            ).add_to(m)

            folium.PolyLine(
                locations=hole_cutout + [hole_cutout[0]],
                color="#28a745",
                weight=3,
                opacity=0.9,
                tooltip=spatial_data['address']
            ).add_to(m)

            # 🚌 Transport Markers
            for t_node in spatial_data['transports']:
                place_name = t_node['name']
                folium.Marker(
                    location=[t_node['lat'], t_node['lng']],
                    tooltip=folium.Tooltip(f"🚌 {place_name}", permanent=False),
                    icon=folium.Icon(color="blue", icon="info-sign")
                ).add_to(m)

            # 🏫 Real School Markers
            for s_node in spatial_data['schools']:
                place_name = s_node['name']
                folium.Marker(
                    location=[s_node['lat'], s_node['lng']],
                    tooltip=folium.Tooltip(f"🏫 {place_name}", permanent=False),
                    icon=folium.Icon(color="orange", icon="star")
                ).add_to(m)

            st_folium(m, width="100%", height=380, returned_objects=[])

        # ----------------------------------------------------
        # 5. USER INTENT & PREDICTION INPUTS
        # ----------------------------------------------------
        st.subheader("2. Select Your Intent")
        btn_col1, btn_col2, _ = st.columns([1, 1, 3])

        with btn_col1:
            if st.button("🏠 Own", use_container_width=True):
                st.session_state.user_role = "Own"

        with btn_col2:
            if st.button("🔑 Rent", use_container_width=True):
                st.session_state.user_role = "Rent"

        if st.session_state.user_role == "Own":
            st.markdown("---")
            st.subheader("3. Enter Property Details (Own)")
            col1, col2 = st.columns(2)

            with col1:
                sqft = st.slider("Square Feet", min_value=500, max_value=5000, value=1200, step=50)
                bedrooms = st.slider("Bedrooms (BHK)", min_value=1, max_value=6, value=2)
            with col2:
                bathrooms = st.slider("Bathrooms", min_value=1, max_value=5, value=2)
                age = st.slider("Property Age (Years)", min_value=0, max_value=30, value=5)

            if st.button("Predict Buying Price", type="primary"):
                total_price = (
                    (sqft * base_rate_est)
                    + (bedrooms * 250000)
                    + (bathrooms * 150000)
                    - (age * (base_rate_est * 4))
                    + (spatial_data['public_transport_count'] * 80000)
                    + (spatial_data['school_count'] * 100000)
                )

                if total_price >= 10000000:
                    formatted_price = f"₹{total_price:,.2f} ({total_price/10000000:.2f} Cr)"
                else:
                    formatted_price = f"₹{total_price:,.2f} ({total_price/100000:.2f} Lakhs)"

                st.success(f"### Estimated Property Purchase Price: **{formatted_price}**")

        elif st.session_state.user_role == "Rent":
            st.markdown("---")
            st.subheader("3. Enter Property Details (Rent)")
            col1, col2 = st.columns(2)

            with col1:
                bedrooms = st.slider("Bedrooms (BHK)", min_value=1, max_value=6, value=2)
                bathrooms = st.slider("Bathrooms", min_value=1, max_value=5, value=2)
            with col2:
                age = st.slider("Property Age (Years)", min_value=0, max_value=30, value=5)

            if st.button("Predict Monthly Rent", type="primary"):
                if loc_tier == 1:
                    base_1bhk_rent = 14000
                    extra_bhk_cost = 8500
                    bathroom_cost = 2500
                    age_depreciation = 180
                elif loc_tier == 2:
                    base_1bhk_rent = 7500
                    extra_bhk_cost = 4500
                    bathroom_cost = 1500
                    age_depreciation = 100
                else:
                    base_1bhk_rent = 4000
                    extra_bhk_cost = 2500
                    bathroom_cost = 1000
                    age_depreciation = 50

                monthly_rent = (
                    base_1bhk_rent
                    + ((bedrooms - 1) * extra_bhk_cost)
                    + (bathrooms * bathroom_cost)
                    - (age * age_depreciation)
                    + (spatial_data['public_transport_count'] * 300)
                    + (spatial_data['school_count'] * 300)
                )

                formatted_rent = f"₹{max(2500, int(monthly_rent)):,} / month"
                st.success(f"### Estimated Monthly Rent: **{formatted_rent}**")

    else:
        st.markdown(
            """
            <style>
            div[data-baseweb="input"] {
                border: 2px solid #dc3545 !important;
                border-radius: 8px !important;
                background-color: #fff5f5 !important;
            }
            </style>
            """,
            unsafe_allow_html=True
        )
        st.error("❌ **Invalid Input:** Please enter a valid City, Locality, or Pincode.")
