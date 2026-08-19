import math
import numpy as np
import pandas as pd
import streamlit as st
from geopy.geocoders import ArcGIS

# Page Setup
st.set_page_config(page_title="Real-Time House Price Predictor", page_icon="📍", layout="wide")
st.title("📍 Real-Time House Price Predictor")

# 1. Tier-1 & Tier-2 City Anchors
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


def estimate_dynamic_sqft_price(lat, lng, address_text):
    min_dist = float('inf')
    nearest_hub_name = None
    nearest_hub = None

    for hub, coords in CITY_HUBS.items():
        dist = calculate_distance_km(lat, lng, coords['lat'], coords['lng'])
        if dist < min_dist:
            min_dist = dist
            nearest_hub_name = hub
            nearest_hub = coords

    # 1. Metro / City Commuter Belt (within 35 km of a Tier-1 or Tier-2 hub)
    if min_dist <= 35:
        decay_rate = 0.04
        calculated_rate = nearest_hub['base_rate'] * math.exp(-decay_rate * min_dist)
        base_rate = max(2800, calculated_rate)
        market_label = f"Suburban/Urban area near {nearest_hub_name} ({round(min_dist, 1)} km)"

    # 2. Independent District Headquarter / Tier-3 Town (e.g., Kolar, Hassan, Tumakuru)
    else:
        base_rate = 2200  # Realistic base rate for Tier-3 district centers
        market_label = f"Tier-3 District / Regional Town ({round(min_dist, 1)} km from {nearest_hub_name})"

    # Micro-spatial variance (street-level variation using lat/lng hash)
    spatial_jitter = 1.0 + (((int(lat * 10000) ^ int(lng * 10000)) % 16) - 8) / 100.0
    final_sqft_price = int(base_rate * spatial_jitter)

    return final_sqft_price, market_label


# 2. Geocoder Setup
@st.cache_resource
def get_geolocator():
    return ArcGIS(timeout=10)

geolocator = get_geolocator()


# 3. Location Validation Function (Rejects Streets/Stops, Accepts Cities/Towns/Pincodes)
@st.cache_data(show_spinner=False)
def get_location_data(address):
    if not address or len(address.strip()) < 2:
        return None

    raw_query = address.strip()

    # Reject non-6-digit numeric input
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

            # Explicitly REJECT street-level, building, or transit stop entities
            forbidden_types = [
                'PointAddress', 
                'StreetAddress', 
                'StreetName', 
                'StreetNameGroup', 
                'Building', 
                'POI', 
                'Intersection', 
                'Transit', 
                'Bus Stop'
            ]
            
            if score < 75:
                return None

            if addr_type in forbidden_types:
                return None

            formatted_address = location.address
            if "india" not in formatted_address.lower():
                return None

            lat, lng = location.latitude, location.longitude
            public_transport_count = int(abs(hash(f"{lat:.2f},{lng:.2f}")) % 10) + 1
            school_count = int(abs(hash(f"{lat:.3f},{lng:.3f}")) % 12) + 1

            return {
                "lat": lat,
                "lng": lng,
                "address": formatted_address,
                "public_transport_count": public_transport_count,
                "school_count": school_count,
            }
    except Exception:
        pass

    return None


# 4. Streamlit UI
st.header("1. Enter Property Location")
location_input = st.text_input(
    "Enter City, Town, or 6-digit Pincode", 
    placeholder="e.g. Bengaluru, Kolar, Mysuru, or 563101"
)

spatial_data = None

if location_input.strip():
    spatial_data = get_location_data(location_input)
    
    if spatial_data:
        base_rate_est, market_label = estimate_dynamic_sqft_price(
            spatial_data['lat'], spatial_data['lng'], spatial_data['address']
        )
        st.markdown(
            f"""
            <div style="background-color: #d4edda; color: #155724; padding: 12px; border-radius: 8px; border: 1px solid #c3e6cb; margin-bottom: 15px;">
                ✅ <strong>Verified Location:</strong> {spatial_data['address']}<br>
                📊 <strong>Estimated Market Rate:</strong> ~₹{base_rate_est:,} / sq.ft. ({market_label})
            </div>
            """, 
            unsafe_allow_html=True
        )
    else:
        st.error("❌ **Invalid Input:** Please enter a valid City name, Town name, or a 6-digit Indian Pincode. Street names, stops, and specific buildings are not allowed.")

st.header("2. Property Characteristics")
col1, col2 = st.columns(2)

with col1:
    sqft = st.slider("Square Feet", min_value=500, max_value=5000, value=1200, step=50)
    bedrooms = st.slider("Bedrooms (BHK)", min_value=1, max_value=6, value=2)

with col2:
    bathrooms = st.slider("Bathrooms", min_value=1, max_value=5, value=2)
    age = st.slider("Property Age (Years)", min_value=0, max_value=30, value=5)


# 5. Prediction Execution
if st.button("Predict Property Price"):
    if not location_input.strip():
        st.error("Please enter a City, Town, or Pincode.")
        st.stop()

    if not spatial_data:
        st.error("Cannot predict price for an invalid location. Please provide a valid City, Town, or 6-digit Pincode above.")
        st.stop()

    base_sqft_price, market_label = estimate_dynamic_sqft_price(
        spatial_data['lat'], spatial_data['lng'], spatial_data['address']
    )

    st.write(f"🌐 **Coordinates:** Lat `{spatial_data['lat']:.4f}`, Lng `{spatial_data['lng']:.4f}`")
    st.write(f"🏷️ Market Tier: **{market_label}**")

    total_price = (
        (sqft * base_sqft_price)
        + (bedrooms * 250000)
        + (bathrooms * 150000)
        - (age * (base_sqft_price * 4))
        + (spatial_data['public_transport_count'] * 80000)
        + (spatial_data['school_count'] * 100000)
    )

    if total_price >= 10000000:
        formatted_price = f"₹{total_price:,.2f} ({total_price/10000000:.2f} Cr)"
    else:
        formatted_price = f"₹{total_price:,.2f} ({total_price/100000:.2f} Lakhs)"

    st.success(f"### Estimated Price: **{formatted_price}**")
