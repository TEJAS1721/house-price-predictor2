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

    # Check Tier assignment based on commuter distance
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


# 2. Geocoder Setup
@st.cache_resource
def get_geolocator():
    return ArcGIS(timeout=10)

geolocator = get_geolocator()


# 3. Location Validation Function
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
            
            if score < 70 or addr_type in forbidden_types:
                return None

            lat, lng = location.latitude, location.longitude
            public_transport_count = int(abs(hash(f"{lat:.2f},{lng:.2f}")) % 10) + 1
            school_count = int(abs(hash(f"{lat:.3f},{lng:.3f}")) % 12) + 1

            return {
                "lat": lat,
                "lng": lng,
                "address": location.address,
                "public_transport_count": public_transport_count,
                "school_count": school_count,
            }
    except Exception:
        pass

    return None


# 4. Streamlit UI: Minimized Location Input
st.subheader("1. Property Location")

col_loc, _ = st.columns([1, 1])

with col_loc:
    location_input = st.text_input(
        "Enter City, Town, or 6-digit Pincode", 
        placeholder="e.g. Bengaluru, Kolar, 563114"
    )

spatial_data = None

if location_input.strip():
    spatial_data = get_location_data(location_input)
    
    if spatial_data:
        base_rate_est, market_label, loc_tier = estimate_location_details(
            spatial_data['lat'], spatial_data['lng']
        )
        st.markdown(
            f"""
            <div style="background-color: #d4edda; color: #155724; padding: 12px; border-radius: 8px; border: 1px solid #c3e6cb; margin-bottom: 15px;">
                ✅ <strong>Verified Location:</strong> {spatial_data['address']}<br>
                📍 <strong>Location Classification:</strong> Tier {loc_tier} ({market_label})
            </div>
            """, 
            unsafe_allow_html=True
        )

        # 5. User Type Selection
        st.subheader("2. Select Your Intent")
        
        user_role = st.radio(
            "Are you looking to Buy or Rent?",
            ["Buyer", "Renter"],
            horizontal=True,
            key="user_role"
        )

        st.subheader(f"3. Property Details ({user_role})")
        col1, col2 = st.columns(2)

        # BUYER PATH
        if user_role == "Buyer":
            with col1:
                sqft = st.slider("Square Feet", min_value=500, max_value=5000, value=1200, step=50)
                bedrooms = st.slider("Bedrooms (BHK)", min_value=1, max_value=6, value=2)
            with col2:
                bathrooms = st.slider("Bathrooms", min_value=1, max_value=5, value=2)
                age = st.slider("Property Age (Years)", min_value=0, max_value=30, value=5)

            if st.button("Predict Buying Price"):
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

        # RENTER PATH (Tier-Based Rental Calculation)
        else:
            with col1:
                bedrooms = st.slider("Bedrooms (BHK)", min_value=1, max_value=6, value=2)
                bathrooms = st.slider("Bathrooms", min_value=1, max_value=5, value=2)
            with col2:
                age = st.slider("Property Age (Years)", min_value=0, max_value=30, value=5)

            if st.button("Predict Monthly Rent"):
                # Independent Tier-Based Rental Math
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
                else:  # Tier 3 (e.g., Kolar)
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
        st.error("❌ **Invalid Input:** Please enter a valid City name, Town name, or a 6-digit Indian Pincode. Street names, stops, and specific buildings are not allowed.")
