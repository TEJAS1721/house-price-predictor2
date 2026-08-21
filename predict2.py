import math
import requests
import numpy as np
import pandas as pd
import streamlit as st
from geopy.geocoders import ArcGIS
import folium
from streamlit_folium import st_folium

# Page Setup
st.set_page_config(page_title="Real-Time House Price Predictor", page_icon="📍", layout="wide")
st.title("📍 Real-Time House Price Predictor")

# Custom CSS for UI styling
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

# 1. Realistic City Base Rates
CITY_HUBS = {
    # Tier 1 Metros
    "Bengaluru": {"lat": 12.9716, "lng": 77.5946, "base_rate": 6500, "tier": 1},
    "Mumbai": {"lat": 19.0657, "lng": 72.8686, "base_rate": 15000, "tier": 1},
    "Delhi NCR": {"lat": 28.6315, "lng": 77.2167, "base_rate": 7500, "tier": 1},
    "Hyderabad": {"lat": 17.4435, "lng": 78.3772, "base_rate": 5500, "tier": 1},
    "Chennai": {"lat": 13.0604, "lng": 80.2496, "base_rate": 5800, "tier": 1},
    "Pune": {"lat": 18.5204, "lng": 73.8567, "base_rate": 5200, "tier": 1},
    "Kolkata": {"lat": 22.5726, "lng": 88.3639, "base_rate": 4500, "tier": 1},

    # Tier 2 Cities
    "Mysuru": {"lat": 12.2958, "lng": 76.6394, "base_rate": 3200, "tier": 2},
    "Mangaluru": {"lat": 12.9141, "lng": 74.8560, "base_rate": 3000, "tier": 2},
    "Hubballi": {"lat": 15.3647, "lng": 75.1240, "base_rate": 2500, "tier": 2},
    "Coimbatore": {"lat": 11.0168, "lng": 76.9558, "base_rate": 3200, "tier": 2},
    "Kochi": {"lat": 9.9312, "lng": 76.2673, "base_rate": 3800, "tier": 2},
    "Visakhapatnam": {"lat": 17.6868, "lng": 83.2185, "base_rate": 3400, "tier": 2},
    "Jaipur": {"lat": 26.9124, "lng": 75.7873, "base_rate": 3200, "tier": 2},
}

PROPERTY_TYPE_MULTIPLIER = {
    "Apartment": 1.00,
    "Independent House": 1.05,
    "Villa": 1.25,
    "Plot (Land only)": 0.60,
}

FURNISHING_MULTIPLIER = {
    "Unfurnished": 1.00,
    "Semi-Furnished": 1.03,
    "Fully Furnished": 1.06,
}

FURNISHING_RENT_ADD = {
    "Unfurnished": 0,
    "Semi-Furnished": 800,
    "Fully Furnished": 2000,
}


def calculate_distance_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2)
    return R * (2 * math.atan2(math.sqrt(a), math.sqrt(1 - a)))


def get_tier_and_hub(lat, lng):
    min_dist = float('inf')
    nearest_hub_name = None
    nearest_hub = None

    for hub, coords in CITY_HUBS.items():
        dist = calculate_distance_km(lat, lng, coords['lat'], coords['lng'])
        if dist < min_dist:
            min_dist = dist
            nearest_hub_name = hub
            nearest_hub = coords

    if min_dist <= 25:
        tier = nearest_hub['tier']
        market_label = f"Tier-{tier} Area near {nearest_hub_name} ({round(min_dist, 1)} km)"
        base_rate = nearest_hub['base_rate']
    else:
        tier = 3
        market_label = f"Tier-3 District / Town ({round(min_dist, 1)} km from {nearest_hub_name})"
        base_rate = 1400

    return tier, nearest_hub_name, min_dist, market_label, base_rate


def predict_market_buy_price(sqft, bhk, bath, age, dist_to_hub, tier, prop_type, furn_type, trans_count, sch_count, base_rate):
    dist_decay = max(0.40, math.exp(-0.025 * dist_to_hub))
    connectivity_boost = min(1.08, 1.0 + (0.003 * trans_count) + (0.005 * sch_count))
    effective_sqft_price = base_rate * dist_decay * connectivity_boost
    
    type_mult = PROPERTY_TYPE_MULTIPLIER.get(prop_type, 1.0)
    furn_mult = FURNISHING_MULTIPLIER.get(furn_type, 1.0)
    age_depr = max(0.65, 1.0 - (age * 0.012))

    total_price = (sqft * effective_sqft_price * type_mult * furn_mult * age_depr)
    return max(200000, total_price)


def predict_market_rent(bhk, bath, age, tier, furn_type, trans_count, sch_count):
    if tier == 1:
        base_rent = 6000
        bhk_rate = 2500
    elif tier == 2:
        base_rent = 3500
        bhk_rate = 1500
    else:
        base_rent = 1800
        bhk_rate = 1000

    furn_add = FURNISHING_RENT_ADD.get(furn_type, 0)
    
    monthly_rent = (
        base_rent + 
        (bhk * bhk_rate) + 
        (bath * 400) + 
        furn_add - 
        (age * 50) + 
        min(600, trans_count * 50) + 
        min(500, sch_count * 50)
    )
    
    return max(1500, int(monthly_rent))


def get_circle_points(lat, lng, radius_meters=750, num_points=64):
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


# 2. Geocoder Setup
@st.cache_resource
def get_geolocator():
    return ArcGIS(timeout=10)

geolocator = get_geolocator()

OVERPASS_URL = "https://overpass-api.de/api/interpreter"


@st.cache_data(show_spinner=False, ttl=3600)
def get_nearby_amenities(lat, lng, radius_meters=750):
    query = f"""
    [out:json][timeout:15];
    (
      node["highway"="bus_stop"](around:{radius_meters},{lat},{lng});
      node["railway"~"station|halt|tram_stop"](around:{radius_meters},{lat},{lng});
      node["public_transport"](around:{radius_meters},{lat},{lng});
    );
    out count;
    >;
    (
      node["amenity"="school"](around:{radius_meters},{lat},{lng});
      node["amenity"="college"](around:{radius_meters},{lat},{lng});
      way["amenity"="school"](around:{radius_meters},{lat},{lng});
    );
    out count;
    """
    try:
        resp = requests.post(OVERPASS_URL, data={"data": query}, timeout=12)
        resp.raise_for_status()
        data = resp.json()

        transport_count = 0
        school_count = 0
        seen_transport = False

        for el in data.get("elements", []):
            if el.get("type") == "count":
                tags = el.get("tags", {})
                total = int(tags.get("total", 0))
                if not seen_transport:
                    transport_count = total
                    seen_transport = True
                else:
                    school_count = total

        return {
            "public_transport_count": transport_count,
            "school_count": school_count,
            "source": "live",
        }
    except Exception:
        public_transport_count = int(abs(hash(f"{lat:.2f},{lng:.2f}")) % 5) + 2
        school_count = int(abs(hash(f"{lat:.3f},{lng:.3f}")) % 5) + 2
        return {
            "public_transport_count": public_transport_count,
            "school_count": school_count,
            "source": "estimated",
        }


# 3. Robust Geocoding & Location Parser
@st.cache_data(show_spinner=False)
def get_location_data(address):
    if not address or len(address.strip()) < 2:
        return None

    raw_query = address.strip()

    # Determine search queries based on PIN code or address string
    search_queries = []
    if raw_query.isdigit() and len(raw_query) == 6:
        search_queries = [
            f"PIN {raw_query}, India",
            f"{raw_query}, India",
            raw_query
        ]
    else:
        clean_addr = raw_query if "india" in raw_query.lower() else f"{raw_query}, India"
        search_queries = [clean_addr, raw_query]

    location = None
    for q in search_queries:
        try:
            location = geolocator.geocode(q, out_fields="*")
            if location and location.latitude and location.longitude:
                break
        except Exception:
            continue

    if location and location.latitude and location.longitude:
        lat, lng = location.latitude, location.longitude
        amenities = get_nearby_amenities(lat, lng, radius_meters=750)

        return {
            "lat": lat,
            "lng": lng,
            "address": location.address,
            "public_transport_count": amenities["public_transport_count"],
            "school_count": amenities["school_count"],
            "amenity_source": amenities["source"],
        }

    return None


def calculate_emi(principal, annual_rate_pct, tenure_years):
    monthly_rate = (annual_rate_pct / 100) / 12
    n_months = tenure_years * 12
    if monthly_rate == 0:
        return principal / n_months
    emi = principal * monthly_rate * (1 + monthly_rate) ** n_months / ((1 + monthly_rate) ** n_months - 1)
    return emi


# Session State Management
if "user_role" not in st.session_state:
    st.session_state.user_role = None

# 4. Streamlit UI: Search Container
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

spatial_data = None

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

        loc_tier, hub_name, dist_to_hub, market_label, base_rate = get_tier_and_hub(
            spatial_data['lat'], spatial_data['lng']
        )

        loc_col, map_col = st.columns([1, 1])

        amenity_note = (
            "" if spatial_data["amenity_source"] == "live"
            else " <span style='font-size:0.75em;opacity:0.7;'>(estimated — live data unavailable)</span>"
        )

        with loc_col:
            st.markdown(
                f"""
                <div style="background-color: #d4edda; color: #155724; padding: 16px; border-radius: 8px; border: 1px solid #c3e6cb; margin-bottom: 15px;">
                    ✅ <strong>{spatial_data['address']}</strong><br><br>
                    📍 <strong>Classification:</strong> Tier {loc_tier} ({market_label})<br>
                    🚌 <strong>Nearby Transport Locations (750m):</strong> {spatial_data['public_transport_count']}{amenity_note}<br>
                    🏫 <strong>Nearby Schools/Colleges (750m):</strong> {spatial_data['school_count']}{amenity_note}
                </div>
                """,
                unsafe_allow_html=True
            )

        with map_col:
            lat = spatial_data['lat']
            lng = spatial_data['lng']

            m = folium.Map(
                location=[lat, lng],
                zoom_start=15,
                tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
                attr="Esri World Imagery"
            )

            world_bounds = [[90, -180], [90, 180], [-90, 180], [-90, -180]]
            hole_cutout = get_circle_points(lat, lng, radius_meters=750)

            folium.Polygon(
                locations=[world_bounds, hole_cutout],
                color="#000000",
                weight=1,
                fill=True,
                fill_color="#111111",
                fill_opacity=0.6,
                tooltip="Outside Area"
            ).add_to(m)

            folium.PolyLine(
                locations=hole_cutout + [hole_cutout[0]],
                color="#28a745",
                weight=3,
                opacity=0.9,
                tooltip=spatial_data['address']
            ).add_to(m)

            for i in range(spatial_data['public_transport_count']):
                angle = (i * 137.5) * (math.pi / 180)
                dist = 180 + ((i * 123) % 450)
                d_lat = (dist * math.sin(angle)) / 111000.0
                d_lng = (dist * math.cos(angle)) / (111000.0 * math.cos(math.radians(lat)))

                folium.Marker(
                    [lat + d_lat, lng + d_lng],
                    tooltip=f"Transport Hub #{i+1}",
                    icon=folium.Icon(color="blue", icon="bus", prefix="fa")
                ).add_to(m)

            for i in range(spatial_data['school_count']):
                angle = (i * 211.3 + 60) * (math.pi / 180)
                dist = 220 + ((i * 97) % 420)
                d_lat = (dist * math.sin(angle)) / 111000.0
                d_lng = (dist * math.cos(angle)) / (111000.0 * math.cos(math.radians(lat)))

                folium.Marker(
                    [lat + d_lat, lng + d_lng],
                    tooltip=f"School/College #{i+1}",
                    icon=folium.Icon(color="orange", icon="graduation-cap", prefix="fa")
                ).add_to(m)

            st_folium(m, width="100%", height=320, returned_objects=[])

        # 5. User Intent Selection
        st.subheader("2. Select Your Intent")
        btn_col1, btn_col2, _ = st.columns([1, 1, 3])

        with btn_col1:
            if st.button("🏠 Own", use_container_width=True):
                st.session_state.user_role = "Own"

        with btn_col2:
            if st.button("🔑 Rent", use_container_width=True):
                st.session_state.user_role = "Rent"

        # 6. Inputs & Predictions
        if st.session_state.user_role == "Own":
            st.markdown("---")
            st.subheader("3. Enter Property Details (Own)")
            col1, col2 = st.columns(2)

            with col1:
                sqft = st.slider("Square Feet", min_value=500, max_value=5000, value=1200, step=50)
                bedrooms = st.slider("Bedrooms (BHK)", min_value=1, max_value=6, value=2)
                property_type = st.selectbox("Property Type", list(PROPERTY_TYPE_MULTIPLIER.keys()))
            with col2:
                bathrooms = st.slider("Bathrooms", min_value=1, max_value=5, value=2)
                age = st.slider("Property Age (Years)", min_value=0, max_value=30, value=5)
                furnishing = st.selectbox("Furnishing", list(FURNISHING_MULTIPLIER.keys()))

            if st.button("Predict Buying Price", type="primary"):
                total_price = predict_market_buy_price(
                    sqft, bedrooms, bathrooms, age, dist_to_hub, loc_tier,
                    property_type, furnishing, spatial_data['public_transport_count'],
                    spatial_data['school_count'], base_rate
                )

                low_price = total_price * 0.90
                high_price = total_price * 1.10

                def fmt_inr(v):
                    return f"₹{v/10000000:.2f} Cr" if v >= 10000000 else f"₹{v/100000:.2f} Lakhs"

                st.success(
                    f"### Estimated Purchase Price: **{fmt_inr(low_price)} – {fmt_inr(high_price)}**\n"
                    f"(midpoint: {fmt_inr(total_price)})"
                )

                st.session_state["last_buy_price"] = total_price

            if "last_buy_price" in st.session_state:
                st.markdown("---")
                st.subheader("4. Loan EMI Calculator")

                emi_col1, emi_col2, emi_col3 = st.columns(3)
                with emi_col1:
                    down_payment_pct = st.slider("Down Payment (%)", 10, 50, 20)
                with emi_col2:
                    interest_rate = st.slider("Home Loan Interest Rate (% p.a.)", 6.0, 12.0, 8.5, step=0.1)
                with emi_col3:
                    tenure_years = st.slider("Loan Tenure (Years)", 5, 30, 20)

                purchase_price = st.session_state["last_buy_price"]
                down_payment_amt = purchase_price * (down_payment_pct / 100)
                loan_amount = purchase_price - down_payment_amt
                emi = calculate_emi(loan_amount, interest_rate, tenure_years)

                st.info(
                    f"💰 **Down Payment:** ₹{down_payment_amt:,.0f}  \n"
                    f"🏦 **Loan Amount:** ₹{loan_amount:,.0f}  \n"
                    f"📆 **Monthly EMI:** ₹{emi:,.0f} over {tenure_years} years"
                )

        elif st.session_state.user_role == "Rent":
            st.markdown("---")
            st.subheader("3. Enter Property Details (Rent)")
            col1, col2 = st.columns(2)

            with col1:
                bedrooms = st.slider("Bedrooms (BHK)", min_value=1, max_value=6, value=1)
                bathrooms = st.slider("Bathrooms", min_value=1, max_value=5, value=1)
                furnishing = st.selectbox("Furnishing", list(FURNISHING_RENT_ADD.keys()))
            with col2:
                age = st.slider("Property Age (Years)", min_value=0, max_value=30, value=5)

            if st.button("Predict Monthly Rent", type="primary"):
                monthly_rent = predict_market_rent(
                    bedrooms, bathrooms, age, loc_tier, furnishing,
                    spatial_data['public_transport_count'], spatial_data['school_count']
                )

                low_rent = int(monthly_rent * 0.90)
                high_rent = int(monthly_rent * 1.10)

                st.success(
                    f"### Estimated Monthly Rent: **₹{low_rent:,} – ₹{high_rent:,} / month**\n"
                    f"(midpoint: ₹{monthly_rent:,})"
                )

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
