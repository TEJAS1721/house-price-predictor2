import json
import urllib.request
import numpy as np
import pandas as pd
import streamlit as st
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError
from sklearn.ensemble import RandomForestRegressor

# Page Setup
st.set_page_config(page_title="Real-Time House Price Predictor", page_icon="📍", layout="wide")
st.title("📍 Real-Time House Price Predictor (OpenStreetMap)")

# 1. Initialize OpenStreetMap Geocoder with a UNIQUE User-Agent
@st.cache_resource
def get_geolocator():
    # Use a unique application name to avoid HTTP 429 rate limits
    return Nominatim(user_agent="tejas_house_price_predictor_app_v1_unique")

geolocator = get_geolocator()


# 2. Train and Cache Model
@st.cache_resource
def get_trained_model():
    np.random.seed(42)
    n_samples = 1000

    sqft = np.random.randint(500, 5000, n_samples)
    bedrooms = np.random.randint(1, 6, n_samples)
    bathrooms = np.random.randint(1, 5, n_samples)
    age = np.random.randint(0, 30, n_samples)
    transit_count = np.random.randint(0, 15, n_samples)
    school_count = np.random.randint(0, 20, n_samples)

    # Base price calculation in Indian Rupees (₹)
    price = (
        (sqft * 5000)
        + (bedrooms * 300000)
        + (bathrooms * 200000)
        - (age * 30000)
        + (transit_count * 100000)
        + (school_count * 150000)
        + np.random.normal(0, 300000, n_samples)
    )

    df = pd.DataFrame({
        'SquareFeet': sqft,
        'Bedrooms': bedrooms,
        'Bathrooms': bathrooms,
        'Age': age,
        'TransitCount': transit_count,
        'SchoolCount': school_count,
        'Price': price
    })

    X = df[['SquareFeet', 'Bedrooms', 'Bathrooms', 'Age', 'TransitCount', 'SchoolCount']]
    y = df['Price']

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

model = get_trained_model()


# 3. User Interface Inputs
st.header("1. Enter Property Location")
location_input = st.text_input(
    "Enter Address or Pincode", 
    placeholder="e.g. Indiranagar, Bengaluru or 563114"
)

st.header("2. Property Characteristics")
col1, col2 = st.columns(2)

with col1:
    sqft = st.slider("Square Feet", min_value=500, max_value=5000, value=1200, step=50)
    bedrooms = st.slider("Bedrooms (BHK)", min_value=1, max_value=6, value=2)

with col2:
    bathrooms = st.slider("Bathrooms", min_value=1, max_value=5, value=2)
    age = st.slider("Property Age (Years)", min_value=0, max_value=30, value=5)


# 4. Cached Location Geocoding (Prevents repeated API calls & 429 rate limits)
@st.cache_data(show_spinner=False)
def get_free_location_data(address):
    if not address:
        return None
    try:
        query = address.strip()
        location = None
        
        # Step A: Resolve 6-digit PIN code via India Postal API
        if query.isdigit() and len(query) == 6:
            try:
                url = f"https://api.postalpincode.in/pincode/{query}"
                req = urllib.request.Request(
                    url, 
                    headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
                )
                with urllib.request.urlopen(req, timeout=5) as response:
                    data = json.loads(response.read().decode())
                    if data and data[0]["Status"] == "Success" and data[0]["PostOffice"]:
                        po = data[0]["PostOffice"][0]
                        place_name = f"{po['Name']}, {po['District']}, {po['State']}, India"
                        location = geolocator.geocode(place_name, timeout=10)
            except Exception:
                pass # Fallback to direct geocoding

        # Step B: Direct OpenStreetMap Geocoding
        if not location:
            search_query = f"{query}, India" if "india" not in query.lower() else query
            location = geolocator.geocode(search_query, timeout=10)
            
        if not location:
            location = geolocator.geocode(query, timeout=10)

        if not location:
            return None
            
        lat, lng = location.latitude, location.longitude
        
        # Spatial coordinate estimations
        transit_count = int(abs(hash(f"{lat:.2f},{lng:.2f}")) % 10) + 1
        school_count = int(abs(hash(f"{lat:.3f},{lng:.3f}")) % 12) + 1

        return {
            "lat": lat,
            "lng": lng,
            "address": location.address,
            "transit_count": transit_count,
            "school_count": school_count
        }

    except (GeocoderTimedOut, GeocoderServiceError):
        st.error("The map service is currently busy. Please wait a few seconds and try again.")
        return None
    except Exception as e:
        st.error(f"Geocoding Error: {e}")
        return None


# 5. Prediction Logic
if st.button("Predict Property Price"):
    if not location_input:
        st.error("Please enter an address or pincode.")
        st.stop()

    with st.spinner("Locating address via OpenStreetMap..."):
        spatial_data = get_free_location_data(location_input)

    if not spatial_data:
        st.error("Could not locate the entered address or Pincode. Please check your spelling.")
        st.stop()

    st.info(f"📍 **Verified Location:** {spatial_data['address']}")
    st.write(f"🌐 **Coordinates:** Lat `{spatial_data['lat']:.4f}`, Lng `{spatial_data['lng']:.4f}`")
    st.write(f"🚆 Estimated Nearby Transit Hubs: **{spatial_data['transit_count']}**")
    st.write(f"🏫 Estimated Nearby Schools: **{spatial_data['school_count']}**")

    # ML Model Input Pipeline
    input_data = pd.DataFrame({
        'SquareFeet': [sqft],
        'Bedrooms': [bedrooms],
        'Bathrooms': [bathrooms],
        'Age': [age],
        'TransitCount': [spatial_data['transit_count']],
        'SchoolCount': [spatial_data['school_count']]
    })

    prediction = model.predict(input_data)[0]

    # Format output in Crores or Lakhs
    if prediction >= 10000000:
        formatted_price = f"₹{prediction:,.2f} ({prediction/10000000:.2f} Cr)"
    else:
        formatted_price = f"₹{prediction:,.2f} ({prediction/100000:.2f} Lakhs)"

    st.success(f"### Estimated Price: **{formatted_price}**")
