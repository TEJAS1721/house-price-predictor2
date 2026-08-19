import numpy as np
import pandas as pd
import googlemaps
import streamlit as st
from sklearn.ensemble import RandomForestRegressor

# Page Setup
st.set_page_config(page_title="Real-Time House Price Predictor", page_icon="📍", layout="wide")
st.title("📍 Real-Time House Price Predictor with Google Maps")

# 1. Sidebar for API Key
api_key = st.sidebar.text_input("Enter Google Maps API Key", type="password")

@st.cache_resource
def init_gmaps(key):
    if key:
        return googlemaps.Client(key=key)
    return None

gmaps = init_gmaps(api_key)

# 2. Train and Cache Model (Includes Location Amenities as Features)
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

    # Base price + amenity value boosts (₹1 Lakh/transit station, ₹1.5 Lakhs/school)
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
    placeholder="e.g. Indiranagar, Bengaluru or 560038"
)

st.header("2. Property Characteristics")
col1, col2 = st.columns(2)

with col1:
    sqft = st.slider("Square Feet", min_value=500, max_value=5000, value=1200, step=50)
    bedrooms = st.slider("Bedrooms (BHK)", min_value=1, max_value=6, value=2)

with col2:
    bathrooms = st.slider("Bathrooms", min_value=1, max_value=5, value=2)
    age = st.slider("Property Age (Years)", min_value=0, max_value=30, value=5)

# 4. Geospatial Feature Extraction
def get_spatial_features(address, client):
    if not client or not address:
        return None

    try:
        geocode_res = client.geocode(address)
        if not geocode_res:
            st.error("Address or Pincode not found on Google Maps.")
            return None

        lat = geocode_res[0]["geometry"]["location"]["lat"]
        lng = geocode_res[0]["geometry"]["location"]["lng"]
        formatted_address = geocode_res[0]["formatted_address"]

        transit_res = client.places_nearby(
            location=(lat, lng), radius=2000, type="transit_station"
        )
        transit_count = len(transit_res.get("results", []))

        school_res = client.places_nearby(
            location=(lat, lng), radius=2000, type="school"
        )
        school_count = len(school_res.get("results", []))

        return {
            "lat": lat,
            "lng": lng,
            "transit_count": transit_count,
            "school_count": school_count,
            "address": formatted_address,
        }
    except Exception as e:
        st.error(f"Error fetching Google Maps location features: {e}")
        return None

# 5. Prediction Logic
if st.button("Predict Property Price"):
    if not location_input:
        st.error("Please enter an address or pincode.")
        st.stop()

    transit_count, school_count = 0, 0
    
    if api_key and gmaps:
        spatial_data = get_spatial_features(location_input, gmaps)
        if spatial_data:
            transit_count = spatial_data["transit_count"]
            school_count = spatial_data["school_count"]
            
            st.info(f"**Verified Location:** {spatial_data['address']}")
            st.write(f"🚆 Nearby Transit Stations (2km radius): **{transit_count}**")
            st.write(f"🏫 Nearby Schools (2km radius): **{school_count}**")
    else:
        st.warning("No API key provided. Predicting without real-time amenity lookup.")

    # Prepare input for ML prediction
    input_data = pd.DataFrame({
        'SquareFeet': [sqft],
        'Bedrooms': [bedrooms],
        'Bathrooms': [bathrooms],
        'Age': [age],
        'TransitCount': [transit_count],
        'SchoolCount': [school_count]
    })

    prediction = model.predict(input_data)[0]

    # Display in Lakhs or Crores
    if prediction >= 10000000:
        formatted_price = f"₹{prediction:,.2f} ({prediction/10000000:.2f} Cr)"
    else:
        formatted_price = f"₹{prediction:,.2f} ({prediction/100000:.2f} Lakhs)"

    st.success(f"### Estimated Price: **{formatted_price}**")
