import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

# 1. Load Real Housing Dataset
# Example dataset columns: ['City', 'Property_Type', 'Area_SqFt', 'Bedrooms', 'Bathrooms', 'Price_Lakhs']
df = pd.read_csv("india_real_estate_dataset.csv")

# Clean & Prepare Features
features = ['City', 'Property_Type', 'Area_SqFt', 'Bedrooms', 'Bathrooms']
X = df[features]
y = df['Price_Lakhs']

# 2. Preprocessing Pipeline for Categorical & Numerical Features
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['City', 'Property_Type']),
        ('num', 'passthrough', ['Area_SqFt', 'Bedrooms', 'Bathrooms'])
    ]
)

# 3. Build & Train Random Forest Model
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model_pipeline.fit(X_train, y_train)

# Output Model Evaluation
score = model_pipeline.score(X_test, y_test)
print(f"Model R² Score: {score:.2f}")

# 4. Save Trained Model to File
joblib.dump(model_pipeline, "india_housing_model.pkl")
print("Model saved successfully as india_housing_model.pkl!")
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(page_title="Data-Driven ML House Price Predictor", page_icon="🤖", layout="wide")
st.title("🤖 ML-Powered Indian Real Estate Predictor")

MODEL_PATH = "india_housing_model.pkl"

# 1. Load Saved Machine Learning Pipeline
@st.cache_resource
def load_ml_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None

model = load_ml_model()

if model is None:
    st.error("⚠️ **Machine Learning Model File (`india_housing_model.pkl`) Not Found!**")
    st.info("Please run **Step 1 (Training Script)** first using your dataset to generate the `.pkl` model file.")
else:
    st.success("✅ Machine Learning Model Loaded Successfully!")
    
    st.subheader("Enter Property Details")
    col1, col2 = st.columns(2)

    cities = ["Mumbai", "Bengaluru", "Delhi NCR", "Hyderabad", "Pune", "Chennai", "Kolkata", "Ahmedabad"]
    prop_types = ["Apartment", "Independent House", "Villa", "Penthouse"]

    with col1:
        city = st.selectbox("Select City", cities)
        prop_type = st.selectbox("Property Type", prop_types)
        sqft = st.number_input("Area (Square Feet)", min_value=300, max_value=10000, value=1200, step=50)

    with col2:
        bhk = st.slider("Bedrooms (BHK)", 1, 6, 2)
        bath = st.slider("Bathrooms", 1, 6, 2)

    if st.button("Predict Price using ML Model", type="primary"):
        # Format input data for the model pipeline
        input_data = pd.DataFrame([{
            'City': city,
            'Property_Type': prop_type,
            'Area_SqFt': sqft,
            'Bedrooms': bhk,
            'Bathrooms': bath
        }])

        # Predict price in Lakhs
        predicted_price_lakhs = model.predict(input_data)[0]
        price_inr = predicted_price_lakhs * 100000

        def fmt_inr(v):
            return f"₹{v/10000000:.2f} Cr" if v >= 10000000 else f"₹{v/100000:.2f} Lakhs"

        st.markdown("---")
        st.subheader("🎯 Prediction Results (Trained on Real Listings)")
        st.success(f"### Estimated Market Price: **{fmt_inr(price_inr)}**")
        
        # Display feature summary
        st.json({
            "City": city,
            "Property Type": prop_type,
            "Area": f"{sqft} sq.ft",
            "Configuration": f"{bhk} BHK, {bath} Bath",
            "ML Algorithm": "Random Forest Regressor"
        })
