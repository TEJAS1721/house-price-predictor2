import math

# 1. Expanded City Anchors (Tier-1, Tier-2, & Regional Capitals)
CITY_HUBS = {
    # Tier 1 Metros
    "Mumbai": {"lat": 19.0657, "lng": 72.8686, "base_rate": 35000},
    "Delhi NCR": {"lat": 28.6315, "lng": 77.2167, "base_rate": 22000},
    "Bengaluru": {"lat": 12.9716, "lng": 77.5946, "base_rate": 16000},
    "Hyderabad": {"lat": 17.4435, "lng": 78.3772, "base_rate": 11000},
    "Chennai": {"lat": 13.0604, "lng": 80.2496, "base_rate": 12000},
    "Kolkata": {"lat": 22.5726, "lng": 88.3639, "base_rate": 10000},
    "Pune": {"lat": 18.5204, "lng": 73.8567, "base_rate": 10000},
    "Ahmedabad": {"lat": 23.0225, "lng": 72.5714, "base_rate": 8000},

    # Tier 2 Hubs
    "Jaipur": {"lat": 26.9124, "lng": 75.7873, "base_rate": 6500},
    "Lucknow": {"lat": 26.8467, "lng": 80.9462, "base_rate": 6000},
    "Chandigarh": {"lat": 30.7333, "lng": 76.7794, "base_rate": 8500},
    "Kochi": {"lat": 9.9312, "lng": 76.2673, "base_rate": 7000},
    "Indore": {"lat": 22.7196, "lng": 75.8577, "base_rate": 5500},
    "Nagpur": {"lat": 21.1458, "lng": 79.0882, "base_rate": 5000},
    "Coimbatore": {"lat": 11.0168, "lng": 76.9558, "base_rate": 6000},
    "Visakhapatnam": {"lat": 17.6868, "lng": 83.2185, "base_rate": 6000},
    "Bhubaneswar": {"lat": 20.2961, "lng": 85.8245, "base_rate": 5500},
    "Patna": {"lat": 25.5941, "lng": 85.1376, "base_rate": 5000},
    "Guwahati": {"lat": 26.1445, "lng": 91.7362, "base_rate": 5000},
}

# 2. Distance Calculation
def calculate_distance_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2 + 
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2)
    return R * (2 * math.atan2(math.sqrt(a), math.sqrt(1 - a)))

# 3. Universal Price Estimator (Works Anywhere)
def estimate_dynamic_sqft_price(lat, lng, formatted_address):
    address_lower = formatted_address.lower()
    min_dist = float('inf')
    nearest_hub_name = None
    nearest_hub = None

    # Find closest known anchor
    for hub, coords in CITY_HUBS.items():
        dist = calculate_distance_km(lat, lng, coords['lat'], coords['lng'])
        if dist < min_dist:
            min_dist = dist
            nearest_hub_name = hub
            nearest_hub = coords

    # If within 40km of a known city hub, apply distance decay
    if min_dist <= 40:
        decay_rate = 0.04
        calculated_rate = nearest_hub['base_rate'] * math.exp(-decay_rate * min_dist)
        base_rate = max(3000, calculated_rate)
        location_type = f"Near {nearest_hub_name} ({round(min_dist, 1)} km from center)"
    
    # If far from major hubs, determine tier from geocoded address
    else:
        # Tier 3 urban check
        if any(term in address_lower for term in ['city', 'nagar', 'town', 'district', 'pur']):
            base_rate = 3800
            location_type = "Tier-3 Town / Regional Area"
        # Rural / Village default
        else:
            base_rate = 2200
            location_type = "Rural / Small Settlement"

    # Micro-spatial variance (street-level variation using lat/lng hash)
    spatial_jitter = 1.0 + (((int(lat * 10000) ^ int(lng * 10000)) % 16) - 8) / 100.0
    final_sqft_price = int(base_rate * spatial_jitter)

    return final_sqft_price, location_type
