@st.cache_data(show_spinner=False)
def fetch_real_nearby_pois(address, lat, lng):
    """Fetches real schools and transit hubs using precise GPS bounding boxes."""
    real_schools = []
    real_transport = []
    seen_names = set()

    # Define a bounding box around the target location (~5km radius)
    delta_lat = 0.045
    delta_lng = 0.045
    s, w, n, e = lat - delta_lat, lng - delta_lng, lat + delta_lat, lng + delta_lat

    # 1. Direct Overpass Bounding Box Search (All Educational Institutions)
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
        headers = {"User-Agent": "RealHousePricePredictor/3.0"}
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

    # 2. Query ArcGIS Category POIs (No Synthetic/Fake Name Fallbacks)
    try:
        geo = ArcGIS(timeout=10)
        clean_loc = address.split(',')[0].strip()
        candidates = geo.geocode(f"School in {clean_loc}", exactly_one=False, max_results=10) or []
        
        for item in candidates:
            s_name = item.address.split(',')[0].strip()
            # Ignore generic address matches that are not actual named schools
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

    # 3. Fetch Real Railway & Bus Stations
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

    # Sort strictly by distance
    real_schools = sorted(real_schools, key=lambda x: x['dist'])
    real_transport = sorted(real_transport, key=lambda x: x['dist'])

    return real_schools, real_transport
