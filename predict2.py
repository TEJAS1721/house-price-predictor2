import folium
from folium.plugins import Fullscreen, MarkerCluster
from streamlit_folium import st_folium

# --- Custom Styling for Map Container ---
st.markdown("""
<style>
    .map-card {
        background: #1e222d;
        padding: 12px;
        border-radius: 16px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.25);
        margin-bottom: 20px;
    }
    .map-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        color: #ffffff;
        padding: 4px 8px 12px 8px;
    }
    .map-title {
        font-size: 15px;
        font-weight: 600;
        letter-spacing: 0.5px;
    }
    .badge-tier {
        background: linear-gradient(135deg, #007bff, #00d2ff);
        color: white;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# --- Map Rendering Function ---
def render_enhanced_location_preview(spatial_data, loc_tier, market_label):
    lat = spatial_data['lat']
    lng = spatial_data['lng']
    address = spatial_data['address']

    # 1. Initialize Map with Realistic Base Tiles
    m = folium.Map(
        location=[lat, lng],
        zoom_start=16,
        max_zoom=19,
        tiles=None # We will add custom tile layers
    )

    # High-Res Satellite View (Esri Clarity / World Imagery)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri World Imagery",
        name="Satellite Aerial",
        max_zoom=19
    ).add_to(m)

    # Clean Vector Street Map (CartoDB Positron)
    folium.TileLayer(
        tiles="https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png",
        attr="CartoDB Voyager",
        name="Street View",
        max_zoom=19
    ).add_to(m)

    # 2. Add Animated Pulsing Pin for Target Location
    custom_pin_html = f"""
    <div style="
        position: relative;
        width: 30px;
        height: 30px;
        background: #ff3b30;
        border: 3px solid #ffffff;
        border-radius: 50%;
        box-shadow: 0 0 15px rgba(255, 59, 48, 0.8);
        display: flex;
        align-items: center;
        justify-content: center;
    ">
        <div style="
            width: 10px;
            height: 10px;
            background: white;
            border-radius: 50%;
        "></div>
    </div>
    """
    
    folium.Marker(
        [lat, lng],
        popup=folium.Popup(
            f"""
            <div style="font-family: sans-serif; padding: 5px; width: 200px;">
                <h4 style="margin: 0 0 5px 0; color: #1e222d;">Selected Locality</h4>
                <p style="margin: 0; font-size: 12px; color: #555;">{address}</p>
            </div>
            """,
            max_width=250
        ),
        tooltip=f"📍 {address}",
        icon=folium.DivIcon(
            html=custom_pin_html,
            icon_size=(30, 30),
            icon_anchor=(15, 15)
        )
    ).add_to(m)

    # 3. Add Amenity Layer Groups with Custom Styling
    transport_group = folium.FeatureGroup(name="Transit Hubs")
    school_group = folium.FeatureGroup(name="Schools & Colleges")

    # Add Transit Markers
    for i in range(spatial_data['public_transport_count']):
        angle = (i * 137.5) * (math.pi / 180)
        dist = 180 + ((i * 123) % 450)
        d_lat = (dist * math.sin(angle)) / 111000.0
        d_lng = (dist * math.cos(angle)) / (111000.0 * math.cos(math.radians(lat)))

        folium.CircleMarker(
            location=[lat + d_lat, lng + d_lng],
            radius=7,
            color="#007bff",
            fill=True,
            fill_color="#007bff",
            fill_opacity=0.8,
            popup=f"Transit Hub #{i+1}",
            tooltip="🚌 Public Transit Point"
        ).add_to(transport_group)

    # Add School Markers
    for i in range(spatial_data['school_count']):
        angle = (i * 211.3 + 60) * (math.pi / 180)
        dist = 220 + ((i * 97) % 420)
        d_lat = (dist * math.sin(angle)) / 111000.0
        d_lng = (dist * math.cos(angle)) / (111000.0 * math.cos(math.radians(lat)))

        folium.CircleMarker(
            location=[lat + d_lat, lng + d_lng],
            radius=7,
            color="#ff9500",
            fill=True,
            fill_color="#ff9500",
            fill_opacity=0.8,
            popup=f"School/College #{i+1}",
            tooltip="🏫 Educational Institution"
        ).add_to(school_group)

    transport_group.add_to(m)
    school_group.add_to(m)

    # 4. Interactive Tools
    folium.LayerControl(position="topright", collapsed=True).add_to(m)
    Fullscreen(position="topleft").add_to(m)

    return m

# --- Render in Streamlit UI ---
with map_col:
    st.markdown(f"""
    <div class="map-header">
        <span class="map-title">🌐 Interactive Location Preview</span>
        <span class="badge-tier">Tier {loc_tier} Locality</span>
    </div>
    """, unsafe_allow_html=True)
    
    m = render_enhanced_location_preview(spatial_data, loc_tier, market_label)
    
    map_output = st_folium(
        m,
        width="100%",
        height=380,
        key="enhanced_interactive_map",
        returned_objects=["last_clicked"]
    )

    # Capture map clicks for dynamic pin adjustments
    if map_output and map_output.get("last_clicked"):
        clicked_lat = map_output["last_clicked"]["lat"]
        clicked_lng = map_output["last_clicked"]["lng"]
        if (st.session_state.custom_coords is None or 
            st.session_state.custom_coords["lat"] != clicked_lat):
            st.session_state.custom_coords = {"lat": clicked_lat, "lng": clicked_lng}
            st.rerun()
