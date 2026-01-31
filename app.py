import streamlit as st
import pandas as pd
import joblib
import json
import os
import base64
import folium
from streamlit_folium import st_folium
import plotly.graph_objects as go
from datetime import datetime
from geopy.distance import geodesic

try:
    from src.utils.config import MODEL_PATH
except ImportError:
    MODEL_PATH = 'models/final_pipeline.joblib'

COORD_FILE = 'data/location_coords.json'

st.set_page_config(page_title="Cab Demand Forecasting", page_icon=None, layout="wide")

vehicle_images = {
    'Bike': 'data/images/bike.png',
    'eBike': 'data/images/ebike.webp',
    'Auto': 'data/images/auto.jpg',
    'Go Mini': 'data/images/go mini.jpg',
    'Go Sedan': 'data/images/go sedan.png',
    'Premier Sedan': 'data/images/premier sedan.jpg',
    'Uber XL': 'data/images/uber xl.jpg'
}

if 'selected_vehicle' not in st.session_state:
    st.session_state.selected_vehicle = "Go Sedan"
if 'show_results' not in st.session_state:
    st.session_state.show_results = False

st.markdown("""
<style>
    /* BUTTON STYLES - FIXED HEIGHT ENFORCED */
    /* BUTTON STYLING */
    div.stButton > button {
        width: 100%;
        border-radius: 4px;
        background-color: #1e2130; 
        color: #eeeeee;
        font-size: 13px; /* Slightly smaller for better fit */
        font-weight: 600;
        
        /* STRICT FIXED HEIGHT CONFIGURATION */
        height: 60px !important; 
        min-height: 60px !important;
        max-height: 60px !important;
        
        /* Text alignment logic */
        white-space: pre-wrap !important; 
        line-height: 1.2 !important;
        padding: 4px !important;
        
        /* Flex centering */
        display: flex;
        align-items: center;
        justify-content: center;
        text-align: center;
        
        transition: all 0.2s ease;
    }
    
    div.stButton > button:hover {
        border-color: #3498db;
        color: #3498db;
        background-color: #262b3d;
    }
    
    /* CARD IMAGE CONTAINER */
    .vehicle-card {
        background-color: #ffffff;
        border-radius: 8px;
        padding: 5px;
        height: 100px;
        width: 100%;
        display: flex;
        align-items: center;
        justify-content: center;
        margin-bottom: 5px;
    }

    /* IMAGE ITSELF */
    .vehicle-img {
        max-width: 100%;
        max-height: 100%;
        object-fit: contain; 
        filter: drop-shadow(0px 4px 4px rgba(0,0,0,0.1)); 
    }

    /* HEADERS */
    .main-header { font-size: 2.2rem; font-weight: 700; color: #6c5ce7; margin-bottom: 0.5rem; }
    .sub-header { font-size: 1.3rem; font-weight: 600; color: #a29bfe; margin-top: 1rem; margin-bottom: 1rem;}
    .price-display { font-size: 2.5rem; font-weight: 700; color: #00b894; margin: 0; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_assets():
    model = None
    coords = None
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
    if os.path.exists(COORD_FILE):
        with open(COORD_FILE, 'r') as f:
            coords = json.load(f)
    return model, coords

def get_img_as_base64(file_path):
    if not os.path.exists(file_path):
        return None
    with open(file_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

pipeline, LOC_COORDS = load_assets()

def explain_price(distance, vehicle_type, predicted_price):
    base_fares = {'Bike': 25, 'eBike': 20, 'Auto': 40, 'Go Mini': 50, 'Go Sedan': 80, 'Premier Sedan': 100, 'Uber XL': 150}
    rates_per_km = {'Bike': 6, 'eBike': 5, 'Auto': 10, 'Go Mini': 12, 'Go Sedan': 17, 'Premier Sedan': 20, 'Uber XL': 28}
    base = base_fares.get(vehicle_type, 80)
    rate = rates_per_km.get(vehicle_type, 17)
    distance_cost = distance * rate
    surge = predicted_price - (base + distance_cost)
    if surge < 0: surge = 0
    return base, distance_cost, surge

with st.sidebar:
    st.header("Plan Your Ride")
    col1, col2 = st.columns(2)
    with col1: ride_date = st.date_input("Date", datetime.today())
    with col2: ride_time = st.time_input("Time", datetime.now().time())
    
    event_list = ['Normal Day', 'Monsoon', 'Wedding_Season', 'Diwali', 'New Year']
    selected_event = st.selectbox("Event", event_list)
    st.markdown("---")
    
    loc_options = sorted(list(LOC_COORDS.keys())) if LOC_COORDS else ["Connaught Place", "Noida Sector 62"]
    pickup = st.selectbox("Pickup", loc_options, index=0)
    drop = st.selectbox("Drop", loc_options, index=1)
    
    p_coords = LOC_COORDS.get(pickup) if LOC_COORDS else None
    d_coords = LOC_COORDS.get(drop) if LOC_COORDS else None
    
    if p_coords and d_coords:
        distance = geodesic(p_coords, d_coords).km * 1.3
        st.write(f"**Dist:** {distance:.1f} km")
    else:
        distance = 10.0

    st.markdown("---")
    if st.button("Find Price", type="primary", use_container_width=True):
        st.session_state.show_results = True

st.markdown('<div class="main-header">Cab Demand Forecasting</div>', unsafe_allow_html=True)

if st.session_state.show_results:
    vehicles = ['Bike', 'eBike', 'Auto', 'Go Mini', 'Go Sedan', 'Premier Sedan', 'Uber XL']
    price_map = {}
    
    for v in vehicles:
        try:
            input_data = pd.DataFrame({
                'datetime': [datetime.combine(ride_date, ride_time)],
                'Event': [selected_event], 'Vehicle Type': [v],
                'Pickup Location': [pickup], 'Drop Location': [drop],
                'Ride Distance': [distance]
            })
            if pipeline:
                predicted = int(pipeline.predict(input_data)[0])
            else:
                raise Exception
        except:
            b, d, s = explain_price(distance, v, 0)
            predicted = int(b + d + s)
        price_map[v] = predicted

    st.markdown('<div class="sub-header">Select Vehicle</div>', unsafe_allow_html=True)

    cols = st.columns(7, gap="small")
    
    for i, v_type in enumerate(vehicles):
        price = price_map[v_type]
        img_path = vehicle_images.get(v_type)
        
        with cols[i]:
            img_b64 = get_img_as_base64(img_path)
            if img_b64:
                ext = os.path.splitext(img_path)[1].lower().replace('.', '')
                mime = 'jpeg' if ext == 'jpg' else ext
                st.markdown(f"""
                <div class="vehicle-card">
                    <img src="data:image/{mime};base64,{img_b64}" class="vehicle-img">
                </div>
                """, unsafe_allow_html=True)
            else:
                 st.markdown('<div class="vehicle-card">🚖</div>', unsafe_allow_html=True)

            if st.session_state.selected_vehicle == v_type:
                label = f" {v_type}\n₹{price}"
            else:
                label = f"{v_type}\n₹{price}"
                
            if st.button(label, key=f"btn_{v_type}"):
                st.session_state.selected_vehicle = v_type
                st.rerun()

    st.markdown("---")
    c1, c2 = st.columns([2, 1])
    
    with c1:
        if p_coords and d_coords:
            m = folium.Map(location=[(p_coords[0]+d_coords[0])/2, (p_coords[1]+d_coords[1])/2], zoom_start=11)
            folium.Marker(p_coords, icon=folium.Icon(color="green")).add_to(m)
            folium.Marker(d_coords, icon=folium.Icon(color="red")).add_to(m)
            st_folium(m, height=400, width="100%")
            
    with c2:
        sel_v = st.session_state.selected_vehicle
        if sel_v not in price_map: sel_v = vehicles[0]
        sel_p = price_map[sel_v]
        
        st.markdown(f"### {sel_v}")
        st.markdown(f"<div class='price-display'>₹ {sel_p}</div>", unsafe_allow_html=True)
        
        b, d, s = explain_price(distance, sel_v, sel_p)
        fig = go.Figure(data=[go.Pie(labels=['Base', 'Distance', 'Surge'], values=[b, d, s], hole=.6)])
        fig.update_layout(height=300, margin=dict(t=0,b=0,l=0,r=0), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

else:
    st.info("Please use the sidebar to find a price.")