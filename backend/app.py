from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime
from geopy.distance import geodesic
import pandas as pd
import joblib
import json
import os

app = FastAPI(title="Cab Demand Forecasting API")

# Enable CORS for the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models/final_pipeline.joblib')
COORD_FILE = os.path.join(os.path.dirname(__file__), 'data/location_coords.json')

# Try to load models and coords
pipeline = None
LOC_COORDS = {}

try:
    if os.path.exists(MODEL_PATH):
        pipeline = joblib.load(MODEL_PATH)
except Exception as e:
    print(f"Error loading model: {e}")

try:
    if os.path.exists(COORD_FILE):
        with open(COORD_FILE, 'r') as f:
            LOC_COORDS = json.load(f)
except Exception as e:
    print(f"Error loading coords: {e}")

vehicle_types = ['Bike', 'eBike', 'Auto', 'Go Mini', 'Go Sedan', 'Premier Sedan', 'Uber XL']

class RideRequest(BaseModel):
    pickup: str
    drop: str
    date: str
    time: str
    event: str

def calculate_distance(p_coords, d_coords):
    return geodesic(p_coords, d_coords).km * 1.3

def explain_price(distance, vehicle_type, predicted_price=0):
    base_fares = {'Bike': 25, 'eBike': 20, 'Auto': 40, 'Go Mini': 50, 'Go Sedan': 80, 'Premier Sedan': 100, 'Uber XL': 150}
    rates_per_km = {'Bike': 6, 'eBike': 5, 'Auto': 10, 'Go Mini': 12, 'Go Sedan': 17, 'Premier Sedan': 20, 'Uber XL': 28}
    base = base_fares.get(vehicle_type, 80)
    rate = rates_per_km.get(vehicle_type, 17)
    return base + (distance * rate)

@app.get("/locations")
def get_locations():
    return list(LOC_COORDS.keys()) if LOC_COORDS else ["Connaught Place", "Noida Sector 62"]

@app.get("/events")
def get_events():
    return ['Normal Day', 'Monsoon', 'Wedding_Season', 'Diwali', 'New Year']

@app.post("/estimate")
def get_estimate(req: RideRequest):
    # Get distance
    p_coords = LOC_COORDS.get(req.pickup)
    d_coords = LOC_COORDS.get(req.drop)
    
    if p_coords and d_coords:
        distance = calculate_distance(p_coords, d_coords)
    else:
        distance = 10.0 # Default fallback
        
    prices = []
    for v in vehicle_types:
        try:
            input_df = pd.DataFrame({
                'datetime': [pd.to_datetime(f"{req.date} {req.time}")],
                'Event': [req.event], 
                'Vehicle Type': [v],
                'Pickup Location': [req.pickup], 
                'Drop Location': [req.drop],
                'Ride Distance': [distance]
            })
            if pipeline:
                predicted = int(pipeline.predict(input_df)[0])
            else:
                predicted = int(explain_price(distance, v))
        except Exception as e:
            predicted = int(explain_price(distance, v))

        # Build simulated breakdown logic similar to what might be graphed
        base_fares = {'Bike': 25, 'eBike': 20, 'Auto': 40, 'Go Mini': 50, 'Go Sedan': 80, 'Premier Sedan': 100, 'Uber XL': 150}
        rates_per_km = {'Bike': 6, 'eBike': 5, 'Auto': 10, 'Go Mini': 12, 'Go Sedan': 17, 'Premier Sedan': 20, 'Uber XL': 28}
        b_fare = base_fares.get(v, 80)
        d_fare = int(distance * rates_per_km.get(v, 17))
        surge = max(0, predicted - (b_fare + d_fare))
        
        breakdown = [
            {"label": "Base Fare", "value": b_fare},
            {"label": f"Distance ({round(distance,1)}km)", "value": d_fare},
            {"label": "Time/Demand Factor", "value": surge}
        ]

        prices.append({
            "vehicle": v,
            "price": predicted,
            "capacity": 4 if v in ['Go Sedan', 'Premier Sedan', 'Go Mini'] else (6 if v == 'Uber XL' else (1 if v in ['Bike', 'eBike'] else 3)),
            "desc": "Affordable ride" if "Go" in v else "Premium experience" if "Premier" in v else "Quick travel",
            "breakdown": breakdown
        })

    return {
        "distance": round(distance, 1),
        "estimates": prices,
        "pickup_coords": p_coords,
        "drop_coords": d_coords
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
