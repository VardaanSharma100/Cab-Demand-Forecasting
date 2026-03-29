import pandas as pd
import json
import os
import sys
import time
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.utils.config import PROCESSED_DATA_PATH
COORD_FILE='data/location_coords.json'

def main():
    print('Starting Coordinate Fetch ...')
    df=pd.read_csv(PROCESSED_DATA_PATH)
    all_locations=pd.concat([df['Pickup Location'],df['Drop Location']]).unique()

    print(f"found {len(all_locations)} unique locations")

    geolocator=Nominatim(user_agent='Cab Demand Forecasting_v1')
    coords_map={}

    for loc in all_locations:
        try:
            search_query=f'{loc},India'
            location=geolocator.geocode(search_query,timeout=10)
            if location:
                coords_map[loc]=[location.latitude,location.longitude]
            else:
                coords_map[loc]=[28.6304,77.2177]
                print(f"not found {loc}")

            time.sleep(1.0)
        except Exception as e:
            print(f"Error with {loc} : {e}")
            coords_map[loc]=[28.6304,77.2177]
    os.makedirs(os.path.dirname(COORD_FILE),exist_ok=True)

    with open(COORD_FILE,'w') as f:
        json.dump(coords_map,f)
if __name__=='__main__':
    main()