import sys
import os
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'../../')))

from src.utils.config import MODEL_PATH
from src.utils.common import load_object

def main():
    model=load_object(MODEL_PATH)

    sample_input = pd.DataFrame({
        'datetime': ['2024-04-15 14:30:00'],
        'Event': ['Normal Day'],
        'Vehicle Type': ['Premier'],
        'Pickup Location': ['Connaught Place'],
        'Drop Location': ['Noida Sector 18'],
        'Ride Distance': [12.5]
    })

    print(model.predict(sample_input))
if __name__=='__main___':
    main()