# Cab Demand & Dynamic Fare Forecasting

![Cab Demand Forecasting Demo](demo.gif)

This project is an end-to-end Machine Learning pipeline and forecasting application designed to predict dynamic cab fares accurately. It factors in variables like location, time, vehicle type, traffic distances, and special events to compute an optimal price, mimicking real-world dynamic pricing engines (e.g., Surge Pricing in Uber/Ola).

## 🧠 Machine Learning Architecture

The core of the system relies on an **XGBoost Regression Model** that handles complex, non-linear relationships between the time of day, weather, events, and distance metrics. 

### Model Performance
The predictive model has achieved high accuracy on the test dataset:
*   **R² Score**: 0.89
*   **MAE (Mean Absolute Error)**: 15.15
*   **RMSE (Root Mean Square Error)**: 24.78

### Explainable Pricing (Pricing Breakdown)
Instead of acting as a black box, the model predictions are mathematically broken down to offer a transparent price explanation. Predictions output clear metrics mapping out exactly *why* a price is what it is:
1. **Base Fare**: Fixed base cost correlated to the chosen vehicle capacity.
2. **Distance Contribution**: Mileage rates dynamically mapped via routing APIs.
3. **Surge Factor**: Algorithmic capability triggering price surges mapped to Events (e.g., Monsoon, Diwali, Rush Hour).

### Feature Engineering
The pipeline automatically extracts robust features from raw coordinates and logs, specifically focusing on:
- **Haversine & Routing Distance:** Generating real-world metrics from coordinate mappings.
- **Temporal Cycles**: Encoding time and day periods to capture peak demand trends.
- **Categorical Intersections**: Specialized encodings for geographic nodes and interacting vehicle combinations.

## 📁 Project Structure

```text
├── data/               # Raw and processed datasets
├── models/             # Trained serialized pipeline models (.joblib)
├── notebooks/          # Exploratory Data Analysis (EDA) & experimentation
├── src/                # Machine Learning source code
│   ├── data/           # ETL, ingestion & cleaning
│   ├── features/       # Feature engineering and transformation
│   ├── models/         # XGBoost training & evaluation scripts
│   └── utils/          # ML utility tools
├── backend/            # FastAPI layer serving the ML model & explainability
├── frontend/           # React/Vite UI visualizing geospatial ML predictions
└── requirements.txt    # ML and Backend dependencies
```

## ⚙️ Model Pipeline Execution

Run the ML lifecycle to recreate the dataset and retrain the XGBoost predictor:

1.  **Process Data**:
    ```bash
    python src/data/make_data.py
    ```
    This prepares the latest events and spatial data.

2.  **Train Model**:
    ```bash
    python src/models/train_model.py
    ```
    This trains the XGBoost model and serializes the pipeline to `models/final_pipeline.joblib`.

## 🚀 Serving the Application

To interact with the model predictions via the React and FastAPI stack:

1. **Start the ML Inference Server (Backend)**:
   ```bash
   cd backend
   pip install -r requirements.txt
   uvicorn app:app --reload
   ```

2. **Start the Maps & Prediction UI (Frontend)**:
   ```bash
   cd frontend
   npm install
   npm run dev
   ```
