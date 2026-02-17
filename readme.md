# Cab Demand & Fare Forecasting

This project is a machine learning-based application designed to predict dynamic cab fares and demand. It utilizes a robust data pipeline and an interactive Streamlit dashboard to provide insights and fare estimations based on location, time, vehicle type, and special events.

## 🚀 Features

*   **Interactive Dashboard**: A user-friendly web interface built with Streamlit.
*   **Dynamic Fare Prediction**: Accurate fare estimates using an XGBoost regression model.
*   **Location Visualizations**: Integration with Folium for interactive map-based location selection.
*   **Event-Aware Pricing**: Accounts for special events (e.g., Festivals, Weather conditions) that impact demand and pricing.
*   **Vehicle Variety**: Supports multiple vehicle types including Auto, Premier Sedan, Bike, Uber XL, and more.

## 📊 Model Performance

The predictive model has achieved high accuracy on the test dataset:

*   **R² Score (Accuracy)**: 0.89
*   **MAE (Mean Absolute Error)**: 15.15
*   **RMSE (Root Mean Square Error)**: 24.78

## 📂 Project Structure

```text
├── data/               # Raw and processed datasets
├── logs/               # Application logs
├── models/             # Trained serialized models (.joblib)
├── notebooks/          # Jupyter notebooks for EDA and experimentation
├── src/                # Source code directory
│   ├── data/           # Scripts for data ingestion and processing
│   ├── features/       # Feature engineering and transformation
│   ├── models/         # Model training and evaluation scripts
│   └── utils/          # Utility functions and configuration
├── app.py              # Main Streamlit application entry point
├── readme.md           # Project documentation
└── requirements.txt    # (Recommended) List of dependencies
```

## 🛠️ Installation

1.  **Clone the repository**:
    ```bash
     git clone <repository-url>
     cd "Cab Demand Forecasting"
    ```

2.  **Install dependencies**:
    Ensure you have Python installed. It is recommended to use a virtual environment.
    ```bash
    pip install pandas streamlit joblib folium streamlit-folium plotly geopy xgboost scikit-learn
    ```

## 💡 Usage

### Running the Web Application
To launch the interactive dashboard:
```bash
streamlit run app.py
```

### Data Pipeline & Training
To reproduce the data processing and model training steps:

1.  **Process Data**:
    ```bash
    python src/data/make_data.py
    ```
    This cleans the raw data and saves it to `data/processed/`.

2.  **Train Model**:
    ```bash
    python src/models/train_model.py
    ```
    This trains the XGBoost model and saves the pipeline to `models/final_pipeline.joblib`.

## 📈 Data & Events

The model considers various factors including:

*   **Vehicle Types**: Auto, Premier Sedan, Bike, Go Mini, Go Sedan, Uber XL, eBike.
*   **Events**: Monsoon, Wedding Season, Diwali, Christmas, New Year, and standard days.
