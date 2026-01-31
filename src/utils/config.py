import os

PROJECT_ROOT=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

RAW_DATA_PATH=os.path.join(PROJECT_ROOT,'data','raw','ncr_rides_final_event_dataset_2024.csv')
PROCESSED_DATA_PATH=os.path.join(PROJECT_ROOT,'data','processed','train_ready.csv')
MODEL_PATH=os.path.join(PROJECT_ROOT,'models','final_pipeline.joblib')

DATETIME_COL='datetime'
TARGET_COL='Booking Value'

CATEGORICAL_COLS=['Event','Vehicle Type','Pickup Location','Drop Location']
NUMERICAL_COLS=['Ride Distance']

ALL_FEATURES=[DATETIME_COL] + CATEGORICAL_COLS + NUMERICAL_COLS

HYPERPARAMETERS={'n_estimators': 577,
 'max_depth': 8,
 'learning_rate': 0.05972362157207607,
 'subsample': 0.7101105091676063,
 'colsample_bytree': 0.7718765912856783,
 'min_child_weight': 6,
 'gamma': 0.14390799526208975,
 'random_state':42,
 'n_jobs':-1
 }