import sys
import os 
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'../../')))

from src.features.preprocessing import TemporalFeatureExtractor, build_feature_pipeline
from src.utils.config import PROCESSED_DATA_PATH,TARGET_COL,ALL_FEATURES

def main():
    df=pd.read_csv(PROCESSED_DATA_PATH)
    X=df[ALL_FEATURES]
    y=df[TARGET_COL].astype(float)

    x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

    xgb_params = {
        'n_estimators': 577,
        'max_depth': 8,
        'learning_rate': 0.05972362157207607,
        'subsample': 0.7101105091676063,
        'colsample_bytree': 0.7718765912856783,
        'min_child_weight': 6,
        'gamma': 0.14390799526208975,
        'random_state': 42,
        'n_jobs': -1
    }
    pipeline = Pipeline([
        ('extractor', TemporalFeatureExtractor()),
        ('preprocessor', build_feature_pipeline()),
        ('model', XGBRegressor(**xgb_params))
    ])
    pipeline.fit(x_train,y_train)

    y_pred=pipeline.predict(x_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    print(f" R2 Score (Accuracy): {r2:.4f}")
    print(f" MAE (Avg Error):     {mae:.2f}")
    print(f" RMSE (Root Error):   {rmse:.2f}")

if __name__=='__main__':
    main()
