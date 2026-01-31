import os
import pandas as pd
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from src.features.preprocessing import TemporalFeatureExtractor,build_feature_pipeline
from src.utils.common import setup_logger,save_object
from src.utils.config import PROCESSED_DATA_PATH, MODEL_PATH, TARGET_COL, ALL_FEATURES,HYPERPARAMETERS

logger=setup_logger()

def main():
    logger.info(f'Loading data from{PROCESSED_DATA_PATH}...')
    if not os.path.exists(PROCESSED_DATA_PATH):
        logger.error("Data not found! Please run 'src/data/make_dataset.py' first.")
        return
    df=pd.read_csv(PROCESSED_DATA_PATH)
    X=df[ALL_FEATURES]
    y=df[TARGET_COL].astype(float)

    logger.info(f'Training Data Shape:{X.shape}')

    xgb_params=HYPERPARAMETERS

    model_pipeline=Pipeline([
        ('extractor',TemporalFeatureExtractor()),

        ('preprocesser',build_feature_pipeline()),

        ('model',XGBRegressor(**xgb_params))
    ])

    logger.info("Training model with XGBoost....")

    model_pipeline.fit(X,y)

    save_object(model_pipeline,MODEL_PATH)
    logger.info("Training complete pipeline saved successfully")

if __name__=='__main__':
    main()
