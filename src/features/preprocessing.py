import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from src.utils.config import DATETIME_COL,CATEGORICAL_COLS,NUMERICAL_COLS

class TemporalFeatureExtractor(BaseEstimator,TransformerMixin):
    def fit(self,X,y): return self
    def transform(self,X):
        X[DATETIME_COL]=pd.to_datetime(X[DATETIME_COL])
        X['month']=X[DATETIME_COL].dt.month
        X['weekday']=X[DATETIME_COL].dt.weekday
        X['minute']=X[DATETIME_COL].dt.minute
        X['hour']=X[DATETIME_COL].dt.hour

        X=X.drop(columns=[DATETIME_COL])
        return X
    
class CyclicFeatureTransformer(BaseEstimator,TransformerMixin):
    def __init__(self,column_name,max_val):
        self.column_name=column_name
        self.max_val=max_val
    def fit(self,X,y):return self
    def transform(self,X):
        X[f'{self.column_name}_sin']=np.sin(2*np.pi*X[self.column_name]/self.max_val)
        X[f'{self.column_name}_cos']=np.cos(2*np.pi*X[self.column_name]/self.max_val)
        X=X.drop(columns=[self.column_name])
        return X
def build_feature_pipeline():

    hour_trans=CyclicFeatureTransformer('hour',24)
    minute_trans=CyclicFeatureTransformer('minute',60)
    weekday_trans=CyclicFeatureTransformer('weekday',7)
    month_trans=CyclicFeatureTransformer('month',12)

    cat_trans=OneHotEncoder(drop='first')
    num_trans=StandardScaler()

    preprocesser=ColumnTransformer(
        transformers=[
            ('cat',cat_trans,CATEGORICAL_COLS),
            ('num',num_trans,NUMERICAL_COLS),

            ('cyclic_h',hour_trans,['hour']),
            ('cyclic_m',minute_trans,['minute']),
            ('cyclic_w',weekday_trans,['weekday']),
            ('cyclic_mo',month_trans,['month'])
        ],
        remainder='passthrough'
    )
     
    return preprocesser