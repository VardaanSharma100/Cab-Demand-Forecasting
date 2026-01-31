import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from src.utils.config import DATETIME_COL,CATEGORICAL_COLS,NUMERICAL_COLS
from sklearn.preprocessing import TargetEncoder
class TemporalFeatureExtractor(BaseEstimator,TransformerMixin):
    def fit(self,X,y=None): return self
    def transform(self,X):
        X_=X.copy()
        X_[DATETIME_COL]=pd.to_datetime(X_[DATETIME_COL])
        X_['month']=X_[DATETIME_COL].dt.month
        X_['weekday']=X_[DATETIME_COL].dt.weekday
        X_['minute']=X_[DATETIME_COL].dt.minute
        X_['hour']=X_[DATETIME_COL].dt.hour

        X_=X_.drop(columns=[DATETIME_COL])
        return X_
    
class CyclicFeatureTransformer(BaseEstimator,TransformerMixin):
    def __init__(self,column_name,max_val):
        self.column_name=column_name
        self.max_val=max_val
    def fit(self,X,y=None):return self
    def transform(self,X):
        X_=X.copy()
        X_[f'{self.column_name}_sin']=np.sin(2*np.pi*X_[self.column_name]/self.max_val)
        X_[f'{self.column_name}_cos']=np.cos(2*np.pi*X_[self.column_name]/self.max_val)
        X_=X_.drop(columns=[self.column_name])
        return X_
def build_feature_pipeline():

    hour_trans=CyclicFeatureTransformer('hour',24)
    minute_trans=CyclicFeatureTransformer('minute',60)
    weekday_trans=CyclicFeatureTransformer('weekday',7)
    month_trans=CyclicFeatureTransformer('month',12)

    target_enc_cols = ['Pickup Location', 'Drop Location'] 
    ohe_cols = ['Vehicle Type', 'Event']

    num_trans=StandardScaler()

    ohe_trans = OneHotEncoder(drop='first',sparse_output=False)
    target_enc_trans = TargetEncoder(target_type='continuous',smooth=10)
    preprocesser=ColumnTransformer(
        transformers=[
            ('ohe',ohe_trans,ohe_cols),
            ('target_enc',target_enc_trans,target_enc_cols),
            ('num',num_trans,NUMERICAL_COLS),

            ('cyclic_h',hour_trans,['hour']),
            ('cyclic_m',minute_trans,['minute']),
            ('cyclic_w',weekday_trans,['weekday']),
            ('cyclic_mo',month_trans,['month'])
        ],
        remainder='passthrough'
    )
     
    return preprocesser