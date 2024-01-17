from typing import Iterable
from mlflow.pyfunc import PythonModel, PythonModelContext
from sklearn.base import BaseEstimator
import pandas as pd
import numpy as np 

class artifact(PythonModel):
    def __init__(
            self,
            model: BaseEstimator,
            ordinal_encoder : BaseEstimator,
            target_endcoder: BaseEstimator,
            ordinal_encoded_features: Iterable[str],
            target_encoded_features: Iterable[str]

    ):
        self.model = model
        self.ordinal_encoder = ordinal_encoder
        self.target_encoder = target_endcoder
        self.ordinal_encoded_features = ordinal_encoded_features
        self.target_encoder_features = target_encoded_features

    def predict(self, context : PythonModelContext, model_input_df: pd.DataFrame) -> np.ndarray:
        if isinstance(model_input_df , pd.DataFrame):
            prepared_df = self._prepare(model_input_df)
            predictions = self.model.predict(prepared_df)
            return predictions
        
        raise TypeError("This implementation can only take pandas DataFrame as inputs")
    
    def _prepare(self, model_input_df : pd.DataFrame) -> pd.DataFrame:
        prepared_df = model_input_df.drop(
            [*self.ordinal_encoded_features, *self.target_encoded_features],
            axis = 1
        )
        prepared_df[self.ordinal_encoded_features] = self.ordinal_encoded.transform(
            model_input_df[self.ordinal_encoded_features]
        )
        prepared_df[self.target_encoder_features] = self.target_encoder.transform(
            model_input_df[self.target_encoder_features]
        )
        return prepared_df