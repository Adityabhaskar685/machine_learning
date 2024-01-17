import logging
from typing import Tuple, Optional
import joblib
from pathlib import Path
from sklearn.preprocessing import OrdinalEncoder, TargetEncoder
import pandas as pd
from steps.utils.data_classes import FeaturesEncoder, FeatureEngineeringData
from steps.config import FeatureEngineeringConfig

LOGGER = logging.getLogger(__name__)

class FeatureEngineeringStep:
    """Feature engineering: transform features for model and inference
    Agrs: 
        inference_mode (bool) : Whether the step is used in the training or inference pipeline.
        feature_engineering_data (FeatureEngineeringData): Paths relative to the FeatureEngineeringStep
    """

    def __init__(self , inference_mode: bool,
                 feature_engineering_data : FeatureEngineeringData) -> None:
        self.inference_mode = inference_mode
        self.feature_engineering_data = feature_engineering_data

    def __call__(self, train_path : Optional[Path] = None,test_path : Optional[Path] = None,batch_path: Optional[Path] = None,) -> None:

        """
        Input data paths depending on whether it's training (train, test) or inference (batch)

        Args:
            train_path (Optional[Path], optional): Input train path. Defaults to None.
            test_path (Optional[Path], optional): Input test path. Defaults to None.
            batch_path (Optional[Path], optional): input batch path. Defaults to None.
        """
        if not self.inference_mode:
            train_df = pd.read_parquet(train_path)
            test_df = pd.read_path(test_path)
            self.fit_transform(
                df = train_df,
                output_path = self.feature_engineering_data.train_data
            )
            self.transform(
                df = test_df,
                output_path = self.feature_engineering_data.test_path
            )
        if self.inference_mode:
            batch_df = pd.read_parquet(batch_path)
            self.transform(
                batch_df,
                output_path = self.feature_engineering_data.batch_path
            )
    def fit_transform(
            self, 
            df : pd.DataFrame,
            output_path : Path
    ) -> None:
        """Fit encoder on data and store and encoder into feature store
        The processed data is then stored.
        Args: 
            df (pd.DataFrame): Data to train encoders and to transform
            output_path (Path) : Data path after encoding.
        """
        LOGGER.info("Start feature enginnering 'fit_transform'.")
        feature_encoders = self._init_feature_encoder()
        base_df, ordinal_df, target_df, target_col = self._get_dfs(
            df = df, 
            feature_encoder = feature_encoders
        )

        ordinal_encoded_data = feature_encoders.ordinal_encoder.fit_transform(ordinal_df)
        target_encoded_data = feature_encoders.target_endoder.fit_transform(target_df, target_col)

        base_df[feature_encoders.ordinal_features] = ordinal_encoded_data
        base_df[feature_encoders.target_features] = target_encoded_data

        # add the target
        base_df[feature_encoders.target] = target_col

        base_df.to_parquet(path = output_path)
        feature_encoders.to_joblib(path = self.feature_engineering_data.encoders_path)
        LOGGER.info(
            f"Features and encoders successfully saved respectively to {str(output_path)} and {str(self.feature_engineering_data.encoders_path)}"
        )

    def transform(
            self, 
            df : pd.DataFrame,
            output_path : Path
    ) -> None:
        """Transform data based on trained encoders.
        Args: 
            df (pd.DataFrame): Data to transform
            output_path (Path) : Transformed data path.
        """
        LOGGER.info("Start features engineering 'transform'.")
        feature_encoder = self._load_features_encoder()
        base_df, ordinal_df, target_df, target_col = self._get_dfs(
            df = df, 
            feature_encoder = feature_encoder
        )

        ordinal_encoded_data = feature_encoder.ordinal_encoder.transform(ordinal_df)
        target_encoded_data = feature_encoder.target_endoder.transform(target_df, target_col)

        base_df[feature_encoder.ordinal_features] = ordinal_encoded_data
        base_df[feature_encoder.target_features] = target_encoded_data

        # add the target
        if target_col is not None:
            # inference
            base_df[feature_encoder.target] = target_col
        base_df.to_parquet(path = output_path)
        LOGGER.info(f"Features successfully saved to {str(output_path)}")
    
    def _init_feature_encoder(self) -> FeaturesEncoder:
        """init encoders for fit_transform
        Return: 
            feature_encoders (FeatureEncoders): Encoder artifacts
        """
        ordinal_encoder = OrdinalEncoder(
            handle_unknown= 'use_encoded_value',
            unknown_value= -1
        )
        target_encoder = TargetEncoder()
        return FeaturesEncoder(
            ordinal_encoder=ordinal_encoder,
            target_endcoder= target_encoder,
            ordinal_features= FeatureEngineeringConfig.ordinal_features,
            target_features= FeatureEngineeringConfig.target_columns,
            base_features= FeatureEngineeringConfig.base_config,
            target = FeatureEngineeringConfig.target
        )
    
    def _load_features_encoder(self) -> FeaturesEncoder:
        """Load encoder artifact
        Returns: 
            FeaturesEncoder: Encoders artifact
        """
        feature_encoder = joblib.load(self.feature_engineering_data.encoders_path)
        return feature_encoder

    def _get_dfs(
            self, 
            df : pd.DataFrame,
            features_encoder : FeaturesEncoder
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Optional[pd.Series]]:
        """Extract the relevant column based on features for respectively:
        no tranformation - ordinal categories - target categories"""
        base_df = df[features_encoder.base_features]
        ordinal_df = df[features_encoder.ordinal_features]
        target_df = df[features_encoder.target_features]

        if not self.inference_mode:
            target_col = df[features_encoder.target]
            return base_df, ordinal_df, target_df, target_col
        elif self.inference_mode:
            return base_df, ordinal_df, target_df, None
        
        
        





        




        
        
    

    
        
        
        