from datetime import datetime
from airflow import DAG 
from airflow.operators.python import PythonOperator
from steps.preprocess_step import PreprocessStep
from steps.inference_step import InferenceStep
from steps.feature_engineering_step import FeatureEngineeringStep
from steps.utils.data_classes import PreprocessingData, FeatureEnginneringData
from steps.config import (
    FeatureEngineeringConfig,
    INFERENCE_DATA_PATH,
    PreprocessConfig,
)

# preparation
inference_mode = True
preprocessing_data = PreprocessingData(
    batch_path=PreprocessingData.batch_path
)

feature_engineering_data = FeatureEnginneringData(
    batch_path=FeatureEngineeringConfig.batch_config,
    encoder_path=FeatureEngineeringConfig.encoders_path
)

# steps
preprocess_step = PreprocessStep(
    inference_mode = inference_mode,
    preprocessing_data = preprocessing_data
)

feature_engineering_step = FeatureEngineeringStep(
    inference_mode= inference_mode,
    feature_engineering_data=feature_engineering_data
)

inference_step = InferenceStep()

default_args = {
    'owner' :"user",
    'depends_on_past' : False,
    'retries': 0,
    'catchup': False
}

with DAG(
    'inference-pipeline',
    default_args = default_args,
    start_date = datetime(2024,1,16),
    tags = ['inference'],
    schedule= None,
) as dag:
    preprocess_task = PythonOperator(
        task_id = 'preprocessing',
        python_callable=preprocess_step,
        op_kwargs={
            'data_path' : INFERENCE_DATA_PATH
        }
    )

    feature_engineering_task = PythonOperator(
        task_id = "feature_engineering",
        python_callable=feature_engineering_step,
        op_kwargs={
            'batch_path': preprocessing_data.batch_path
        }
    )

    inference_task = PythonOperator(
        task_id = 'inference',
        python_callable=inference_step,
        op_kwargs={
            'batch_path' : feature_engineering_data.batch_path
        }
    )
    
    preprocess_task >> feature_engineering_task >> inference_task