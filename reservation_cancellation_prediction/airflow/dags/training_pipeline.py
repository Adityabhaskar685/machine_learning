from datetime import datetime

from airflow import DAG  
from airflow.operators.python import PythonOperator
from steps.preprocess_step import PreprocessingStep
from steps.train_step import TrainStep
from steps.condition_step import ConditionStep
from steps.feature_engineering_step import FeatureEngineeringStep
from steps.utils.data_classes import PreprocessingData, FeatureEnginneringData

from steps.config import (
    TRAINING_DATA_PATH,
    TrainerConfig,
    ConditionConfig,
    PreprocessConfig,
    FeatureEngineeringConfig
)


# preparation
inference_mode = False
preprocessing_data = PreprocessingData(
    train_path = PreprocessingData.train_path,
    test_path = PreprocessConfig.test_path
)

feature_engineering_data = FeatureEnginneringData(
    train_path = FeatureEngineeringConfig.train_path,
    test_path = FeatureEngineeringConfig.test_path,
    encoder_path = FeatureEngineeringConfig.encoders_path
)

target = FeatureEngineeringConfig.target

#steps 

preprocess_step = PreprocessingStep(
    inference_mode = inference_mode,
    preprocess_data = Preprocessing_data
)

feature_engineering_step = FeatureEngineeringStep(
    inference_mode=inference_mode,
    feature_engineering_data=feature_engineering_data
)

train_step = TrainStep(
    params = TrainerConfig.params
)

condition_step = ConditionStep(
    criteria=ConditionConfig.criteria,
    metric = ConditionConfig.metric
)

default_args = {
    'owner' : 'user',                    # user's name
    "depends_on_past": False,            # keeps a task from getting triggered if the previous schedule for the task hasn’t succeeded.
    "retries": 0,                        # Number of retries for a dag 
    "catchup": False,                    # Run the dag from the start_date to today in respect to the trigger frequency 

}

with DAG(
    "training-pipeline", 
    default_args=default_args,
    start_date = datetime(2024,01,17),
    tags = ['training'],
    schedule= None
) as dag:
    preprocessing_task = PythonOperator(
        task_id = 'preprocessing',
        python_callable=preprocess_step,
        op_kwargs={'data_path': TRAINING_DATA_PATH}
    )

    feature_engineering_task = PythonOperator(
        task_id = "feature_engineering",
        python_callable=feature_engineering_step,
        op_kwargs={
            'train_path' : preprocessing_data.train_path,
            'test_path' : preprocessing_data.test_path,
        }
    )

    training_task = PythonOperator(
        task_id = "training",
        python_callable=train_step,
        op_kwargs={
            'train_path' : feature_engineering_data.train_path,
            'test_path' : feature_engineering_data.test_path,
            'target' : target
        }
    )

    validation_task = PythonOperator(
        task_id = 'validation',
        python_callable=condition_step,
        op_kwargs=training_task.output
    )

    preprocessing_task >> feature_engineering_task >> training_task >> validation_task