TRAINING_DATA_PATH = 'data\hotel_bookings.parquet'
INFERENCE_DATA_PATH = 'data\sample_for_inference.parquet'

class PreprocessConfig:
    train_path = "data/preprocessed/train.parquet"
    test_path = 'data/preprocessed/test.parquet'
    batch_path = 'data/preprocessed/batch.parquet'


class TrainerConfig:
    model_name = 'gradient-boosting'
    random_state = 42
    train_size = 0.2
    shuffle = True
    params =  {
        'n_estimators' : 100,
        'min_sample_split' : 2,
        'min_sample_leaf': 1
    }

class ConditionConfig:
    criteria = 0.05
    metric = 'roc_auc'

class MlFlowConfig:
    uri = 'http://0.0.0.0:8000'
    experiment_name = 'cancelation_estimator'
    artifact_path = 'model-artifact'
    registered_model_name = 'cancelation_estimator'


class FeatureEngineeringConfig:
    train_path = 'data/features_store/train.parquet'
    test_path = 'data/features_store/test.parquet'
    batch_path = 'data/features_store/batch.parquet'
    encoders_path = 'artifacts/encoders.joblib'
    base_features = [
        "lead_time",
        "arrival_date_year",
        "arrival_date_week_number",
        "arrival_date_day_of_month",
        "stays_in_weekend_nights",
        "stays_in_week_nights",
        "adults",
        "children",
        "babies",
        "is_repeated_guest",
        "previous_cancellations",
        "previous_bookings_not_canceled",
        "days_in_waiting_list",
        "adr",
        "required_car_parking_spaces",
        "total_of_special_requests"
    ]
    ordinal_features = [
        "arrival_date_month",
        "meal",
        "market_segment",
        "distribution_channel",
        "reserved_room_type",
        "assigned_room_type",
        "customer_type"
    ]
    target_features = [
        "country",
        "booking_changes",
        "agent",
        "company"
    ]
    target = "is_canceled"