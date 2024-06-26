import os
from dataclasses import dataclass, field
from datetime import datetime
from src.NLP.constants.train_pipeline import *


@dataclass
class DataIngestionConfig:
    data_ingestion_dir: str = os.path.join(
        os.getcwd(), ARTIFACTS_DIR, DATA_INGESTION_DIR_NAME
    )

    feature_store_file_path: str = os.path.join(
        data_ingestion_dir, DATA_INGESTION_FEATURE_STORE_DIR
    )

    imbalanced_data_file_path: str = os.path.join(
        feature_store_file_path, IMBALANCED_DATA
    )

    raw_data_file_path: str = os.path.join(
        feature_store_file_path, RAW_DATA
    )

    data_download_url: str = DATA_DOWNLOAD_URL


@dataclass
class DataValidationConfig:
    data_validation_dir: str = os.path.join(
        os.getcwd(), ARTIFACTS_DIR, DATA_VALIDATION_ARTIFACTS_DIR
    )
    
    report_file_path: str = os.path.join(
        data_validation_dir, DATA_VALIDATION_REPORT_DIR, DATA_VALIDATION_REPORT_FILE_NAME
    )

    imbalanced_data_file_path: str = os.path.join(os.getcwd(), ARTIFACTS_DIR, DATA_INGESTION_DIR_NAME, DATA_INGESTION_FEATURE_STORE_DIR, IMBALANCED_DATA)
    raw_data_file_path: str = os.path.join(os.getcwd(), ARTIFACTS_DIR, DATA_INGESTION_DIR_NAME, DATA_INGESTION_FEATURE_STORE_DIR, RAW_DATA)

    schema_mapping: dict = field(default_factory= lambda: {
        'imbalanced_data.csv': IMBALANCED_DATA_SCHEMA,
        'raw_data.csv': RAW_DATA_SCHEMA
    })


@dataclass
class DataTransformationConfig:
    def __init__(self):
        self.DATA_TRANSFORMATION_ARTIFACTS_DIR: str = os.path.join(os.getcwd(), ARTIFACTS_DIR, DATA_TRANSFORMATION_ARTIFACTS_DIR)
        self.TRANSFORMED_FILE_NAME = os.path.join(self.DATA_TRANSFORMATION_ARTIFACTS_DIR, TRANSFORMED_FILE_NAME)
        self.ID = ID
        self.AXIS = AXIS
        self.INPLACE = INPLACE
        self.IGNORE_INDEX = IGNORE_INDEX
        self.DROP_COLUMNS = DROP_COULMNS
        self.CLASS = CLASS
        self.LABEL = LABEL
        self.TWEET = TWEET


@dataclass
class ModelTrainerConfig:
    def __init__(self):
        self.MODEL_TRAINER_ARTIFACTS_DIR: str = os.path.join(os.getcwd(), ARTIFACTS_DIR, MODEL_TRAINER_ARTIFACTS_DIR)
        self.TRAINED_MODEL_DIR: str = os.path.join(self.MODEL_TRAINER_ARTIFACTS_DIR, TRAINED_MODEL_DIR)
        self.TRAINED_MODEL_PATH: str = os.path.join(self.TRAINED_MODEL_DIR, TRAINED_MODEL_NAME)
        self.TOKENIZER_PATH: str = os.path.join(self.TRAINED_MODEL_DIR, TOKENIZER_NAME)
        self.X_TEST_DATA_PATH: str = os.path.join(self.TRAINED_MODEL_DIR, X_TEST_FILE_NAME)
        self.Y_TEST_DATA_PATH: str = os.path.join(self.TRAINED_MODEL_DIR, Y_TEST_FILE_NAME)
        self.X_TRAIN_DATA_PATH: str = os.path.join(self.TRAINED_MODEL_DIR, X_TRAIN_FILE_NAME)
        self.RANDOM_STATE = RANDOM_STATE
        self.TEST_SIZE = TEST_SIZE
        self.EPOCH = EPOCH
        self.BATCH_SIZE = BATCH_SIZE
        self.VALIDATION_SPLIT = VALIDATION_SPLIT
        self.MAX_WORDS = MAX_WORDS
        self.MAX_LEN = MAX_LEN
        self.LOSS = LOSS
        self.METRICS = METRICS
        self.ACTIVATION = ACTIVATION
        self.LABEL = LABEL
        self.TWEET = TWEET


@dataclass
class ModelEvaluationConfig:
    def __init__(self):
        self.MODEL_EVALUATION_DIR = os.path.join(os.getcwd(), ARTIFACTS_DIR, MODEL_EVALUATION_ARTIFACTS_DIR)
        self.BEST_MODEL_DIR_PATH = os.path.join(self.MODEL_EVALUATION_DIR, BEST_MODEL_DIR)
        self.MODEL_NAME = MODEL_NAME


@dataclass
class ModelPusherConfig:
    def __init__(self):
        self.MODEL_PUSHER_DIR = os.path.join(os.getcwd(), ARTIFACTS_DIR, MODEL_PUSHER_ARTIFACTS_DIR)
        self.BEST_MODEL_DIR = os.path.join(self.MODEL_PUSHER_DIR, BEST_MODEL_DIR)
        self.MODEL_NAME = MODEL_NAME
        self.TOKENIZER_NAME = TOKENIZER_NAME
        self.TOKENIZER_PATH = os.path.join(self.BEST_MODEL_DIR, self.TOKENIZER_NAME)


@dataclass
class PredictionPipelineConfig:
    def __init__(self):
        self.BEST_MODEL_DIR = os.path.join(os.getcwd(), ARTIFACTS_DIR, MODEL_PUSHER_ARTIFACTS_DIR, BEST_MODEL_DIR)
        self.MODEL_PATH = os.path.join(self.BEST_MODEL_DIR, MODEL_NAME)
        self.TOKENIZER_PATH: str = os.path.join(self.BEST_MODEL_DIR, TOKENIZER_NAME)
        self.MAX_LEN = MAX_LEN
