import os
from dataclasses import dataclass, field
from datetime import datetime
from src.NLP.constants.train_pipeline import *

TIMESTAMP: str = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")

@dataclass
class DataIngestionPipelineConfig:
    artifacts_dir: str = os.path.join(ARTIFACTS_DIR, TIMESTAMP)


data_ingestion_pipeline_config: DataIngestionPipelineConfig = DataIngestionPipelineConfig()

@dataclass
class DataIngestionConfig:
    data_ingestion_dir: str = os.path.join(
        data_ingestion_pipeline_config.artifacts_dir, DATA_INGESTION_DIR_NAME
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
        ARTIFACTS_DIR, TIMESTAMP, DATA_VALIDATION_ARTIFACTS_DIR
    )
    
    report_file_path: str = os.path.join(
        data_validation_dir, DATA_VALIDATION_REPORT_DIR, DATA_VALIDATION_REPORT_FILE_NAME
    )

    imbalanced_data_file_path: str = os.path.join(ARTIFACTS_DIR, TIMESTAMP, DATA_INGESTION_DIR_NAME, DATA_INGESTION_FEATURE_STORE_DIR, IMBALANCED_DATA)
    raw_data_file_path: str = os.path.join(ARTIFACTS_DIR, TIMESTAMP, DATA_INGESTION_DIR_NAME, DATA_INGESTION_FEATURE_STORE_DIR, RAW_DATA)

    schema_mapping: dict = field(default_factory= lambda: {
        'imbalanced_data.csv': IMBALANCED_DATA_SCHEMA,
        'raw_data.csv': RAW_DATA_SCHEMA
    })

data_validation_config: DataValidationConfig = DataValidationConfig()


@dataclass
class DataTransformationConfig:
    def __init__(self):
        self.DATA_TRANSFORMATION_ARTIFACTS_DIR: str = os.path.join(ARTIFACTS_DIR, TIMESTAMP, DATA_TRANSFORMATION_ARTIFACTS_DIR)
        self.TRANSFORMED_FILE_NAME = os.path.join(self.DATA_TRANSFORMATION_ARTIFACTS_DIR, TRANSFORMED_FILE_NAME)
        self.ID = ID
        self.AXIS = AXIS
        self.INPLACE = INPLACE
        self.IGNORE_INDEX = IGNORE_INDEX
        self.DROP_COLUMNS = DROP_COULMNS
        self.CLASS = CLASS
        self.LABEL = LABEL
        self.TWEET = TWEET



