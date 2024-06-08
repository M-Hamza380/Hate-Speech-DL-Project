import os
from dataclasses import dataclass
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

'''


@dataclass
class DataValidationConfig:
    def __init__(self):
        self.DATA_VALIDATION_ARTIFACTS_DIR: str = os.path.join(os.getcwd(), ARTIFACTS_DIR, DATA_VALIDATION_ARTIFACTS_DIR)
        self.ID = IMBALANCED_DATA_ID


@dataclass
class DataTransformationConfig:
    def __init__(self):
        self.DATA_TRANSFORMATION_ARTIFACTS_DIR: str = os.path.join(os.getcwd(), ARTIFACTS_DIR, DATA_TRANSFORMATION_ARTIFACTS_DIR)
        self.TRANSFORMED_FILE_NAME = os.path.join(self.DATA_TRANSFORMATION_ARTIFACTS_DIR, TRANSFORMED_FILE_NAME)
        self.ID = ID
        self.AXIS = AXIS
        self.INPLACE = INPLACE
        self.DROP_COLUMNS = DROP_COULMNS
        self.CLASS = CLASS
        self.LABEL = LABEL
        self.TWEET = TWEET
'''


