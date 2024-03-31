import sys

from src.NLP.utils.exception import CustomException
from src.NLP.utils.logger import logging
from src.NLP.components.data_ingestion import DataIngestion
from src.NLP.entity.config_entity import (DataIngestionConfig)
from src.NLP.entity.artifact_entity import (DataIngestionArtifact)


class TrainPipeline:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()
    
    def start_data_ingestion(self) -> DataIngestionArtifact:
        try:
            logging.info("Entered the satrt_data_ingestion method of TrainPipeline class")

            data_ingestion = DataIngestion(
                data_ingestion_config = self.data_ingestion_config
            )

            data_ingestion_artifact = data_ingestion.initiate_data_ingestion()

            logging.info(f"Exited the start_data_ingestion method of TrainPipeline class")

            return data_ingestion_artifact
        except Exception as e:
            raise CustomException(e, sys)
    
    def run_pipeline(self):
        try:
            data_ingestion_artifact = self.start_data_ingestion()
        
        except Exception as e:
            raise CustomException(e, sys)

        
