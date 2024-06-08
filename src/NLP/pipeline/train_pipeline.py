import sys

from src.NLP.utils.exception import CustomException
from src.NLP.utils.logger import logging
from src.NLP.components.data_ingestion import DataIngestion
from src.NLP.components.data_transformation import DataTransformation
from src.NLP.entity.config_entity import (DataIngestionConfig,)
from src.NLP.entity.artifact_entity import (DataIngestionArtifact,)


class TrainPipeline:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()
        # self.data_transformation_config = DataTransformationConfig()
    
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
            raise CustomException(e, sys) from e
    
    '''
    
    def start_data_transformation(self, data_ingestion_artifacts = DataIngestionArtifact) -> DataTransformationArtifact:
        try:
            logging.info("Entered the satrt_data_transformation method of TrainPipeline class")

            data_transformation = DataTransformation(
                data_ingestion_artifacts = data_ingestion_artifacts,
                data_transformation_config = self.data_transformation_config
            )

            data_transformation_artifact = data_transformation.initiate_data_transformation()
            logging.info("Exited the start_data_transformation method of TrainPipeline class")

            return data_transformation_artifact

        except Exception as e:
            raise CustomException(e, sys) from e
    '''
    
    def run_pipeline(self):
        logging.info("Entered the run_pipeline method of TrainPipeline class")
        try:
            data_ingestion_artifact = self.start_data_ingestion()

            '''
            data_transformation_artifact = self.start_data_transformation(
                data_ingestion_artifacts = data_ingestion_artifact
            )
            '''

            logging.info("Exited the run_pipeline method of TrainPipeline class")
        
        except Exception as e:
            raise CustomException(e, sys)

        
