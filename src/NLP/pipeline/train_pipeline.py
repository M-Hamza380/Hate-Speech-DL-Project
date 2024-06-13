import sys

from src.NLP.utils.exception import CustomException
from src.NLP.utils.logger import logging
from src.NLP.components.data_ingestion import DataIngestion
from src.NLP.components.data_transformation import DataTransformation
from src.NLP.components.data_validation import DataValidation
from src.NLP.components.model_trainer import ModelTrainer
from src.NLP.components.model_evaluation import ModelEvaluation
from src.NLP.components.model_pusher import ModelPusher

from src.NLP.entity.config_entity import (DataIngestionConfig, 
                                          DataValidationConfig, 
                                          DataTransformationConfig, 
                                          ModelTrainerConfig,
                                          ModelEvaluationConfig,
                                          ModelPusherConfig)
from src.NLP.entity.artifact_entity import (DataIngestionArtifact, 
                                            DataValidationArtifact,
                                            DataTransformationArtifact, 
                                            ModelTrainerArtifact,
                                            ModelEvaluationArtifact,
                                            ModelPusherArtifact)


class TrainPipeline:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()
        self.data_validation_config = DataValidationConfig()
        self.data_transformation_config = DataTransformationConfig()
        self.model_trainer_config = ModelTrainerConfig()
        self.model_evaluation_config = ModelEvaluationConfig()
        self.model_pusher_config = ModelPusherConfig()
    
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
    
    def start_data_validation(self, data_ingestion_artifacts: DataIngestionArtifact) -> DataValidationArtifact:
        try:
            logging.info(f"Entered the start_data_validation method of TrainPipeline class")
            data_validation = DataValidation(data_validation_config=self.data_validation_config)
            data_validation_artifact = data_validation.initiate_data_validation()
            logging.info(f"Exited the start_data_validation method of TrainPipeline class")
            return data_validation_artifact
        except Exception as e:
            raise CustomException(e, sys) from e
    
    
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
    
    def start_model_trainer(self, data_transformation_artifact: DataTransformationArtifact) -> ModelTrainerArtifact:
        try:
            logging.info(f"Entered the start_model_trainer method of TrainPipeline class")
            model_trainer = ModelTrainer(
                    data_transformation_artifacts= data_transformation_artifact,
                    model_trainer_config= self.model_trainer_config
                )
            model_trainer_artifact = model_trainer.initiate_model_trainer()
            logging.info(f"Exited the start_model_trainer method of TrainPipeline class")
            return model_trainer_artifact
        except Exception as e:
            raise CustomException(e, sys) from e
    
    def start_model_evaluation(self, model_trainer_artifact: ModelTrainerArtifact, data_transformation_artifact: DataTransformationArtifact) -> ModelEvaluationArtifact:
        try:
            logging.info(f"Entered the start_model_evaluation method of TrainPipeline class")
            model_evaluation = ModelEvaluation(
                data_transformation_artifact= data_transformation_artifact,
                model_evaluation_config= self.model_evaluation_config,
                model_trainer_artifact= model_trainer_artifact,
                model_pusher_config= self.model_pusher_config
            )
            model_evaluation_artifact = model_evaluation.initiate_model_evaluation()
            logging.info(f"Exited the start_model_evaluation method of TrainPipeline class")
            return model_evaluation_artifact
        except Exception as e:
            raise CustomException(e, sys) from e
    
    def start_model_pusher(self, trained_model_path: str, tokenizer_path: str) -> ModelPusherArtifact:
        try:
            logging.info("Entered the start_model_pusher method of TrainPipeline class")
            model_pusher = ModelPusher(
                model_pusher_config = self.model_pusher_config,
            )
            model_pusher_artifact = model_pusher.initiate_model_pusher(trained_model_path, tokenizer_path)
            logging.info("Exited the start_model_pusher method of TrainPipeline class")
            return model_pusher_artifact
        except Exception as e:
            raise CustomException(e, sys)
    


    def run_pipeline(self):
        logging.info("Entered the run_pipeline method of TrainPipeline class")
        try:
            data_ingestion_artifact = self.start_data_ingestion()
            data_validation_artifact = self.start_data_validation(
                    data_ingestion_artifacts = data_ingestion_artifact
                )
    
            data_transformation_artifact = self.start_data_transformation(
                data_ingestion_artifacts = data_ingestion_artifact
            )

            model_trainer_artifact = self.start_model_trainer(
                data_transformation_artifact= data_transformation_artifact
            )

            model_evaluation_artifact = self.start_model_evaluation(
                model_trainer_artifact = model_trainer_artifact,
                data_transformation_artifact = data_transformation_artifact
            )

            if model_evaluation_artifact.is_model_accepted:
                model_pusher_artifact = self.start_model_pusher(
                    model_trainer_artifact.trained_model_path,
                    model_trainer_artifact.tokenizer_path
                    )
                logging.info(f"Model pusher artifact: {model_pusher_artifact}")
            else:
                logging.info("The current trained model was not accepted.")

            logging.info("Exited the run_pipeline method of TrainPipeline class")
        except Exception as e:
            raise CustomException(e, sys)
