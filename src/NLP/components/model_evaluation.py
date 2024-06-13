import os, sys, pickle
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras
from keras.utils import pad_sequences
from sklearn.metrics import confusion_matrix

from src.NLP.utils.logger import logging
from src.NLP.utils.exception import CustomException
from src.NLP.constants.train_pipeline import *
from src.NLP.entity.config_entity import ModelEvaluationConfig
from src.NLP.entity.artifact_entity import (ModelEvaluationArtifact,
                                            ModelTrainerArtifact,
                                            DataTransformationArtifact)
from src.NLP.entity.config_entity import ModelPusherConfig
from src.NLP.components.model_pusher import ModelPusher


class ModelEvaluation:
    def __init__(self, model_evaluation_config: ModelEvaluationConfig,
                 model_trainer_artifact: ModelTrainerArtifact,
                 data_transformation_artifact: DataTransformationArtifact,
                 model_pusher_config: ModelPusherConfig
                 ):
        self.model_evaluation_config = model_evaluation_config
        self.model_trainer_artifact = model_trainer_artifact
        self.data_transformation_artifact = data_transformation_artifact
        self.model_pusher_config = model_pusher_config
    
    def get_best_model(self) -> str:
        try:
            logging.info(f"Entered the get_best_model method of ModelEvaluation class")
            os.makedirs(self.model_evaluation_config.BEST_MODEL_DIR_PATH, exist_ok=True)
            best_model_path = os.path.join(self.model_evaluation_config.BEST_MODEL_DIR_PATH,
                                           self.model_evaluation_config.MODEL_NAME)
            logging.info(f"Exited the get_best_model method of ModelEvaluation class")
            return best_model_path
        except Exception as e:
            raise CustomException(e, sys) from e
        
    def evaluation(self, model_path: str, tokenizer_path: str):
        try:
            logging.info(f"Entered the evaluation method of ModelEvaluation class")
            print(self.model_trainer_artifact.x_test_path)
            x_test = pd.read_csv(self.model_trainer_artifact.x_test_path, index_col=0)
            y_test = pd.read_csv(self.model_trainer_artifact.y_test_path, index_col=0)

            with open(tokenizer_path, 'rb') as handle:
                tokenizer = pickle.load(handle)
            
            load_model = keras.models.load_model(model_path)

            x_test = x_test['tweet'].astype(str)
            x_test = x_test.squeeze()
            y_test = y_test.squeeze()

            test_sequences = tokenizer.texts_to_sequences(x_test)
            test_sequences_matrix = pad_sequences(test_sequences, maxlen=MAX_LEN)

            accuracy = load_model.evaluate(test_sequences_matrix, y_test)
            print("-------- Accuracy --------")
            print(f" {accuracy} ")
            logging.info(f"The test accuracy is: {accuracy}")

            lstm_pred = load_model.predict(test_sequences_matrix)
            # Perform prediction
            res = [1 if prediction[0] >= 0.5 else 0 for prediction in lstm_pred]
            
            print("-------- Confusion_Matrix --------")
            print(f" {confusion_matrix(y_test, res)} ")
            logging.info(f"The confusion_matrix is: {confusion_matrix(y_test, res)}")
            logging.info(f"Exited the evaluation method of ModelEvaluation class")
            return accuracy[1]
        except Exception as e:
            raise CustomException(e, sys) from e
    
    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:
        try:
            logging.info("Entered the initiate_model_evaluation method of ModelEvaluation class")
            
            logging.info("Loading and evaluating the current trained model")
            current_model_accuracy = self.evaluation(
                self.model_trainer_artifact.trained_model_path,
                self.model_trainer_artifact.tokenizer_path
            )

            best_model_path = self.get_best_model()
            if not os.path.isfile(best_model_path):
                logging.info("No best model found locally. Accepting the current trained model")
                is_model_accepted = True
            else:
                logging.info("Loading and evaluating the best model")
                best_model_accuracy = self.evaluation(best_model_path, self.model_pusher_config.TOKENIZER_NAME)
                is_model_accepted = current_model_accuracy > best_model_accuracy
                print("-------- Model_Accepted --------")
                print(is_model_accepted)

            model_evaluation_artifact = ModelEvaluationArtifact(is_model_accepted=is_model_accepted)

            if is_model_accepted:
                logging.info("Updating the best model with the current trained model")
                model_pusher = ModelPusher(self.model_pusher_config)
                model_pusher_artifact = model_pusher.initiate_model_pusher(
                    self.model_trainer_artifact.trained_model_path,
                    self.model_trainer_artifact.tokenizer_path
                )
                logging.info(f"Model pusher artifact: {model_pusher_artifact}")

            logging.info("Exited the initiate_model_evaluation method of ModelEvaluation class")
            return model_evaluation_artifact
        except Exception as e:
            raise CustomException(e, sys) from e

