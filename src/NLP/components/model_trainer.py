import os, sys, pickle
import pandas as pd

from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences

from src.NLP.utils.logger import logging
from src.NLP.utils.exception import CustomException
from src.NLP.constants.train_pipeline import * 
from src.NLP.dl.model import ModelArchitecture
from src.NLP.entity.config_entity import ModelTrainerConfig
from src.NLP.entity.artifact_entity import ModelTrainerArtifact, DataTransformationArtifact

class ModelTrainer:
    def __init__(self, model_trainer_config: ModelTrainerConfig, data_transformation_artifacts: DataTransformationArtifact):
        self.data_transformation_artifact = data_transformation_artifacts
        self.model_trainer_config = model_trainer_config
    
    def spliting_data(self, csv_path):
        try:
            logging.info(f"Entered the spliting_data method")
            df = pd.read_csv(csv_path)
            logging.info(f"Apliting the data into x & y")
            x = df[TWEET].astype(str)
            y = df[LABEL]
            logging.info(f"Applying train_test_split on the data")
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = self.model_trainer_config.TEST_SIZE, random_state= self.model_trainer_config.RANDOM_STATE)
            print(type(x_train), type(y_train))
            logging.info(f"Exited the spliting_data method")
            return x_train,x_test,y_train,y_test
        except Exception as e:
            raise CustomException(e, sys) from e
    
    def tokenizing(self, x_train):
        try:
            logging.info(f"Entered tokenization method and applying tokenization on the data")
            tokenizer = Tokenizer(num_words= self.model_trainer_config.MAX_WORDS)
            tokenizer.fit_on_texts(x_train)
            sequences = tokenizer.texts_to_sequences(x_train)
            logging.info(f"Converting test into sequences: {sequences}")
            sequences_matrix = pad_sequences(sequences, maxlen= self.model_trainer_config.MAX_LEN)
            logging.info(f"Exited the tokenization method")
            return sequences_matrix, tokenizer
        except Exception as e:
            raise CustomException(e, sys) from e
    
    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            logging.info(f"Entered initiate_model_trainer method")
            x_train, x_test, y_train, y_test = self.spliting_data(csv_path= self.data_transformation_artifact.transformed_data_path)
            model_architecture = ModelArchitecture()
            model = model_architecture.get_model()
            sequences_matrix, tokenizer = self.tokenizing(x_train)

            model.fit(sequences_matrix, y_train, batch_size= self.model_trainer_config.BATCH_SIZE,
                      epochs= self.model_trainer_config.EPOCH, 
                      validation_split= self.model_trainer_config.VALIDATION_SPLIT)
            logging.info(f"Model training finished and saving the model or tokenization")
            with open('tokenizer.pickle', 'w') as handle:
                pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
            os.makedirs(self.model_trainer_config.TRAINED_MODEL_DIR, exist_ok=True)
            model.save(self.model_trainer_config.TRAINED_MODEL_PATH)
            x_test.to_csv(self.model_trainer_config.X_TEST_DATA_PATH)
            y_test.to_csv(self.model_trainer_config.Y_TEST_DATA_PATH)
            x_train.to_csv(self.model_trainer_config.X_TRAIN_DATA_PATH)

            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_path= self.model_trainer_config.TRAINED_MODEL_PATH,
                x_test_path = self.model_trainer_config.X_TEST_DATA_PATH,
                y_test_path= self.model_trainer_config.Y_TEST_DATA_PATH
            )
            logging.info(f"Exited the initiate_model_trainer method")
            return model_trainer_artifact
        except Exception as e:
            raise CustomException(e, sys) from e

            
            
        


