import os, sys, pickle
from tensorflow import keras
from keras.utils import pad_sequences

from src.NLP.utils.logger import logging
from src.NLP.utils.exception import CustomException
from src.NLP.entity.config_entity import PredictionPipelineConfig


class PredictionPipeline:
    def __init__(self, config: PredictionPipelineConfig):
        self.config = config
        self.model = self.load_model(self.config.MODEL_PATH)
        self.tokenizer = self.load_tokenizer(self.config.TOKENIZER_PATH)
    
    def load_model(self, model_path: str):
        try:
            logging.info("Entered the load_model method of PredictionPipeline class")
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"No file or directory found at {model_path}")
            
            model = keras.models.load_model(model_path)
            logging.info(f"Model loaded from {model_path}")
            logging.info("Exited the load_model method of PredictionPipeline class")
            return model
        except Exception as e:
            raise CustomException(e, sys) from e
    
    def load_tokenizer(self, tokenizer_path: str):
        try:
            logging.info("Entered the load_tokenizer method of PredictionPipeline class")
            with open(tokenizer_path, 'rb') as handle:
                tokenizer = pickle.load(handle)
            logging.info("Tokenizer loaded successfully")
            logging.info("Exited the load_tokenizer method of PredictionPipeline class")
            return tokenizer
        except Exception as e:
            raise CustomException(e, sys) from e
    
    def preprocess_input(self, texts):
        try:
            logging.info("Entered the preprocess_input method of PredictionPipeline class")
            sequences = self.tokenizer.texts_to_sequences(texts)
            sequences_matrix = pad_sequences(sequences, maxlen=self.config.MAX_LEN)
            logging.info("Exited the preprocess_input method of PredictionPipeline class")
            return sequences_matrix
        except Exception as e:
            raise CustomException(e, sys) from e
    
    def prediction(self, texts):
        try:
            logging.info("Entered the prediction method of PredictionPipeline class")
            processed_input = self.preprocess_input(texts)
            pred = self.model.predict(processed_input)
            logging.info("Exited the prediction method of PredictionPipeline class")
            return ["Hate & Abusive" if p >= 0.5 else "Normal" for p in pred]
        except Exception as e:
            raise CustomException(e, sys) from e
    
    def run_prediction_pipeline(self, texts):
        try:
            logging.info("Entered the run_prediction_pipeline method of PredictionPipeline class")
            result = self.prediction(texts)
            logging.info("Exited the run_prediction_pipeline method of PredictionPipeline class")
            return result
        except Exception as e:
            raise CustomException(e, sys) from e
