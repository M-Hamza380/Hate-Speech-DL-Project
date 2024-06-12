import os, sys, pickle
from tensorflow import keras
from keras.utils import pad_sequences

from src.NLP.utils.logger import logging
from src.NLP.utils.exception import CustomException
from src.NLP.entity.config_entity import PredictionPipelineConfig


class PredictionPipeline:
    def __init__(self, config: PredictionPipelineConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
    
    def load_model(self):
        try:
            logging.info("Entered the load_model method of PredictionPipeline class")
            model_path = os.path.join(self.config.BEST_MODEL_DIR, self.config.MODEL_NAME)
            self.model = keras.models.load_model(model_path)
            logging.info(f"Model loaded from {model_path}")
            logging.info("Exited the load_model method of PredictionPipeline class")
        except Exception as e:
            raise CustomException(e, sys) from e
    
    def load_tokenizer(self):
        try:
            logging.info("Entered the load_tokenizer method of PredictionPipeline class")
            with open(self.config.TOKENIZER_PATH, 'rb') as handle:
                tokenizer = pickle.load(handle)
            logging.info("Tokenizer loaded successfully")
            
            logging.info("Exited the load_tokenizer method of PredictionPipeline class")
        except Exception as e:
            raise CustomException(e, sys) from e
    
    def preprocess_input(self, texts):
        try:
            logging.info("Entered the preprocess_input method of PredictionPipeline class")
            sequences = self.tokenizer.texts_to_sequences(texts)
            sequences_matrix = pad_sequences(sequences, maxlen= self.config.MAX_LEN)
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
            if pred[0] >= 0.5:
                return "Hate & Abusive"
            else:
                return "Normal"
        except Exception as e:
            raise CustomException(e, sys) from e
    
    def run_prediction_pipeline(self, texts):
        try:
            logging.info("Entered the run_prediction_pipeline method of PredictionPipeline class")
            self.load_model()
            self.load_tokenizer()
            result = self.prediction(texts)
            logging.info("Exited the run_prediction_pipeline method of PredictionPipeline class")
            return result
        except Exception as e:
            raise CustomException(e, sys) from e
        

