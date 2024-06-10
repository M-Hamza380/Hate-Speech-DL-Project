'''
    Creating model architecture here
'''
import sys
from keras.models import Sequential
from keras.optimizers import RMSprop
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import LSTM, Activation, Dense, Embedding, SpatialDropout1D

from src.NLP.utils.exception import CustomException
from src.NLP.utils.logger import logging
from src.NLP.constants.train_pipeline import *

class ModelArchitecture:
    def __init__(self):
        pass

    def get_model(self):
        try:
            logging.info(f"Entered model architecture method")
            model = Sequential()
            model.add(Embedding(MAX_WORDS, 100, input_length=MAX_LEN))
            model.add(SpatialDropout1D(0.2))
            model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2, return_sequences=True))
            model.add(LSTM(60, dropout=0.2, recurrent_dropout=0.2, return_sequences=True))
            model.add(SpatialDropout1D(0.1))
            model.add(LSTM(20, dropout=0.1, recurrent_dropout=0.1))
            model.add(Dense(1, activation= ACTIVATION))
            model.compile(loss= LOSS, optimizer=RMSprop(), metrics= METRICS)

            model.summary()
            logging.info(f"Exited model architecture method")
            return model
        except Exception as e:
            raise CustomException(e, sys) from e

