from src.NLP.utils.logger import logging
from src.NLP.utils.exception import CustomException
import sys

logging.info('Custom Exception successfully run')

try:
    a = 7/ '0'
except Exception as e:
    raise CustomException(e, sys)