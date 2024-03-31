import os, sys, re, string
import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split

from src.NLP.utils.exception import CustomException
from src.NLP.utils.logger import logging
from src.NLP.entity.config_entity import DataTransformationConfig
from src.NLP.entity.artifact_entity import DataIngestionArtifact, DataTransformationArtifact


class DataTransformation:
    def __init__(self, data_transformation_config: DataTransformationConfig, data_ingestion_artifacts: DataIngestionArtifact):
        self.data_transformation_config = data_transformation_config
        self.data_ingestion_artifacts = data_ingestion_artifacts
    
    def imbalanced_data_cleaning(self):
        try:
            logging.info("Entered into the imbalanced_data_cleaning function")
            imbalanced_data = pd.read_csv(self.data_ingestion_artifacts.feature_store_path)
        except Exception as e:
            raise CustomException(e, sys) from e
