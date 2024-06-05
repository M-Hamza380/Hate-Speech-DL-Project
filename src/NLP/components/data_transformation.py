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
            imbalanced_data = pd.read_csv(self.data_transformation_config.IMBALANCED_DATA)
            imbalanced_data.drop(self.data_transformation_config.ID, axis=self.data_transformation_config.AXIS, inplace= self.data_transformation_config.INPLACE)
            logging.info(f"Exited the imbalanced data_cleaning function and returned imbalanced data {imbalanced_data}")
            return imbalanced_data
        except Exception as e:
            raise CustomException(e, sys) from e
    

    def raw_data_cleaning(self):
        try:
            logging.info("Entered into the raw_data_cleaning function")
            raw_data = pd.read_csv(self.data_transformation_config.RAW_DATA)
            raw_data.drop(columns= self.data_transformation_config.DROP_COLUMNS, axis= self.data_transformation_config.AXIS, inplace= self.data_transformation_config.INPLACE)

            raw_data[raw_data[self.data_transformation_config.CLASS] == 0][self.data_transformation_config.CLASS] = 1

            # Replace the value of 0 to 1
            raw_data[self.data_transformation_config.CLASS].replace({0:1}, inplace=True)

            # Replace the value of 2 to 0
            raw_data[self.data_transformation_config.CLASS].replace({2:0}, inplace=True)

            # Change the name of the class to label
            raw_data.rename(columns={self.data_transformation_config.CLASS: self.data_transformation_config.LABEL}, inplace=True)

            logging.info(f'Exited the raw_data_cleaning function and returned the raw data {raw_data}')

            return raw_data
        except Exception as e:
            raise CustomException(e, sys) from e
    

    def concat_dataframe(self):
        try:
            logging.info("Entered into the concat_dataframe function")
            # Concatenate both the data into a single data frame
            frame = [self.raw_data_cleaning(), self.imbalanced_data_cleaning()]
            df = pd.concat(frame)
            logging.info(f"Returned the concatinated dataframe {df}")
            return df
        except Exception as e:
            raise CustomException(e, sys) from e
    

    def concate_data_cleaning(self, words):
        try:
            logging.info("Entered into the concat_data_cleaning function")
            # Apply stemming and stopwords on the data
            stemming = nltk.SnowballStemmer('english')
            stopword = set(stopwords.words("english"))
            words = str(words).lower()
            words = re.sub(r'[^a-zA-Z0-9\s]', '', words)
            words = re.sub('\[.*?]', "", words)
            words = re.sub('https?://\S+|www\.\S+', "", words)
            words = re.sub('<.*?>+', '', words)
            words = re.sub('[%s]' % re.escape(string.punctuation), '', words)
            words = re.sub('\n', '', words)
            words = re.sub('\w*\d\w*', '', words)
            words = [word for word in words.split(" ") if word not in stopword]
            words = [stemming.stem(word) for word in words]
            words = " ".join(words)
            logging.info("Exited the concat_data_cleaning function")
            return words
        except Exception as e:
            raise CustomException(e, sys) from e
    
    
    def initiate_data_transformation(self) -> DataTransformationArtifact:
        try:
            logging.info('Entered into the initiate_data_transformation function')
            self.imbalanced_data_cleaning()
            self.raw_data_cleaning()
            df = self.concat_dataframe()
            df[self.data_transformation_config.TWEET] = df[self.data_transformation_config.TWEET].apply(self.concate_data_cleaning)

            os.makedirs(self.data_transformation_config.DATA_TRANSFORMATION_ARTIFACTS_DIR, exist_ok=True)
            df.to_csv(self.data_transformation_config.TRANSFORMED_FILE_NAME, index=False, header=True)

            data_transformation_artifact = DataTransformationArtifact(
                transformed_data_path = self.data_transformation_config.TRANSFORMED_FILE_NAME
            )
            logging.info("Exited the initiate_data_transformation function")
            return data_transformation_artifact

        except Exception as e:
            raise CustomException(e, sys) from e
