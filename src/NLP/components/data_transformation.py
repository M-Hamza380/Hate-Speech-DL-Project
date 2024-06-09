'''

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
            df = pd.concat(frame, ignore_index=True)
            logging.info(f"Returned the concatinated dataframe:\n {df}")
            return df
        except Exception as e:
            raise CustomException(e, sys) from e
        
    # Expand Contractions
    def expand_contractions(self, text, contractions_dict):
        try:
            contractions_re = re.compile('(%s)' % '|'.join(contractions_dict.keys()))
            def replace(self, match):
                return contractions_dict[match.group(0)]
            return contractions_re.sub(replace, text)
        except Exception as e:
            raise CustomException(e, sys)
    

    def concate_data_cleaning(self, words):
        try:
            logging.info("Entered into the concat_data_cleaning function")
            # Apply stemming and stopwords on the data
            stemming = nltk.SnowballStemmer('english')
            stopword = set(stopwords.words("english"))
            words = str(words).lower()
            words = self.expand_contractions(words, contractions_dict)  # Expand contractions
            words = re.sub(r'[^a-zA-Z0-9\s]', '', words)
            words = re.sub('\[.*?]', "", words)
            words = re.sub('https?://\S+|www\.\S+', "", words)
            words = re.sub('<.*?>+', '', words)
            words = re.sub('[%s]' % re.escape(string.punctuation), '', words)
            words = re.sub('\n', '', words)
            words = re.sub('\w*\d\w*', '', words)
            words = re.sub(r'\s+', ' ', words).strip()
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


contractions_dict = {
    "won't": "will not",
    "n't": " not",
    "'re": " are",
    "'s": " is",
    "'d": " would",
    "'ll": " will",
    "'t": " not",
    "'ve": " have",
    "'m": " am",
    "ain't": "am not",
    "aren't": "are not",
    "can't": "can not",
    "can't've": "can not have",
    "'cause": "because",
    "cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "couldn't've": "could not have",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hadn't've": "had not have",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'd've": "he would have",
    "he'll": "he will",
    "he'll've": "he will have",
    "he's": "he is",
    "how'd": "how did",
    "how'd'y": "how do you",
    "how'll": "how will",
    "how's": "how is",
    "i'd": "i would",
    "i'd've": "i would have",
    "i'll": "i will",
    "i'll've": "i will have",
    "i'm": "i am",
    "i've": "i have",
    "isn't": "is not",
    "it'd": "it would",
    "it'd've": "it would have",
    "it'll": "it will",
    "it'll've": "it will have",
    "it's": "it is",
    "let's": "let us",
    "ma'am": "madam",
    "mayn't": "may not",
    "might've": "might have",
    "mightn't": "might not",
    "mightn't've": "might not have",
    "must've": "must have",
    "mustn't": "must not",
    "mustn't've": "must not have",
    "needn't": "need not",
    "needn't've": "need not have",
    "o'clock": "of the clock",
    "oughtn't": "ought not",
    "oughtn't've": "ought not have",
    "shan't": "shall not",
    "sha'n't": "shall not",
    "shan't've": "shall not have",
    "she'd": "she would",
    "she'd've": "she would have",
    "she'll": "she will",
    "she'll've": "she will have",
    "she's": "she is",
    "should've": "should have",
    "shouldn't": "should not",
    "shouldn't've": "should not have",
    "so've": "so have",
    "so's": "so as",
    "that'd": "that would",
    "that'd've": "that would have",
    "that's": "that is",
    "there'd": "there would",
    "there'd've": "there would have",
    "there's": "there is",
    "they'd": "they would",
    "they'd've": "they would have",
    "they'll": "they will",
    "they'll've": "they will have",
    "they're": "they are",
    "they've": "they have",
    "to've": "to have",
    "wasn't": "was not",
    "we'd": "we would",
    "we'd've": "we would have",
    "we'll": "we will",
    "we'll've": "we will have",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what will",
    "what'll've": "what will have",
    "what're": "what are",
    "what's": "what is",
    "what've": "what have",
    "when's": "when is",
    "when've": "when have",
    "where'd": "where did",
    "where's": "where is",
    "where've": "where have",
    "who'll": "who will",
    "who'll've": "who will have",
    "who's": "who is",
    "who've": "who have",
    "why's": "why is",
    "why've": "why have",
    "will've": "will have",
    "won't": "will not",
    "won't've": "will not have",
    "would've": "would have",
    "wouldn't": "would not",
    "wouldn't've": "would not have",
    "y'all": "you all",
    "y'all'd": "you all would",
    "y'all'd've": "you all would have",
    "y'all're": "you all are",
    "y'all've": "you all have",
    "you'd": "you would",
    "you'd've": "you would have",
    "you'll": "you will",
    "you'll've": "you will have",
    "you're": "you are",
    "you've": "you have",
}
'''