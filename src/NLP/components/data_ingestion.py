import os, sys, shutil, zipfile
from six.moves import urllib

from src.NLP.utils.exception import CustomException
from src.NLP.utils.logger import logging
from src.NLP.entity.config_entity import DataIngestionConfig
from src.NLP.entity.artifact_entity import DataIngestionArtifact


class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig = DataIngestionConfig()):
        try:
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise CustomException(e, sys)
        
    
    def download_data(self) -> str:
        try:
            dataset_url = self.data_ingestion_config.data_download_url
            zip_download_dir = self.data_ingestion_config.data_ingestion_dir
            os.makedirs(zip_download_dir, exist_ok=True)
            data_file_name = os.path.basename(dataset_url)
            zip_file_path = os.path.join(zip_download_dir, data_file_name)
            logging.info(f"Downloading data from {dataset_url} into file {zip_file_path}")
            urllib.request.urlretrieve(dataset_url, zip_file_path)
            logging.info(f"Downloaded the dataset")
            return zip_file_path
        except Exception as e:
            raise CustomException(e, sys)
    
    def extract_zip_file(self, zip_file_path: str) -> str:
        try:
            feature_store_path = self.data_ingestion_config.feature_store_file_path
            os.makedirs(feature_store_path, exist_ok= True)
            with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                zip_ref.extractall(feature_store_path)

            logging.info(f"Extracting zip file: {zip_file_path} into dir: {feature_store_path}")
            return feature_store_path
        except Exception as e:
            raise CustomException(e, sys)
    
    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        logging.info(f"Entered initiate_data_ingestion method of Data_Ingestion class")

        try:
            zip_file_path = self.download_data()
            feature_store_path = self.extract_zip_file(zip_file_path)

            # Ensure the imbalanced data directory exists or create it
            imbalanced_data_dir = self.data_ingestion_config.imbalanced_data_file_path
            os.makedirs(imbalanced_data_dir, exist_ok=True)

            # Ensure the raw data directory exists or create it
            raw_data_dir = self.data_ingestion_config.raw_data_file_path
            os.makedirs(raw_data_dir, exist_ok=True)

            # Move csv files to their respective folders
            for root, dirs, files in os.walk(feature_store_path):
                for file in files:
                    if file.endswith(".csv"):
                        source_file_path = os.path.join(root, file)
                        if 'imbalanced_data' in file.lower():
                            destination_dir = self.data_ingestion_config.imbalanced_data_file_path
                        elif 'raw_data' in file.lower():
                            destination_dir = self.data_ingestion_config.raw_data_file_path
                        else:
                            logging.warning(f"File {file} does not match any condition")
                            continue
                        
                        destination_path = os.path.join(destination_dir, file)
                        
                        # Move the file, overwriting if it exists
                        shutil.move(source_file_path, destination_path)
                        logging.info(f"Moved file {file} to {destination_path}")

            data_ingestion_artifact = DataIngestionArtifact(
                data_zip_file_path = zip_file_path,
                feature_store_path = feature_store_path,
                imbalanced_data_path = self.data_ingestion_config.imbalanced_data_file_path,
                raw_data_path = self.data_ingestion_config.raw_data_file_path,
            )

            logging.info(f"Data ingestion artifact: {data_ingestion_artifact}")

            return data_ingestion_artifact
        except Exception as e:
            raise CustomException(e, sys)
        



