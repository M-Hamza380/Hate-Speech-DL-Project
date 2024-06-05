"""
Data Ingestion related constant start with DATA_INGESTION VAR NAME
"""
import os

# Common constants
ARTIFACTS_DIR: str = "artifacts"


# Data ingestion constants
DATA_INGESTION_DIR_NAME: str = 'Data_Ingestion_Artifacts'

DATA_INGESTION_FEATURE_STORE_DIR: str = 'feature_store'
IMBALANCED_DATA = "imbalanced_data"
RAW_DATA = "raw_data"

DATA_DOWNLOAD_URL: str = "https://github.com/M-Hamza380/Hate-Speech-DL-Project/raw/main/Dataset/Hate-Speech.zip"

# Data transformation constants
DATA_TRANSFORMATION_ARTIFACTS_DIR: str = "Data_Transformation_Artifacts"
TRANSFORMED_FILE_NAME = 'final.csv'
DATA_DIR = "data"
ID = "id"
AXIS = 1
INPLACE = True
DROP_COULMNS = ['Unnamed: 0', 'count', 'hate_speech', 'offensive_language', 'neither']
CLASS = 'class'
LABEL = 'label'
TWEET = 'tweet'



