"""
Data Ingestion related constant start with DATA_INGESTION VAR NAME
"""

# Common constants
ARTIFACTS_DIR: str = "artifacts"


# Data ingestion constants
DATA_INGESTION_DIR_NAME: str = 'Data_Ingestion_Artifacts'
DATA_INGESTION_FEATURE_STORE_DIR: str = 'feature_store'
IMBALANCED_DATA: str = "imbalanced_data"
RAW_DATA: str = "raw_data"

DATA_DOWNLOAD_URL: str = "https://github.com/M-Hamza380/Hate-Speech-DL-Project/raw/main/Dataset/Hate-Speech.zip"


# Data validation constants
DATA_VALIDATION_ARTIFACTS_DIR: str = "Data_Validation_Artifacts"
DATA_VALIDATION_REPORT_DIR: str = "report"
DATA_VALIDATION_REPORT_FILE_NAME: str = "report.yaml"

IMBALANCED_DATA_SCHEMA = {
    'id': 'int64',
    'label': 'int64',
    'tweet': 'object'
}

RAW_DATA_SCHEMA = {
    'Unnamed: 0': 'int64',
    'count': 'int64',
    'hate_speech': 'int64',
    'offensive_language': 'int64',
    'neither': 'int64',
    'class': 'int64',
    'tweet': 'object'
}


# Data transformation constants
DATA_TRANSFORMATION_ARTIFACTS_DIR: str = "Data_Transformation_Artifacts"
TRANSFORMED_FILE_NAME = 'final.csv'
DATA_DIR = "data"
ID = "id"
AXIS = 1
INPLACE = True
IGNORE_INDEX = True
DROP_COULMNS = ['Unnamed: 0', 'count', 'hate_speech', 'offensive_language', 'neither']
CLASS = 'class'
LABEL = 'label'
TWEET = 'tweet'


# Model training constants
MODEL_TRAINER_ARTIFACTS_DIR: str = "Model_Trainer_Artifacts"
TRAINED_MODEL_DIR: str = "trained_model"
TRAINED_MODEL_NAME: str = "model.h5"
X_TEST_FILE_NAME: str = "x_test.csv"
Y_TEST_FILE_NAME: str = "y_test.csv"
X_TRAIN_FILE_NAME: str = "x_train.csv"

TEST_SIZE = 0.3
RANDOM_STATE = 42
EPOCH = 1
BATCH_SIZE = 128
VALIDATION_SPLIT = 0.2

# Model architecture constants
MAX_WORDS = 50000
MAX_LEN = 300
LOSS = "binary_crossentropy"
METRICS = ['accuracy']
ACTIVATION = "sigmoid"
