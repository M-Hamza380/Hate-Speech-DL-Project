import os
import logging
from pathlib import Path

logging.basicConfig(level= logging.INFO, format= "[%(asctime)s] : %(name)s : %(message)s : ")

Project_Name = 'NLP'

list_of_files=[
    f"src/{Project_Name}/__init__.py",
    f"src/{Project_Name}/components/__init__.py",
    f"src/{Project_Name}/components/data_ingestion.py",
    f"src/{Project_Name}/components/data_transformation.py",
    f"src/{Project_Name}/components/model_trainer.py",
    f"src/{Project_Name}/components/model_evaluation.py",
    f"src/{Project_Name}/components/model_pusher.py",
    f"src/{Project_Name}/configuration/__init__.py",
    f"src/{Project_Name}/configuration/s3_operations.py",
    f"src/{Project_Name}/constants/__init__.py",
    f"src/{Project_Name}/entity/__init__.py",
    f"src/{Project_Name}/entity/config_entity.py",
    f"src/{Project_Name}/entity/artifact_entity.py",
    f"src/{Project_Name}/pipeline/__init__.py",
    f"src/{Project_Name}/pipeline/train_pipeline.py",
    f"src/{Project_Name}/pipeline/prediction_pipeline.py",
    f"src/{Project_Name}/dl/__init__.py",
    f"src/{Project_Name}/dl/model.py",
    f"src/{Project_Name}/utils/__init__.py",
    f"src/{Project_Name}/utils/exception.py",
    f"src/{Project_Name}/utils/logger.py",
    f"src/{Project_Name}/utils/common.py",
    "notebook/model_building.ipynb",
    "templates/index.html",
    "app.py",
    "demo.py",
    "requirements.txt",
    "Dockerfile",
    ".dockerignore",
    "setup.py"
]


for filepath in list_of_files:
    filepath = Path(filepath)

    filedir, filename = os.path.split(filepath)

    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directory: {filedir} for the file: {filename}")
    
    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, 'w') as f:
            pass
            logging.info(f"Creating empty file: {filepath}")
    else:
        logging.info(f"{filename} is already exists.")
