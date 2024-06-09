import os, sys, yaml
import pandas as pd

from src.NLP.utils.exception import CustomException
from src.NLP.utils.logger import logging
from src.NLP.entity.config_entity import DataValidationConfig
from src.NLP.entity.artifact_entity import DataValidationArtifact

class DataValidation:
    def __init__(self, data_validation_config: DataValidationConfig):
        try:
            self.data_validation_config = data_validation_config
        except Exception as e:
            raise CustomException(e, sys) from e
    
    def validate_data(self, csv_file_path: str, schema: dict) -> dict:
        logging.info(f"Perform data validation on dataset")

        validation_result = {
            "file_path": csv_file_path,
            "is_valid": True,
            "error": []
        }

        try:
            # read csv file
            df = pd.read_csv(csv_file_path)

            # check for expected columns
            missing_columns = [col for col in schema.keys() if col not in df.columns]
            if missing_columns:
                validation_result['is_valid'] = False
                validation_result['error'].append(f"Missing columns: {missing_columns}")
            
            # check for correct data types
            for column, expected_dtype in schema.items():
                if column in df.columns:
                    actual_dtype = str(df[column].dtype)
                    if actual_dtype != expected_dtype:
                        validation_result['is_valid'] = False
                        validation_result['error'].append(f"Column '{column}' expected_dtype '{expected_dtype}' but got '{actual_dtype}")
            
            # check for missing values
            missing_values = df.isnull().sum()
            column_with_missing_values = missing_values[missing_values > 0].to_dict()
            if column_with_missing_values:
                validation_result['is_valid'] = False
                validation_result['error'].append(f"Columns with missing values: {column_with_missing_values}")


        except Exception as e:
            validation_result['is_valid'] = False
            validation_result['error'].append(f"Error reading csv file: {str(e)}")
            raise CustomException(e, sys) from e
        
        return validation_result
    
    def generate_report(self, validation_results: list):
        try:
            report_dir = os.path.dirname(self.data_validation_config.report_file_path)
            os.makedirs(report_dir, exist_ok=True)
            with open(self.data_validation_config.report_file_path, 'w') as report_file:
                yaml.dump(validation_results, report_file)
        except Exception as e:
            raise CustomException(e, sys) from e
    
    def initiate_data_validation(self) -> DataValidationArtifact:
        logging.info(f"Entered initiate_data_validation method of DataValidation class")

        try:
            validation_results = []
            for filename, schema in self.data_validation_config.schema_mapping.items():
                if filename == 'imbalanced_data.csv':
                    csv_file_path = os.path.join(self.data_validation_config.imbalanced_data_file_path, filename)
                elif filename == 'raw_data.csv':
                    csv_file_path = os.path.join(self.data_validation_config.raw_data_file_path, filename)
                
                if os.path.exists(csv_file_path):
                    validation_result = self.validate_data(csv_file_path, schema)
                    validation_results.append(validation_result)
                else:
                    validation_results.append({
                        'file_path': csv_file_path,
                        'is_valid': False,
                        'error': [f"File {filename} does not exist"]
                    })
            
            self.generate_report(validation_results)

            data_validation_artifact = DataValidationArtifact(
                report_file_path= self.data_validation_config.report_file_path
            )

            logging.info(f"Data validation artifact: {data_validation_artifact}")
            return data_validation_artifact
        except Exception as e:
            raise CustomException(e, sys) from e
        
    



