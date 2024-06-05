from dataclasses import dataclass

@dataclass
class DataIngestionArtifact:
    data_zip_file_path: str
    feature_store_path: str
    imbalanced_data_path: str
    raw_data_path: str

@dataclass
class DataTransformationArtifact:
    transformed_data_path: str