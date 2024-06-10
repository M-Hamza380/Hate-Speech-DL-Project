from dataclasses import dataclass

@dataclass(frozen=True)
class DataIngestionArtifact:
    data_zip_file_path: str
    feature_store_path: str
    imbalanced_data_path: str
    raw_data_path: str

@dataclass(frozen=True)
class DataValidationArtifact:
    report_file_path: str


@dataclass(frozen=True)
class DataTransformationArtifact:
    transformed_data_path: str


@dataclass(frozen=True)
class ModelTrainerArtifact:
    trained_model_path: str
    x_test_path: list
    y_test_path: list