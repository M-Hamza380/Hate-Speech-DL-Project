import os, sys, shutil

from src.NLP.utils.logger import logging
from src.NLP.utils.exception import CustomException

from src.NLP.entity.config_entity import ModelPusherConfig
from src.NLP.entity.artifact_entity import ModelPusherArtifact

class ModelPusher:
    def __init__(self, model_pusher_config: ModelPusherConfig):
        self.model_pusher_config = model_pusher_config
    
    def initiate_model_pusher(self, traned_model_path: str) -> ModelPusherArtifact:
        try:
            logging.info("Entered the initiate_model_pusher method of ModelPusher class")
            best_model_dir = self.model_pusher_config.BEST_MODEL_DIR
            model_name = self.model_pusher_config.MODEL_NAME

            os.makedirs(best_model_dir, exist_ok=True)
            best_model_path = os.path.join(best_model_dir, model_name)

            shutil.copy(traned_model_path, best_model_path)
            is_model_pushed = os.path.isfile(best_model_path)
            logging.info(f"Model pushed: {is_model_pushed}, Best model path: {best_model_path}")
            model_pusher_artifact = ModelPusherArtifact(
                is_model_pushed = is_model_pushed,
                best_model_path = best_model_path
            )
            logging.info("Exited the initiate_model_pusher method of ModelPusher class")
            return model_pusher_artifact
        except Exception as e:
            raise CustomException(e, sys) from e
