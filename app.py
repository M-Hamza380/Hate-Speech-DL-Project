import sys, uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from fastapi.responses import RedirectResponse, Response

from src.NLP.utils.logger import logging
from src.NLP.utils.exception import CustomException
from src.NLP.pipeline.train_pipeline import TrainPipeline
from src.NLP.entity.config_entity import PredictionPipelineConfig
from src.NLP.pipeline.prediction_pipeline import PredictionPipeline
from src.NLP.constants.train_pipeline import *

app = FastAPI()

class PredictRequest(BaseModel):
    texts: List[str] 


config = PredictionPipelineConfig()
prediction_pipeline = PredictionPipeline(config = config)


@app.on_event("startup")
async def startup_event():
    try:
        prediction_pipeline.load_model()
        prediction_pipeline.load_tokenizer()
        logging.info("Model and Tokenizer loaded successfully as startip_event")
    except Exception as e:
        raise CustomException(e, sys) from e


@app.get("/train")
async def training():
    try:
        train_pipeline = TrainPipeline()
        train_pipeline.run_pipeline()
        return Response("Training successful!")
    except Exception as e:
        raise CustomException(e, sys) from e

@app.post("/predict")
async def predict_route(request: PredictRequest):
    try:
        texts = request.texts
        predictions = prediction_pipeline.run_prediction_pipeline(texts)
        return {"predictions": predictions.to_list()}
    except Exception as e:
        raise HTTPException(status_code=500, detail="prediction failed")
    
if __name__ == "__main__":
    uvicorn.run(app, host=APP_HOST, port=APP_PORT)
