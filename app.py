import sys
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from fastapi.responses import Response, RedirectResponse

from src.NLP.utils.logger import logging
from src.NLP.utils.exception import CustomException
from src.NLP.pipeline.train_pipeline import TrainPipeline
from src.NLP.entity.config_entity import PredictionPipelineConfig
from src.NLP.pipeline.prediction_pipeline import PredictionPipeline
from src.NLP.constants.train_pipeline import APP_HOST, APP_PORT

app = FastAPI()

class PredictRequest(BaseModel):
    texts: List[str]

config = PredictionPipelineConfig()
prediction_pipeline = PredictionPipeline(config=config)

@app.get("/", tags=["authentication"])
async def index():
    try:
        logging.info("Model and Tokenizer loaded successfully as startup_event")
        return RedirectResponse(url="/docs")
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
        return {"predictions": predictions}
    except Exception as e:
        raise HTTPException(status_code=500, detail="prediction failed")

if __name__ == "__main__":
    uvicorn.run(app, host=APP_HOST, port=APP_PORT)
