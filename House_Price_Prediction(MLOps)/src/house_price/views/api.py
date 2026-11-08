from fastapi import FastAPI, APIRouter, HTTPException
from functools import lru_cache
import os
from house_price.config import Settings
from house_price.config import settings as default_settings
from house_price.models.schemas import InputDataSchema, PredictionResult
from house_price.controllers.model_controller import ModelController

base_router=APIRouter(
    prefix="/api/v1",
    tags=["base"]
)

data_router=APIRouter(
    prefix="/api/v1/data",
    tags=["data"]
)
@lru_cache
def get_controller() -> ModelController:
    """One controller for the whole app, so artifacts load only once."""
    return ModelController(default_settings)


@base_router.get("/health")
def welcome():
    
    return {"Welcome to the House Price Prediction API!"} 

@data_router.post("/predict", response_model=PredictionResult)
def predict_price(input_data: InputDataSchema):
    """Predict the house price (in lacs) from the input data."""
    try:
        return get_controller().predict_price(input_data)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))