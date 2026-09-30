from fastapi import FastAPI, APIRouter, Depends
from prometheus_fastapi_instrumentator import Instrumentator
import os
from house_price.views import api
app = FastAPI()
app.include_router(api.base_router)
app.include_router(api.data_router) 

# Automatic metrics: request count, latency, status codes per endpoint
Instrumentator().instrument(app).expose(app)