from pydantic import BaseModel, Field   

class InputDataSchema(BaseModel):
    POSTED_BY: str
    UNDER_CONSTRUCTION: int
    RERA: int
    BHK_NO: int
    BHK_OR_RK: str
    SQUARE_FT: float
    READY_TO_MOVE: int
    RESALE: int
    LONGITUDE: float
    LATITUDE: float



class PredictionResult(BaseModel):
    predicted_price: float
    