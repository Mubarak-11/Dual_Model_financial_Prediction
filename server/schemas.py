from typing import Dict, Union, Optional
from pydantic import BaseModel, Field

Number = Union[int, float]

class PredictRequest(BaseModel):
    #accept a flexible payload;
    #the preprocessing.py will handle key/values
    features: Dict[str, Union[Number, str]] = Field(..., description="Raw Feature Map")

class PredictClassifyResponse(BaseModel):
    fraud_prob: float
    model_version: Optional[str] = None

class PredictRegressResponse(BaseModel):
    amount_log: float
    amount: float
    model_version: Optional[str] = None

class PredictBothResponse(BaseModel):
    fraud_prob: float
    amount_log: float
    amount: float
    versions: Dict[str, str]
    