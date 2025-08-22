import os, logging
import numpy as np
from fastapi import FastAPI, HTTPException
from typing import Dict, Any
import pandas as pd
import joblib

from server.schemas import (
    PredictRequest, PredictClassifyResponse, PredictRegressResponse, PredictBothResponse
)
from server.model_def import load_cls_state_dict, load_reg_state_dict, forward_classify, forward_regress, warmup

MODEL_CKPT_CLS = os.getenv("MODEL_CKPT_CLS", "model_cls.pt")
MODEL_CKPT_REG = os.getenv("MODEL_CKPT_REG", "model_reg.pt")
PREPROC_CLS    = os.getenv("PREPROC_CLS",    "preprocessor_cls.pkl")
PREPROC_REG    = os.getenv("PREPROC_REG",    "preprocessor_reg.pkl")
MODEL_VER_CLS  = os.getenv("MODEL_VER_CLS",  "v1")
MODEL_VER_REG  = os.getenv("MODEL_VER_REG",  "v1")

BASE_COLS = [
    "step", "type", "amount",
    "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest",
    "isFlaggedFraud"
]

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("paysim-api")

app = FastAPI(title="Paysim Dual Model API", version="1.0")
MODEL_CLS = MODEL_REG = None
PROC_CLS = PROC_REG = None

def row_to_df(features: Dict[str, Any]) -> pd.DataFrame:
    row = {c: features.get(c, np.nan) for c in BASE_COLS}
    return pd.DataFrame([row])

def transform_with(proc, df_row: pd.DataFrame) -> np.ndarray:
    xt = proc.transform(df_row)
    xt = np.asarray(xt, dtype=np.float32)
    if xt.ndim != 2 or xt.shape[0] != 1:
        raise ValueError(f"Preprocessor produced shape {xt.shape}, expected (1, F)")
    return xt

@app.on_event("startup")
def _startup():
    global MODEL_CLS, MODEL_REG, PROC_CLS, PROC_REG

    logger.info("Loading preprocessors")
    PROC_CLS = joblib.load(PREPROC_CLS)
    PROC_REG = joblib.load(PREPROC_REG)

    # Build one sample row to get feature dims from each preprocessor
    sample = {
        "step": 1, "type": "CASH_OUT", "amount": 0.0,
        "oldbalanceOrg": 0.0, "newbalanceOrig": 0.0,
        "oldbalanceDest": 0.0, "newbalanceDest": 0.0,
        "isFlaggedFraud": 0
    }
    df1   = row_to_df(sample)
    x_cls = transform_with(PROC_CLS, df1)
    x_reg = transform_with(PROC_REG, df1)

    n_feat_cls = x_cls.shape[1]
    n_feat_reg = x_reg.shape[1]

    logger.info("Loading models....")
    MODEL_CLS = load_cls_state_dict(MODEL_CKPT_CLS, n_feat_cls)  # <-- state_dict loader
    MODEL_REG = load_reg_state_dict(MODEL_CKPT_REG, n_feat_reg)

    try:
        _ = forward_classify(MODEL_CLS, x_cls)   # uses classifier's exact feature shape
        _ = forward_regress(MODEL_REG, x_reg)    # uses regressor's exact feature shape
        logger.info("Warmup ok.")
    except Exception as e:
        logger.warning(f"Warmup skipped: {e}")

@app.get('/health')
def health():
    ok = all([MODEL_CLS, MODEL_REG, PROC_CLS, PROC_REG])
    return {"ok": bool(ok), "cls_version": MODEL_VER_CLS, "reg_version": MODEL_VER_REG}

@app.post("/predict/classify", response_model=PredictClassifyResponse)
def predict_classify(req: PredictRequest):
    try:
        df_row = row_to_df(req.features)
        x = transform_with(PROC_CLS, df_row)
        prob = forward_classify(MODEL_CLS, x)
        return {"fraud_prob": prob, "model_version": MODEL_VER_CLS}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Classification error: {e}")

@app.post("/predict/regress", response_model=PredictRegressResponse)
def predict_regress(req: PredictRequest):
    try:
        df_row = row_to_df(req.features)
        x = transform_with(PROC_REG, df_row)
        y_log, y_real = forward_regress(MODEL_REG, x)
        return {"amount_log": y_log, "amount": y_real, "model_version": MODEL_VER_REG}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"regression error: {e}")

@app.post("/predict", response_model=PredictBothResponse)
def predict_both(req: PredictRequest):
    try:
        df_row = row_to_df(req.features)
        x_cls = transform_with(PROC_CLS, df_row)
        x_reg = transform_with(PROC_REG, df_row)
        prob = forward_classify(MODEL_CLS, x_cls)
        y_log, y_real = forward_regress(MODEL_REG, x_reg)
        return {
            "fraud_prob": prob,
            "amount_log": y_log,
            "amount": y_real,
            "versions": {"cls": MODEL_VER_CLS, "reg": MODEL_VER_REG}
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Predict error: {e}")
