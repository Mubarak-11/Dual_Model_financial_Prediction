import os
import torch
import numpy as np
from typing import Tuple

Device = "cuda" if torch.cuda.is_available() else "cpu"

#import classese from training script
from torch_classify import classmodel
from torch_regress import regressmodel

def load_cls_state_dict(weights_path: str, in_features: int)-> torch.nn.Module:
    model = classmodel(in_features)    
    state = torch.load(weights_path, map_location=Device, weights_only=True)

    model.load_state_dict(state)
    model.eval() #enter evaluate 
    return model.to(Device)

def load_reg_state_dict(weights_path: str, in_features: int)-> torch.nn.Module:
    model = regressmodel(in_features)    
    state = torch.load(weights_path, map_location=Device, weights_only=True)

    model.load_state_dict(state)
    model.eval() #enter evaluate 
    return model.to(Device)

@torch.inference_mode()
def forward_classify(model: torch.nn.Module, x:np.ndarray) ->float:
    """x: shape(1,F) numpy -> returns prob in [0,1]"""

    t = torch.tensor(x, dtype=torch.float32, device=Device)
    logits = model(t)   #shape (1,1) or just (1,)

    logits = logits.view(-1) #will adjust for size issues
    prob = torch.sigmoid(logits)[0].item()

    return float(prob)

@torch.inference_mode()
def forward_regress(model: torch.nn.Module, x:np.ndarray) -> Tuple[float, float]:
    """ Returns (log1p_pred, real_pred) using expm1 for inverse transform"""

    t = torch.tensor(x, dtype=torch.float32, device=Device)
    y_log = model(t).view(-1)[0].item()
    y_real = float(np.expm1(y_log)) #expm1 is inverse of log1p

    return float(y_log), y_real

def warmup(model_cls: torch.nn.Module, model_reg: torch.nn.Module, n_features: int)-> None:
    """ Run a dummy forward to catch a shape/device mismatch/issues on startup"""

    fake = torch.zeros((1, n_features), dtype=torch.float32, device=Device)
    with torch.inference_mode():
        _ = torch.sigmoid(model_cls(fake).view(-1))
        _ = model_reg(fake).view(-1)
        
