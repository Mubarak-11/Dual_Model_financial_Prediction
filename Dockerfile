# ---- Base ----
FROM python:3.10-slim

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# ---- Python deps (kept cacheable) ----
COPY requirements.txt .
RUN pip install -r requirements.txt

# ---- App code ----
# Package + helpers used by server/model_def.py
COPY server/ server/
COPY preprocessing.py torch_classify.py torch_regress.py ./

# ---- Artifacts (bake into image under /models) ----
# Preprocessors at repo root:
COPY preprocessor_cls.pkl preprocessor_reg.pkl /models/

# Checkpoints from the exact folders you provided:
#  - C:\...\torch_proj#1_classify\artificats\paysim_classify_model_2025-08-14_15-49-35\model_state.pt
#  - C:\...\torch_proj#1_classify\regress_artificats\paysim_regress_model_2025-08-18_13-14-55\model_state.pt
COPY artificats/paysim_classify_model_2025-08-14_15-49-35/model_state.pt /models/model_cls.pt
COPY regress_artificats/paysim_regress_model_2025-08-18_13-14-55/model_state.pt /models/model_reg.pt

# Provide stable defaults your code can read (override in ACA if you ever need)
ENV MODEL_CKPT_CLS=/models/model_cls.pt \
    MODEL_CKPT_REG=/models/model_reg.pt \
    PREPROC_CLS=/models/preprocessor_cls.pkl \
    PREPROC_REG=/models/preprocessor_reg.pkl \
    MODEL_VER_CLS=v1 \
    MODEL_VER_REG=v1

# ---- Network / startup ----
EXPOSE 8000
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]

#to prevenet GUI backend issues
ENV MPLBACKEND=Agg MPLCONFIGDIR=/tmp/matplotlib

