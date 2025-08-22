# Dockerfile (at repo root)
FROM python:3.10-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Upgrade pip (safer/faster builds)
RUN python -m pip install --upgrade pip

# Install deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App code + artifacts
COPY server ./server
COPY preprocessor_cls.pkl preprocessor_reg.pkl ./artifacts/

# ⚠️ Use EXACT repo paths (Linux paths, relative to repo root)
COPY artificats/paysim_classify_model_2025-08-14_15-47-07/model_state.pt ./artifacts/cls_model_state.pt
COPY regress_artificats/paysim_regress_model_2025-08-18_13-14-55/model_state.pt ./artifacts/reg_model_state.pt

EXPOSE 8000
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
