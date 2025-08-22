#--- Base (use cpu-Only, small image) ---
FROM python:3.13.5

# Set working dir
WORKDIR /app

# Install system deps (needed for torch, uvicorn, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY server ./server
COPY preprocessor_cls.pkl preprocessor_reg.pkl ./artifacts/
COPY artificats/paysim_classify_model_2025-08-14_15-47-07/model_state.pt ./artifacts/cls_model_state.pt
COPY regress_artificats/paysim_regress_model_2025-08-18_13-14-55/model_state.pt ./artifacts/reg_model_state.pt

# Expose FastAPI port
EXPOSE 8000

# Run app
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
