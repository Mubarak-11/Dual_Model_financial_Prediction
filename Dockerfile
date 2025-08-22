# PyTorch CPU image with Python 3.10
FROM pytorch/pytorch:2.4.0-cpu

WORKDIR /app

# (optional) faster builds
RUN python -m pip install --upgrade pip

# Copy requirements first for cache
COPY requirements.txt .

# Install your other deps (torch is already present)
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code + artifacts
COPY server ./server
COPY preprocessor_cls.pkl preprocessor_reg.pkl ./artifacts/
# 👇 adjust these two lines to EXACT paths in your repo
COPY artificats/paysim_classify_model_2025-08-14_15-47-07/model_state.pt ./artifacts/cls_model_state.pt
COPY regress_artificats/paysim_regress_model_2025-08-18_13-14-55/model_state.pt ./artifacts/reg_model_state.pt

EXPOSE 8000
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
