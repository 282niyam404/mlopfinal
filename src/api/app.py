from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib
import torch
import torch.nn as nn
import os

app = FastAPI(title="HMM + LSTM API")

# =========================
# Config
# =========================
MODEL_DIR = os.getenv("MODEL_DIR", "src/models")

# =========================
# Globals
# =========================
meta = None
models = {}

# =========================
# Model Class (same as training)
# =========================
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=2, batch_first=True)
        self.fc = nn.Linear(hidden_size, 3)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

# =========================
# Load Models
# =========================
@app.on_event("startup")
async def load_model():
    global meta, models

    try:
        meta = joblib.load(os.path.join(MODEL_DIR, "meta.pkl"))

        input_size = meta["input_size"]

        # load LSTM models
        for state in [0, 1]:
            path = os.path.join(MODEL_DIR, f"lstm_state_{state}.pt")

            if os.path.exists(path):
                m = LSTMModel(input_size)
                m.load_state_dict(torch.load(path, map_location="cpu"))
                m.eval()
                models[state] = m

        print("✅ Models loaded successfully")

    except Exception as e:
        print(f"❌ Error loading model: {e}")

# =========================
# Request Schema
# =========================
class PredictionRequest(BaseModel):
    sequence: list  # shape: [time_steps, num_features]

# =========================
# Helper
# =========================
def softmax(x):
    e = np.exp(x)
    return e / np.sum(e)

# =========================
# Routes
# =========================
@app.get("/")
async def root():
    return {"message": "HMM + LSTM API running"}

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "meta_loaded": meta is not None,
        "num_models": len(models)
    }

# =========================
# Prediction
# =========================
@app.post("/predict")
async def predict(request: PredictionRequest):

    if meta is None or len(models) == 0:
        return {"error": "Model not loaded properly"}

    scaler = meta["scaler"]
    hmm = meta["hmm"]
    time_steps = meta["time_steps"]

    try:
        # input → numpy
        seq = np.array(request.sequence)

        # scale
        seq_scaled = scaler.transform(seq)

        # reshape for LSTM
        seq_tensor = torch.tensor(seq_scaled, dtype=torch.float32).unsqueeze(0)

        # HMM probs (use last log_ret column assumption)
        log_ret = seq[:, -1].reshape(-1, 1)
        probs = hmm.predict_proba(log_ret)[-1]

        preds = []

        for s, model in models.items():
            pred = model(seq_tensor).detach().numpy()[0]
            preds.append(pred)

        preds = np.array(preds)

        final_pred = np.sum(
            probs[:len(models)].reshape(-1, 1) * preds,
            axis=0
        )

        weights = softmax(final_pred)

        return {
            "weights": weights.tolist()
        }

    except Exception as e:
        return {"error": str(e)}

# =========================
# Run
# =========================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)