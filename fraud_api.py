from fastapi import FastAPI
from pydantic import BaseModel
import xgboost as xgb
import numpy as np
import joblib

# 🎯 Load model once during startup
model = joblib.load("xgboost_fraud_model.pkl")

# 🚀 Start FastAPI app
app = FastAPI(title="Fraud Detection API", version="0.1.0")


# 🧾 Define request schema
class Transaction(BaseModel):
    TX_AMOUNT: float
    is_high_amount: int
    terminal_fraud_history_28d: int
    is_unusual_for_customer: int


# ✅ Root route
@app.get("/")
def read_root():
    return {"message": "Welcome to the Fraud Detection API!"}


# 🔍 Prediction route
@app.post("/predict/")
def predict_fraud(tx: Transaction):
    # Turn the input into a 2D array
    input_data = np.array(
        [
            [
                tx.TX_AMOUNT,
                tx.is_high_amount,
                tx.terminal_fraud_history_28d,
                tx.is_unusual_for_customer,
            ]
        ]
    )

    # Predict probability and label
    probability = float(model.predict_proba(input_data)[0][1])
    prediction = int(model.predict(input_data)[0])

    return {"fraud_probability": round(probability, 4), "prediction": prediction}
