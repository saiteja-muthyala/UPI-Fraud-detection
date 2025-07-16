import pandas as pd
import joblib

# Load trained model
model = joblib.load("xgboost_fraud_model.pkl")
print("✅ Model loaded.")

# Load sample data (e.g., last 5 transactions from test set)
df = pd.read_csv("final_dataset_for_model.csv")
sample_data = df.drop(columns=["TX_FRAUD"]).tail(5)

# Predict fraud probabilities
proba = model.predict_proba(sample_data)[:, 1]
predictions = (proba >= 0.9).astype(int)  # threshold from Day 9

# Show results
for i, prob in enumerate(proba):
    print(f"\n🔍 Transaction {i+1}")
    print(f"Probability of fraud: {prob:.4f}")
    print(f"Prediction (0=Legit, 1=Fraud): {predictions[i]}")
print("\n✅ Predictions complete.")
