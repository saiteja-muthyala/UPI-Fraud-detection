import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import joblib

# Load final dataset
df = pd.read_csv("final_dataset_for_model.csv")
X = df.drop(columns=["TX_FRAUD"])
y = df["TX_FRAUD"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Compute scale_pos_weight for imbalance
scale_pos_weight = sum(y_train == 0) / sum(y_train == 1)

# Train XGBoost model
model = XGBClassifier(
    scale_pos_weight=scale_pos_weight,
    eval_metric="logloss",
    max_depth=4,
    learning_rate=0.1,
    n_estimators=100,
    random_state=42,
)
model.fit(X_train, y_train)

# Save model to file
joblib.dump(model, "xgboost_fraud_model.pkl")
print("✅ Model saved as xgboost_fraud_model.pkl")
