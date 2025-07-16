import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
import matplotlib.pyplot as plt
from xgboost import XGBClassifier

# 1. Load data
df = pd.read_csv("final_dataset_for_model.csv")
print(f"✅ Loaded: {df.shape}")

# 2. Split X and y
X = df.drop(columns=["TX_FRAUD"])
y = df["TX_FRAUD"]

# 3. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 4. Compute scale_pos_weight (handle imbalance)
num_legit = sum(y_train == 0)
num_fraud = sum(y_train == 1)
scale_pos_weight = num_legit / num_fraud
print(f"⚖️ Class Imbalance: Legit = {num_legit}, Fraud = {num_fraud}")
print(f"🧮 scale_pos_weight = {scale_pos_weight:.2f}")

# 5. Train XGBoost
model = XGBClassifier(
    use_label_encoder=False,
    eval_metric="logloss",
    scale_pos_weight=scale_pos_weight,
    max_depth=4,
    learning_rate=0.1,
    n_estimators=100,
    random_state=42,
)
model.fit(X_train, y_train)
print("✅ XGBoost model trained.")

# 6. Predict
y_pred = model.predict(X_test)

# 7. Evaluate
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred, digits=4))

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm, display_labels=["Legit (0)", "Fraud (1)"]
)
disp.plot(cmap="Blues")
plt.title("Confusion Matrix - XGBoost")
plt.tight_layout()
plt.show()
