import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from xgboost import XGBClassifier, plot_importance
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_recall_curve,
)

# 1. Load dataset
df = pd.read_csv("final_dataset_for_model.csv")
X = df.drop(columns=["TX_FRAUD"])
y = df["TX_FRAUD"]

# 2. Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 3. Handle class imbalance
scale_pos_weight = sum(y_train == 0) / sum(y_train == 1)

# 4. Train XGBoost
model = XGBClassifier(
    scale_pos_weight=scale_pos_weight,
    eval_metric="logloss",
    max_depth=4,
    learning_rate=0.1,
    n_estimators=100,
    random_state=42,
)
model.fit(X_train, y_train)
print("✅ Model trained.")

# 5. Plot Feature Importance (by gain)
plt.figure(figsize=(8, 6))
plot_importance(model, importance_type="gain", title="Feature Importance (Gain)")
plt.tight_layout()
plt.show()

# 6. Predict probabilities
y_proba = model.predict_proba(X_test)[:, 1]

# 7. Plot Precision–Recall curve
precision, recall, thresholds = precision_recall_curve(y_test, y_proba)

plt.figure(figsize=(8, 5))
plt.plot(recall, precision, marker=".")
plt.title("Precision-Recall Curve")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.grid(True)
plt.tight_layout()
plt.show()

# 8. Evaluate at custom threshold
THRESHOLD = 0.90
y_pred_thresh = (y_proba >= THRESHOLD).astype(int)

# 9. Show classification report
print(f"\n🧪 Evaluation at threshold = {THRESHOLD}")
print(classification_report(y_test, y_pred_thresh, digits=4))

# 10. Confusion matrix
cm = confusion_matrix(y_test, y_pred_thresh)
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm, display_labels=["Legit (0)", "Fraud (1)"]
)
disp.plot(cmap="Blues")
plt.title(f"Confusion Matrix @ Threshold {THRESHOLD}")
plt.tight_layout()
plt.show()
