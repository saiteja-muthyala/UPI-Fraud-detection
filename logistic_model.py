import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
import matplotlib.pyplot as plt

# 1. Load final dataset
print("📥 Loading data...")
df = pd.read_csv("final_dataset_for_model.csv")
print(f"✅ Loaded: {df.shape}")

# 2. Split X and y
X = df.drop(columns=["TX_FRAUD"])
y = df["TX_FRAUD"]

# 3. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"🔄 Split: {X_train.shape[0]} train rows, {X_test.shape[0]} test rows")

# 4. Train logistic regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
print("✅ Logistic Regression model trained.")

# 5. Predict
y_pred = model.predict(X_test)

# 6. Evaluation
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred, digits=4))

# 7. Confusion matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm, display_labels=["Legit (0)", "Fraud (1)"]
)
disp.plot(cmap="Blues")
plt.title("Confusion Matrix - Logistic Regression")
plt.tight_layout()
plt.show()
