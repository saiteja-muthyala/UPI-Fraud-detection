# 🚨 Credit Card or UPI Fraud Detection System

A complete end-to-end fraud detection pipeline using custom feature engineering and XGBoost, built for real-time and batch inference.

---

## 🧠 Project Highlights

- **Data Source**: Simulated credit card transactions (`.pkl` files)
- **Objective**: Predict whether a transaction is fraudulent
- **Model**: XGBoost Classifier with class imbalance handling
- **Features**:
  - `TX_AMOUNT`
  - `is_high_amount` (rule-based)
  - `terminal_fraud_history_28d` (past frauds per terminal)
  - `is_unusual_for_customer` (behavioral outliers)
- **Deployment**:
  - 📊 Streamlit dashboard (`dashboard_app.py`)
  - 🚀 FastAPI backend (`fraud_api.py`)
  - 🧠 Trained model saved as `xgboost_fraud_model.pkl`

---

## 📦 Folder Structure

```

fraud\_detection/
├── data/                           # Raw .pkl files
├── final\_dataset\_for\_model.csv     # Cleaned + feature-engineered data
├── dashboard\_app.py                # Streamlit dashboard app
├── fraud\_api.py                    # FastAPI backend
├── xgboost\_fraud\_model.pkl         # Trained model
├── predict\_fraud\_day10.py          # CLI predictions
├── save\_model\_day10.py             # Save model to disk
├── day6\_finalize\_dataset.py        # Final dataset creation
├── day7\_logistic\_model.py
├── day8\_xgboost\_model.py
├── day9\_feature\_importance\_and\_threshold.py
└── README.md                       # You're here

````

---

## 🚀 How to Run

### 1. Install Requirements
```bash
pip install -r requirements.txt
````

### 2. Run Dashboard

```bash
streamlit run dashboard_app.py
```

### 3. Run FastAPI Server

```bash
uvicorn fraud_api:app --reload
```

* Access Swagger docs at: `http://127.0.0.1:8000/docs`

---

## 🎯 Model Performance

| Metric    | Value  |
| --------- | ------ |
| Accuracy  | 99.6%  |
| Precision | 23.91% |
| Recall    | 78.57% |
| AUC-PR    | High   |

---

## 🧪 Sample API Input

```json
{
  "TX_AMOUNT": 300.5,
  "is_high_amount": 1,
  "terminal_fraud_history_28d": 3,
  "is_unusual_for_customer": 1
}
```

**Output**:

```json
{
  "fraud_probability": 0.9984,
  "prediction": 1
}
```

---

## 📬 Contact

**Author**: Muthyala Sai Teja
📧 Email: [tejasai48548012@gmail.com](mailto:tejasai48548012@gmail.com)

---

> Production-ready, interpretable, and extendable fraud detection pipeline.

```
