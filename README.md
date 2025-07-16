# 🚨 Fraud Detection System

A complete end-to-end fraud detection pipeline using custom feature engineering and XGBoost, built for real-time and batch inference.

---

## 🧠 Project Highlights

* **Data Source**: Simulated credit card transactions (`.pkl` files)
* **Objective**: Predict whether a transaction is fraudulent
* **Model**: XGBoost Classifier with class imbalance handling
* **Features**:

  * `TX_AMOUNT`
  * `is_high_amount` (rule-based)
  * `terminal_fraud_history_28d` (past frauds per terminal)
  * `is_unusual_for_customer` (behavioral outliers)
* **Deployment**:

  * 📊 Streamlit dashboard (`dashboard_app.py`)
  * 🚀 FastAPI backend (`fraud_api.py`)
  * 🧠 Trained model saved as `xgboost_fraud_model.pkl`

---

## 📦 Folder Structure

```
fraud_detection/
├── data/                           # Raw .pkl files
├── final_dataset_for_model.csv     # Cleaned + feature-engineered data
├── dashboard_app.py                # Streamlit dashboard app
├── fraud_api.py                    # FastAPI backend
├── xgboost_fraud_model.pkl         # Trained model
├── predict_fraud_day10.py          # CLI predictions
├── save_model_day10.py             # Save model to disk
├── day6_finalize_dataset.py        # Final dataset creation
├── day7_logistic_model.py
├── day8_xgboost_model.py
├── day9_feature_importance_and_threshold.py
└── README.md                       # You're here
```

---

## 🚀 How to Run

### 1. Install Requirements

```bash
pip install -r requirements.txt
```

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
