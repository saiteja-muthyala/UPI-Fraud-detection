# Fraud Transaction Detection System

This project is a complete, production-ready fraud detection system built using **XGBoost**, **Streamlit**, and **FastAPI**. It uses advanced feature engineering, rule-based logic, and machine learning to detect fraudulent credit card transactions with a high degree of accuracy and interpretability.

---

## 🚀 Features

* **Custom Feature Engineering**:

  * `is_high_amount`: Flags large transactions
  * `terminal_fraud_history_28d`: Counts past frauds at terminal
  * `is_unusual_for_customer`: Detects outlier behavior per user

* **Modeling**:

  * Baseline: Logistic Regression (Day 7)
  * Final Model: XGBoost with threshold tuning (Day 8-9)
  * Saved model (`xgboost_fraud_model.pkl`) used in backend API & dashboard

* **Interactive Dashboard (Streamlit)**:

  * Upload CSV and get live fraud predictions
  * View fraud probability distribution and fraud count
  * Fraud prediction results in table format with color highlights

* **API Interface (FastAPI)**:

  * Swagger UI documentation
  * POST endpoint `/predict/` for JSON-based prediction

---

## 📊 Model Performance (on Test Set)

| Metric         | Value          |
| -------------- | -------------- |
| Accuracy       | 99.60%         |
| Precision      | 23.91%         |
| Recall         | 78.57%         |
| F1-Score       | 36.67%         |
| True Positives | 11 / 14 frauds |

---

## 📁 Folder Structure

```
fraud_detection/
├── data/                            # Original .pkl files
├── dashboard_app.py                 # Streamlit UI app
├── fraud_api.py                     # FastAPI backend
├── xgboost_fraud_model.pkl          # Saved XGBoost model
├── final_dataset_for_model.csv      # Final processed dataset
├── processed_data_day4.csv          # Feature: terminal fraud history
├── processed_data_day5.csv          # Feature: customer behavior
├── day6_finalize_dataset.py         # Drop columns, prepare model input
├── day7_logistic_model.py           # Logistic regression model
├── day8_xgboost_model.py            # XGBoost training
├── day9_feature_importance_and_threshold.py # Threshold tuning & plots
├── save_model_day10.py              # Save trained model
├── predict_fraud_day10.py           # Test predictions
├── load_data.py                     # Initial data load & exploration
├── README.md                        # You're here!
├── fraud_detection_report.pdf       # 📄 Project report (optional)
```

---

## 📦 Setup Instructions

```bash
# Step 1: Clone repo
$ git clone https://github.com/yourname/fraud_detection
$ cd fraud_detection

# Step 2: Create virtual environment
$ python -m venv venv
$ source venv/bin/activate  # On Windows: venv\Scripts\activate

# Step 3: Install dependencies
$ pip install -r requirements.txt

# Step 4: Run Streamlit app (dashboard)
$ streamlit run dashboard_app.py

# Step 5: Run FastAPI server (API)
$ uvicorn fraud_api:app --reload

# Open Swagger Docs:
# http://127.0.0.1:8000/docs
```

---

## 📥 API Sample Input (JSON)

```json
{
  "TX_AMOUNT": 300.5,
  "is_high_amount": 1,
  "terminal_fraud_history_28d": 3,
  "is_unusual_for_customer": 1
}
```

## ✅ API Output

```json
{
  "fraud_probability": 0.9984,
  "prediction": 1
}
```

---

## 📈 Example Dashboard Screens

* Prediction Results Table with Probabilities
* Histogram of Fraud Probabilities
* Bar Chart of Fraud Counts

> 📸 Add screenshots in your GitHub repo!

---

## 📌 Tech Stack

* Python
* Pandas, Seaborn, Matplotlib
* Scikit-learn, XGBoost
* Streamlit (for dashboard)
* FastAPI (for API)
* Uvicorn (API server)

---

## 📄 Report

Want to go deeper? See [fraud\_detection\_report.pdf](./fraud_detection_report.pdf) for a detailed breakdown of methodology, features, and evaluation.

---

## 📬 Contact

**Muthyala Sai Teja**
Email: [tejasai48548012@gmail.com](mailto:tejasai48548012@gmail.com)
LinkedIn: [linkedin.com/in/muthyala-sai-teja](https://www.linkedin.com/in/muthyala-sai-teja)

---

## ⭐️ Star This Project

If you found this useful, give it a star and share with others who may benefit from a complete fraud detection pipeline!

---

*Built with ❤️ to stop fraud in its tracks.*
