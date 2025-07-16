import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Set Streamlit page config
st.set_page_config(page_title="Fraud Detection Dashboard", layout="wide")


# Load trained model
@st.cache_resource
def load_model():
    return joblib.load("xgboost_fraud_model.pkl")


model = load_model()

st.title("🔍 Fraud Transaction Detection Dashboard")
st.markdown(
    "Upload transaction data and detect fraudulent activity using your trained XGBoost model."
)

# Upload CSV file
uploaded_file = st.file_uploader("📂 Upload CSV File", type=["csv"])

if uploaded_file:
    # Load data
    data = pd.read_csv(uploaded_file)
    st.success("✅ File uploaded and read successfully!")

    # Check required columns
    required_cols = [
        "TX_AMOUNT",
        "is_high_amount",
        "terminal_fraud_history_28d",
        "is_unusual_for_customer",
    ]
    if not all(col in data.columns for col in required_cols):
        st.error(
            f"❌ Uploaded file must contain these columns:\n{', '.join(required_cols)}"
        )
    else:
        # Make predictions
        probas = model.predict_proba(data[required_cols])[:, 1]
        preds = (probas >= 0.9).astype(int)  # Threshold set from Day 9

        data["fraud_probability"] = probas
        data["prediction"] = preds

        # Display preview
        st.subheader("📊 Prediction Results")
        st.dataframe(data.head(10), use_container_width=True)

        # Fraud summary
        fraud_count = data["prediction"].sum()
        total = len(data)
        st.metric("🚨 Total Fraudulent Transactions Detected", fraud_count, delta=None)
        st.metric("📈 Fraud Detection Rate (%)", f"{(fraud_count/total)*100:.2f}%")

        # Visualizations
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Fraud Probability Distribution")
            fig, ax = plt.subplots()
            sns.histplot(
                data["fraud_probability"], bins=30, kde=True, ax=ax, color="orange"
            )
            st.pyplot(fig)

        with col2:
            st.subheader("Fraud Predictions Count")
            fig2, ax2 = plt.subplots()
            sns.countplot(x="prediction", data=data, palette="Set2", ax=ax2)
            ax2.set_xticklabels(["Legit (0)", "Fraud (1)"])
            st.pyplot(fig2)

        # Download results
        st.subheader("⬇️ Download Results")
        st.download_button(
            "Download CSV with Predictions",
            data.to_csv(index=False),
            file_name="fraud_predictions.csv",
            mime="text/csv",
        )
else:
    st.info("📌 Upload a `.csv` file with transaction data to begin.")
