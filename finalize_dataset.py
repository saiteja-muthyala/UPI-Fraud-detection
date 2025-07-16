import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load enriched dataset from Day 5
print("📥 Loading data from processed_data_day5.csv ...")
df = pd.read_csv("processed_data_day5.csv")
print(f"✅ Dataset loaded with shape: {df.shape}")

# Drop unnecessary columns
drop_cols = [
    "TRANSACTION_ID",
    "TX_DATETIME",
    "CUSTOMER_ID",
    "TERMINAL_ID",
    "TX_TIME_SECONDS",
    "TX_TIME_DAYS",
    "TX_FRAUD_SCENARIO",  # contains label-based leakage
]

df_model = df.drop(columns=drop_cols)
print(f"🧹 Dropped irrelevant columns. New shape: {df_model.shape}")

# Show the final feature columns
print("\n📦 Final feature columns:")
print(df_model.columns.tolist())

# Correlation heatmap
print("\n📊 Generating correlation heatmap...")
plt.figure(figsize=(8, 6))
sns.heatmap(df_model.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()

# Save final dataset
df_model.to_csv("final_dataset_for_model.csv", index=False)
print("\n💾 Saved: final_dataset_for_model.csv (ready for ML modeling)")
