import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set up display options
pd.set_option("display.max_columns", None)

# Path to your data folder
DATA_FOLDER = "data"

# Load all .pkl files from the data folder
files = sorted([f for f in os.listdir(DATA_FOLDER) if f.endswith(".pkl")])

print(f"Found {len(files)} data files. Loading the first 5 days for exploration...")

# Load the first 5 files to keep it lightweight for now
dfs = []
for f in files[:5]:
    df = pd.read_pickle(os.path.join(DATA_FOLDER, f))
    dfs.append(df)

# Combine into one DataFrame
data = pd.concat(dfs, ignore_index=True)

# Print basic info
print(f"\n✅ Shape of dataset: {data.shape}")
print("\n✅ Column names:")
print(data.columns)

print("\n✅ First few records:")
print(data.head())

# Check for missing values
print("\n🔍 Missing values per column:")
print(data.isnull().sum())

# Summary statistics
print("\n📊 Summary statistics:")
print(data.describe())

# Plot class distribution of TX_FRAUD
plt.figure(figsize=(6, 4))
sns.countplot(data=data, x="TX_FRAUD")
plt.title("Fraudulent vs Legitimate Transactions")
plt.xlabel("TX_FRAUD (0 = Legitimate, 1 = Fraud)")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# Optional: Show percentages
fraud_percent = data["TX_FRAUD"].value_counts(normalize=True) * 100
print("\n📈 Fraud class distribution (%):")
print(fraud_percent)

# Optional: Print unique fraud scenario codes
print("\n📌 Unique fraud scenarios:")
print(data["TX_FRAUD_SCENARIO"].unique())

# Ensure datetime format
data["TX_DATETIME"] = pd.to_datetime(data["TX_DATETIME"])

# Group fraud by date
fraud_per_day = data[data["TX_FRAUD"] == 1].groupby(data["TX_DATETIME"].dt.date).size()

# Plot
fraud_per_day.plot(kind="bar", title="Fraud Transactions per Day", figsize=(8, 4))
plt.ylabel("Fraud Count")
plt.xlabel("Date")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 4))
sns.boxplot(data=data, x="TX_FRAUD", y="TX_AMOUNT")
plt.title("Transaction Amounts by TX_FRAUD")
plt.xlabel("TX_FRAUD (0 = Legit, 1 = Fraud)")
plt.ylabel("TX_AMOUNT")
plt.tight_layout()
plt.show()

# Top 10 terminals with most fraud
top_terminals = data[data["TX_FRAUD"] == 1]["TERMINAL_ID"].value_counts().head(10)
top_terminals.plot(kind="bar", title="Top Terminals with Most Frauds", figsize=(6, 3))
plt.ylabel("Fraud Count")
plt.xlabel("TERMINAL_ID")
plt.tight_layout()
plt.show()

# Top 10 customers with most fraud
top_customers = data[data["TX_FRAUD"] == 1]["CUSTOMER_ID"].value_counts().head(10)
top_customers.plot(kind="bar", title="Top Customers with Most Frauds", figsize=(6, 3))
plt.ylabel("Fraud Count")
plt.xlabel("CUSTOMER_ID")
plt.tight_layout()
plt.show()

scenario_counts = data[data["TX_FRAUD"] == 1]["TX_FRAUD_SCENARIO"].value_counts()

# Bar plot
scenario_counts.plot(kind="bar", title="Fraud Scenarios", figsize=(6, 3))
plt.xlabel("Scenario Code")
plt.ylabel("Fraud Count")
plt.tight_layout()
plt.show()

print("\n📌 Fraud scenario counts:\n", scenario_counts)


from sklearn.metrics import confusion_matrix, classification_report

# 1. Create a new feature
data["is_high_amount"] = data["TX_AMOUNT"] > 220

# 2. Compare with actual fraud labels
y_true = data["TX_FRAUD"]
y_pred = data["is_high_amount"].astype(int)  # convert boolean to 0/1

# 3. Confusion matrix
cm = confusion_matrix(y_true, y_pred)
print("\n📉 Confusion Matrix:\n", cm)

# 4. Classification report (precision, recall, f1)
print("\n📊 Classification Report:")
print(classification_report(y_true, y_pred, digits=4))

# 5. Basic insight
high_amount_frauds = data[data["is_high_amount"] & (data["TX_FRAUD"] == 1)].shape[0]
total_frauds = data[data["TX_FRAUD"] == 1].shape[0]
print(
    f"\n🎯 Rule Accuracy: {high_amount_frauds}/{total_frauds} frauds correctly flagged by amount > 220"
)

print("\n🚀 Starting Day 4: Feature Engineering – Terminal Fraud History (28 days)")

# Group by terminal
terminal_groups = data.groupby("TERMINAL_ID")

# Initialize column
data["terminal_fraud_history_28d"] = 0

# Loop through each terminal
for terminal_id, group in terminal_groups:
    group = group.sort_values("TX_TIME_DAYS")

    fraud_count_list = []

    for idx, row in group.iterrows():
        # Look back 28 days (excluding current txn day)
        past_txns = group[
            (group["TX_TIME_DAYS"] < row["TX_TIME_DAYS"])
            & (group["TX_TIME_DAYS"] >= row["TX_TIME_DAYS"] - 28)
        ]
        # Count frauds in this terminal in the last 28 days
        count_fraud = past_txns["TX_FRAUD"].sum()
        fraud_count_list.append(count_fraud)

    # Assign values to original dataframe
    data.loc[group.index, "terminal_fraud_history_28d"] = fraud_count_list

print("\n✅ Completed computing terminal_fraud_history_28d feature.")

# Preview new feature
print(
    data[
        ["TERMINAL_ID", "TX_TIME_DAYS", "TX_FRAUD", "terminal_fraud_history_28d"]
    ].head(10)
)

# Visualize fraud vs. non-fraud distribution for this feature
plt.figure(figsize=(6, 4))
sns.boxplot(data=data, x="TX_FRAUD", y="terminal_fraud_history_28d")
plt.title("Fraud History at Terminal (28d) vs TX_FRAUD")
plt.xlabel("TX_FRAUD (0 = Legit, 1 = Fraud)")
plt.ylabel("Fraud Count in Terminal (last 28 days)")
plt.tight_layout()
plt.show()

# Optional: Save processed data with new features
data.to_csv("processed_data_day4.csv", index=False)
print("\n💾 Saved enriched dataset as processed_data_day4.csv")

print("\n🚀 Starting Day 5: Feature Engineering – Customer Behavior (Scenario 3)")

# Group by customer
customer_groups = data.groupby("CUSTOMER_ID")

# Initialize new feature column
data["is_unusual_for_customer"] = 0

# Loop through each customer's transactions
for customer_id, group in customer_groups:
    group = group.sort_values("TX_TIME_DAYS")

    unusual_flags = []

    for idx, row in group.iterrows():
        # Look back 14 days
        past_txns = group[
            (group["TX_TIME_DAYS"] < row["TX_TIME_DAYS"])
            & (group["TX_TIME_DAYS"] >= row["TX_TIME_DAYS"] - 14)
        ]

        # Compute customer avg + std for past 14 days
        mean_amt = past_txns["TX_AMOUNT"].mean()
        std_amt = past_txns["TX_AMOUNT"].std()

        threshold = mean_amt + 3 * std_amt if pd.notna(std_amt) else float("inf")

        # Flag if current TX_AMOUNT > threshold
        is_unusual = row["TX_AMOUNT"] > threshold
        unusual_flags.append(int(is_unusual))

    # Save the flag back
    data.loc[group.index, "is_unusual_for_customer"] = unusual_flags

print("\n✅ Completed computing is_unusual_for_customer feature.")

plt.figure(figsize=(6, 4))
sns.boxplot(data=data, x="TX_FRAUD", y="is_unusual_for_customer")
plt.title("Unusual Spending vs TX_FRAUD")
plt.xlabel("TX_FRAUD (0 = Legit, 1 = Fraud)")
plt.ylabel("is_unusual_for_customer")
plt.tight_layout()
plt.show()

data.to_csv("processed_data_day5.csv", index=False)
print("\n💾 Saved updated dataset as processed_data_day5.csv")
