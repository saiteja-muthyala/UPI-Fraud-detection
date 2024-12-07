import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd

# Load dataset
upi_fraud_data = pd.read_csv(r"E:\\21eg107b58\\synthetic_upi_fraud_data.csv")

# Preprocessing
sender_encoder = LabelEncoder()
receiver_encoder = LabelEncoder()
upi_fraud_data['Sender_ID'] = sender_encoder.fit_transform(upi_fraud_data['Sender_ID'])
upi_fraud_data['Receiver_ID'] = receiver_encoder.fit_transform(upi_fraud_data['Receiver_ID'])       

# Extract hours and minutes from 'Timestamp'
upi_fraud_data['Timestamp'] = pd.to_datetime(upi_fraud_data['Timestamp'])
upi_fraud_data['Hour'] = upi_fraud_data['Timestamp'].dt.hour
upi_fraud_data['Minute'] = upi_fraud_data['Timestamp'].dt.minute

upi_fraud_data = upi_fraud_data.drop(columns=['Timestamp'])

# Extract features and labels
X = upi_fraud_data.drop(columns=['Is_Fraud'])
y = upi_fraud_data['Is_Fraud']

# Normalize 'Amount' feature
scaler = StandardScaler()
X['Amount'] = scaler.fit_transform(X[['Amount']])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

# Define the Neural Network model
class FraudDetectionModel(nn.Module):
    def __init__(self, input_dim):
        super(FraudDetectionModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.sigmoid(self.fc4(x))
        return x

# Initialize the model, loss function, and optimizer
input_dim = X_train.shape[1]
model = FraudDetectionModel(input_dim)

criterion = nn.BCELoss()  # Binary Cross Entropy Loss for binary classification
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
epochs = 10
batch_size = 32
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()

    # Forward pass
    y_pred_train = model(X_train_tensor)
    loss = criterion(y_pred_train, y_train_tensor)

    # Backward pass and optimization
    loss.backward()
    optimizer.step()

    # Print the training loss
    if (epoch + 1) % 1 == 0:
        print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}')

# Evaluate the model
model.eval()
with torch.no_grad():
    y_pred_test = model(X_test_tensor)
    y_pred_test = (y_pred_test > 0.5).float()  # Apply a threshold of 0.5

    accuracy = (y_pred_test.eq(y_test_tensor).sum() / y_test_tensor.shape[0]).item()
    print(f'Test Accuracy: {(accuracy)*100}')
