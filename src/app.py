import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load data
@st.cache_data
def load_data():
    data = pd.read_csv('data/raw/1_y_train.csv')
    data['date'] = pd.to_datetime(data['date'])
    data.set_index('date', inplace=True)

    # Feature Engineering
    data['month'] = data.index.month
    data['day_of_year'] = data.index.dayofyear
    data['is_weekend'] = (data.index.weekday >= 5).astype(int)

    features = data[['relative_humidity_2m', 'month', 'day_of_year', 'is_weekend']]
    labels = data[['temperature_2m']]

    # Scaling
    scaler_features = MinMaxScaler(feature_range=(0, 1))
    scaler_labels = MinMaxScaler(feature_range=(0, 1))
    features_scaled = scaler_features.fit_transform(features)
    labels_scaled = scaler_labels.fit_transform(labels)
    
    return features_scaled, labels_scaled, scaler_features, scaler_labels

# Create dataset function
def create_dataset(features, labels, time_step=1):
    X, y = [], []
    for i in range(len(features) - time_step - 1):
        X.append(features[i:(i + time_step), :])
        y.append(labels[i + time_step, 0])
    return np.array(X), np.array(y)

# Define LSTM Model
class LSTMModel(nn.Module):
    def __init__(self, input_size=4, hidden_size=256, num_layers=4, output_size=1):  # Increased num_layers to 4
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.3)
        self.fc1 = nn.Linear(hidden_size, 128)
        self.fc2 = nn.Linear(128, 64)  # Added an extra fully connected layer
        self.fc3 = nn.Linear(64, output_size)  # Final output layer
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc1(self.relu(out[:, -1, :]))
        out = self.dropout(out)
        out = self.fc2(self.relu(out))  # Pass through the extra layer
        out = self.fc3(out)  # Final output
        return out
    
# Streamlit app layout
st.title("LSTM Training Dashboard")
st.sidebar.header("Training Configuration")

# Hyperparameters
num_epochs = st.sidebar.number_input("Number of Epochs", min_value=1, value=300)
learning_rate = st.sidebar.number_input(
    "Learning Rate", 
    min_value=0.00001, 
    max_value=0.1, 
    value=0.001, 
    step=0.00001
)

# Display the formatted learning rate
st.sidebar.write(f"Current Learning Rate: {learning_rate:.4f}")

batch_size = st.sidebar.number_input("Batch Size", min_value=1, value=64)

# Load data
features_scaled, labels_scaled, scaler_features, scaler_labels = load_data()
X, y = create_dataset(features_scaled, labels_scaled, time_step=48)

# Prepare DataLoader
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)
dataset = TensorDataset(X_tensor, y_tensor)

# Define training function
def train_model(num_epochs, learning_rate, batch_size):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMModel().to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    # Create a placeholder for epoch losses
    epoch_losses = []
    
    # Create a progress bar
    progress_bar = st.progress(0)
    
    # Placeholder for loss display
    loss_display = st.empty()  # Empty widget to update later

    for epoch in range(num_epochs):
        model.train()
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        train_loss = 0.0

        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch.to(device))
            loss = criterion(outputs, y_batch.unsqueeze(1).to(device))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        epoch_losses.append(train_loss)

        # Log the training loss
        logging.info(f"Epoch {epoch + 1}/{num_epochs} - Training Loss: {train_loss:.9f}")

        # Update Streamlit progress bar
        progress_bar.progress((epoch + 1) / num_epochs)

        # Update loss display
        loss_display.markdown(f"**Epoch {epoch + 1}/{num_epochs}**<br>**Training Loss:** {train_loss:.6f}", unsafe_allow_html=True)

    # Save model checkpoint
    checkpoint_path = "model.pth"
    torch.save(model.state_dict(), checkpoint_path)
    logging.info(f"Model checkpoint saved at {checkpoint_path}")

    return model

# Define testing function
def test_model(model, scaler_features, scaler_labels, num_samples=200):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    # Prepare test data (Use the last known data points for testing)
    test_features = features_scaled[-(48 + num_samples):]  # Last 48 time steps plus num_samples for predictions
    test_features_tensor = torch.tensor(test_features[-(48):].reshape(1, 48, -1), dtype=torch.float32).to(device)

    predictions = []
    actuals = []

    with torch.no_grad():
        for _ in range(num_samples):
            # Generate prediction
            pred = model(test_features_tensor)
            predictions.append(pred.item())
            # Shift features for the next prediction
            new_feature = test_features_tensor[0, -1, :].cpu().numpy()
            new_feature[0] = pred.item()  # Replace temperature in feature set
            new_feature = scaler_features.transform(new_feature.reshape(1, -1))  # Scale new feature
            test_features_tensor = torch.cat((test_features_tensor, torch.tensor(new_feature.reshape(1, 1, -1), dtype=torch.float32).to(device)), dim=1)
            test_features_tensor = test_features_tensor[:, 1:, :]  # Remove oldest timestep

            # Append actuals (using last known humidity values for demonstration)
            actuals.append(scaler_labels.inverse_transform([[new_feature[0, 0]]])[0, 0])  # Replace with real actuals

    return predictions, actuals

# Start Training Button
if st.button("Start Training"):
    model = train_model(num_epochs, learning_rate, batch_size)

    # Automatically start testing after training
    predictions, actuals = test_model(model, scaler_features, scaler_labels)

    # Create a DataFrame for results
    results_df = pd.DataFrame({
        "Predicted Temperature": predictions,
        "Actual Temperature": actuals
    })

    # Display results in a table
    st.subheader("Testing Results")
    st.write(results_df)

# Display all checkpoints in sidebar
st.sidebar.subheader("Model Checkpoints")
if os.path.exists("."):
    checkpoint_files = [f for f in os.listdir(".") if f.endswith(".pth")]
    for f in checkpoint_files:
        st.sidebar.write(f)

# Show GPU Info
st.sidebar.subheader("GPU Information")
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**2)  # Convert to MB
    st.sidebar.write(f"GPU Name: {gpu_name}")
    st.sidebar.write(f"Total GPU Memory: {gpu_memory:.2f} MB")
else:
    st.sidebar.write("CUDA not available. Running on CPU.")
