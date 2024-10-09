import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import os
import logging
import time
import psutil

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

    # Adding additional features (e.g., previous day temperature and humidity)
    data['prev_temperature'] = data['temperature_2m'].shift(1)
    data['prev_humidity'] = data['relative_humidity_2m'].shift(1)

    # Drop NaN values
    data.dropna(inplace=True)

    features = data[['relative_humidity_2m', 'month', 'day_of_year', 'is_weekend', 'prev_temperature', 'prev_humidity']]
    labels = data[['temperature_2m']]

    # Scaling
    scaler_features = MinMaxScaler(feature_range=(0, 1))
    scaler_labels = MinMaxScaler(feature_range=(0, 1))
    features_scaled = scaler_features.fit_transform(features)
    labels_scaled = scaler_labels.fit_transform(labels)

    return features_scaled, labels_scaled, scaler_labels

# Create dataset function
def create_dataset(features, labels, time_step=1):
    X, y = [], []
    for i in range(len(features) - time_step - 1):
        X.append(features[i:(i + time_step), :])
        y.append(labels[i + time_step, 0])
    return np.array(X), np.array(y)

# Define LSTM Model
class LSTMModel(nn.Module):
    def __init__(self, input_size=6, hidden_size=256, num_layers=3, dropout_rate=0.3, output_size=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_rate)
        self.fc1 = nn.Linear(hidden_size, 128)
        self.fc2 = nn.Linear(128, output_size)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc1(self.relu(out[:, -1, :]))
        out = self.dropout(out)
        out = self.fc2(out)
        return out

# Streamlit app layout
st.title("LSTM Training Dashboard")
st.sidebar.header("Training Configuration")

# Control Model Parameters
hidden_size = st.sidebar.number_input("Hidden Size", min_value=32, max_value=1024, value=256, step=32)
dropout_rate = st.sidebar.slider("LSTM Dropout Rate", min_value=0.0, max_value=0.5, value=0.3, step=0.05)

# Hyperparameters
num_epochs = st.sidebar.number_input("Number of Epochs", min_value=1, value=800)
learning_rate = st.sidebar.number_input(
    "Learning Rate",
    min_value=0.00001,
    max_value=0.1,
    value=0.001,
    step=0.00001
)
batch_size = st.sidebar.number_input("Batch Size", min_value=1, value=64)
validation_split = st.sidebar.slider("Validation Split (%)", min_value=0, max_value=50, value=20)

# Load data
features_scaled, labels_scaled, scaler_labels = load_data()
X, y = create_dataset(features_scaled, labels_scaled, time_step=48)

# Prepare DataLoader
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)

# Split data into training and validation sets
val_size = int(len(X_tensor) * validation_split / 100)
train_size = len(X_tensor) - val_size
train_dataset, val_dataset = random_split(TensorDataset(X_tensor, y_tensor), [train_size, val_size])

# Define training function
def train_model(num_epochs, learning_rate, batch_size):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMModel(hidden_size=hidden_size, dropout_rate=dropout_rate).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    epoch_losses = []
    val_losses = []

    # Create a placeholder for the loss plot
    fig = go.Figure()
    fig.update_layout(
        title='Training and Validation Loss',
        xaxis_title='Epochs',
        yaxis_title='Loss',
        template='plotly_white'
    )

    # Create a progress bar
    progress_bar = st.progress(0)

    for epoch in range(num_epochs):
        start_time = time.time()
        model.train()
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
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

        # Validation step
        model.eval()
        val_loss = 0.0
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        with torch.no_grad():
            for X_val, y_val in val_loader:
                outputs = model(X_val.to(device))
                loss = criterion(outputs, y_val.unsqueeze(1).to(device))
                val_loss += loss.item()
        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        # Log the training and validation loss
        logging.info(f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {train_loss:.6f}, Validation Loss: {val_loss:.6f}")

        # Update Streamlit progress bar
        progress_bar.progress((epoch + 1) / num_epochs)

        # Update plot every 10 epochs
        if (epoch + 1) % 10 == 0:
            fig.add_trace(go.Scatter(x=list(range(len(epoch_losses))), y=epoch_losses, mode='lines', name='Train Loss', line=dict(color='blue')))
            fig.add_trace(go.Scatter(x=list(range(len(val_losses))), y=val_losses, mode='lines', name='Validation Loss', line=dict(color='orange')))
            st.plotly_chart(fig, use_container_width=True)

    return model  # Return the trained model for testing

# Define test function
def test_model(model, features_scaled, labels_scaled, time_step=48):
    # Create test dataset using the last `time_step` elements from the training data
    test_features, test_labels = create_dataset(features_scaled, labels_scaled, time_step)

    # Use only the last 200 examples for testing
    if len(test_features) > 200:
        test_features = test_features[-200:]
        test_labels = test_labels[-200:]

    # Convert to tensors
    test_features_tensor = torch.tensor(test_features, dtype=torch.float32).to(next(model.parameters()).device)

    # Make predictions
    model.eval()
    with torch.no_grad():
        predictions = model(test_features_tensor).cpu().numpy()

    # Inverse scaling to get actual values
    predictions = scaler_labels.inverse_transform(predictions)
    test_labels = scaler_labels.inverse_transform(test_labels.reshape(-1, 1))

    return predictions, test_labels

# Start Training Button
if st.button("Start Training"):
    model = train_model(num_epochs, learning_rate, batch_size)

# Show GPU Info and System Details
st.sidebar.subheader("System Information")
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**2)  # Convert to MB
    st.sidebar.write(f"GPU Name: {gpu_name}")
    st.sidebar.write(f"Total GPU Memory: {gpu_memory:.2f} MB")
else:
    st.sidebar.write("CUDA not available. Running on CPU.")

cpu_usage = psutil.cpu_percent(interval=1)
ram_usage = psutil.virtual_memory().percent
st.sidebar.write(f"CPU Usage: {cpu_usage}%")
st.sidebar.write(f"RAM Usage: {ram_usage}%")

# Display model architecture
st.sidebar.subheader("Model Architecture")
model = LSTMModel(hidden_size=hidden_size, dropout_rate=dropout_rate)
st.sidebar.write(model)

# Display Predictions after training
if st.button("Test Model"):
    # Ensure the model is available for testing
    if 'model' in locals():
        predictions, actuals = test_model(model, features_scaled, labels_scaled)

        # Display results in a DataFrame
        results_df = pd.DataFrame({
            'Predictions': predictions.flatten(),
            'Actual': actuals.flatten()
        })
        st.write(results_df)

        # Plot predictions vs actual values
        fig_results = go.Figure()
        fig_results.add_trace(go.Scatter(x=results_df.index, y=results_df['Predictions'], mode='lines', name='Predictions', line=dict(color='blue')))
        fig_results.add_trace(go.Scatter(x=results_df.index, y=results_df['Actual'], mode='lines', name='Actual', line=dict(color='orange')))
        fig_results.update_layout(title='Predictions vs Actual', xaxis_title='Index', yaxis_title='Temperature', template='plotly_white')
        st.plotly_chart(fig_results, use_container_width=True)
    else:
        st.error("No trained model found. Please train the model first.")
