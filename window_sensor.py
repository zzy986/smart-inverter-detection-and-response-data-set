import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

import time

class LSTMRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3, dropout=0.2):
        super(LSTMRegressor, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        lstm_out = self.layer_norm(lstm_out)
        out = self.fc(self.dropout(lstm_out[:, -1, :]))
        return out

# SMAPE metric
def smape(y_true, y_pred):
    return 100 / len(y_true) * np.sum(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))

# Calculate metrics
def calculate_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    smape_value = smape(y_true, y_pred)
    return mae, mse, rmse, r2, smape_value

# Visualization function
def visualize_results(y_test_original, predictions_original, model_name):
    plt.figure(figsize=(8, 6))
    plt.plot(y_test_original[:, 0], label='True Voltage (V)', color='blue')
    plt.plot(predictions_original[:, 0], label=f'{model_name} Predicted Voltage (V)', color='orange')
    plt.xlabel('Time Step')
    plt.ylabel('Voltage (V)')
    plt.title(f'{model_name} - Voltage (V) Test Predictions')
    plt.legend()
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.plot(y_test_original[:, 1], label='True Current (A)', color='blue')
    plt.plot(predictions_original[:, 1], label=f'{model_name} Predicted Current (A)', color='orange')
    plt.xlabel('Time Step')
    plt.ylabel('Current (A)')
    plt.title(f'{model_name} - Current (A) Test Predictions')
    plt.legend()
    plt.show()

# Data loading and preprocessing
np.random.seed(42)
torch.manual_seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load data
x_data_2023 = pd.read_csv('Sensor_2023_daytime_1.csv')
y_data_2023 = pd.read_csv('Inverter_2023_daytime_1.csv')
x_data_2022 = pd.read_csv('Sensor_2022_daytime_1.csv')
y_data_2022 = pd.read_csv('Inverter_2022_daytime_1.csv')

x_data_combined = pd.concat([x_data_2022, x_data_2023], axis=0, ignore_index=True)
y_data_combined = pd.concat([y_data_2022, y_data_2023], axis=0, ignore_index=True)

# Prepare feature matrix and target matrix
X = np.stack([x_data_combined['Module_Temperature_degF'].values,
              x_data_combined['Ambient_Temperature_degF'].values,
              x_data_combined['Solar_Irradiation_Wpm2'].values,
              x_data_combined['Wind_Speed_mps'].values], axis=1)

y = np.stack([y_data_combined['DC_Voltage'].values, y_data_combined['DC_Current'].values], axis=1)

scaler_X = StandardScaler()
scaler_y = StandardScaler()

X = scaler_X.fit_transform(X)
y = scaler_y.fit_transform(y)


n_samples = X.shape[0]

# 计算训练集大小 (80%)
train_size = int(n_samples * 0.8)

# 划分训练集
X_train = X[:train_size]
y_train = y[:train_size]

# 剩余的数据（20%），用于划分验证集和测试集
X_temp = X[train_size:]
y_temp = y[train_size:]

# 获取剩余数据的样本数量
n_temp_samples = X_temp.shape[0]

# 将剩余的数据按50%划分为验证集和测试集
valid_size = int(n_temp_samples * 0.5)

X_valid = X_temp[:valid_size]
y_valid = y_temp[:valid_size]
X_test = X_temp[valid_size:]
y_test = y_temp[valid_size:]

X_train_tensor = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
X_valid_tensor = torch.tensor(X_valid, dtype=torch.float32).unsqueeze(1).to(device)
y_valid_tensor = torch.tensor(y_valid, dtype=torch.float32).to(device)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1).to(device)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(device)

print(f"Training set size: {X_train_tensor.size(0)} samples")
print(f"Validation set size: {X_valid_tensor.size(0)} samples")
print(f"Test set size: {X_test_tensor.size(0)} samples")
print(f"Training set size: {X_train_tensor.size(0)} samples")
print(f"Validation set size: {X_valid_tensor.size(0)} samples")
print(f"Test set size: {y_test_tensor.size(0)} samples")


# Load model
model = LSTMRegressor(input_dim=X_test_tensor.shape[2],hidden_dim=128, output_dim=y_test_tensor.shape[1]).to(device)
model.load_state_dict(torch.load('LSTMRegressor_best_model_1A.pth'))
model.eval()
start_time = time.time()
# Perform prediction
with torch.no_grad():
    predictions = model(X_test_tensor)
end_time = time.time()  # 结束计时

# 计算并打印运行时间
execution_time = end_time - start_time
print(f"Function execution time: {execution_time:.4f} seconds")
# Convert predictions back to original scale
predicted_values = scaler_y.inverse_transform(predictions.cpu().numpy())
y_test_original = scaler_y.inverse_transform(y_test_tensor.cpu().numpy())

# Visualize the results
visualize_results(y_test_original, predicted_values, "LSTM")

# Calculate evaluation metrics
mae_voltage, mse_voltage, rmse_voltage, r2_voltage, smape_voltage = calculate_metrics(y_test_original[:, 0], predicted_values[:, 0])
mae_current, mse_current, rmse_current, r2_current, smape_current = calculate_metrics(y_test_original[:, 1], predicted_values[:, 1])

print(f"Voltage(V) - MAE: {mae_voltage:.4f}, MSE: {mse_voltage:.4f}, RMSE: {rmse_voltage:.4f}, R²: {r2_voltage:.4f}, SMAPE: {smape_voltage:.2f}%")
print(f"Current(A) - MAE: {mae_current:.4f}, MSE: {mse_current:.4f}, RMSE: {rmse_current:.4f}, R²: {r2_current:.4f}, SMAPE: {smape_current:.2f}%")

# Calculate power and energy error
time_interval_hours = 5 / 60  # If it's a 5-minute interval

true_power = y_test_original[:, 0] * y_test_original[:, 1]  # True power = True voltage * True current
predicted_power = predicted_values[:, 0] * predicted_values[:, 1]  # Predicted power = Predicted voltage * Predicted current

true_energy_kwh = np.sum(true_power) * time_interval_hours / 1000  # Convert to kWh
predicted_energy_kwh = np.sum(predicted_power) * time_interval_hours / 1000  # Convert to kWh

energy_error_kwh = predicted_energy_kwh - true_energy_kwh
percentage_error = (energy_error_kwh / true_energy_kwh) * 100

# Print energy error results
print(f"True cumulative energy (kWh): {true_energy_kwh:.4f} kWh")
print(f"Predicted cumulative energy (kWh): {predicted_energy_kwh:.4f} kWh")
print(f"Energy error (kWh): {energy_error_kwh:.4f} kWh")
print(f"Percentage error: {percentage_error:.2f}%")
