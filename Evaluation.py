import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

class TransformerRegressor(nn.Module):
    def __init__(self, input_dim, output_dim, d_model=32, nhead=4, num_encoder_layers=2, num_decoder_layers=2,
                 dim_feedforward=32, dropout=0.2):
        super(TransformerRegressor, self).__init__()
        self.input_projection = nn.Linear(input_dim, d_model)
        self.transformer = nn.Transformer(d_model=d_model, nhead=nhead, num_encoder_layers=num_encoder_layers,
                                          num_decoder_layers=num_decoder_layers, dim_feedforward=dim_feedforward,
                                          dropout=dropout)
        self.output_projection = nn.Linear(d_model, output_dim)

    def forward(self, src):
        src = self.input_projection(src)
        src = src.permute(1, 0, 2)
        transformer_output = self.transformer(src, src)
        transformer_output = transformer_output.permute(1, 0, 2)
        output = self.output_projection(transformer_output[:, -1, :])
        return output

class LSTMRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3, dropout=0.2):
        super(LSTMRegressor, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        lstm_out = self.layer_norm(lstm_out)  # 对LSTM输出进行Layer Normalization
        out = self.fc(self.dropout(lstm_out[:, -1, :]))  # 只取最后一个时间步的输出并应用Dropout
        return out


class GRURegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3,):
        super(GRURegressor, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True,dropout=0.2)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # 初始化隐藏层状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)

        # 前向传播GRU
        out, _ = self.gru(x, h0)
        # 取最后一个时间步的输出
        out = self.fc(out[:, -1, :])
        return out


class RNNRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3):
        super(RNNRegressor, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_dim, hidden_dim, num_layers, batch_first=True,dropout=0.2)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # 初始化隐藏层状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)

        # 前向传播RNN
        out, _ = self.rnn(x, h0)
        # 取最后一个时间步的输出
        out = self.fc(out[:, -1, :])
        return out
# 读取数据
x_data_2023 = pd.read_csv('Sensor_2023_daytime_1.csv')
y_data_2023 = pd.read_csv('Inverter_2023_daytime_1.csv')
x_data_2022 = pd.read_csv('Sensor_2022_daytime_1.csv')
y_data_2022 = pd.read_csv('Inverter_2022_daytime_1.csv')

# 合并数据
x_data_combined = pd.concat([x_data_2022, x_data_2023], axis=0, ignore_index=True)
y_data_combined = pd.concat([y_data_2022, y_data_2023], axis=0, ignore_index=True)

# 选择特定的输入特征
X = np.stack([x_data_combined['Module_Temperature_degF'].values,
              x_data_combined['Ambient_Temperature_degF'].values,
              x_data_combined['Solar_Irradiation_Wpm2'].values,
              x_data_combined['Wind_Speed_mps'].values], axis=1)

y = np.stack([y_data_combined['DC_Voltage'].values, y_data_combined['DC_Current'].values], axis=1)

# 重新创建StandardScaler并应用到所有数据
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_standardized = scaler_X.fit_transform(X)
y_standardized = scaler_y.fit_transform(y)

# 划分训练集、验证集和测试集（按之前的比例）
n_samples = X_standardized.shape[0]
train_size = int(n_samples * 0.9)
X_train, X_test = X_standardized[:train_size], X_standardized[train_size:]
y_train, y_test = y_standardized[:train_size], y_standardized[train_size:]

# 转换为Tensor
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)  # 添加时间步维度
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)



model = LSTMRegressor(input_dim=X_test_tensor.shape[2],hidden_dim=128,output_dim=y_test_tensor.shape[1]).to('cuda')

model.load_state_dict(torch.load('LSTMRegressor_best_model_1A.pth'))

# 确认加载权重后模型状态
#print("After loading weights, model is in train mode:", transformer_model.training)  # 应该还是 True

# 切换到评估模式
model.eval()

# 再次确认模型是否进入评估模式
#print("After calling eval(), model is in eval mode:", not transformer_model.training)  # 应该返回 False (评估模式)

# 对测试集进行预测
with torch.no_grad():
    predictions =model(X_test_tensor.to('cuda'))

# 将预测值转换为 NumPy 数组
predictions_np = predictions.cpu().numpy()

# 逆标准化预测结果
predictions_original = scaler_y.inverse_transform(predictions_np)

# 逆标准化真实值
y_test_original = scaler_y.inverse_transform(y_test_tensor.numpy())

# 评估模型
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

y_test_voltage = y_test_original[:, 0]
y_test_current = y_test_original[:, 1]
y_pred_voltage = predictions_original[:, 0]
y_pred_current = predictions_original[:, 1]

# 分别计算电压和电流的 MAE, MSE, RMSE
def calculate_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    smape_value = smape(y_true, y_pred)
    return mae, mse, rmse, r2, smape_value
def smape(y_true, y_pred):
    return 100 / len(y_true) * np.sum(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))
r2_voltage = r2_score(y_test_voltage, y_pred_voltage)
r2_current = r2_score(y_test_current, y_pred_current)


dc_voltage = y_data_combined['DC_Voltage'].values
dc_voltage_min = np.min(dc_voltage)
dc_voltage_max = np.max(dc_voltage)
mae_voltage, mse_voltage, rmse_voltage, r2_voltage, smape_voltage = calculate_metrics(y_test_original[:, 0], predictions_original[:, 0])
mae_current, mse_current, rmse_current, r2_current, smape_current = calculate_metrics(y_test_original[:, 1], predictions_original[:, 1])
print(f"DC Voltage Range: {dc_voltage_min} to {dc_voltage_max}")
print(
    f" - Voltage(V)- MAE: {mae_voltage:.4f}, MSE: {mse_voltage:.4f}, RMSE: {rmse_voltage:.4f}, R²: {r2_voltage:.4f}, SMAPE: {smape_voltage:.2f}%")
print(
    f" - Current(A)- MAE: {mae_current:.4f}, MSE: {mse_current:.4f}, RMSE: {rmse_current:.4f}, R²: {r2_current:.4f}, SMAPE: {smape_current:.2f}%")
# 可视化结果
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))
plt.plot(y_test_original[0:1000, 0], label='True Voltage (V)', color='blue')
plt.plot(predictions_original[0:1000, 0], label='Predicted Voltage (V)', color='orange')
plt.xlabel('Time Step')
plt.ylabel('Voltage (V)')
plt.title('LSTM - Voltage (V) Test Predictions')
plt.legend()
#plt.savefig('LSTM_voltage_2_year_V.pdf')
plt.show()

plt.figure(figsize=(8, 6))
plt.plot(y_test_original[0:1000, 1], label='True Current (A)', color='blue')
plt.plot(predictions_original[0:1000, 1], label='Predicted Current (A)', color='orange')
plt.xlabel('Time Step')
plt.ylabel('Current (A)')
plt.title('LSTM- Current (A) Test Predictions')
plt.legend()
#plt.savefig('LSTM_Current_2_year_I.pdf')
plt.show()
