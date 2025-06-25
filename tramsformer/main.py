import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error

from config import Configs
from model.Transformer import Model
from data_loader import load_weather_data

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def train_and_evaluate(days, lr=1e-3, epochs=5):
    print(f"\n=== 预测未来 {days} 天（{24 * days} 小时） ===")

    configs = Configs()
    configs.pred_len = 24 * days
    configs.label_len = configs.seq_len // 2  # 可调整
    configs.d_model = 512
    configs.n_heads = 8
    configs.d_ff = 2048
    configs.factor = 1
    configs.dropout = 0.1

    device = configs.device

    train_loader, test_loader, scaler = load_weather_data('./data/weather.csv', configs)

    model = Model(configs).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for x_enc, x_dec, y, mask in train_loader:
            x_enc, x_dec, y, mask = x_enc.to(device), x_dec.to(device), y.to(device), mask.to(device)

            optimizer.zero_grad()
            out = model(x_enc, None, x_dec, None, mask=mask)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x_enc.size(0)

        print(f"[{days}天预测] Epoch {epoch+1}/{epochs} Loss: {total_loss / len(train_loader.dataset):.4f}")

    torch.save(model.state_dict(), f'model_transformer_{days}d.pth')
    print(f"模型已保存为 model_transformer_{days}d.pth")

    # =================== 评估 ====================
    model.eval()
    preds, trues = [], []

    with torch.no_grad():
        for x_enc, x_dec, y, mask in test_loader:
            x_enc, x_dec, y, mask = x_enc.to(device), x_dec.to(device), y.to(device), mask.to(device)

            out = model(x_enc, None, x_dec, None, mask=mask)

            preds.append(out.cpu().numpy())
            trues.append(y.cpu().numpy())

    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)

    # 反归一化
    for i in range(preds.shape[2]):
        preds[:, :, i] = preds[:, :, i] * scaler.scale_[i] + scaler.mean_[i]
        trues[:, :, i] = trues[:, :, i] * scaler.scale_[i] + scaler.mean_[i]

    # 打印每个特征的指标
    for i in range(preds.shape[2]):
        pred_i = preds[:, :, i].reshape(-1)
        true_i = trues[:, :, i].reshape(-1)
        mse_val = mean_squared_error(true_i, pred_i)
        mae_val = mean_absolute_error(true_i, pred_i)
        rmse_val = rmse(true_i, pred_i)
        print(f"特征 {configs.features[i]} -> MSE: {mse_val:.4f}, MAE: {mae_val:.4f}, RMSE: {rmse_val:.4f}")

    # 保存预测结果
    data_to_save = {}
    for i in range(preds.shape[2]):
        data_to_save[f'Pred_{configs.features[i]}'] = preds[:, :, i].reshape(-1)
        data_to_save[f'True_{configs.features[i]}'] = trues[:, :, i].reshape(-1)

    df = pd.DataFrame(data_to_save)
    df.to_csv(f'prediction_transformer_{days}d.csv', index=False)
    print(f"[{days}天预测] 测试集结果已保存到 prediction_transformer_{days}d.csv")

if __name__ == '__main__':
    for days in [1,2,3]:  # 预测 1 天和 3 天
        train_and_evaluate(days, lr=1e-4, epochs=6)
