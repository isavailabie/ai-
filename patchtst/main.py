import torch
from config import Configs
from model.PatchTST import Model
from data_loader import load_weather_data
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error


torch.cuda.is_available = lambda: False

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def train_and_evaluate(days, lr=1e-3, epochs=5):
    print(f"\n=== 预测未来 {days} 天（{24*days} 小时） ===")

    configs = Configs()
    configs.pred_len = 24 * days

    
    configs.d_model = 512
    configs.n_heads = 8
    configs.d_ff = 2048
    configs.factor = 1
    configs.dropout = 0.1

    train_loader, test_loader, scaler = load_weather_data('./data/weather.csv', configs)

    model = Model(configs).to(configs.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(configs.device)
            batch_y = batch_y.to(configs.device)

            outputs = model(batch_x, None, None, None)
            loss = criterion(outputs, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"[{days}天预测] Epoch {epoch+1} Loss: {total_loss / len(train_loader):.4f}")

    torch.save(model.state_dict(), f'model_patchtst_{days}d.pth')
    print(f"模型已保存为 model_patchtst_{days}d.pth")

    model.eval()
    preds = []
    trues = []

    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(configs.device)
            batch_y = batch_y.to(configs.device)

            output = model(batch_x, None, None, None)

            preds.append(output.cpu().numpy())
            trues.append(batch_y.cpu().numpy())

    preds = np.concatenate(preds, axis=0)  
    trues = np.concatenate(trues, axis=0)

    
    for i in range(preds.shape[2]):
        preds[:, :, i] = preds[:, :, i] * scaler.scale_[i] + scaler.mean_[i]
        trues[:, :, i] = trues[:, :, i] * scaler.scale_[i] + scaler.mean_[i]

    
    for i in range(preds.shape[2]):
        pred_i = preds[:, :, i].reshape(-1)
        true_i = trues[:, :, i].reshape(-1)
        mse_val = mean_squared_error(true_i, pred_i)
        mae_val = mean_absolute_error(true_i, pred_i)
        rmse_val = rmse(true_i, pred_i)
        print(f"特征 {configs.features[i]} 指标 -> MSE: {mse_val:.4f}, MAE: {mae_val:.4f}, RMSE: {rmse_val:.4f}")

    
    data_to_save = {}
    for i in range(preds.shape[2]):
        data_to_save[f'Pred_{configs.features[i]}'] = preds[:, :, i].reshape(-1)
        data_to_save[f'True_{configs.features[i]}'] = trues[:, :, i].reshape(-1)

    df = pd.DataFrame(data_to_save)
    df.to_csv(f'prediction_vs_actual_all_features_{days}d.csv', index=False)
    print(f"[{days}天预测] 测试集所有特征结果已保存到 prediction_vs_actual_all_features_{days}d.csv")

if __name__ == '__main__':
    
    for days in [1,2,3]:
        train_and_evaluate(days, lr=1e-4, epochs=10)  
