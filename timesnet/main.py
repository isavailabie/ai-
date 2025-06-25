import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from config import Configs
from model.TimesNet import Model as TimesNet
from data_load import load_timesnet_data
from sklearn.metrics import mean_squared_error, mean_absolute_error

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def train_and_evaluate(days, lr=1e-4, epochs=10):
    print(f"\n=== TimesNet 预测未来 {days} 天（{24*days} 小时） ===")

    configs = Configs()
    configs.pred_len = 24 * days
    configs.learning_rate = lr
    configs.epochs = epochs

    # 加载数据，并划分为训练集和测试集（0.8 / 0.2）
    train_loader, test_loader, train_scaler = load_timesnet_data(
        data_path=configs.data_path,
        seq_len=configs.seq_len,
        label_len=configs.label_len,
        pred_len=configs.pred_len,
        features='M',
        batch_size=configs.batch_size,
        shuffle=True,
        split_ratio=0.8
    )

    model = TimesNet(configs).to(configs.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(configs.device)
            batch_y = batch_y[:, -configs.pred_len:, :].to(configs.device)

            output = model(batch_x, None, None, None)
            loss = criterion(output, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"[{days}天预测] Epoch {epoch+1}/{epochs} Loss: {total_loss / len(train_loader):.4f}")

    torch.save(model.state_dict(), f'model_timesnet_{days}d.pth')
    print(f"模型已保存为 model_timesnet_{days}d.pth")

    # 开始评估
    model.eval()
    preds = []
    trues = []
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(configs.device)
            batch_y = batch_y[:, -configs.pred_len:, :].to(configs.device)

            output = model(batch_x, None, None, None)

            preds.append(output.cpu().numpy())
            trues.append(batch_y.cpu().numpy())

    preds = np.concatenate(preds, axis=0)  # (B, pred_len, C)
    trues = np.concatenate(trues, axis=0)

    # ===== 反归一化 =====
    shape = preds.shape
    preds = train_scaler.inverse_transform(preds.reshape(-1, shape[2])).reshape(shape)
    trues = train_scaler.inverse_transform(trues.reshape(-1, shape[2])).reshape(shape)

    # ===== 计算指标 =====
    for i in range(preds.shape[2]):
        pred_i = preds[:, :, i].reshape(-1)
        true_i = trues[:, :, i].reshape(-1)
        mse_val = mean_squared_error(true_i, pred_i)
        mae_val = mean_absolute_error(true_i, pred_i)
        rmse_val = rmse(true_i, pred_i)
        print(f"特征 {configs.features[i]} 指标 -> MSE: {mse_val:.4f}, MAE: {mae_val:.4f}, RMSE: {rmse_val:.4f}")

    # ===== 保存预测结果 =====
    data_to_save = {}
    for i in range(preds.shape[2]):
        data_to_save[f'Pred_{configs.features[i]}'] = preds[:, :, i].reshape(-1)
        data_to_save[f'True_{configs.features[i]}'] = trues[:, :, i].reshape(-1)

    df = pd.DataFrame(data_to_save)
    df.to_csv(f'prediction_vs_actual_timesnet_{days}d.csv', index=False)
    print(f"[{days}天预测] 预测结果已保存到 prediction_vs_actual_timesnet_{days}d.csv")




if __name__ == '__main__':
    for days in [1,2,3]:
        train_and_evaluate(days, lr=1e-4, epochs=6)
