import os
import torch
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from .config import Configs
from .model.Transformer import Model as Transformer
from sklearn.preprocessing import StandardScaler

def predict_with_transformer(target_date_str, model_path='model/transformer/model_transformer_1d.pth'):
    # 初始化配置
    configs = Configs()
    configs.seq_len = 96
    configs.pred_len = 24
    configs.label_len = configs.seq_len // 2
    configs.d_model = 512
    configs.n_heads = 8
    configs.d_ff = 2048
    configs.factor = 1
    configs.dropout = 0.1
    configs.device = torch.device("cpu")  # ✅ 强制使用 CPU，兼容网页

    # Step 1: 加载数据
    data_csv_path = os.path.join('data', 'shanghai_2025weather.csv')
    df = pd.read_csv(data_csv_path)
    df['date'] = pd.to_datetime(df['date'])

    # Step 2: 获取输入数据
    target_date = pd.to_datetime(target_date_str)
    input_start = target_date - timedelta(days=4)
    input_end = target_date - timedelta(hours=1)
    input_data = df[(df['date'] >= input_start) & (df['date'] <= input_end)].copy()

    if len(input_data) != configs.seq_len:
        raise ValueError(f"需要 {configs.seq_len} 小时数据，但只找到 {len(input_data)} 小时")

    X = input_data[configs.features].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Step 3: 构造模型输入
    x_enc = torch.tensor(X_scaled, dtype=torch.float32).unsqueeze(0).to(configs.device)
    x_dec = torch.zeros((1, configs.pred_len, X_scaled.shape[1]), dtype=torch.float32).to(configs.device)
    mask = torch.ones((1, configs.seq_len, configs.seq_len), dtype=torch.float32).to(configs.device)

    # Step 4: 加载 Transformer 模型并预测
    model = Transformer(configs).to(configs.device)
    model.load_state_dict(torch.load(model_path, map_location=configs.device))
    model.eval()

    with torch.no_grad():
        pred = model(x_enc, None, x_dec, None, mask=mask)
        pred = pred.squeeze(0).cpu().numpy()

    # Step 5: 反归一化
    for i in range(pred.shape[1]):
        pred[:, i] = pred[:, i] * scaler.scale_[i] + scaler.mean_[i]

    # Step 6: 组装结果
    hours = pd.date_range(start=target_date, periods=configs.pred_len, freq='h')
    result = pd.DataFrame(pred, columns=configs.features)
    result.insert(0, 'date', hours)

    if 't2m' in configs.features:
        result['temp'] = result['t2m'] - 273.15
    if 'd2m' in configs.features:
        result['humidity'] = (result['d2m'] - 273.15).clip(0, 100)

    # Step 7: 保存结果
    data_dir = os.path.join('data', 'predictions')
    os.makedirs(data_dir, exist_ok=True)
    save_path = os.path.join(data_dir, f'transformer_prediction_{target_date_str}.csv')
    result.to_csv(save_path, index=False)
    print(f"[✅] Transformer预测结果已保存到 {save_path}")

    return result
