import os
import torch
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from .config import Configs
from .model.PatchTST import Model
from sklearn.preprocessing import StandardScaler

def predict_specific_day(target_date_str, model_path='model/patchtst/model_patchtst_1d.pth'):
    configs = Configs()
    configs.seq_len = 96  # 4天
    configs.pred_len = 24  # 预测未来1天
    configs.device = torch.device("cpu")

    configs.d_model = 512
    configs.d_ff = 2048
    configs.n_heads = 8
    configs.dropout = 0.1

    # Step 1: 读取数据（相对项目根目录）
    data_csv_path = os.path.join('data', 'shanghai_2025weather.csv')
    df = pd.read_csv(data_csv_path)
    df['date'] = pd.to_datetime(df['date'])

    # Step 2: 选择预测目标日期和输入区间
    target_date = pd.to_datetime(target_date_str)
    input_start = target_date - timedelta(days=4)
    input_end = target_date - timedelta(hours=1)

    input_data = df[(df['date'] >= input_start) & (df['date'] <= input_end)].copy()
    if len(input_data) != configs.seq_len:
        raise ValueError(f"输入数据不足 {configs.seq_len} 条（你只有 {len(input_data)} 条），请检查时间范围")

    X = input_data[configs.features].values

    # Step 3: 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = torch.tensor(X_scaled, dtype=torch.float32).unsqueeze(0).to(configs.device)

    # Step 4: 加载模型预测
    model = Model(configs).to(configs.device)
    model.load_state_dict(torch.load(model_path, map_location=configs.device))
    model.eval()

    with torch.no_grad():
        pred = model(X_scaled, None, None, None)
        pred = pred.squeeze(0).cpu().numpy()

    # Step 5: 反归一化
    pred_unscaled = scaler.inverse_transform(pred)

    # Step 6: 组装结果 DataFrame
    hours = pd.date_range(start=target_date, periods=24, freq='h')
    columns = ['date'] + configs.features
    result = pd.DataFrame(pred_unscaled, columns=configs.features)
    result.insert(0, 'date', hours)
    result = result[columns]

    # ✅ 添加 temp 和 humidity 两列
    result['temp'] = result['t2m'] - 273.15
    result['humidity'] = (result['d2m'] - 273.15).clip(lower=0, upper=100)

    # 保存到data目录（新增代码）
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    os.makedirs(data_dir, exist_ok=True)  # 确保目录存在
    save_path = os.path.join(data_dir, f'prediction_{target_date_str}.csv')

    result.to_csv(save_path, index=False)
    print(f"[✅] 已保存预测结果到 {save_path}")

    return result
