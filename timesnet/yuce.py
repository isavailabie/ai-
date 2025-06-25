import os
import torch
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from config import Configs
from model.TimesNet import Model as TimesNet
from sklearn.preprocessing import StandardScaler

def predict_with_timesnet(target_date_str, model_path='model_timesnet_1d.pth'):
    # 初始化配置
    configs = Configs()
    configs.seq_len = 96  # 4天的数据 (24*4=96小时)
    configs.pred_len = 24  # 预测未来24小时(1天)
    configs.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Step 1: 读取数据
    data_csv_path = os.path.join('data', 'shanghai_2025weather.csv')  # 根据你的实际路径调整
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
    X_scaled = torch.tensor(X_scaled, dtype=torch.float32).unsqueeze(0).to(configs.device)  # 添加batch维度
    
    # Step 4: 加载TimesNet模型并进行预测
    model = TimesNet(configs).to(configs.device)
    model.load_state_dict(torch.load(model_path, map_location=configs.device))
    model.eval()
    
    with torch.no_grad():
        # TimesNet的forward参数顺序可能与PatchTST不同，请根据实际模型定义调整
        pred = model(X_scaled, None, None, None)
        pred = pred.squeeze(0).cpu().numpy()  # 移除batch维度
    
    # Step 5: 反归一化
    pred_unscaled = scaler.inverse_transform(pred)
    
    # Step 6: 组装结果DataFrame
    hours = pd.date_range(start=target_date, periods=24, freq='h')
    columns = ['date'] + configs.features
    result = pd.DataFrame(pred_unscaled, columns=configs.features)
    result.insert(0, 'date', hours)
    result = result[columns]
    
    # 可选：添加转换后的温度和湿度列（如果需要）
    if 't2m' in configs.features:
        result['temp'] = result['t2m'] - 273.15
    if 'd2m' in configs.features:
        result['humidity'] = (result['d2m'] - 273.15).clip(lower=0, upper=100)
    
    # 保存结果
    data_dir = os.path.join('data', 'predictions')
    os.makedirs(data_dir, exist_ok=True)
    save_path = os.path.join(data_dir, f'timesnet_prediction_{target_date_str}.csv')
    result.to_csv(save_path, index=False)
    print(f"预测结果已保存到 {save_path}")
    
    return result

# 使用示例
if __name__ == '__main__':
    # 预测2025-01-01的天气
    prediction = predict_with_timesnet("2025-01-06", model_path="model_timesnet_1d.pth")
    print(prediction.head())