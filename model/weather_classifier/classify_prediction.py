import pandas as pd
import joblib
import numpy as np
import os

# 模型和标签
model_path = os.path.join('model', 'weather_classifier', 'weather_model.joblib')
model = joblib.load(model_path)
features = ['u10', 'v10', 'd2m', 't2m', 'hcc', 'lcc', 'mcc', 'tp']
labels = ['晴', '多云', '阴', '小雨', '中雨', '大雨', '暴雨', '雾', '雨夹雪', '小雪']

def get_time_weight(hour: int) -> float:
    if 0 <= hour <= 6 or 21 <= hour <= 23:
        return 0.3
    else:
        return 1.0

def classify_prediction(csv_path: str):
    """
    给定预测数据路径，返回：
    - weather_classes: 长度为24的天气分类列表（逐小时）
    - weighted_class: 加权后的一天的天气预测（str）
    - weighted_prob: 对应天气概率（float）
    """
    df = pd.read_csv(csv_path, parse_dates=['date'])
    df = df[features + ['date']].copy()

    # 单独天气分类（逐小时）
    weather_classes = []
    hourly_probs = []

    for _, row in df.iterrows():
        x = row[features].to_frame().T
        probs = model.predict_proba(x)[0]
        weather_classes.append(labels[np.argmax(probs)])
        hourly_probs.append(probs)

    # 计算一天的加权天气预测
    sum_probs = np.zeros(len(labels))
    for prob, timestamp in zip(hourly_probs, df['date']):
        w = get_time_weight(timestamp.hour)
        sum_probs += prob * w

    sum_probs /= sum_probs.sum()
    max_idx = np.argmax(sum_probs)
    weighted_class = labels[max_idx]
    weighted_prob = float(sum_probs[max_idx]) * 100

    return weather_classes, weighted_class, weighted_prob
