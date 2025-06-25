import joblib
import numpy as np
import pandas as pd
# 加载模型
model = joblib.load('weather_model.joblib')

features = ['u10', 'v10', 'd2m', 't2m', 'hcc', 'lcc', 'mcc', 'tp']
labels = ['晴', '多云', '阴', '小雨', '中雨', '大雨', '暴雨', '雾', '雨夹雪', '小雪']

df = pd.read_csv('test.csv', parse_dates=['date'])
df = df[features + ['date']]

def get_time_weight(hour: int) -> float:
    """时间权重，凌晨0-6点和夜晚21-23点权重低，白天权重高"""
    if 0 <= hour <= 6:
        return 0.3  # 凌晨权重较低
    elif 21 <= hour <= 23:
        return 0.3  # 夜晚权重较低
    else:
        return 1.0  # 白天权重正常

# 先加个日期字段（只保留年月日）
df['day'] = df['date'].dt.date

# 存放每一天的加权概率总和
daily_probs = {}

for day, group in df.groupby('day'):
    # 初始化当天概率累加器
    sum_probs = np.zeros(len(labels))
    
    for _, row in group.iterrows():
        input_data = row[features].to_frame().T
        probs = model.predict_proba(input_data)[0]
        w = get_time_weight(row['date'].hour)
        sum_probs += probs * w
    
    # 归一化概率（让它们加起来=1）
    sum_probs /= sum_probs.sum()
    daily_probs[day] = sum_probs

# 输出每天预测概率最高的天气
for day, probs in daily_probs.items():
    top_idx = probs.argmax()
    print(f"{day} 预测天气: {labels[top_idx]}，概率: {probs[top_idx]:.2%}")