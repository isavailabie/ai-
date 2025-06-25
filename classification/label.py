import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib



data = pd.read_csv("labeled_weather.csv")  
X = data.drop(["date", "label"], axis=1)  
y = data["label"]                         


labels = ['晴', '多云', '阴', '小雨', '中雨', '大雨', '暴雨', '雾', '雨夹雪', '小雪']



base_cost_matrix = np.array([
    [0,   1,   2,   4,   5,   6,   7,   5,   6,    7],
    [1,   0,   1,   3,   4,   5,   6,   4,   5,    6],
    [2,   1,   0,   2,   3,   4,   5,   4,   4,    5],
    [4,   3,   2,   0,   1,   2,   3,   4,   3,    4],
    [5,   4,   3,   1,   0,   1,   2,   5,   2,    3],
    [6,   5,   4,   2,   1,   0,   1,   6,   3,    3],
    [7,   6,   5,   3,   2,   1,   0,   7,   4,    4],
    [5,   4,   4,   4,   5,   6,   7,   0,   5,    5],
    [6,   5,   4,   3,   2,   3,   4,   5,   0,    1],
    [7,   6,   5,   4,   3,   3,   4,   5,   1,    0]
])


cost_matrix = np.where(base_cost_matrix == 0, 0, base_cost_matrix + 4)

print(cost_matrix)


def weighted_accuracy(y_true, y_pred, cost_matrix, labels):
    """计算考虑天气相似度的加权准确率"""
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    total_cost = np.sum(cm * cost_matrix)
    max_cost = np.sum(cm) * np.max(cost_matrix)
    return 1 - (total_cost / max_cost), total_cost


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)



sample_weights = np.array([1 / (cost_matrix[labels.index(y), :].mean()) for y in y_train])


model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    class_weight='balanced',  
    random_state=42
)


model.fit(X_train, y_train, sample_weight=sample_weights)


y_pred = model.predict(X_test)

print("=== 标准分类报告 ===")
print(classification_report(y_test, y_pred, target_names=labels))

print("\n=== 代价敏感评估 ===")
acc, penalty = weighted_accuracy(y_test, y_pred, cost_matrix, labels)
print(f"加权准确率: {acc:.4f}")
print(f"总惩罚分数: {penalty}")



joblib.dump(model, 'weather_model.joblib')


import json
with open('feature_columns.json', 'w') as f:
    json.dump(X_train.columns.tolist(), f)


np.savez('model_meta.npz', labels=labels, cost_matrix=cost_matrix)

print("模型及元数据已保存！")