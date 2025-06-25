import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

def generate_square_subsequent_mask(sz):
    # 下三角矩阵掩码，True表示遮挡，False表示可见
    return torch.triu(torch.ones(sz, sz), diagonal=1).bool()

class WeatherDataset(Dataset):
    def __init__(self, data, seq_len, label_len, pred_len):
        """
        data: 归一化后的 numpy 数组，shape [时间, 特征数]
        seq_len: encoder输入长度
        label_len: decoder已知序列长度（历史观测）
        pred_len: decoder预测长度（未来）
        """
        self.data = data
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len

    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len + 1

    def __getitem__(self, idx):
        # Encoder 输入：过去 seq_len 步数据
        x_enc = self.data[idx:idx+self.seq_len]

        # Decoder 输入 = 已知历史 label_len + 预测的 pred_len 的占位（未来部分通常用0填充）
        # 先取 label_len 个真实历史，后面 pred_len 个用0补
        x_dec_known = self.data[idx+self.seq_len - self.label_len: idx+self.seq_len]
        x_dec_pred = np.zeros((self.pred_len, self.data.shape[1]), dtype=np.float32)
        x_dec = np.concatenate([x_dec_known, x_dec_pred], axis=0)

        # 目标预测 y，decoder预测目标是未来 pred_len
        y = self.data[idx+self.seq_len: idx+self.seq_len+self.pred_len]

        # 转成tensor
        x_enc = torch.tensor(x_enc, dtype=torch.float32)
        x_dec = torch.tensor(x_dec, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)

        # 生成decoder掩码，shape (label_len+pred_len, label_len+pred_len)
        mask = generate_square_subsequent_mask(self.label_len + self.pred_len)

        return x_enc, x_dec, y, mask

def load_weather_data(file_path, configs, split_ratio=0.8):
    df = pd.read_csv(file_path)
    data = df[configs.features].values.astype(np.float32)

    scaler = StandardScaler()
    data = scaler.fit_transform(data)

    num_train = int(len(data) * split_ratio)
    train_data = data[:num_train]
    test_data = data[num_train:]

    train_dataset = WeatherDataset(train_data, configs.seq_len, configs.label_len, configs.pred_len)
    test_dataset = WeatherDataset(test_data, configs.seq_len, configs.label_len, configs.pred_len)

    train_loader = DataLoader(train_dataset, batch_size=configs.batch_size, shuffle=True, num_workers=configs.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=configs.batch_size, shuffle=False, num_workers=configs.num_workers)

    return train_loader, test_loader, scaler
