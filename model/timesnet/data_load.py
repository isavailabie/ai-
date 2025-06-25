import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler

class TimesNetDataset(Dataset):
    def __init__(self, data, seq_len, label_len, pred_len):
        """
        data: 已归一化后的 numpy 数组 (样本数, 特征数)
        """
        self.data = data
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len

    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data[s_begin:s_end]
        seq_y = self.data[r_begin:r_end]

        return torch.tensor(seq_x, dtype=torch.float), torch.tensor(seq_y, dtype=torch.float)

def load_timesnet_data(data_path, seq_len, label_len, pred_len, features='M', batch_size=32, shuffle=True):
    df = pd.read_csv(data_path)

    # 去掉时间列
    data_columns = df.columns.tolist()
    if 'date' in data_columns:
        data_columns.remove('date')

    selected_features = data_columns if features == 'M' else [features]

    raw_data = df[selected_features].values.astype(np.float32)

    # 标准化
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(raw_data)

    dataset = TimesNetDataset(data_scaled, seq_len, label_len, pred_len)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return loader, scaler
