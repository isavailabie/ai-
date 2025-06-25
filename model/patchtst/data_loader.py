import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler

class WeatherDataset(Dataset):
    def __init__(self, data, seq_len, pred_len):
        self.data = data
        self.seq_len = seq_len
        self.pred_len = pred_len

    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len + 1

    def __getitem__(self, idx):
        x = self.data[idx:idx+self.seq_len]
        y = self.data[idx+self.seq_len:idx+self.seq_len+self.pred_len]
        return x, y

def load_weather_data(file_path, configs, split_ratio=0.8):
    df = pd.read_csv(file_path)
    data = df[configs.features].values.astype(np.float32)

    # 划分前8列标准化，后2列归一化
    std_data = data[:, :8]
    norm_data = data[:, 8:]

    std_scaler = StandardScaler()
    norm_scaler = MinMaxScaler()

    std_data = std_scaler.fit_transform(std_data)
    norm_data = norm_scaler.fit_transform(norm_data)

    # 合并处理后的数据
    data = np.concatenate([std_data, norm_data], axis=1)

    # 划分训练集和测试集
    num_train = int(len(data) * split_ratio)
    train_data = data[:num_train]
    test_data = data[num_train:]

    # 构建数据集和数据加载器
    train_dataset = WeatherDataset(train_data, configs.seq_len, configs.pred_len)
    test_dataset = WeatherDataset(test_data, configs.seq_len, configs.pred_len)

    train_loader = DataLoader(train_dataset, batch_size=configs.batch_size, shuffle=True, num_workers=configs.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=configs.batch_size, shuffle=False, num_workers=configs.num_workers)

    return train_loader, test_loader, std_scaler, norm_scaler
