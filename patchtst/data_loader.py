import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

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

    scaler = StandardScaler()
    data = scaler.fit_transform(data)

    num_train = int(len(data) * split_ratio)
    train_data = data[:num_train]
    test_data = data[num_train:]

    train_dataset = WeatherDataset(train_data, configs.seq_len, configs.pred_len)
    test_dataset = WeatherDataset(test_data, configs.seq_len, configs.pred_len)

    train_loader = DataLoader(train_dataset, batch_size=configs.batch_size, shuffle=True, num_workers=configs.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=configs.batch_size, shuffle=False, num_workers=configs.num_workers)

    # 多返回 mean 和 std，用于反归一化
    return train_loader, test_loader, scaler

