import torch

class Configs:
    def __init__(self):
        self.task_name = 'long_term_forecast'

        self.seq_len = 96         # encoder 输入长度
        self.label_len = 48       # decoder 已知历史长度
        self.pred_len = 24        # decoder 要预测的长度

        self.d_model = 64
        self.n_heads = 4
        self.d_ff = 128
        self.e_layers = 2
        self.d_layers = 1         # decoder 层数（你模型里用到了）
        self.dropout = 0.1
        self.activation = 'gelu'
        self.factor = 3           # Attention 稀疏因子

        self.enc_in = 10          # 输入维度
        self.dec_in = 10          # decoder 输入维度
        self.c_out = 10           # 输出维度（预测多少个特征）

        self.embed = 'timeF'      # embedding 类型（例如 'timeF'、'fixed' 等）
        self.freq = 'h'           # 时间粒度，比如 'h' 表示每小时一个时间点

        self.batch_size = 32
        self.num_workers = 2
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 特征字段名，记得和 CSV 文件里的列名保持一致
        self.features = ['u10', 'v10', 'd2m', 't2m', 'hcc', 'lcc', 'mcc', 'tp', 'ssr', 'sshf']
