import torch

class Configs:
    def __init__(self):
        # 基本设置
        self.task_name = 'long_term_forecast'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 输入输出序列长度
        self.seq_len = 96         # 输入长度（4 天）
        self.label_len = 48       # decoder 输入长度
        self.pred_len = 24        # 预测长度（1 天），可在训练函数中覆盖

        # 特征设置
        self.features = ['u10', 'v10', 'd2m', 't2m', 'hcc', 'lcc', 'mcc', 'tp', 'ssr', 'sshf']
        self.target = 'u10'       # 预测目标（也可以是全量）
        self.enc_in = len(self.features)
        self.dec_in = len(self.features)
        self.c_out = len(self.features)

        # 模型结构参数
        self.d_model = 64
        self.n_heads = 4
        self.e_layers = 2
        self.d_ff = 128
        self.dropout = 0.1
        self.activation = 'gelu'


        self.factor = 3
        self.top_k = 3             # 找3个主周期
        self.num_kernels = 3
        self.embed = 'fixed'
        self.freq = 'h'            # 每小时


        # 训练参数（可在调用时覆盖）
        self.learning_rate = 1e-4
        self.epochs = 10
        self.batch_size = 32
        self.num_workers = 2

        # 数据路径
        self.data_path = './data/weather.csv'