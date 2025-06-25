import torch
class Configs:
    def __init__(self):
        self.task_name = 'long_term_forecast'

        self.seq_len = 96  # 输入长度：过去4天，每小时1条，共96小时
        self.pred_len = 24  # 预测1天(24小时)，可改成48、72

        self.d_model = 64
        self.n_heads = 4
        self.d_ff = 128
        self.e_layers = 2
        self.dropout = 0.1
        self.activation = 'gelu'
        self.enc_in = 10  # 10个气象特征字段（去掉时间字段）
        self.factor = 3

        self.batch_size = 32
        self.num_workers = 2
        self.device = torch.device("cpu")

        # 你的气象字段（除时间外）
        self.features = ['u10', 'v10', 'd2m', 't2m', 'hcc', 'lcc', 'mcc', 'tp', 'ssr', 'sshf']
