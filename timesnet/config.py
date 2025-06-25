import torch

class Configs:
    def __init__(self):
        
        self.task_name = 'long_term_forecast'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.seq_len = 96         
        self.label_len = 48       
        self.pred_len = 24        

        self.features = ['u10', 'v10', 'd2m', 't2m', 'hcc', 'lcc', 'mcc', 'tp', 'ssr', 'sshf']
        self.target = 'u10'       
        self.enc_in = len(self.features)
        self.dec_in = len(self.features)
        self.c_out = len(self.features)

        self.d_model = 64
        self.n_heads = 4
        self.e_layers = 2
        self.d_ff = 128
        self.dropout = 0.1
        self.activation = 'gelu'

        self.factor = 3
        self.top_k = 3             
        self.num_kernels = 3
        self.embed = 'fixed'
        self.freq = 'h'            

        self.learning_rate = 1e-4
        self.epochs = 10
        self.batch_size = 32
        self.num_workers = 2

        
        self.data_path = './data/weather.csv'