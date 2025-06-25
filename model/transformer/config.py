import torch

class Configs:
    def __init__(self):
        self.task_name = 'long_term_forecast'

        self.seq_len = 96         
        self.label_len = 48       
        self.pred_len = 24        

        self.d_model = 64
        self.n_heads = 4
        self.d_ff = 128
        self.e_layers = 2
        self.d_layers = 1         
        self.dropout = 0.1
        self.activation = 'gelu'
        self.factor = 3           

        self.enc_in = 10          
        self.dec_in = 10          
        self.c_out = 10           

        self.embed = 'timeF'      
        self.freq = 'h'           

        self.batch_size = 32
        self.num_workers = 2
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        
        self.features = ['u10', 'v10', 'd2m', 't2m', 'hcc', 'lcc', 'mcc', 'tp', 'ssr', 'sshf']
