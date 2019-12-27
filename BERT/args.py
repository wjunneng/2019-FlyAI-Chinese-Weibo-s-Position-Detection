# -------------------------arg-------------------------
import torch
from BERT.models import LSTM, IAN, MemNet, RAM, TD_LSTM, Cabasc, ATAE_LSTM, TNet_LF, AOA, MGAN, LCF_BERT, BERT_SPC, \
    AEN_BERT

# ############################### model parameters
model_name = 'bert_spc'
dataset = 'evasampledata4'
optimizer = 'adam'
initializer = 'xavier_uniform_'
# 'try 5e-5, 2e-5 for BERT, 1e-3 for others'
learning_rate = 2e-5
dropout = 0.1
l2reg = 0.01
# try larger number for non-BERT models
num_epoch = 10
# try 16, 32, 64 for BERT models
batch_size = 16
log_step = 5
embed_dim = 300
hidden_dim = 300
bert_dim = 768
# default='bert-base-uncased'
pretrained_bert_name = '/home/wjunneng/Ubuntu/2019-FlyAI-Chinese-Weibo-s-Position-Detection/BERT/pretrained_model'
max_seq_len = 80
polarities_dim = 3
hops = 3
# e.g. cuda:0
device = None
# set seed for reproducibility
seed = None
# set ratio between 0 and 1 for validation support
valset_ratio = 0
# local context focus mode, cdw or cdm
local_context_focus = 'cdm'
# semantic-relative-distance, see the paper of LCF-BERT model
SRD = 3

# ############################### other parameters
# default hyper-parameters for LCF-BERT model is as follws:
# lr: 2e-5
# l2: 1e-5
# batch size: 16
# num epochs: 5
model_classes = {
    'lstm': LSTM,
    'td_lstm': TD_LSTM,
    'atae_lstm': ATAE_LSTM,
    'ian': IAN,
    'memnet': MemNet,
    'ram': RAM,
    'cabasc': Cabasc,
    'tnet_lf': TNet_LF,
    'aoa': AOA,
    'mgan': MGAN,
    'bert_spc': BERT_SPC,
    'aen_bert': AEN_BERT,
    'lcf_bert': LCF_BERT
}
dataset_files = {
    'evasampledata4': {
        'train': '../data/evasampledata4-TaskAA.txt'
    }
}
input_colses = {
    'lstm': ['text_raw_indices'],
    'td_lstm': ['text_left_with_aspect_indices', 'text_right_with_aspect_indices'],
    'atae_lstm': ['text_raw_indices', 'aspect_indices'],
    'ian': ['text_raw_indices', 'aspect_indices'],
    'memnet': ['text_raw_without_aspect_indices', 'aspect_indices'],
    'ram': ['text_raw_indices', 'aspect_indices', 'text_left_indices'],
    'cabasc': ['text_raw_indices', 'aspect_indices', 'text_left_with_aspect_indices',
               'text_right_with_aspect_indices'],
    'tnet_lf': ['text_raw_indices', 'aspect_indices', 'aspect_in_text'],
    'aoa': ['text_raw_indices', 'aspect_indices'],
    'mgan': ['text_raw_indices', 'aspect_indices', 'text_left_indices'],
    'bert_spc': ['text_bert_indices', 'bert_segments_ids'],
    'aen_bert': ['text_raw_bert_indices', 'aspect_bert_indices'],
    'lcf_bert': ['text_bert_indices', 'bert_segments_ids', 'text_raw_bert_indices', 'aspect_bert_indices'],
}
initializers = {
    'xavier_uniform_': torch.nn.init.xavier_uniform_,
    'xavier_normal_': torch.nn.init.xavier_normal,
    'orthogonal_': torch.nn.init.orthogonal_,
}
optimizers = {
    'adadelta': torch.optim.Adadelta,  # default lr=1.0
    'adagrad': torch.optim.Adagrad,  # default lr=0.01
    'adam': torch.optim.Adam,  # default lr=0.001
    'adamax': torch.optim.Adamax,  # default lr=0.002
    'asgd': torch.optim.ASGD,  # default lr=0.01
    'rmsprop': torch.optim.RMSprop,  # default lr=0.01
    'sgd': torch.optim.SGD,
}
