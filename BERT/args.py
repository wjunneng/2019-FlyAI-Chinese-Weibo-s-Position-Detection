# -------------------------arg-------------------------
import os
import sys

os.chdir(sys.path[0])
import torch

from models.lstm import LSTM
from models.ian import IAN
from models.memnet import MemNet
from models.ram import RAM
from models.td_lstm import TD_LSTM
from models.cabasc import Cabasc
from models.atae_lstm import ATAE_LSTM
from models.tnet_lf import TNet_LF
from models.aoa import AOA
from models.mgan import MGAN
from models.lcf_bert import LCF_BERT
from models.bert_spc import BERT_SPC
from net import Net

# ############################### model parameters
dataset = 'evasampledata4'
data_type = 'csv'
labels = ['NONE', 'FAVOR', 'AGAINST']

# topics = ['IphoneSE', '春节放鞭炮', '深圳禁摩限电', '俄罗斯在叙利亚的反恐行动', '开放二胎']
topics = None

# dataset = 'acl-14-short-data'
# data_type = 'txt'

# default='bert-base-uncased'
pretrained_bert_name = 'data/input/model'
# 最优模型保存路径
best_model_path = 'data/input'
# log path
log_path = 'log'
# 模型名称
model_name = 'lcf_bert'
# 优化算法
optimizer = 'adam'
# 初始化方式
initializer = 'xavier_uniform_'
# 'try 5e-5, 2e-5 for BERT, 1e-3 for others'
learning_rate = 2e-5
# 随机失活率
dropout = 0.1
# 权重
l2reg = 0.01
# 步长
log_step = 5
# 嵌入的维度
embed_dim = 300
# 隐藏层神经元个数
hidden_dim = 300
# bert 维度
bert_dim = 768
# 序例最大的长度
max_seq_len = 140
# 极性的维度
polarities_dim = 3
# hops
hops = 3
# e.g. cuda:0
device = None
# set seed for reproducibility
seed = None
# set ratio between 0 and 1 for validation support
valset_ratio = 0.2
# local context focus mode, cdw or cdm
local_context_focus = 'cdm'
# semantic-relative-distance, see the paper of LCF-BERT model
SRD = 3
# k-fold cross validation
cross_val_fold = 8

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
    'lcf_bert': LCF_BERT,
    'aen_bert': Net
}
dataset_files = {
    'evasampledata4': {
        'train': '../data/evasampledata4/evasampledata4-TaskAA.txt'
    },
    'acl-14-short-data': {
        'train': '../data/acl-14-short-data/train.raw',
        'test': '../data/acl-14-short-data/test.raw'
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
