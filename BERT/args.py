# -------------------------arg-------------------------
# ############################### model parameters
model_name = 'bert_spc'
dataset = 'laptop'
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
pretrained_bert_name = '/home/wjunneng/Ubuntu/NLP/情感分析/ABSA-PyTorch/pytorched_pretrain'
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
