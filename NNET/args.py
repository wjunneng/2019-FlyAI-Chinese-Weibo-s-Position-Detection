# -*- coding:utf-8 -*-
import sys

sys.path.append('../')

# -------------------------arg-------------------------
# model save dir
model_dir = 'data/input/model.pt'
# decide portion to spare for training and validation
portion = 0.8
# sgns_300 vector path
sgns_dir = 'data/input/model/sgns'
# all of labels
labels = ['NONE', 'FAVOR', 'AGAINST']
# feat names
feat_names = ["xIndexes", "xQuestions", "yLabels"]
# use the all vocabulary
big_voc = False
# max time step of sentence sequence
sen_max_len = 50
# max time step of sentence sequence
ask_max_len = 25

# ############################### model parameters
# type of model to use for Stance Project
model = "Net"
# number of classes to predict
nclass = 3
# size of RNN hidden layer
nhid = 50
# number of layers of LSTM
nlayers = 1
# learning rate
lr = 1e-4
# dropout rate
dropout = 0.5
# max time step of answer sequence
ans_len = 50
# max time step of question sequence
ask_len = 25
# number of attention hops for RoomConditional models
nhops = 3

# ############################## test purpose
# random seed for reproduction
seed = 123456
# flag for training model or only test
is_test = False
# flag for continue training on current model
proceed = False
