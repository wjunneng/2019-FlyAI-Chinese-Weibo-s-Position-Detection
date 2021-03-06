# -*- coding:utf-8 -*-
import os
import sys

# 将路径设置成都当前目录。
os.chdir(sys.path[0])

import torch
import numpy as np
import argparse
import time
import jieba
import torch.nn as nn
import torch.optim as optim
from progress.bar import Bar
from torch.autograd import Variable

import args as arguments
from net import Net
from vec_utils import read_emb
from vec_text import make_datasets, load_tvt
from model_utils import load_torch_model, test, classify_batch
from flyai.dataset import Dataset
from flyai.utils import remote_helper

torch.manual_seed(arguments.seed)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.manual_seed(arguments.seed)


remote_helper.get_remote_date('https://www.flyai.com/m/sgns.weibo.word.bz2')


class StanceDetection(object):
    def __init__(self, exec_type='train'):
        # 项目的超参
        parser = argparse.ArgumentParser()
        parser.add_argument("-e", "--EPOCHS", default=50, type=int, help="train epochs")
        parser.add_argument("-b", "--BATCH", default=2, type=int, help="batch size")
        self.args = parser.parse_args()
        self.dataset = Dataset(epochs=self.args.EPOCHS, batch=self.args.BATCH)
        self.model_dir = os.path.join(os.getcwd(), arguments.model_dir)

        # 1. Split the data, read into defined format
        label2idx = dict((arguments.labels[i], i) for i in range(len(arguments.labels)))
        target_text, stance, _, _ = self.dataset.get_all_data()

        indexes = [" ".join(jieba.cut(i['TARGET'].lower(), cut_all=False)) for i in target_text]
        questions = [" ".join(jieba.cut(i['TEXT'].lower(), cut_all=False)) for i in target_text]
        labels = [i['STANCE'] for i in stance]
        data = [indexes, questions, labels]
        assert len(data[0]) == len(data[1]) == len(data[2])

        # 2. Data follows this order: train, test
        train_num = int(len(data[0]) * arguments.portion)
        train_data = [d[:train_num] for d in data]
        dev_data = [d[train_num:] for d in data]

        # 3. Read the vocab text file and get VOCAB dictionary
        vocab = read_emb(filename=os.path.join(os.getcwd(), arguments.sgns_dir), stat_lines=1)

        # 4. Transform text into indexes
        self.datasets, word2idx, embeddings = make_datasets(vocab=vocab,
                                                            raw_data={'training': train_data, 'validation': dev_data},
                                                            label2idx=label2idx,
                                                            big_voc=arguments.big_voc,
                                                            feat_names=arguments.feat_names)
        self.datasets_train = load_tvt(tvt_set=self.datasets['training'],
                                       max_lens=[arguments.ans_len, arguments.ask_len],
                                       feat_names=arguments.feat_names)
        self.datasets_dev = load_tvt(tvt_set=self.datasets['validation'],
                                     max_lens=[arguments.ans_len, arguments.ask_len],
                                     feat_names=arguments.feat_names)

        idx2word = dict((v, k) for k, v in word2idx.items())
        self.datasets["word2idx"] = word2idx
        self.datasets["idx2word"] = idx2word

        self.embeddings = torch.from_numpy(np.asarray(embeddings, dtype=np.float32))

        if exec_type == 'train':
            self.main()
        else:
            model = load_torch_model(self.model_dir)
            test(model=model, dataset=self.datasets, test_set=None)

    def main(self):
        """ continue training or not """
        if arguments.proceed:
            if os.path.exists(self.model_dir):
                with open(self.model_dir, "rb") as saved_model:
                    model = torch.load(saved_model)
        else:
            models = {"Net": Net}
            model = models[arguments.model](embeddings=self.embeddings,
                                            input_dim=self.embeddings.size(1),
                                            hidden_dim=arguments.nhid,
                                            num_layers=arguments.nlayers,
                                            output_dim=arguments.nclass,
                                            max_step=[arguments.ans_len, arguments.ask_len],
                                            dropout=arguments.dropout)
            if arguments.model in ["Net"]:
                model.nhops = arguments.nhops

        # train
        model.to(device=DEVICE)
        # 优化器
        optimizer = optim.Adam(model.parameters(), lr=arguments.lr, weight_decay=5e-5)
        # 损失函数
        criterion = nn.CrossEntropyLoss()

        best_f1_test, best_p_valid, best_f1_valid = -np.inf, -np.inf, -np.inf
        epoch_f1_test, epoch_f1_valid, epoch_f1_cur = 0, 0, 0
        batches_per_epoch = len(self.datasets_train) // self.args.BATCH
        max_train_steps = int(self.args.EPOCHS * batches_per_epoch)

        print("--------------\nEpoch 0 begins!")
        bar = Bar("  Processing", max=max_train_steps)
        print(max_train_steps, self.args.EPOCHS, len(self.datasets_train), self.args.BATCH)

        for step in range(max_train_steps):
            bar.next()
            training_batch = self.datasets_train.next_batch(self.args.BATCH)
            features, seq_lens, mask_matrice, labels = training_batch
            (answers, answers_seqlen, answers_mask), (questions, questions_seqlen, questions_mask) \
                = zip(features, seq_lens, mask_matrice)

            assert self.args.BATCH == len(labels) == len(questions) == len(answers)

            # Prepare data and prediction
            labels_ = Variable(torch.LongTensor(labels)).to(DEVICE)

            # necessary for Room model
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=0.25)

            # zero grad
            model.train()
            model.zero_grad()
            outputs = classify_batch(model=model,
                                     features=[answers, answers_seqlen, answers_mask, questions, questions_seqlen,
                                               questions_mask],
                                     max_lens=(arguments.ans_len, arguments.ask_len))
            loss = criterion(outputs[0].view(len(labels_), -1), labels_)

            loss.backward()
            optimizer.step()

            # Test after each epoch
            if (step + 1) % batches_per_epoch == 0:
                tic = time.time()
                f1_score, p_score = test(model=model,
                                         log_result=False,
                                         dataset=self.datasets,
                                         test_set=self.datasets_dev,
                                         batch_size=self.args.BATCH)

                print("\n  Begin to predict the results on Valid")
                print("  using %.5fs" % (time.time() - tic))
                print("  ----Old best F1 on Valid is %f on epoch %d" % (best_f1_valid, epoch_f1_valid))
                print("  ----Old best F1 on Test is %f on epoch %d" % (best_f1_test, epoch_f1_test))

                if f1_score > best_f1_valid:
                    with open(self.model_dir, 'wb') as to_save:
                        torch.save(model, to_save)

                    best_f1_valid = f1_score
                    print("  ----New best F1 on Valid is %f" % f1_score)
                    epoch_f1_valid = self.datasets_train.epochs_completed
                print("--------------\nEpoch %d begins!" % (self.datasets_train.epochs_completed + 1))

        bar.finish()


if __name__ == "__main__":
    print("------------------------------")
    StanceDetection('train')
