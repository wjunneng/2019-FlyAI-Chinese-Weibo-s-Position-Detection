# -*- coding:utf-8 -*-
import sys

sys.path.append('../')

import os
import torch
import numpy as np
import argparse
import time
import torch.nn as nn
import torch.optim as optim
from progress.bar import Bar
from torch.autograd import Variable

from NNET.net import Net
import NNET.args as arguments
from NNET.utils.file_utils import pickle_to_data
from NNET.utils.model_utils import load_torch_model, test, classify_batch

from flyai.dataset import Dataset

torch.manual_seed(arguments.seed)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.manual_seed(arguments.seed)


class StanceDetection(object):
    def __init__(self, exec_type='train'):
        # 项目的超参
        parser = argparse.ArgumentParser()
        parser.add_argument("-e", "--EPOCHS", default=10, type=int, help="train epochs")
        parser.add_argument("-b", "--BATCH", default=8, type=int, help="batch size")
        self.args = parser.parse_args()
        self.dataset = Dataset(epochs=self.args.EPOCHS, batch=self.args.BATCH)

        if exec_type == 'train':
            pass
        elif exec_type == 'validation':
            pass
        else:
            self.features_data = pickle_to_data(arguments.features_baike_dir)
            word2idx = pickle_to_data(arguments.word2idx_baike_dir)
            idx2word = dict((v, k) for k, v in word2idx.items())
            self.features_data["word2idx"] = word2idx
            self.features_data["idx2word"] = idx2word
            model = load_torch_model(arguments.model_dir)
            test(model, self.features_data)

    def main(self):
        """ make sure the folder to save models exist """
        if not os.path.exists(arguments.model_dir):
            os.mkdir(arguments.model_dir)

        """ continue training or not """
        if arguments.proceed:
            if os.path.exists(arguments.model_dir + "/model.pt"):
                with open(arguments.model_dir + "/model.pt", "rb") as saved_model:
                    model = torch.load(saved_model)
        else:
            embeddings = pickle_to_data(arguments.embeddings_baike_dir)
            # from_numpy
            emb_np = np.asarray(embeddings, dtype=np.float32)
            emb = torch.from_numpy(emb_np)

            models = {"Net": Net}
            model = models[arguments.model](embeddings=emb,
                                            input_dim=emb.size(1),
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
        # 加载训练集
        training_set = pickle_to_data(arguments.features_baike_training_dir)

        best_f1_test, best_p_valid, best_f1_valid = -np.inf, -np.inf, -np.inf
        epoch_f1_test, epoch_f1_valid, epoch_f1_cur = 0, 0, 0
        cur_f1_test = -np.inf
        batches_per_epoch = len(training_set) // self.args.BATCH
        max_train_steps = int(self.args.EPOCHS * batches_per_epoch)

        print("--------------\nEpoch 0 begins!")
        bar = Bar("  Processing", max=max_train_steps)
        print(max_train_steps, self.args.EPOCHS, len(training_set), self.args.BATCH)

        for step in range(max_train_steps):
            bar.next()
            training_batch = training_set.next_batch(self.args.batch_size)
            model.train()
            # xIndexes, xQuestions, yLabels
            features, seq_lens, mask_matrice, labels = training_batch
            (answers, answers_seqlen, answers_mask), (questions, questions_seqlen, questions_mask) \
                = zip(features, seq_lens, mask_matrice)
            assert self.args.BATCH == len(labels) == len(questions) == len(answers)
            feats = [answers, answers_seqlen, answers_mask, questions, questions_seqlen, questions_mask]

            # Prepare data and prediction
            labels_ = Variable(torch.LongTensor(labels)).to(DEVICE)

            # necessary for Room model
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=0.25)

            model.zero_grad()
            outputs = classify_batch(model=model, features=feats, max_lens=(arguments.ans_len, arguments.ask_len))
            probs = outputs[0]
            loss = criterion(probs.view(len(labels_), -1), labels_)

            loss.backward()
            optimizer.step()

            # Test after each epoch
            if (step + 1) % batches_per_epoch == 0:
                tic = time.time()
                self.features_data = pickle_to_data(arguments.features_baike_dir)
                word2idx = pickle_to_data(arguments.word2idx_baike_dir)
                idx2word = dict((v, k) for k, v in word2idx.items())
                self.features_data["word2idx"] = word2idx
                self.features_data["idx2word"] = idx2word
                f1_score, p_score = test(model, log_result=False, dataset=self.features_data, data_part="validation")
                print("\n  Begin to predict the results on Valid")
                print("  using %.5fs" % (time.time() - tic))
                print("  ----Old best F1 on Valid is %f on epoch %d" % (best_f1_valid, epoch_f1_valid))
                print("  ----Old best F1 on Test is %f on epoch %d" % (best_f1_test, epoch_f1_test))
                print("  ----Cur save F1 on Test is %f on epoch %d" % (cur_f1_test, epoch_f1_valid))
                if f1_score > best_f1_valid:
                    with open(arguments.model_dir + "/model.pt", 'wb') as to_save:
                        torch.save(model, to_save)
                    best_f1_valid = f1_score
                    print("  ----New best F1 on Valid is %f" % f1_score)
                    epoch_f1_valid = training_set.epochs_completed

                print("--------------\nEpoch %d begins!" % (training_set.epochs_completed + 1))

        bar.finish()


if __name__ == "__main__":
    print("------------------------------")
    StanceDetection('train').main()
