# -*- coding:utf-8 -*-

import os
import sys
import torch
import numpy as np
import argparse
import time
import torch.nn as nn
import torch.optim as optim
from progress.bar import Bar

sys.path.append('../')
from NNET.net import Net
import NNET.args as arguments
from NNET.utils.file_utils import pickle_to_data
from NNET.utils.model_utils import gen_model_path_by_args, load_torch_model, train_step, test

torch.manual_seed(arguments.seed)

# 项目的超参
parser = argparse.ArgumentParser()
parser.add_argument("-e", "--EPOCHS", default=10, type=int, help="train epochs")
parser.add_argument("-b", "--BATCH", default=32, type=int, help="batch size")
args = parser.parse_args()

# 判断gpu是否可用
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
device = torch.device(device)


def main():
    # 1. define location to save the model and mkdir if not exists
    if arguments.save == "__":  # Net_100_batch8_45_25_3k10k20k_bias
        _, arguments.save = gen_model_path_by_args("../data/model/",
                                                   [arguments.model, arguments.nhid, arguments.ans_len,
                                                    arguments.ask_len,
                                                    arguments.batch_size, arguments.input, arguments.nhops])
    if not os.path.exists(arguments.save):
        os.mkdir(arguments.save)

    # 2. load dataset
    dataset_fn = "../data/output/features_%s.pkl" % (arguments.embtype)
    word2idx_fn = "../data/output/word2idx_%s.pkl" % (arguments.embtype)
    dataset = pickle_to_data(dataset_fn)
    word2idx = pickle_to_data(word2idx_fn)
    idx2word = dict((v, k) for k, v in word2idx.items())
    dataset["word2idx"] = word2idx
    dataset["idx2word"] = idx2word

    # 3. test, proceed, train
    if arguments.is_test:
        model = load_torch_model(arguments.save, use_cuda=arguments.cuda)
        test(model, dataset)

    else:
        """ make sure the folder to save models exist """
        if not os.path.exists(arguments.save):
            os.mkdir(arguments.save)

        """ continue training or not """
        if arguments.proceed:
            if os.path.exists(arguments.save + "/model.pt"):
                with open(arguments.save + "/model.pt", "rb") as saved_model:
                    model = torch.load(saved_model)
        else:
            emb_fn = "../data/output/embeddings_%s.pkl" % (arguments.embtype)
            embeddings = pickle_to_data(emb_fn)
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

        if torch.cuda.is_available():
            if not arguments.cuda:
                print("Waring: You have a CUDA device, so you should probably run with --cuda")
            else:
                torch.cuda.manual_seed(arguments.seed)
                model.cuda()

        # train
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-5)
        criterion = nn.CrossEntropyLoss()

        training_set = pickle_to_data("../data/output/features_%s_%s.pkl" % (args.embtype, "training"))
        best_f1_test, best_p_valid, best_f1_valid = -np.inf, -np.inf, -np.inf
        epoch_f1_test, epoch_f1_valid, epoch_f1_cur = 0, 0, 0
        cur_f1_test = -np.inf
        batches_per_epoch = len(training_set) // args.batch_size
        max_train_steps = int(args.epochs * batches_per_epoch)

        print("--------------\nEpoch 0 begins!")
        bar = Bar("  Processing", max=max_train_steps)
        print(max_train_steps, args.epochs, len(training_set), args.batch_size)
        for step in range(max_train_steps):
            bar.next()
            training_batch = training_set.next_batch(args.batch_size)
            # xIndexes, xQuestions, yLabels
            train_step(model, training_batch, optimizer, criterion)

            # Test after each epoch
            if (step + 1) % batches_per_epoch == 0:
                tic = time.time()
                f1_score, p_score = test(model, log_result=False, dataset=dataset, data_part="validation")
                print("\n  Begin to predict the results on Valid")
                print("  using %.5fs" % (time.time() - tic))
                print("  ----Old best F1 on Valid is %f on epoch %d" % (best_f1_valid, epoch_f1_valid))
                print("  ----Old best F1 on Test is %f on epoch %d" % (best_f1_test, epoch_f1_test))
                print("  ----Cur save F1 on Test is %f on epoch %d" % (cur_f1_test, epoch_f1_valid))
                if f1_score > best_f1_valid:
                    with open(args.save + "/model.pt", 'wb') as to_save:
                        torch.save(model, to_save)
                    best_f1_valid = f1_score
                    print("  ----New best F1 on Valid is %f" % f1_score)
                    epoch_f1_valid = training_set.epochs_completed

                print("--------------\nEpoch %d begins!" % (training_set.epochs_completed + 1))

        bar.finish()


if __name__ == "__main__":
    print("------------------------------")
    main()
