# -*- coding:utf-8 -*-

import argparse
import os
import sys
import time

import numpy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from progress.bar import Bar
from NNET.net import Net

sys.path.append('../')
from file_utils import pickle_to_data
from model_utils import gen_model_path_by_args, load_torch_model, tensors_to_numpy, gen_used_text
from vec_utils import get_batch
from eval_utils import count_label, cal_prf, cal_acc

parser = argparse.ArgumentParser(description="PyTorch Net for Stance Project")
''' load data and save model'''
parser.add_argument("--input", type=str, default="3k10k",
                    help="location of dataset")
parser.add_argument("--save", type=str, default="__",
                    help="path to save the model")
''' model parameters '''
parser.add_argument("--model", type=str, default="Net",
                    help="type of model to use for Stance Project")
parser.add_argument("--embtype", type=str, default="baike",
                    help="type of word embeddings")
parser.add_argument("--nclass", type=int, default=3,
                    help="number of classes to predict")
parser.add_argument("--nhid", type=int, default=50,
                    help="size of RNN hidden layer")
parser.add_argument("--nlayers", type=int, default=1,
                    help="number of layers of LSTM")
parser.add_argument("--lr", type=float, default=1e-4,
                    help="learning rate")
parser.add_argument("--epochs", type=int, default=10,
                    help="number of training epoch")
parser.add_argument("--batch_size", type=int, default=8,
                    help="batch size")
parser.add_argument("--dropout", type=float, default=0.5,
                    help="dropout rate")
parser.add_argument("--ans_len", type=int, default=50,
                    help="max time step of answer sequence")
parser.add_argument("--ask_len", type=int, default=25,
                    help="max time step of question sequence")
parser.add_argument("--nhops", type=int, default=3,
                    help="number of attention hops for RoomConditional models")

''' test purpose'''
parser.add_argument("--seed", type=int, default=123456,
                    help="random seed for reproduction")
parser.add_argument("--cuda", action="store_true",
                    help="use CUDA")
parser.add_argument("--is_test", action="store_true",
                    help="flag for training model or only test")
parser.add_argument("--verify", action="store_true",
                    help="flag for testing correctness of program on 10 training records")
parser.add_argument("--proceed", action="store_true",
                    help="flag for continue training on current model")

args = parser.parse_args()

torch.manual_seed(args.seed)


def classify_batch(model, features, use_cuda=True, max_lens=(45, 25)):
    """
    !!! Specify the mode of model before calling
    Predict a single batch return probabilities & max_att_index
        For both train, test and evaluation
    :param model:
    :param features:
    :param use_cuda:
    :param max_lens:
    :return:
    """
    [answers, answers_seqlen, answers_mask, questions, questions_seqlen, questions_mask] = features
    batch_size = len(answers)
    ans_len, ask_len = max_lens

    questions_ = Variable(torch.LongTensor(questions).view(batch_size, ask_len))
    questions_seqlen_ = Variable(torch.LongTensor(questions_seqlen).view(batch_size, 1))
    questions_mask_ = Variable(torch.LongTensor(questions_mask).view(batch_size, ask_len))
    answers_ = Variable(torch.LongTensor(answers).view(batch_size, ans_len))
    answers_seqlen_ = Variable(torch.LongTensor(answers_seqlen).view(batch_size, 1))
    answers_mask_ = Variable(torch.LongTensor(answers_mask).view(batch_size, ans_len))

    if use_cuda:
        questions_ = questions_.cuda()
        questions_seqlen_ = questions_seqlen_.cuda()
        questions_mask_ = questions_mask_.cuda()
        answers_ = answers_.cuda()
        answers_seqlen_ = answers_seqlen_.cuda()
        answers_mask_ = answers_mask_.cuda()

    assert len(answers) == len(questions)

    outputs = model((answers_, answers_seqlen_, answers_mask_),
                    (questions_, questions_seqlen_, questions_mask_))
    return outputs


def classify_batches(batch_size, model, features,
                     use_cuda=True, max_lens=(45, 25)):
    """

    :param batch_size:
    :param model:
    :param features:
    :param use_cuda:
    :param max_lens:
    :return:
    """
    total_num = len(features[0])
    batches_to_classify = get_batch(batch_size, total_num, features=features)  # generator
    y_pred = []
    max_indexes = []
    max_probs = []
    for one_batch in batches_to_classify:
        outputs = classify_batch(model, one_batch,
                                 use_cuda, max_lens=max_lens)

        probs, max_idx = outputs[0], outputs[1]
        max_prob_batch, pred_batch = torch.max(probs, dim=1)
        pred_batch, max_prob_batch, max_index_batch = tensors_to_numpy(use_cuda, [pred_batch, max_prob_batch, max_idx])
        y_pred.extend(pred_batch)
        max_indexes.extend(max_index_batch)
        max_probs.extend(max_prob_batch)
    return y_pred, max_indexes, max_probs


def log_prf_single(y_pred, y_true, model_name="RoomConditional", data_part="Test"):
    """
    cal prf and macro-f1 for single model
    :param y_true:
    :param y_pred:
    :param model_name:
    :return:
    """
    print("-------------------------------")
    print("  PRF for %s  " % model_name)

    accuracy = cal_acc(y_pred, y_true)
    # for All kinds of classes
    pred, right, gold = count_label(y_pred, y_true, include_class=[0, 1, 2])
    prf_result = cal_prf(pred, right, gold, formation=False)
    p = prf_result['p']
    r = prf_result['r']
    f1 = prf_result['f']
    macro_f1 = prf_result["macro"][-1]
    micro_f1 = prf_result["micro"][-1]

    print("  *** Cons|Neu|Pros ***\n  ***", pred, right, gold)
    print("   *Accuracy is %d/%d = %f" % (sum(right), sum(gold), accuracy))
    print("    Precision: %s" % p)
    print("    Recall   : %s" % r)
    print("    F1 score : %s" % f1)
    print("    Macro F1 score on is %f" % macro_f1)
    print("    Micro F1 score on is %f" % micro_f1)

    # for classes of interest
    pred, right, gold = count_label(y_pred, y_true, include_class=[0, 2])
    prf_result = cal_prf(pred, right, gold, formation=False)
    p = prf_result['p']
    r = prf_result['r']
    f1 = prf_result['f']
    macro_f1 = prf_result["macro"][-1]
    micro_f1 = prf_result["micro"][-1]

    print("  *** Cons|Pros ***\n  ***", pred, right, gold)
    print("   *Right on test is %d/%d = %f" % (sum(right), sum(gold), sum(right) / sum(gold)))
    print("    Precision: %s" % p)
    print("    Recall   : %s" % r)
    print("    F1 score : %s" % f1)
    print("    Macro F1 score on is %f" % macro_f1)
    print("    Micro F1 score on is %f" % micro_f1)

    # eval_result = [accuracy, macro_f1, micro_f1]
    eval_result = {
        "accuracy": accuracy,
        "macro_f": macro_f1,
        "micro_f": micro_f1,
        "f_score": f1
    }

    return eval_result  # [accuracy, f1{Con/Pro}, macro_f1]


def _gen_text(idx2word, questions, answers):
    if idx2word:
        questions = gen_used_text(idx2word=idx2word, text_idx=questions)
        answers = gen_used_text(idx2word=idx2word, text_idx=answers)
    return questions, answers


def log_text_single(questions, answers, y_pred, y_true, idx2word=None, max_indexes=None):
    total = len(answers)
    # gen_used_text(<word2idx, texts, max_len>, <idx2word, text_idx>)
    q_text, a_text = _gen_text(idx2word, questions, answers)

    for idx in range(total):
        print("**** 问题: %s\n**** 回答: %s" % (q_text[idx], a_text[idx]))
        if max_indexes and idx2word:
            print("Most important word %s" % idx2word[answers[idx][max_indexes[idx]]])
        print("True:%s  Pred:%s\n" % (str(y_true[idx]), str(y_pred[idx])))


def test(model, dataset, log_result=False, data_part="test"):
    """
    1. decide batch_size, batch_num
    2. classify each batch and combine the predictions --> test_batch()
    3. log the result --> log_text_single()
    4. log and return prf scores --> log_prf_single()

    :param model:
    :param dataset:
    :param log_result:
    :param data_part:
    :return:
    """

    ''' One batch for all test data XX
            [answers, questions]
            [answers_len, questions_len]
            labels
        
    '''
    test_set = pickle_to_data("../data/output/features_%s_%s.pkl" % (args.embtype, data_part))

    test_len = len(test_set)
    # always
    # test_len = 600
    # test_len = 1500

    features, seq_lens, mask_matrice, labels = test_set.next_batch(test_len)
    (answers, answers_seqlen, answers_mask), (questions, questions_seqlen, questions_mask) \
        = zip(features, seq_lens, mask_matrice)
    assert test_len == len(answers) == len(labels) == len(questions)
    feats = [answers, answers_seqlen, answers_mask, questions, questions_seqlen, questions_mask]

    tic = time.time()

    batch_size = 100
    model.eval()
    pred, max_indexes, _ = classify_batches(batch_size, model,
                                            features=feats,
                                            use_cuda=args.cuda,
                                            max_lens=(args.ans_len, args.ask_len))

    tit = time.time() - tic
    print("\n  Predicting {:d} examples using {:5.4f} seconds".format(len(test_set), tit))

    labels = numpy.asarray(labels)

    ''' 3. log the result '''
    # if log_result:
    #     log_text_single(questions, answers, pred, labels, dataset["idx2word"], max_indexes)

    ''' 4. log and return prf scores '''
    _, full_model = gen_model_path_by_args("", [args.model, args.nhid, args.ans_len,
                                                args.ask_len, args.batch_size, args.input])
    eval_result = log_prf_single(pred, labels, args.model)
    macro_f1, acc = eval_result["macro_f"], eval_result["accuracy"]

    return macro_f1, acc


def train_step(model, training_data, optimizer, criterion):
    features, seq_lens, mask_matrice, labels = training_data
    (answers, answers_seqlen, answers_mask), (questions, questions_seqlen, questions_mask) \
        = zip(features, seq_lens, mask_matrice)
    assert args.batch_size == len(labels) == len(questions) == len(answers)
    feats = [answers, answers_seqlen, answers_mask, questions, questions_seqlen, questions_mask]

    # verify the correctness: default false
    if args.verify:
        feats = [feat[:10] for feat in feats]
        labels = labels[:10]

    # Prepare data and prediction
    labels_ = Variable(torch.LongTensor(labels))
    if args.cuda:
        labels_ = labels_.cuda()

    # necessary for Room model
    torch.nn.utils.clip_grad_norm(model.parameters(), 0.25)

    model.zero_grad()
    outputs = classify_batch(model,
                             feats,
                             use_cuda=args.cuda,
                             max_lens=(args.ans_len, args.ask_len))
    probs = outputs[0]
    loss = criterion(probs.view(len(labels_), -1), labels_)

    loss.backward()
    optimizer.step()


def train(model, dataset):
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-5)
    criterion = nn.CrossEntropyLoss()

    training_set = pickle_to_data("../data/output/features_%s_%s.pkl" % (args.embtype, "training"))
    best_f1_test, best_p_valid, best_f1_valid = -numpy.inf, -numpy.inf, -numpy.inf
    epoch_f1_test, epoch_f1_valid, epoch_f1_cur = 0, 0, 0
    cur_f1_test = -numpy.inf
    batches_per_epoch = len(training_set) // args.batch_size
    max_train_steps = int(args.epochs * batches_per_epoch)

    print("--------------\nEpoch 0 begins!")
    bar = Bar("  Processing", max=max_train_steps)
    print(max_train_steps, args.epochs, len(training_set), args.batch_size)
    for step in range(max_train_steps):
        bar.next()
        training_batch = training_set.next_batch(args.batch_size)
        train_step(model, training_batch, optimizer, criterion)  # xIndexes, xQuestions, yLabels

        # Test after each epoch
        if (step + 1) % batches_per_epoch == 0:
            tic = time.time()
            f1_score, p_score = test(model, dataset, data_part="validation")
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


def create_model(emb_fn):
    embeddings = pickle_to_data(emb_fn)
    emb_np = numpy.asarray(embeddings, dtype=numpy.float32)  # from_numpy
    emb = torch.from_numpy(emb_np)

    models = {
        "Net": Net
    }
    model = models[args.model](embeddings=emb,
                               input_dim=emb.size(1),
                               hidden_dim=args.nhid,
                               num_layers=args.nlayers,
                               output_dim=args.nclass,
                               max_step=[args.ans_len, args.ask_len],
                               dropout=args.dropout)
    if args.model in ["Net", ]:
        model.nhops = args.nhops
    return model


def load_dataset(dataset_fn, word2idx_fn):
    dataset = pickle_to_data(dataset_fn)
    word2idx = pickle_to_data(word2idx_fn)
    idx2word = dict((v, k) for k, v in word2idx.items())
    dataset["word2idx"] = word2idx
    dataset["idx2word"] = idx2word
    return dataset


def main():
    # 1. define location to save the model and mkdir if not exists
    if args.save == "__":  # RoomConditional_100_batch8_45_25_3k10k20k_bias
        _, args.save = gen_model_path_by_args("../data/model/",
                                              [args.model, args.nhid, args.ans_len, args.ask_len,
                                               args.batch_size, args.input, args.nhops])
    if not os.path.exists(args.save):
        os.mkdir(args.save)

    # 2. load dataset
    dataset_fn = "../data/output/features_%s.pkl" % (args.embtype)
    word2idx_fn = "../data/output/word2idx_%s.pkl" % (args.embtype)
    dataset = load_dataset(dataset_fn, word2idx_fn)

    # 3. test, proceed, train
    if args.is_test:
        model = load_torch_model(args.save, use_cuda=args.cuda)
        print(type(model))
        test(model, dataset)

    else:
        ''' make sure the folder to save models exist '''
        if not os.path.exists(args.save):
            os.mkdir(args.save)

        ''' continue training or not '''
        if args.proceed:
            if os.path.exists(args.save + "/model.pt"):
                with open(args.save + "/model.pt", "rb") as saved_model:
                    model = torch.load(saved_model)
        else:
            emb_fn = "../data/output/embeddings_%s.pkl" % (args.embtype)
            model = create_model(emb_fn)

        if torch.cuda.is_available():
            if not args.cuda:
                print("Waring: You have a CUDA device, so you should probably run with --cuda")
            else:
                torch.cuda.manual_seed(args.seed)
                model.cuda()
        train(model, dataset)


if __name__ == "__main__":
    print("------------------------------")
    main()
