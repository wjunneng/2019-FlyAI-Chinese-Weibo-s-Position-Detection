import os
import time
import torch
import numpy as np
from torch.autograd import Variable

from NNET import args
from NNET.utils.vec_utils import get_mask_matrix, get_padding, sentences_to_idx, get_batch
from NNET.utils.file_utils import read_file2list, read_file2lol, pickle_to_data
from NNET.utils.str_utils import seg_sentences
from NNET.utils.log_utils import log_text_single, log_prf_single


#############################################################
###################
#   load text and labels
###################
def load_test_text(feat_filenames, seged=True):
    """

    :param feat_filenames: filename of input text features
    :param seged:
    :return:
    """
    if not seged:
        test_text = [read_file2list(fn) for fn in feat_filenames]
        test_text = [seg_sentences(text) for text in test_text]
    else:
        test_text = [read_file2lol(fn) for fn in feat_filenames]

    return test_text


def load_test_data(feat_filenames, word2idx_filename, max_lens=(45, 25), seged=True):
    """
    Load data into vectors:
        1. read text and seg text
        2. read word2idx file
        3. sentence to idx: padding,  seq_len, mask matrix
    :param feat_filenames: list, question and answer file name
    :param word2idx_filename: word2idx
    :param max_lens: max length of each feature
    :param seged:
    :return:
    """
    # 1.
    assert len(feat_filenames) == len(max_lens)

    if not seged:
        test_text = [read_file2list(fn) for fn in feat_filenames]
        test_text = [seg_sentences(text) for text in test_text]
    else:
        test_text = [read_file2lol(fn) for fn in feat_filenames]

    # 2.
    word2idx = pickle_to_data(word2idx_filename)

    # 3.
    test_data = []
    for text, ml in zip(test_text, max_lens):
        text = sentences_to_idx(text, word2idx)
        text, text_seqlen = get_padding(text, max_len=ml)
        text_mask = get_mask_matrix(text_seqlen, max_len=ml)
        test_data.extend([text, text_seqlen, text_mask])

    return test_data


def load_test_label(label_filename, label2idx=None):
    if label2idx is None:
        label2idx = {"AGAINST": 0, "NONE": 1, "FAVOR": 2}
    labels = read_file2list(label_filename)
    test_labels = [label2idx[label] for label in labels]
    return test_labels


#############################################################
###################
#   get model paths and names
###################
def gen_model_path_by_args(in_dir, model_params):
    """

    :param in_dir: "../saved_model/sogou/"
    :param model_params: [args.model, args.nhid, args.ans_len, args.ask_len, args.batch_size, args.input]
    :return:
    """
    model_name = model_params[0]

    model_params = [str(param) for param in model_params]
    model_path = "%s%s" % (in_dir, "_".join(model_params))

    if not os.path.exists(model_path):
        os.mkdir(model_path)

    return model_name, model_path


def gen_model_paths_by_args(in_dir, model_params_list):
    """

    :param in_dir: "../saved_model/sogou/"
    :param model_params_list: [[args.model, args.nhid, args.ans_len, args.ask_len, args.batch_size, args.input]]
    :return:
    """
    model_np = [gen_model_path_by_args(in_dir, model_params) for model_params in model_params_list]
    model_names = [mnp[0] for mnp in model_np]
    model_paths = [mnp[1] for mnp in model_np]
    print(model_names)
    print(model_paths)

    return model_names, model_paths


#############################################################
###################
#   torch utilities and load model
###################
def tensor_to_numpy(use_cuda, tensor, dim=(-1,)):
    if use_cuda:
        ndarray = tensor.view(dim).cpu().data.numpy()
    else:
        ndarray = tensor.view(dim).data.numpy()
    return ndarray


def tensors_to_numpy(use_cuda, tensors, dim=(-1,)):
    if use_cuda:
        ndarray = [tensor.view(dim).cpu().data.numpy() for tensor in tensors]
    else:
        # print(use_cuda, type(tensors[2].data))
        ndarray = [tensor.view(dim).data.numpy() for tensor in tensors]
    return ndarray


def load_torch_model(model_path, use_cuda=True):
    with open(model_path + "/model.pt", "rb") as f:
        if use_cuda:
            model = torch.load(f)
        else:
            model = torch.load(f, map_location=lambda storage, loc: storage)
            model.cpu()
        model.eval()
        return model


#############################################################
###################
#   Generate real text for comparison
###################

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

    outputs = model((answers_, answers_seqlen_, answers_mask_), (questions_, questions_seqlen_, questions_mask_))
    return outputs


def classify_batches(batch_size, model, features, use_cuda=True, max_lens=(50, 25)):
    """
    :param batch_size:
    :param model:
    :param features:
    :param use_cuda:
    :param max_lens:
    :return:
    """
    total_num = len(features[0])
    # generator
    batches_to_classify = get_batch(batch_size, total_num, features=features)
    y_pred = []
    max_indexes = []
    max_probs = []
    for one_batch in batches_to_classify:
        outputs = classify_batch(model, one_batch, use_cuda, max_lens=max_lens)

        probs, max_idx = outputs[0], outputs[1]
        max_prob_batch, pred_batch = torch.max(probs, dim=1)
        pred_batch, max_prob_batch, max_index_batch = tensors_to_numpy(use_cuda, [pred_batch, max_prob_batch, max_idx])
        y_pred.extend(pred_batch)
        max_indexes.extend(max_index_batch)
        max_probs.extend(max_prob_batch)

    return y_pred, max_indexes, max_probs


def train_step(model, training_data, optimizer, criterion):
    """
    train step
    :param model:
    :param training_data:
    :param optimizer:
    :param criterion:
    :return:
    """
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
    outputs = classify_batch(model, feats, use_cuda=args.cuda, max_lens=(args.ans_len, args.ask_len))
    probs = outputs[0]
    loss = criterion(probs.view(len(labels_), -1), labels_)

    loss.backward()
    optimizer.step()


def test(model, dataset, log_result=True, data_part="test"):
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

    """ One batch for all test data XX
            [answers, questions]
            [answers_len, questions_len]
            labels

    """
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

    labels = np.asarray(labels)

    """ 3. log the result """
    if log_result:
        log_text_single(questions, answers, pred, labels, dataset["idx2word"], max_indexes)

    """ 4. log and return prf scores """
    _, full_model = gen_model_path_by_args("", [args.model, args.nhid, args.ans_len,
                                                args.ask_len, args.batch_size, args.input])
    eval_result = log_prf_single(y_pred=pred, y_true=labels, model_name=args.model)
    macro_f1, acc = eval_result["macro_f"], eval_result["accuracy"]

    return macro_f1, acc
