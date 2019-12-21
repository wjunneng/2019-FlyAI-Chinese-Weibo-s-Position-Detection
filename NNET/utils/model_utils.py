import torch
from yutils.vec_utils import get_mask_matrix, get_padding, sentences_to_idx
from yutils.file_utils import read_file2list, read_file2lol, pickle_to_data
from yutils.str_utils import seg_sentences


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

    # print(test_text[0][0], len(test_text[0][0]))

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
        label2idx = {"-1": 0, "0": 1, "1": 2}
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
    # args.save = "../saved_model/sogou/%s_%d_%d_%d_%d_%s" % (args.model, args.nhid, args.ans_len, args.ask_len,
    #                                                         args.batch_size, args.input)
    model_name = model_params[0]

    model_params = [str(param) for param in model_params]
    model_path = "%s%s" % (in_dir, "_".join(model_params))

    return model_name, model_path


def gen_model_paths_by_args(in_dir, model_params_list):
    """

    :param in_dir: "../saved_model/sogou/"
    :param model_params_list: [[args.model, args.nhid, args.ans_len, args.ask_len, args.batch_size, args.input]]
    :return:
    """
    # args.save = "../saved_model/sogou/%s_%d_%d_%d_%d_%s" % (args.model, args.nhid, args.ans_len, args.ask_len,
    #                                                         args.batch_size, args.input)

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
def gen_used_text(word2idx=None, texts=None, max_len=None, idx2word=None, text_idx=None, choice="string"):
    """
    Usually for sentences of one model
    Given original text, return the actually model-used text (with _unk, _pad)
        if given text_idx --> restore shorter-texts from text_idx
        else              --> get text_idx with word2idx and do reversal step

    :param word2idx:
    :param texts:
    :param max_len:
    :param idx2word:
    :param text_idx:
    :return:
    """
    # Not given text_idx: generate according to max_len and word2idx
    if text_idx is None:
        if (word2idx is None) or (texts is None) or (max_len is None):
            print("Need more information to gen text_idx")
        else:
            text_idx = sentences_to_idx(texts, word2idx)
            text_idx, _ = get_padding(text_idx, max_len=max_len)

    if idx2word is None:
        idx2word = dict((v, k) for k, v in word2idx.items())
        # idx2word = dict((idx, word) for word, idx in enumerate(word2idx))

    # text has max_length and paddings/unks
    if choice == "string":
        shorter_texts = [" ".join([idx2word[idx] for idx in t]) for t in text_idx]
    else:
        shorter_texts = [[idx2word[idx] for idx in t] for t in text_idx]

    return shorter_texts
