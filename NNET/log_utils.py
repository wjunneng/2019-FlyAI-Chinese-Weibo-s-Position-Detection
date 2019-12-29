# -*- coding:utf-8 -*-
from eval_utils import count_label, cal_prf, cal_acc
from model_utils import sentences_to_idx, get_padding


def log_prf_single(y_pred, y_true, model_name="Net", data_part="Test"):
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

    # [accuracy, f1{Con/Pro}, macro_f1]
    return eval_result


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

    # text has max_length and paddings/unks
    if choice == "string":
        shorter_texts = [" ".join([idx2word[idx] for idx in t]) for t in text_idx]
    else:
        shorter_texts = [[idx2word[idx] for idx in t] for t in text_idx]

    return shorter_texts


def log_text_single(questions, answers, y_pred, y_true, idx2word=None, max_indexes=None):
    total = len(answers)
    q_text = None
    a_text = None
    if idx2word:
        q_text = gen_used_text(idx2word=idx2word, text_idx=questions)
        a_text = gen_used_text(idx2word=idx2word, text_idx=answers)

    for idx in range(total):
        print("**** 问题: %s\n**** 回答: %s" % (q_text[idx], a_text[idx]))
        if max_indexes and idx2word:
            print("Most important word %s" % idx2word[answers[idx][max_indexes[idx]]])
        print("True:%s  Pred:%s\n" % (str(y_true[idx]), str(y_pred[idx])))
