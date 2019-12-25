from NNET.utils.eval_utils import count_label, cal_prf, cal_acc
from NNET.utils.model_utils import gen_used_text


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
