# -*- coding:utf-8 -*-
import os
import sys
from pyltp import Segmentor


###############
# Chinese sentence pre-processing
###############
def seg_sentences(senlist, choice="string", lexicon=None):
    """
    :param senlist: list of raw sentences
    :param choice: "string" for write_list2file(); "list" for further processing
    :param lexicon
    :return: list of strings(for choice="string"); list of list(for choice="else")
    """
    ROOTDIR = os.path.join(os.path.dirname(__file__), os.pardir)
    sys.path = [os.path.join(ROOTDIR, "lib")] + sys.path

    # Set your own model path
    place = decide_run_place()
    MODELDIR = os.path.join(place + "nlp_res/ltp/ltp_data/")

    segmentor = Segmentor()
    if lexicon:
        segmentor.load_with_lexicon(os.path.join(MODELDIR, "cws.model"), lexicon)
    else:
        segmentor.load(os.path.join(MODELDIR, "cws.model"))

    words_list = []

    if choice == "string":
        for s in senlist:
            words = segmentor.segment(s.strip())
            words = list_to_str(list(words))
            # print "------", "|".join(words), "-------"
            # print words
            words_list.append(words)
    else:
        for s in senlist:
            words = segmentor.segment(s)
            words = list(words)
            words_list.append(words)
    segmentor.release()
    return words_list


def load_segmentor(lexicon=None):
    ROOTDIR = os.path.join(os.path.dirname(__file__), os.pardir)
    sys.path = [os.path.join(ROOTDIR, "lib")] + sys.path

    # Set your own model path
    place = decide_run_place()
    MODELDIR = os.path.join(place + "nlp_res/ltp/ltp_data/")

    segmentor = Segmentor()
    if lexicon:
        segmentor.load_with_lexicon(os.path.join(MODELDIR, "cws.model"), lexicon)
    else:
        segmentor.load(os.path.join(MODELDIR, "cws.model"))

    return segmentor


###############
# Text segmentation
###############
def tokenize_sentences(senlist, lan="chn", choice="string", lexicon=None):
    """

    :param senlist:
    :param lan:
    :param choice:
    :param lexicon:
    :param place:
    :return:
    """

    tokenized_sens = seg_sentences(senlist, choice=choice, lexicon=lexicon)

    return tokenized_sens


#############################################################
###############
# String Utilities
##############
def list_to_str(list_of_words, has_blank=True):
    """covert list of segment words into a single string
    """
    low = list_of_words
    s = ""
    if has_blank:
        for i in low:
            if i not in set(["\n", " ", "\n\n"]):
                s += i + " "
    else:
        for i in low:
            if i != "\n" and i != " " and i != "\n\n":
                s += i
    return s


def str_to_list(sentence_in_string):
    """convert strings with '\n' to list of words without '\n' """
    return sentence_in_string.strip().split()  # remove last \n


###############
# English pre-processing
###############
def decide_run_place():
    return "../data/"
