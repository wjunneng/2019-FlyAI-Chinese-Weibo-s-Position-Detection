# -*- coding:utf-8 -*-
import os
import numpy as np
import torch
import pickle
import pandas as pd
from torch.utils.data import Dataset
from pytorch_transformers import BertTokenizer


class Tokenizer4Bert(object):
    def __init__(self, max_seq_len, pretrained_bert_name):
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_bert_name)
        self.max_seq_len = max_seq_len

    def text_to_sequence(self, text, reverse=False, padding='post', truncating='post'):
        sequence = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(text))
        if len(sequence) == 0:
            sequence = [0]
        if reverse:
            sequence = sequence[::-1]
        return pad_and_truncate(sequence, self.max_seq_len, padding=padding, truncating=truncating)


def bulid_tokenizer(fnames, max_seq_len, dat_fname):
    if os.path.exists(dat_fname):
        print('loading tokenizer:', dat_fname)
        tokenizer = pickle.load(open(dat_fname), 'rb')
    else:
        text = ''
        for fname in fnames:
            with open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore') as file:
                lines = file.readlines()
            for i in range(0, len(lines), 3):
                text_left, _, text_right = [s.lower().strip() for s in lines[i].partition('$T$')]
                aspect = lines[i + 1].lower().strip()
                text_raw = text_left + ' ' + aspect + ' ' + text_right
                text += text_raw + ' '

        tokenizer = Tokenizer(max_seq_len)
        tokenizer.fit_on_text(text)
        pickle.dump(tokenizer, open(dat_fname, 'wb'))

    return tokenizer


class Tokenizer(object):
    def __init__(self, max_seq_len, lower=True):
        self.lower = lower
        self.max_seq_len = max_seq_len
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 1

    def fit_on_text(self, text):
        if self.lower:
            text = text.lower()
        for word in text.split():
            if word not in self.word2idx:
                self.word2idx = self.idx
                self.idx2word[self.idx] = word
                self.idx += 1

    def text_to_sequence(self, text, reverse=True, padding='post', truncating='post'):
        if self.lower:
            text = text.lower()
        words = text.split()
        unknownidx = len(self.word2idx) + 1
        sequence = [self.word2idx[w] if w in self.word2idx else unknownidx for w in words]
        if len(sequence) == 0:
            sequence = [0]
        if reverse:
            sequence = sequence[::-1]

        return pad_and_truncate(sequence, self.max_seq_len, padding=padding, truncating=truncating)


def build_embedding_matrix(word2idx, embed_dim, dat_fname):
    if os.path.exists(dat_fname):
        print('loading_embedding_matrix:', dat_fname)
        embedding_matrix = pickle.load(open(dat_fname, 'rb'))
    else:
        print('loading word vectors...')
        # idx 0 and len(word2idx)+1 are all-zeros
        embedding_matrix = np.zeros(shape=(len(word2idx) + 2, embed_dim))
        fname = './glove.twitter.27B/glove.twitter.27B.' + str(
            embed_dim) + 'd.txt' if embed_dim != 300 else './glove.42B.300d.txt'
        word_vec = {}
        with open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore') as file:
            for line in file.readlines():
                tokens = line.rstrip().split()
                if word2idx is None or tokens[0] in word2idx.keys():
                    word_vec[tokens[0]] = np.asarray(tokens[1:], dtype='float32')
        print('building embedding_matrix:', dat_fname)
        for word, index in word2idx.items():
            vec = word_vec.get(word)
            if vec is not None:
                embedding_matrix[index] = vec
        pickle.dump(embedding_matrix, open(dat_fname, 'wb'))

    return embedding_matrix


def pad_and_truncate(sequence, maxlen, dtype='int64', padding='post', truncating='post', value=0):
    x = (np.ones(maxlen) * value).astype(dtype)
    if truncating == 'pre':
        trunc = sequence[-maxlen:]
    else:
        trunc = sequence[:maxlen]
    trunc = np.asarray(trunc, dtype=dtype)
    if padding == 'post':
        x[:len(trunc)] = trunc
    else:
        x[-len(trunc):] = trunc
    return x


class ABSADataset(Dataset):
    def __init__(self, data_type, fname, tokenizer):
        self.fname = fname
        self.tokenizer = tokenizer
        if data_type == 'csv':
            self.data = self._deal_csv()
        else:
            self.data = self._deal_txt()

    def _deal_txt(self):
        with open(self.fname, 'r', encoding='utf-8', newline='\n', errors='ignore') as file:
            lines = file.readlines()

        all_data = []
        for i in range(0, len(lines), 3):
            text_left, _, text_right = [s.lower().strip() for s in lines[i].partition("$T$")]
            aspect = lines[i + 1].lower().strip()
            polarity = lines[i + 2].strip()

            text_raw_indices = self.tokenizer.text_to_sequence(text_left + " " + aspect + " " + text_right)
            text_raw_without_aspect_indices = self.tokenizer.text_to_sequence(text_left + " " + text_right)
            text_left_indices = self.tokenizer.text_to_sequence(text_left)
            text_left_with_aspect_indices = self.tokenizer.text_to_sequence(text_left + " " + aspect)
            text_right_indices = self.tokenizer.text_to_sequence(text_right, reverse=True)
            text_right_with_aspect_indices = self.tokenizer.text_to_sequence(" " + aspect + " " + text_right,
                                                                             reverse=True)
            aspect_indices = self.tokenizer.text_to_sequence(aspect)
            left_context_len = np.sum(text_left_indices != 0)
            aspect_len = np.sum(aspect_indices != 0)
            aspect_in_text = torch.tensor([left_context_len.item(), (left_context_len + aspect_len - 1).item()])
            polarity = int(polarity) + 1

            text_bert_indices = self.tokenizer.text_to_sequence(
                '[CLS] ' + text_left + " " + aspect + " " + text_right + ' [SEP] ' + aspect + " [SEP]")
            bert_segments_ids = np.asarray([0] * (np.sum(text_raw_indices != 0) + 2) + [1] * (aspect_len + 1))
            bert_segments_ids = pad_and_truncate(bert_segments_ids, self.tokenizer.max_seq_len)

            text_raw_bert_indices = self.tokenizer.text_to_sequence(
                "[CLS] " + text_left + " " + aspect + " " + text_right + " [SEP]")
            aspect_bert_indices = self.tokenizer.text_to_sequence("[CLS] " + aspect + " [SEP]")

            data = {
                'text_bert_indices': text_bert_indices,
                'bert_segments_ids': bert_segments_ids,
                'text_raw_bert_indices': text_raw_bert_indices,
                'aspect_bert_indices': aspect_bert_indices,
                'text_raw_indices': text_raw_indices,
                'text_raw_without_aspect_indices': text_raw_without_aspect_indices,
                'text_left_indices': text_left_indices,
                'text_left_with_aspect_indices': text_left_with_aspect_indices,
                'text_right_indices': text_right_indices,
                'text_right_with_aspect_indices': text_right_with_aspect_indices,
                'aspect_indices': aspect_indices,
                'aspect_in_text': aspect_in_text,
                'polarity': polarity,
            }

            all_data.append(data)
        return all_data

    def _deal_csv(self):
        data = pd.read_csv(filepath_or_buffer=self.fname, sep='\t', encoding='utf-8')
        ID = data['ID']
        TARGET = data['TARGET']
        TEXT = data['TEXT']
        STANCE = data['STANCE']

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)
