# -*- coding:utf-8 -*-
import os
import sys

os.chdir(sys.path[0])

import re
import math
import copy
import numpy as np
import torch
import pickle
import jieba
import json
import shutil
import random
from collections import Counter
import pandas as pd
from torch.utils.data import Dataset
from pytorch_transformers import BertTokenizer, BertModel
from sklearn.metrics import f1_score
from concurrent.futures import ThreadPoolExecutor

import zh_wiki
import args


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
        return Util.pad_and_truncate(sequence, self.max_seq_len, padding=padding, truncating=truncating)


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

        return Util.pad_and_truncate(sequence, self.max_seq_len, padding=padding, truncating=truncating)


class ABSADataset(Dataset):
    def __init__(self, data_type, fname, tokenizer):
        self.tokenizer = tokenizer

        if data_type == 'csv':
            self.fname = fname
            self.label2idx = dict((args.labels[i], i) for i in range(len(args.labels)))
            self.data = self._deal_csv()
        elif data_type == 'txt':
            self.fname = fname
            self.data = self._deal_txt()
        else:
            self.label2idx = dict((args.labels[i], i) for i in range(len(args.labels)))
            self.fname = fname
            self.data = self._deal_none()

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

            bert_segments_ids = Util.pad_and_truncate(bert_segments_ids, self.tokenizer.max_seq_len)

            text_raw_bert_indices = self.tokenizer.text_to_sequence(
                "[CLS] " + text_left + " " + aspect + " " + text_right + " [SEP]")

            aspect_bert_indices = self.tokenizer.text_to_sequence("[CLS] " + aspect + " [SEP]")

            data = {
                'text_bert_indices': text_bert_indices,
                'bert_segments_ids': bert_segments_ids,
                'text_raw_bert_indices': text_raw_bert_indices,  # aen_bert
                'aspect_bert_indices': aspect_bert_indices,  # aen_bert
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
        print('drop before shape: {}'.format(data.shape))
        data.dropna(axis=0, how='any', inplace=True)
        print('drop after shape: {}'.format(data.shape))
        ID = data['ID'].values.tolist()
        TARGET = data['TARGET'].values.tolist()
        TEXT = data['TEXT'].values.tolist()
        STANCE = data['STANCE'].values.tolist()

        all_data = []
        for i in range(len(ID)):
            aspect = TARGET[i].strip().lower()
            polarity = STANCE[i]
            text = TEXT[i].strip().lower()

            aspect_indices = self.tokenizer.text_to_sequence(aspect)
            aspect_len = np.sum(aspect_indices != 0)
            text_bert_indices = self.tokenizer.text_to_sequence('[CLS] ' + text + ' [SEP] ' + aspect + " [SEP]")

            text_raw_indices = self.tokenizer.text_to_sequence(text)
            bert_segments_ids = np.asarray([0] * (np.sum(text_raw_indices != 0) + 2) + [1] * (aspect_len + 1))
            bert_segments_ids = Util.pad_and_truncate(bert_segments_ids, self.tokenizer.max_seq_len)

            text_raw_bert_indices = self.tokenizer.text_to_sequence("[CLS] " + text + " [SEP]")
            aspect_bert_indices = self.tokenizer.text_to_sequence("[CLS] " + aspect + " [SEP]")
            polarity = self.label2idx[polarity]

            data = {
                'text_bert_indices': text_bert_indices,
                'bert_segments_ids': bert_segments_ids,
                'text_raw_bert_indices': text_raw_bert_indices,  # aen_bert
                'aspect_bert_indices': aspect_bert_indices,  # aen_bert
                'polarity': polarity,
            }

            all_data.append(data)

        return all_data

    def _deal_none(self):
        (TARGET, TEXT, STANCE) = self.fname
        all_data = []
        for i in range(len(TARGET)):
            aspect = TARGET[i].strip().lower()
            text = TEXT[i].strip().lower()

            aspect_indices = self.tokenizer.text_to_sequence(aspect)
            aspect_len = np.sum(aspect_indices != 0)
            text_bert_indices = self.tokenizer.text_to_sequence('[CLS] ' + text + ' [SEP] ' + aspect + " [SEP]")

            text_raw_indices = self.tokenizer.text_to_sequence(text)
            bert_segments_ids = np.asarray([0] * (np.sum(text_raw_indices != 0) + 2) + [1] * (aspect_len + 1))
            bert_segments_ids = Util.pad_and_truncate(bert_segments_ids, self.tokenizer.max_seq_len)

            text_raw_bert_indices = self.tokenizer.text_to_sequence("[CLS] " + text + " [SEP]")
            aspect_bert_indices = self.tokenizer.text_to_sequence("[CLS] " + aspect + " [SEP]")

            if STANCE is None:
                data = {
                    'text_bert_indices': text_bert_indices,
                    'bert_segments_ids': bert_segments_ids,
                    'text_raw_bert_indices': text_raw_bert_indices,  # aen_bert
                    'aspect_bert_indices': aspect_bert_indices,  # aen_bert
                }

            else:
                polarity = STANCE[i]
                polarity = self.label2idx[polarity]

                data = {
                    'text_bert_indices': text_bert_indices,
                    'bert_segments_ids': bert_segments_ids,
                    'text_raw_bert_indices': text_raw_bert_indices,  # aen_bert
                    'aspect_bert_indices': aspect_bert_indices,  # aen_bert
                    'polarity': polarity,
                }

            all_data.append(data)

        return all_data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


class Util(object):
    @staticmethod
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

    @staticmethod
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

    @staticmethod
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

    @staticmethod
    def print_args(model, logger, args):
        n_trainable_params, n_nontrainable_params = 0, 0
        for p in model.parameters():
            n_params = torch.prod(torch.tensor(p.shape))
            if p.requires_grad:
                n_trainable_params += n_params
            else:
                n_nontrainable_params += n_params

        logger.info(
            'n_trainable_params: {0}, n_nontrainable_params: {1}'.format(n_trainable_params, n_nontrainable_params))
        logger.info('> training arguments:')
        for arg in vars(args):
            logger.info('>>> {0}:{1}'.format(arg, getattr(args, arg)))

    @staticmethod
    def reset_params(model, args):
        for child in model.children():
            if type(child) != BertModel:
                for p in child.parameters():
                    if p.requires_grad:
                        if len(p.shape) > 1:
                            args.initializer(p)
                        else:
                            stdv = 1. / math.sqrt(p.shape[0])
                            torch.nn.init.uniform_(p, a=-stdv, b=stdv)

    @staticmethod
    def evaluate_acc_f1(model, args, data_loader):
        n_correct, n_total = 0, 0
        t_targets_all, t_outputs_all = None, None
        with torch.no_grad():
            for t_batch, t_sample_batched in enumerate(data_loader):
                t_inputs = [t_sample_batched[col].to(args.device) for col in args.inputs_cols]
                t_outputs = model(t_inputs)
                t_targets = t_sample_batched['polarity'].to(args.device)

                n_correct += (torch.argmax(t_outputs, -1) == t_targets).sum().item()
                n_total += len(t_outputs)

                if t_targets_all is None:
                    t_targets_all = t_targets
                    t_outputs_all = t_outputs
                else:
                    t_targets_all = torch.cat((t_targets_all, t_targets), dim=0)
                    t_outputs_all = torch.cat((t_outputs_all, t_outputs), dim=0)

        acc = n_correct / n_total
        f1 = f1_score(y_true=t_targets_all.cpu(), y_pred=torch.argmax(t_outputs_all, -1).cpu(), labels=[0, 1, 2],
                      average='macro')

        return acc, f1

    @staticmethod
    def save_model(model, output_dir):
        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
        output_model_file = os.path.join(output_dir, "pytorch_model.bin")
        torch.save(model_to_save.state_dict(), output_model_file)

    @staticmethod
    def load_model(model, output_dir):
        # Load a trained model that you have fine-tuned
        output_model_file = os.path.join(output_dir, "pytorch_model.bin")
        model.load_state_dict(torch.load(output_model_file))

        return model

    @staticmethod
    def calculate_word_count(train_data):
        topics = ['IphoneSE', '春节放鞭炮', '深圳禁摩限电', '俄罗斯在叙利亚的反恐行动', '开放二胎']
        stance = list(train_data['STANCE'])
        target = list(train_data['TARGET'])
        text = list(train_data['TEXT'])

        results = None
        for topic in topics:
            print('topic: {}'.format(topic))
            none_label = None
            favor_label = None
            against_label = None

            for index in range(train_data.shape[0]):
                if target[index].strip().lower() != topic.lower():
                    continue
                count = Counter(list(
                    jieba.cut(re.sub(r"[^a-zA-Z0-9\u4e00-\u9fa5]", '', text[index].strip()), cut_all=False)))

                label = stance[index].strip()
                if label == 'AGAINST':
                    if against_label is None:
                        against_label = copy.deepcopy(count)
                    else:
                        against_label.update(count)

                elif label == 'FAVOR':
                    if favor_label is None:
                        favor_label = copy.deepcopy(count)
                    else:
                        favor_label.update(count)

                else:
                    if none_label is None:
                        none_label = copy.deepcopy(count)
                    else:
                        none_label.update(count)

            all_label = copy.deepcopy(none_label)
            all_label.update(favor_label)
            all_label.update(against_label)
            all_label_most = all_label.most_common(1000)

            result = pd.DataFrame(data={"WORD": [i for (i, j) in all_label_most]})
            result["NONE"] = [none_label.get(i) for (i, j) in all_label_most]
            result["FAVOR"] = [favor_label.get(i) for (i, j) in all_label_most]
            result["AGAINST"] = [against_label.get(i) for (i, j) in all_label_most]
            result.fillna(value=0, inplace=True)

            result['FREQ'] = [j for (i, j) in all_label_most]
            result = result[result['FREQ'] >= 4]
            result['SUPPORT'] = np.nan_to_num(
                [max(none, favor, against) / (1.0 * (none + favor + against)) for (none, favor, against) in
                 zip(result['NONE'], result['FAVOR'], result['AGAINST'])])

            result = result[result['SUPPORT'] >= 0.8]
            result.sort_values(by='SUPPORT', inplace=True, ascending=False)
            result['TARGET'] = topic

            if results is None:
                results = result[['TARGET', 'WORD', 'NONE', 'FAVOR', 'AGAINST', 'FREQ', 'SUPPORT']]
            else:
                results = pd.concat(
                    [results, result[['TARGET', 'WORD', 'NONE', 'FAVOR', 'AGAINST', 'FREQ', 'SUPPORT']]])

        results.to_csv(path_or_buf=os.path.join(os.getcwd(), args.word_count_dir), index=None)


class PreProcessing(object):
    def __init__(self, fileText):
        self.label = []
        self.maxLen = 0
        self.fileText = fileText.tolist()

        self.refresh_eng_data()
        self.refresh_chn_data()
        self.hant_to_hans()

    def refresh_eng_data(self):
        fileText = []
        for istring in self.fileText:
            # 替换
            istring = re.sub(r"\'s", " \'s", istring)
            istring = re.sub(r"\'ve", " \'ve", istring)
            istring = re.sub(r"n\'t", " n\'t", istring)
            istring = re.sub(r"\'re", " \'re", istring)
            istring = re.sub(r"\'d", " \'d", istring)
            istring = re.sub(r"\'ll", " \'ll", istring)
            istring = re.sub("“", """\"""", istring)
            istring = re.sub("”", """\"""", istring)
            istring = re.sub(r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b', " ", istring)

            # 中文字符转英文字符
            table = {ord(f): ord(t) for (f, t) in
                     zip(r"""！‟＃＄％＆‛（）＊＋，－。／：；＜＝＞？＠【＼】＾＿｀｛｜｝～""", r"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~""")}
            istring = istring.translate(table)

            istring = re.sub(' +', ' ', istring)
            istring = re.sub(',+', ',', istring)
            istring = re.sub('\.+', '.', istring)
            istring = re.sub('!+', '!', istring)
            istring = re.sub('\?+', '?', istring)
            istring = re.sub('%+', '%', istring)
            istring = re.sub('#+', '#', istring)
            istring = re.sub('@+', '@', istring)
            istring = re.sub('&+', '&', istring)
            istring = re.sub('~+', '~', istring)
            istring = re.sub('/+', '/', istring)
            istring = re.sub('-+', '-', istring)
            istring = re.sub('\(+', '(', istring)
            istring = re.sub('\)+', ')', istring)
            istring = re.sub('\|+', '|', istring)

            fileText.append(istring.strip().lower())

            self.fileText = fileText

    def refresh_chn_data(self):
        """
         全角数字转半角
         全角英文字母转半角
         全角中文标点转半角
         转小写(可选)
        """
        fileText = []
        for istring in self.fileText:
            rstring = ""
            for uchar in istring:
                inside_code = ord(uchar)
                if inside_code == 0x3000:
                    inside_code = 0x0020
                else:
                    inside_code -= 0xfee0
                if inside_code < 0x0020 or inside_code > 0x7e:
                    rstring += uchar
                else:
                    rstring += chr(inside_code)
            rstring = re.sub(r'\s+', ' ', rstring).lower()

            fileText.append(rstring)

        self.fileText = fileText

    def hant_to_hans(self):
        _zh2Hant, _zh2Hans = {}, {}
        for old, new in ((zh_wiki.zh2Hant, _zh2Hant), (zh_wiki.zh2Hans, _zh2Hans)):
            for k, v in old.items():
                new[k] = v

        _zh2Hant = {value: key for (key, value) in _zh2Hant.items() if key != value}
        _zh2Hans = {key: value for (key, value) in _zh2Hans.items() if key != value}

        _zh2Hant.update(_zh2Hans)

        for index in range(len(self.fileText)):
            for hant in _zh2Hant.keys():
                self.fileText[index] = self.fileText[index].replace(hant, _zh2Hant[hant])

    def get_file_text(self):
        return np.asarray(self.fileText)


class SynonymsReplacer(object):
    """
    同义句生成
    original text : 吸烟的危害是什么
    result：['吸烟的危害是什么', '吸烟的危害是那些', '吸烟的危害为什么', '吸烟的危害为那些', '吸烟的害处是什么', '吸烟的害处是那些', '吸烟的害处为什么', '吸烟的害处为那些']

    """

    def __init__(self):
        synonyms_file_path = os.path.join(os.getcwd(), args.synomys_dir)
        # 移动文件路径
        shutil.copyfile(os.path.join(os.getcwd(), 'synomys.json'), synonyms_file_path)
        # self.synonyms = self.load_synonyms(synonyms_file_path)
        self.synonyms_file_path = synonyms_file_path
        # self.segmentor = self.segment(cws_model_path)
        # 每个元素为句子中每个词及其同义词构成的列表
        self.candidate_synonym_list = {}
        # 加载停用词表
        self.stop = [line.strip() for line in
                     open(os.path.join(os.getcwd(), args.stop_words_dir), encoding='gbk').readlines()]

    def segment(self, sentence):
        """调用pyltp的分词方法将str类型的句子分词并以list形式返回"""
        result = [i for i in list(jieba.cut(sentence, cut_all=False)) if i not in self.stop]

        return result

    def load_synonyms(self, file_path):
        """
        加载同义词表
        :param file_path: 同义词表路径
        :return: 同义词列表[[xx,xx],[xx,xx]...]
        """
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in json.load(file):
                sign = yield line
                if sign == 'stop':
                    break

    def permutation(self, data):
        """
        排列函数
        :param data: 需要进行排列的数据，列表形式
        :return:
        """
        assert len(data) >= 1, "Length of data must greater than 0."

        # 当data中只剩（有）一个词及其同义词的列表时，程序返回
        if len(data) == 1:
            return data[0]

        else:
            head = data[0]
            # 不断切分到只剩一个词的同义词列表
            tail = data[1:]

        tail = self.permutation(tail)
        permt = []
        # 构建两个词列表的同义词组合
        for h in head:
            for t in tail:
                # 传入的整个data的最后一个元素是一个一维列表，其中每个元素为str
                if isinstance(t, str):
                    permt.extend([[h] + [t]])
                elif isinstance(t, list):
                    permt.extend([[h] + t])
        return permt

    def permutation_one(self, data):
        """
        排列函数 从list中选择一个元素
        :param data: 需要进行排列的数据，列表形式
        :return:
        """
        assert len(data) >= 1, "Length of data must greater than 0."
        random.seed(42)

        result = ""
        for sample in data:
            result += random.sample(sample, 1)[0]

        return result

    def search_synonyms(self, word, word_synonyms, index):
        """
        根据同义词列表，对每一个word做搜寻匹配
        :param word:
        :param word_synonyms:
        :param index:
        :return:
        """
        synonyms_generation = self.load_synonyms(self.synonyms_file_path)
        # 遍历同义词表，syn为其中的一条
        for syn in synonyms_generation:
            try:
                # 如果句子中的词在同义词表某一条目中，将该条目中它的同义词添加到该词的同义词列表中
                if word in syn:
                    syn.remove(word)
                    word_synonyms.extend(syn)
                    synonyms_generation.send('stop')
            except StopIteration:
                break
        return {index: word_synonyms}

    def add_synonyms(self, obj):
        """
        # 添加一个词语的同义词列表
        :param obj:
        :return:
        """
        obj = obj.result()
        self.candidate_synonym_list.update(obj)

    def get_syno_sents_list(self, input_sentence):
        """
        产生同义句，并返回同义句列表，返回的同义句列表没有包含该句本身
        :param input_sentence: 需要制造同义句的原始句子
        :return:
        """

        assert len(input_sentence) > 0, "Length of sentence must greater than 0."

        seged_sentence = self.segment(input_sentence)

        pool = ThreadPoolExecutor(len(seged_sentence))
        for index, word in enumerate(seged_sentence):
            word_synonyms = [word]
            pool.submit(self.search_synonyms, word, word_synonyms, index).add_done_callback(self.add_synonyms)
        pool.shutdown()
        d1 = sorted(self.candidate_synonym_list.items(), key=lambda k: k[0])
        candidate_synonym_list = [k[1] for k in d1]

        # ################# 组合
        # # 将候选同义词列表们排列组合产生同义句
        # perm_sent = self.permutation(candidate_synonym_list)
        #
        # syno_sent_list = []
        # for p in perm_sent:
        #     syno_sent_list.append("".join(p))
        #
        # return syno_sent_list
        # ################# 组合

        perm_sent = self.permutation_one(candidate_synonym_list)

        return perm_sent
