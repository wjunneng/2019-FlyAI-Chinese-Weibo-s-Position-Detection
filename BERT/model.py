# -*- coding: utf-8 -*
import os
import torch
import math
import jieba
import numpy as np
from flyai.model.base import Base
from pytorch_transformers import BertModel
from torch.utils.data import DataLoader, random_split

from data_utils import Util, ABSADataset, Tokenizer4Bert, PreProcessing
import args

__import__('net', fromlist=["Net"])


class Model(Base):
    def __init__(self, data):
        self.net = None
        self.data = data
        self.args = args
        self.idx2label = dict((i, args.labels[i]) for i in range(len(args.labels)))

        self.tokenizer = Tokenizer4Bert(max_seq_len=self.args.max_seq_len,
                                        pretrained_bert_name=os.path.join(os.getcwd(),
                                                                          self.args.pretrained_bert_name))
        bert = BertModel.from_pretrained(os.path.join(os.getcwd(), self.args.pretrained_bert_name))
        model = self.args.model_classes[args.model_name](bert, self.args).to(self.args.device)

        if self.args.topics is not None:
            self.net_0 = Util.load_model(model=model,
                                         output_dir=os.path.join(os.getcwd(), args.best_model_path,
                                                                 self.args.topics[0]))
            self.net_0.eval()
            self.net_1 = Util.load_model(model=model,
                                         output_dir=os.path.join(os.getcwd(), args.best_model_path,
                                                                 self.args.topics[1]))
            self.net_1.eval()
            self.net_2 = Util.load_model(model=model,
                                         output_dir=os.path.join(os.getcwd(), args.best_model_path,
                                                                 self.args.topics[2]))
            self.net_2.eval()
            self.net_3 = Util.load_model(model=model,
                                         output_dir=os.path.join(os.getcwd(), args.best_model_path,
                                                                 self.args.topics[3]))
            self.net_3.eval()
            self.net_4 = Util.load_model(model=model,
                                         output_dir=os.path.join(os.getcwd(), args.best_model_path,
                                                                 self.args.topics[4]))
            self.net_4.eval()
        else:
            self.net = Util.load_model(model=model, output_dir=os.path.join(os.getcwd(), args.best_model_path))

        # ############################# 特征词库的方法 效果不好
        # self.word_count = pd.read_csv(self.args.word_count_dir)
        # self.word_count_word = list(self.word_count['WORD'])
        # self.word_count_none = dict(zip(self.word_count_word, list(self.word_count['NONE'] / self.word_count['FREQ'])))
        # self.word_count_favor = dict(
        #     zip(self.word_count_word, list(self.word_count['FAVOR'] / self.word_count['FREQ'])))
        # self.word_count_against = dict(
        #     zip(self.word_count_word, list(self.word_count['AGAINST'] / self.word_count['FREQ'])))
        # ############################# 特征词库的方法 效果不好

    def do_predict(self, TEXT, TARGET):
        TEXT_1 = PreProcessing(TEXT).get_file_text()
        predict_set = ABSADataset(data_type=None, fname=(TARGET.tolist(), TEXT_1.tolist(), None),
                                  tokenizer=self.tokenizer)
        predict_loader = DataLoader(dataset=predict_set, batch_size=len(TEXT))
        outputs = None
        for i_batch, sample_batched in enumerate(predict_loader):
            inputs = [sample_batched[col].to(self.args.device) for col in self.args.input_colses[self.args.model_name]]
            if self.args.topics is None:
                outputs = self.net(inputs)
            elif self.args.topics.index(TARGET[0]) == 0:
                outputs = self.net_0(inputs)
            elif self.args.topics.index(TARGET[0]) == 1:
                outputs = self.net_1(inputs)
            elif self.args.topics.index(TARGET[0]) == 2:
                outputs = self.net_2(inputs)
            elif self.args.topics.index(TARGET[0]) == 3:
                outputs = self.net_3(inputs)
            elif self.args.topics.index(TARGET[0]) == 4:
                outputs = self.net_4(inputs)

            # ############################# 特征词库的方法 效果不好
            # WORDS = list(jieba.cut(TEXT_1.tolist()[0], cut_all=False))
            # none, favor, against = 0, 0, 0
            # for word in WORDS:
            #     if word in self.word_count_none:
            #         none += len(word) * self.word_count_none[word]
            #     if word in self.word_count_favor:
            #         favor += len(word) * self.word_count_favor[word]
            #     if word in self.word_count_against:
            #         against += len(word) * self.word_count_against[word]
            #
            # none = 0.3 * none + 0.7 * outputs.detach().numpy().tolist()[0][0]
            # favor = 0.3 * favor + 0.7 * outputs.detach().numpy().tolist()[0][1]
            # against = 0.3 * against + 0.7 * outputs.detach().numpy().tolist()[0][2]
            #
            # outputs = [none, favor, against]
            # outputs = outputs.index(max(outputs))
            # print(
            #     '{},        {},        {},        {},        {},        {},        {}'.format(self.idx2label[outputs],
            #                                                                                   round(none, 4),
            #                                                                                   round(favor, 4),
            #                                                                                   round(against, 4),
            #                                                                                   TARGET[0], TEXT[0],
            #                                                                                   TEXT_1[0]))
            # ############################# 特征词库的方法 效果不好

        outputs = torch.argmax(outputs, dim=-1).numpy().tolist()

        return outputs

    def predict(self, **data):
        TARGET, TEXT = self.data.predict_data(**data)

        return self.do_predict(TARGET=TARGET, TEXT=TEXT)

    def predict_all(self, datas):
        """
        预测所有的数据
        :param datas:
        :return:
        """
        labels = []

        if self.args.predict_batch:
            for i in range(math.ceil(len(datas) / self.args.BATCH)):
                batch_x_data = [datas[i] for i in
                                range(i * self.args.BATCH, min((i + 1) * self.args.BATCH, len(datas)))]
                TARGET = [i['TARGET'] for i in batch_x_data]
                TEXT = [i['TEXT'] for i in batch_x_data]
                labels.extend(self.do_predict(TEXT=np.array(TEXT), TARGET=np.asarray(TARGET)))
        else:
            for data in datas:
                predicts = self.predict(TARGET=data['TARGET'], TEXT=data['TEXT'])

                labels.extend(predicts)

        return labels
