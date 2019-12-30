# -*- coding: utf-8 -*
import os
import torch
import jieba
from flyai.model.base import Base

import args
from processor import Processor
from vec_utils import read_emb
from vec_text import make_datasets, load_tvt
from model_utils import classify_batch

__import__('net', fromlist=["Net"])
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Model(Base):
    def __init__(self, data):
        self.net = None
        self.data = data
        self.model_dir = os.path.join(os.getcwd(), args.model_dir)
        self.vocab = read_emb(filename=os.path.join(os.getcwd(), args.sgns_dir), stat_lines=1)
        self.processor = Processor()

    def predict(self, **data):
        if self.net is None:
            self.net = torch.load(self.model_dir)
        TARGET, TEXT = self.data.predict_data(**data)
        indexes = " ".join(jieba.cut(TARGET[0], cut_all=False))
        questions = " ".join(jieba.cut(TEXT[0], cut_all=False))
        self.datasets, word2idx, embeddings = make_datasets(vocab=self.vocab,
                                                            raw_data={'prediction': [[indexes], [questions]]},
                                                            label2idx=None,
                                                            big_voc=args.big_voc)
        self.datasets_prediction = load_tvt(tvt_set=self.datasets['prediction'],
                                            max_lens=[args.ans_len, args.ask_len])
        features, seq_lens, mask_matrice, _ = self.datasets_prediction.next_batch(batch_size=1)
        (answers, answers_seqlen, answers_mask), (questions, questions_seqlen, questions_mask) \
            = zip(features, seq_lens, mask_matrice)

        outputs = classify_batch(model=self.net,
                                 features=[answers, answers_seqlen, answers_mask, questions, questions_seqlen,
                                           questions_mask],
                                 max_lens=(args.ans_len, args.ask_len))

        return torch.argmax(outputs[0]).cpu().numpy().tolist()

    def predict_all(self, datas):
        """
        预测所有的数据
        :param datas:
        :return:
        """
        labels = []
        for data in datas:
            predicts = self.predict(TARGET=data['TARGET'], TEXT=data['TEXT'])

            labels.append(predicts)

        return labels
