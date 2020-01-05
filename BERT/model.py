# -*- coding: utf-8 -*
import os
from flyai.model.base import Base
from pytorch_transformers import BertModel
import torch
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

    def predict(self, **data):
        TARGET, TEXT = self.data.predict_data(**data)
        TEXT_1 = PreProcessing(TEXT).get_file_text()
        predict_set = ABSADataset(data_type=None, fname=(TARGET.tolist(), TEXT_1.tolist(), None),
                                  tokenizer=self.tokenizer)
        predict_set, _ = random_split(predict_set, (len(predict_set), 0))
        predict_loader = DataLoader(dataset=predict_set, batch_size=1, shuffle=True)
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

            none, favor, against = outputs.detach().numpy().tolist()[0]
            outputs = torch.argmax(outputs).numpy().tolist()

            print(
                '{},        {},        {},        {},        {},        {},        {}'.format(self.idx2label[outputs],
                                                                                              round(none, 4),
                                                                                              round(favor, 4),
                                                                                              round(against, 4),
                                                                                              TARGET[0], TEXT[0],
                                                                                              TEXT_1[0]))
        return outputs

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
