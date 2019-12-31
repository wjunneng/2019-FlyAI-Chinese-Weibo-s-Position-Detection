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

    def predict(self, **data):
        if self.net is None:
            model_dir = os.path.join(os.getcwd(), args.best_model_path)
            self.tokenizer = Tokenizer4Bert(max_seq_len=self.args.max_seq_len,
                                            pretrained_bert_name=os.path.join(os.getcwd(),
                                                                              self.args.pretrained_bert_name))
            bert = BertModel.from_pretrained(os.path.join(os.getcwd(), self.args.pretrained_bert_name))
            model = self.args.model_classes[args.model_name](bert, self.args).to(self.args.device)
            self.net = Util.load_model(model=model, output_dir=model_dir)

        TARGET, TEXT = self.data.predict_data(**data)
        TEXT = PreProcessing(TEXT).get_file_text()
        predict_set = ABSADataset(data_type=None, fname=(TARGET.tolist(), TEXT.tolist(), None),
                                  tokenizer=self.tokenizer)
        predict_set, _ = random_split(predict_set, (len(predict_set), 0))
        predict_loader = DataLoader(dataset=predict_set, batch_size=1, shuffle=True)
        self.net.eval()
        outputs = None
        for i_batch, sample_batched in enumerate(predict_loader):
            inputs = [sample_batched[col].to(self.args.device) for col in self.args.input_colses[self.args.model_name]]
            outputs = self.net(inputs)
        outputs = torch.argmax(outputs).numpy().tolist()

        print(' [{}],     {},        {}'.format(self.idx2label[outputs], TARGET[0], TEXT[0]))
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
