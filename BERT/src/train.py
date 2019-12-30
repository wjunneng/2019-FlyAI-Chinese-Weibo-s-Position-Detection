# -*- coding:utf-8 -*-
import os
import random
import numpy as np
import torch
import logging
import sys
import shutil
from torch import nn
from time import strftime, localtime
from pytorch_transformers import BertModel
from torch.utils.data import DataLoader, random_split
from flyai.utils import remote_helper

import argparse
from BERT import args as arguments
from BERT.data_utils import ABSADataset, Tokenizer4Bert
from BERT.data_utils import Util

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))

remote_helper.get_remote_date("https://www.flyai.com/m/chinese_base.zip")
shutil.copyfile(os.path.join(os.getcwd(), 'vocab.txt'),
                os.path.join(os.getcwd(), arguments.pretrained_bert_name, 'vocab.txt'))


class Instructor(object):
    def __init__(self, arguments):
        # 项目的超参
        parser = argparse.ArgumentParser()
        parser.add_argument("-e", "--EPOCHS", default=50, type=int, help="train epochs")
        parser.add_argument("-b", "--BATCH", default=2, type=int, help="batch size")
        self.args = parser.parse_args()
        self.arguments = arguments

        if 'bert' in self.arguments.model_name:
            tokenizer = Tokenizer4Bert(max_seq_len=self.arguments.max_seq_len,
                                       pretrained_bert_name=self.arguments.pretrained_bert_name)
            bert = BertModel.from_pretrained(pretrained_model_name_or_path=self.arguments.pretrained_bert_name)
            self.model = self.arguments.model_class(bert, self.arguments).to(self.arguments.device)
        else:
            tokenizer = Util.bulid_tokenizer(
                fnames=[self.arguments.dataset_file['train'], self.arguments.dataset_file['test']],
                max_seq_len=self.arguments.max_seq_len,
                dat_fname='{0}_tokenizer.dat'.format(self.arguments.dataset)
            )
            embedding_matrix = Util.build_embedding_matrix(
                word2idx=tokenizer.word2idx,
                embed_dim=self.arguments.embed_dim,
                dat_fname='{0}_{1}_embedding_matrix.dat'.format(str(self.arguments.embed_dim), self.arguments.dataset)
            )
            self.model = self.arguments.model_class(embedding_matrix, self.arguments).to(self.arguments.device)

        self.trainset = ABSADataset(data_type=self.arguments.data_type, fname=self.arguments.dataset_file['train'],
                                    tokenizer=tokenizer)

        assert 0 <= self.arguments.valset_ratio < 1
        if self.arguments.valset_ratio > 0:
            valset_len = int(len(self.trainset) * self.arguments.valset_ratio)
            self.trainset, self.valset = random_split(self.trainset, (len(self.trainset) - valset_len, valset_len))
        else:
            self.testset = ABSADataset(data_type=self.arguments.data_type, fname=self.arguments.dataset_file['test'],
                                       tokenizer=tokenizer)
            self.valset = self.testset

        if self.arguments.device.type == 'cuda':
            logger.info(
                'cuda memory allocated: {}'.format(torch.cuda.memory_allocated(device=self.arguments.device.index)))

        Util.print_args(model=self.model, logger=logger, args=self.arguments)

    def run(self):
        # loss and optimizer
        criterion = nn.CrossEntropyLoss()
        _params = filter(lambda x: x.requires_grad, self.model.parameters())
        optimizer = self.arguments.optimizer(_params, lr=self.arguments.learning_rate,
                                             weight_decay=self.arguments.l2reg)

        train_data_loader = DataLoader(dataset=self.trainset, batch_size=self.args.BATCH, shuffle=True)
        val_data_loader = DataLoader(dataset=self.valset, batch_size=self.args.BATCH, shuffle=False)

        Util.reset_params(model=self.model, args=self.arguments)
        # 训练
        max_val_acc = 0
        max_val_f1 = 0
        global_step = 0
        best_model_path = None
        for epoch in range(self.args.EPOCHS):
            logger.info('>' * 100)
            logger.info('epoch: {}'.format(epoch))
            n_correct, n_total, loss_total = 0, 0, 0
            # switch model to training mode
            self.model.train()
            for i_batch, sample_batched in enumerate(train_data_loader):
                global_step += 1
                # clear gradient accumulators
                optimizer.zero_grad()

                inputs = [sample_batched[col].to(self.arguments.device) for col in self.arguments.inputs_cols]
                outputs = self.model(inputs)
                targets = sample_batched['polarity'].to(self.arguments.device)

                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                n_correct += (torch.argmax(outputs, -1) == targets).sum().item()
                n_total += len(outputs)
                loss_total += loss.item() * len(outputs)
                if global_step % self.arguments.log_step == 0:
                    train_acc = n_correct / n_total
                    train_loss = loss_total / n_total
                    logger.info('loss: {:.4f}, acc: {:.4f}'.format(train_loss, train_acc))

            val_acc, val_f1 = Util.evaluate_acc_f1(model=self.model, args=self.arguments, data_loader=val_data_loader)
            logger.info('> val_acc: {:.4f}, val_f1: {:.4f}'.format(val_acc, val_f1))
            if val_acc > max_val_acc:
                max_val_acc = val_acc
                if not os.path.exists('state_dict'):
                    os.mkdir('state_dict')
                best_model_path = 'state_dict/{0}_{1}_val_acc{2}'.format(self.arguments.model_name,
                                                                         self.arguments.dataset,
                                                                         round(val_acc, 4))
                torch.save(self.model.state_dict(), best_model_path)
                logger.info('>> saved: {}'.format(best_model_path))
            if val_f1 > max_val_f1:
                max_val_f1 = val_f1

        logger.info('> max_val_acc: {0} max_val_f1: {1}'.format(max_val_acc, max_val_f1))

        logger.info('> train save model path: {}'.format(best_model_path))

        # 测试
        # test_data_loader = DataLoader(dataset=self.testset, batch_size=self.args.BATCH, shuffle=False)
        # self.model.load_state_dict(torch.load(best_model_path))
        # self.model.eval()
        # test_acc, test_f1 = self._evaluate_acc_f1(test_data_loader)
        # logger.info('>> test_acc: {:.4f}, test_f1: {:.4f}'.format(test_acc, test_f1))


if __name__ == '__main__':
    if arguments.seed is not None:
        random.seed(arguments.seed)
        np.random.seed(arguments.seed)
        torch.manual_seed(arguments.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    arguments.model_class = arguments.model_classes[arguments.model_name]
    arguments.dataset_file = arguments.dataset_files[arguments.dataset]
    arguments.inputs_cols = arguments.input_colses[arguments.model_name]
    arguments.initializer = arguments.initializers[arguments.initializer]
    arguments.optimizer = arguments.optimizers[arguments.optimizer]
    arguments.device = torch.device(
        'cuda:0' if torch.cuda.is_available() else 'cpu') if arguments.device is None else torch.device(
        arguments.device)

    log_file = '{}-{}-{}.log'.format(arguments.model_name, arguments.dataset, strftime('%y%m%d-%H%M', localtime()))
    logger.addHandler(logging.FileHandler(log_file))

    instructor = Instructor(arguments)
    instructor.run()
