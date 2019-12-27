# -*- coding:utf-8 -*-
import os
import random
import numpy as np
import torch
import logging
import sys
import math
from torch import nn
from time import strftime, localtime
from pytorch_transformers import BertModel
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import f1_score

from BERT import args
from BERT.utils.data_utils import ABSADataset
from BERT.utils.data_utils import Tokenizer4Bert, bulid_tokenizer, build_embedding_matrix

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))


class Instructor(object):
    def __init__(self, args):
        self.args = args

        if 'bert' in self.args.model_name:
            tokenizer = Tokenizer4Bert(max_seq_len=self.args.max_seq_len,
                                       pretrained_bert_name=self.args.pretrained_bert_name)
            bert = BertModel.from_pretrained(pretrained_model_name_or_path=self.args.pretrained_bert_name)
            self.model = self.args.model_class(bert, self.args).to(self.args.device)
        else:
            tokenizer = bulid_tokenizer(
                fnames=[self.args.dataset_file['train'], self.args.dataset_file['test']],
                max_seq_len=self.args.max_seq_len,
                dat_fname='{0}_tokenizer.dat'.format(self.args.dataset)
            )
            embedding_matrix = build_embedding_matrix(
                word2idx=tokenizer.word2idx,
                embed_dim=self.args.embed_dim,
                dat_fname='{0}_{1}_embedding_matrix.dat'.format(str(self.args.embed_dim), self.args.dataset)
            )
            self.model = self.args.model_class(embedding_matrix, self.args).to(self.args.device)

        self.trainset = ABSADataset(self.args.dataset_file['train'], tokenizer)
        self.testset = ABSADataset(self.args.dataset_file['test'], tokenizer)
        assert 0 <= self.args.valset_ratio < 1
        if self.args.valset_ratio > 0:
            valset_len = int(len(self.trainset) * self.args.valset_ratio)
            self.trainset, self.valset = random_split(self.trainset, (len(self.trainset) - valset_len, valset_len))
        else:
            self.valset = self.testset

        if self.args.device.type == 'cuda':
            logger.info('cuda memory allocated: {}'.format(torch.cuda.memory_allocated(device=self.args.device.index)))

        self._print_args()

    def _print_args(self):
        n_trainable_params, n_nontrainable_params = 0, 0
        for p in self.model.parameters():
            n_params = torch.prod(torch.tensor(p.shape))
            if p.requires_grad:
                n_trainable_params += n_params
            else:
                n_nontrainable_params += n_params

        logger.info(
            'n_trainable_params: {0}, n_nontrainable_params: {1}'.format(n_trainable_params, n_nontrainable_params))
        logger.info('> training arguments:')
        for arg in vars(self.args):
            logger.info('>>> {0}:{1}'.format(arg, getattr(self.args, arg)))

    def _reset_params(self):
        for child in self.model.children():
            if type(child) != BertModel:
                for p in child.parameters():
                    if p.requires_grad:
                        if len(p.shape) > 1:
                            self.args.initializer(p)
                        else:
                            stdv = 1. / math.sqrt(p.shape[0])
                            torch.nn.init.uniform_(p, a=-stdv, b=stdv)

    def _evaluate_acc_f1(self, data_loader):
        n_correct, n_total = 0, 0
        t_targets_all, t_outputs_all = None, None
        with torch.no_grad():
            for t_batch, t_sample_batched in enumerate(data_loader):
                t_inputs = [t_sample_batched[col].to(self.args.device) for col in self.args.inputs_cols]
                t_outputs = self.model(t_inputs)
                t_targets = t_sample_batched['polarity'].to(self.args.device)

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

    def _train(self, criterion, optimizer, train_data_loader, val_data_loader):
        max_val_acc = 0
        max_val_f1 = 0
        global_step = 0
        path = None
        for epoch in range(self.args.num_epoch):
            logger.info('>' * 100)
            logger.info('epoch: {}'.format(epoch))
            n_correct, n_total, loss_total = 0, 0, 0
            # switch model to training mode
            self.model.train()
            for i_batch, sample_batched in enumerate(train_data_loader):
                global_step += 1
                # clear gradient accumulators
                optimizer.zero_grad()

                inputs = [sample_batched[col].to(self.args.device) for col in self.args.inputs_cols]
                outputs = self.model(inputs)
                targets = sample_batched['polarity'].to(self.args.device)

                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                n_correct += (torch.argmax(outputs, -1) == targets).sum().item()
                n_total += len(outputs)
                loss_total += loss.item() * len(outputs)
                if global_step % self.args.log_step == 0:
                    train_acc = n_correct / n_total
                    train_loss = loss_total / n_total
                    logger.info('loss: {:.4f}, acc: {:.4f}'.format(train_loss, train_acc))

            val_acc, val_f1 = self._evaluate_acc_f1(val_data_loader)
            logger.info('> val_acc: {:.4f}, val_f1: {:.4f}'.format(val_acc, val_f1))
            if val_acc > max_val_acc:
                max_val_acc = val_acc
                if not os.path.exists('state_dict'):
                    os.mkdir('state_dict')
                path = 'state_dict/{0}_{1}_val_acc{2}'.format(self.args.model_name, self.args.dataset,
                                                              round(val_acc, 4))
                torch.save(self.model.state_dict(), path)
                logger.info('>> saved: {}'.format(path))
            if val_f1 > max_val_f1:
                max_val_f1 = val_f1

        logger.info('> max_val_acc: {0} max_val_f1: {1}'.format(max_val_acc, max_val_f1))

        return path

    def run(self):
        # loss and optimizer
        criterion = nn.CrossEntropyLoss()
        _params = filter(lambda x: x.requires_grad, self.model.parameters())
        optimizer = self.args.optimizer(_params, lr=self.args.learning_rate, weight_decay=self.args.l2reg)

        train_data_loader = DataLoader(dataset=self.trainset, batch_size=self.args.batch_size, shuffle=True)
        val_data_loader = DataLoader(dataset=self.valset, batch_size=self.args.batch_size, shuffle=False)

        self._reset_params()
        # 训练
        best_model_path = self._train(criterion=criterion, optimizer=optimizer, train_data_loader=train_data_loader,
                                      val_data_loader=val_data_loader)
        logger.info('> train save model path: {}'.format(best_model_path))

        # 测试
        # test_data_loader = DataLoader(dataset=self.testset, batch_size=self.args.batch_size, shuffle=False)
        # self.model.load_state_dict(torch.load(best_model_path))
        # self.model.eval()
        # test_acc, test_f1 = self._evaluate_acc_f1(test_data_loader)
        # logger.info('>> test_acc: {:.4f}, test_f1: {:.4f}'.format(test_acc, test_f1))


if __name__ == '__main__':
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    args.model_class = args.model_classes[args.model_name]
    args.dataset_file = args.dataset_files[args.dataset]
    args.inputs_cols = args.input_colses[args.model_name]
    args.initializer = args.initializers[args.initializer]
    args.optimizer = args.optimizers[args.optimizer]
    args.device = torch.device(
        'cuda:0' if torch.cuda.is_available() else 'cpu') if args.device is None else torch.device(args.device)

    log_file = '{}-{}-{}.log'.format(args.model_name, args.dataset, strftime('%y%m%d-%H%M', localtime()))
    logger.addHandler(logging.FileHandler(log_file))

    instructor = Instructor(args)
    instructor.run()
