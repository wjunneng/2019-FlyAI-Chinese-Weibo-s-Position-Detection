# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

torch.manual_seed(123456)

"""

Read-over-and-over model (ROOM) for Stance Classification in community QA

For this task, I believe the following feature would be useful for 
    classifying the final stance
1. final representation should both look at the question and answer
2. Q-type and Answer Pattern: answers following certain patterns 
    are more likely to hold stance towards the question
3. Location Knowledge for Which1-type questions: Which one is better, A or B?


Model: Read the question and answer for many times to get more distilled answer

1. Use an RNN to control the reading process!
2. 



"""


class Net(nn.Module):
    """
        Implementation of RooM for Stance Classification Task
        Procedure:
        1. Ans   : Bi-GRU encoder
        2. Ask   : Bi-GRU encoder (with Position-Encoding)
        3. RNN-based stance reader
           h_t = GRU(h_{t-1}, Q_t, A_t)
           Q_t = \sum{a_i^Q * Q_i^t-1)
           A_t = \sum{a_i^A * A_i^t-1)

    """

    def __init__(self, embeddings, input_dim, hidden_dim, num_layers, output_dim, max_step=(45, 25), dropout=0.5,
                 num_hops=5):
        super(Net, self).__init__()

        self.emb = nn.Embedding(num_embeddings=embeddings.size(0),
                                embedding_dim=embeddings.size(1),
                                padding_idx=0)
        self.emb.weight = nn.Parameter(embeddings)
        # self.emb = nn.Embedding(vocab_size, embedding_dim)

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.nhops = num_hops

        # Ask encoder
        self.ask_len = max_step[1]
        self.ask_rnn = nn.GRU(input_size=self.input_dim,
                              hidden_size=self.hidden_dim,
                              num_layers=num_layers,
                              dropout=dropout,
                              batch_first=True,
                              bidirectional=True)

        # Ans encoder
        self.ans_len = max_step[0]
        self.ans_rnn = nn.GRU(input_size=self.input_dim,
                              hidden_size=self.hidden_dim,
                              num_layers=num_layers,
                              dropout=dropout,
                              batch_first=True,
                              bidirectional=True)

        self.roo = nn.GRUCell(2 * self.hidden_dim, 2 * self.hidden_dim)

        # self.ask_fc = nn.Linear(2*self.hidden_dim, 2*self.hidden_dim)
        # self.ans_fc = nn.Linear(2*self.hidden_dim, 2*self.hidden_dim)
        #
        # self.cat_fc = nn.Linear(2*self.hidden_dim, 2*self.hidden_dim)

        self.output = nn.Linear(2 * self.hidden_dim, self.output_dim)

    def attention(self, aspect, memory, mask_matrix):
        """
        Get the attention results (weighted sum of memory content)
        :param aspect: (batch_size, 2*hidden_dim)
        :param memory: (batch_size, max_len, 2*hidden_dim)
        :param mask_matrix: mask matrix for memory block (batch_size, max_len)
        :return:
        """
        '''   Mask Att weights of padding 0s with -float('inf')  '''
        aspect = torch.unsqueeze(aspect, dim=2)
        # (batch_size, max_len, 2*hidden_dim) * (batch_size, 2*hidden_dim) --> (batch_size, max_len)
        att_weights = torch.bmm(memory, aspect)
        att_weights = torch.squeeze(att_weights, dim=2)

        # replace 0 in mask matrix with -inf
        right_part = (1.0 - mask_matrix)
        right_part = (right_part == 1)
        att_weights.data.masked_fill_(right_part.data, -float("inf"))

        # print(att_weights)

        att_weights = F.softmax(att_weights, dim=1)
        # (batch_size, max_len) --> (batch_size, max_len, 1)
        att_weights = torch.unsqueeze(att_weights, dim=2)

        weighted_sum = memory * att_weights.expand_as(memory)
        weighted_avg = torch.sum(weighted_sum, dim=1)

        return weighted_avg

    def forward(self, answers, questions):
        """
        :param answers: (batch, ans_length), tensor for answer sequence
        :param questions: (batch, ask_length), tensor for ask sequence
        :return:
        """

        ''' Embedding Layer | Padding | Sequence_length 25/45'''
        ask_batch_index, ask_lengths, ask_mask_matrix = questions
        ask_batch = self.emb(ask_batch_index)
        ans_batch, ans_lengths, ans_mask_matrix = answers
        ans_batch = self.emb(ans_batch)

        batch_size = len(ask_batch)

        ''' Bi-GRU Layer 
                Batch&Pad: torch.nn.utils.rnn.pad_packed_sequence
        '''
        # _outs: (batch_size, seq_len, features)
        ask_outs, _ = self.ask_rnn(ask_batch.view(batch_size, -1, self.input_dim))
        ans_outs, _ = self.ans_rnn(ans_batch.view(batch_size, -1, self.input_dim))

        # Batch_first only change viewpoint, may not be contiguous
        ans_rnn = ans_outs.contiguous().view(batch_size, -1, 2 * self.hidden_dim)  # (batch, ans_len, 2*hid)
        ask_rnn = ask_outs.contiguous().view(batch_size, -1, 2 * self.hidden_dim)  # (batch, ask_len, 2*hid)

        ''' Individual ATT Layer '''
        ans_mask = ans_mask_matrix.view(batch_size, -1, 1).float()
        ask_mask = ask_mask_matrix.view(batch_size, -1, 1).float()

        ans_mask = ans_mask.expand_as(ans_rnn)  # (batch, ans_len, 2*hid)
        ask_mask = ask_mask.expand_as(ask_rnn)  # (batch, ask_len, 2*hid)

        ans_rnn = ans_rnn * ans_mask  # (batch, ans_len, 2*hid)
        ask_rnn = ask_rnn * ask_mask  # (batch, ask_len, 2*hid)

        # h_0 = Variable(torch.rand(batch_size, 2*self.hidden_dim)).cuda()
        h_0 = Variable(torch.zeros(batch_size, 2 * self.hidden_dim))  # .cuda()
        h_t = h_0
        for step_t in range(self.nhops):
            # for step_t in range(3):
            # attend question first
            q_t = self.attention(h_t, ask_rnn, ask_mask_matrix)
            h_t = self.roo(q_t, h_t)

            h_t = F.tanh(h_t)  # maybe useful

            # attend answer
            a_t = self.attention(h_t, ans_rnn, ans_mask_matrix)
            h_t = self.roo(a_t, h_t)

        representation = h_t
        out = self.output(representation)
        out_scores = F.softmax(out, dim=1)

        key_index = Variable(torch.LongTensor([0] * batch_size))  # .cuda()

        # print out_scores
        return out_scores, key_index.view(-1)
