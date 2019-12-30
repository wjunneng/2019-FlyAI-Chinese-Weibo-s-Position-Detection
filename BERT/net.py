import torch
import torch.nn as nn

from squeeze_embedding import SqueezeEmbedding
from attention import Attention
from point_wise_feed_forward import PositionwiseFeedForward


class Net(nn.Module):
    def __init__(self, bert, opt):
        super(Net, self).__init__()
        self.opt = opt
        self.bert = bert
        self.squeeze_embedding = SqueezeEmbedding()
        self.dropout = nn.Dropout(opt.dropout)

        self.attn_k = Attention(opt.bert_dim, out_dim=opt.hidden_dim, n_head=8, score_function='mlp',
                                dropout=opt.dropout)
        self.attn_q = Attention(opt.bert_dim, out_dim=opt.hidden_dim, n_head=8, score_function='mlp',
                                dropout=opt.dropout)
        self.ffn_c = PositionwiseFeedForward(opt.hidden_dim, dropout=opt.dropout)
        self.ffn_t = PositionwiseFeedForward(opt.hidden_dim, dropout=opt.dropout)

        self.attn_s1 = Attention(opt.hidden_dim, n_head=8, score_function='mlp', dropout=opt.dropout)

        self.dense = nn.Linear(opt.hidden_dim * 3, opt.polarities_dim)

    def forward(self, inputs):
        context, target = inputs[0], inputs[1]
        context_len = torch.sum(context != 0, dim=-1)
        target_len = torch.sum(target != 0, dim=-1)
        context = self.squeeze_embedding(context, context_len)
        context, _ = self.bert(context)
        context = self.dropout(context)
        target = self.squeeze_embedding(target, target_len)
        target, _ = self.bert(target)
        target = self.dropout(target)

        hc, _ = self.attn_k(context, context)
        hc = self.ffn_c(hc)
        ht, _ = self.attn_q(context, target)
        ht = self.ffn_t(ht)

        s1, _ = self.attn_s1(hc, ht)

        context_len = torch.tensor(context_len, dtype=torch.float).to(self.opt.device)
        target_len = torch.tensor(target_len, dtype=torch.float).to(self.opt.device)

        hc_mean = torch.div(torch.sum(hc, dim=1), context_len.view(context_len.size(0), 1))
        ht_mean = torch.div(torch.sum(ht, dim=1), target_len.view(target_len.size(0), 1))
        s1_mean = torch.div(torch.sum(s1, dim=1), context_len.view(context_len.size(0), 1))

        x = torch.cat((hc_mean, s1_mean, ht_mean), dim=-1)
        out = self.dense(x)
        return out
