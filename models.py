import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import MLP


class Contrast_Encoder(nn.Module):
    def __init__(self, graph_encoder, hidden_dim, bert_hidden=768, in_dim=768, dropout_p=0.3):
        super(Contrast_Encoder, self).__init__()
        self.graph_encoder = graph_encoder
        self.common_proj_mlp = MLP(in_dim, in_dim, hidden_dim, dropout_p=dropout_p, act=nn.LeakyReLU())

    def forward(self, p_gfeature, p_adj, docnum, secnum):
        pg = self.graph_encoder(p_gfeature.float(), p_adj.float(), docnum, secnum)
        pg = self.common_proj_mlp(pg)
        return pg


class End2End_Encoder(nn.Module):
    def __init__(self, graph_encoder, in_dim, hidden_dim, dropout_p):
        super(End2End_Encoder, self).__init__()
        self.graph_encoder = graph_encoder
        self.dropout = nn.Dropout(dropout_p)
        self.out_proj_layer_mlp = MLP(in_dim, in_dim, hidden_dim, act=nn.LeakyReLU(), dropout_p=dropout_p, layers=2)
        self.final_layer = nn.Linear(in_dim, 1)

    def forward(self, x, adj, docnum, secnum):
        x = self.graph_encoder(x.float(), adj.float(), docnum, secnum)
        x = x[:, :-docnum-secnum-1, :]
        x = self.out_proj_layer_mlp(x)
        x = self.final_layer(x)
        return x


def _similarity(h1: torch.Tensor, h2: torch.Tensor):
    h1 = F.normalize(h1)
    h2 = F.normalize(h2)
    return h1 @ h2.t()


class InfoNCE(nn.Module):
    def __init__(self, tau):
        super(InfoNCE, self).__init__()
        self.tau = tau

    def forward(self, anchor, sample, pos_mask, neg_mask):
        sim = _similarity(anchor, sample) / self.tau
        if len(anchor) > 1:
            sim, _ = torch.max(sim, dim=0)
        exp_sim = torch.exp(sim) * (pos_mask + neg_mask)
        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True))
        loss = log_prob * pos_mask
#         loss[torch.isnan(loss)] = 0.0
        loss = loss.sum(dim=1) / pos_mask.sum(dim=1)
        return -loss.mean()
