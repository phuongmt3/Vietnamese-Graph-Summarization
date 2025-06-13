import torch
import torch.nn as nn
import torch.nn.functional as F
from graph_layers import MLP, StepWiseGraphConvLayer
from load_data import getPositionEncoding

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PositionVec = torch.stack([torch.from_numpy(getPositionEncoding(i, d=768)) for i in range(200)], dim=0).float().to(device)


class Contrast_Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, heads, act=nn.LeakyReLU(0.1), dropout_p=0.1):
        super(Contrast_Encoder, self).__init__()
        self.graph_encoder = StepWiseGraphConvLayer(in_dim=input_dim, hid_dim=hidden_dim,
                                                    dropout_p=dropout_p, act=act, nheads=heads, iter=1)
        self.common_proj_mlp = MLP([input_dim, hidden_dim, input_dim], layers=2, dropout_p=dropout_p, act=act,
                                   keep_last_layer=False)

    def forward(self, p_gfeature, doc_lens, p_adj, docnum, secnum):
        posVec = torch.cat([PositionVec[:l] for l in doc_lens] + [torch.zeros(secnum+docnum+1, 768).float().to(device)], dim=0)
        p_gfeature = p_gfeature + posVec.unsqueeze(0)
        pg = self.graph_encoder(p_gfeature, p_adj, docnum, secnum)
        pg = self.common_proj_mlp(pg)
        return pg


class End2End_Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, heads, act=nn.LeakyReLU(0.1), dropout_p=0.3):
        super(End2End_Encoder, self).__init__()
        self.graph_encoder = StepWiseGraphConvLayer(in_dim=input_dim, hid_dim=hidden_dim,
                                                    dropout_p=dropout_p, act=act, nheads=heads, iter=1)
        self.dropout = nn.Dropout(dropout_p)
        self.out_proj_layer_mlp = MLP([input_dim, hidden_dim, input_dim], layers=2, dropout_p=dropout_p, act=act,
                                      keep_last_layer=False)
        self.linear = MLP([input_dim, 1], layers=1, dropout_p=dropout_p, act=act, keep_last_layer=True)

    def forward(self, x, doc_lens, adj, docnum, secnum):
        x = self.graph_encoder(x, adj, docnum, secnum)
        x_last = x.clone()
        x = self.out_proj_layer_mlp(x)
        return x, self.linear(x)[:, :-docnum-secnum-1, :]

def _similarity(h1: torch.Tensor, h2: torch.Tensor):
    h1 = F.normalize(h1)
    h2 = F.normalize(h2)
    return h1 @ h2.t()

class InfoNCE(nn.Module):
    def __init__(self, tau):
        super(InfoNCE, self).__init__()
        self.tau = tau

    def forward(self, anchor, sample, pos_mask, *args, **kwargs):
        sim = _similarity(anchor, sample) / self.tau
        if len(anchor) > 1:
            sim, _ = torch.max(sim, dim=0, keepdim=True)
        exp_sim = torch.exp(sim)
        loss = torch.log((exp_sim * pos_mask).sum(dim=1)) - torch.log(exp_sim.sum(dim=1))
        return -loss.mean()
