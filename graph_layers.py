import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, dims: list, layers=2, act=nn.LeakyReLU(), dropout_p=0.1, keep_last_layer=False):
        super(MLP, self).__init__()
        assert len(dims) == layers + 1
        self.layers = layers
        self.act = act
        self.dropout = nn.Dropout(dropout_p)
        self.keep_last = keep_last_layer

        self.mlp_layers = nn.ModuleList([])
        for i in range(self.layers):
            self.mlp_layers.append(nn.Linear(dims[i], dims[i + 1]))

    def forward(self, x):
        for i in range(len(self.mlp_layers) - 1):
            x = self.dropout(self.act(self.mlp_layers[i](x)))
        if self.keep_last:
            x = self.mlp_layers[-1](x)
        else:
            x = self.act(self.mlp_layers[-1](x))
        return x


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, n_heads: int,
                 is_concat: bool = True,
                 dropout: float = 0.6,
                 leaky_relu_negative_slope: float = 0.2):
        super().__init__()

        self.is_concat = is_concat
        self.n_heads = n_heads

        # Calculate the number of dimensions per head
        if is_concat:
            assert out_features % n_heads == 0
            self.n_hidden = out_features // n_heads
        else:
            self.n_hidden = out_features

        self.linear = nn.Linear(in_features, self.n_hidden * n_heads, bias=False)
        self.attn = nn.Linear(self.n_hidden * 2, 1, bias=False)
        self.activation = nn.LeakyReLU(negative_slope=leaky_relu_negative_slope)
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, adj_mat: torch.Tensor, docnum, secnum):
        n_nodes = h.shape[0]
        g = self.linear(h).view(n_nodes, self.n_heads, self.n_hidden)
        g_repeat = g.repeat(n_nodes, 1, 1)
        g_repeat_interleave = g.repeat_interleave(n_nodes, dim=0)
        g_concat = torch.cat([g_repeat_interleave, g_repeat], dim=-1)
        g_concat = g_concat.view(n_nodes, n_nodes, self.n_heads, 2 * self.n_hidden)
        e = self.activation(self.attn(g_concat))

        e = e.squeeze(-1)

        # The adjacency matrix should have shape
        # `[n_nodes, n_nodes, n_heads]` or`[n_nodes, n_nodes, 1]`
        assert adj_mat.shape[0] == 1 or adj_mat.shape[0] == n_nodes
        assert adj_mat.shape[1] == 1 or adj_mat.shape[1] == n_nodes
        assert adj_mat.shape[2] == 1 or adj_mat.shape[2] == self.n_heads
        # Mask $e_{ij}$ based on adjacency matrix.
        # $e_{ij}$ is set to $- \infty$ if there is no edge from $i$ to $j$.
        e = e.masked_fill(adj_mat == 0, float(-1e9))
        a = self.softmax(e)
        a = self.dropout(a)
        attn_res = torch.einsum('ijh,jhf->ihf', a, g)

        # Concatenate the heads
        if self.is_concat:
            return attn_res.reshape(n_nodes, self.n_heads * self.n_hidden)
        # Take the mean of the heads
        else:
            return attn_res.mean(dim=1)


class GAT(nn.Module):
    def __init__(self, in_features: int, n_hidden: int, n_classes: int, n_heads: int, dropout: float):
        super().__init__()
        self.layer1 = GraphAttentionLayer(in_features, n_hidden, n_heads, is_concat=True, dropout=dropout)
        self.activation = nn.ELU()
        self.output = GraphAttentionLayer(n_hidden, n_classes, 1, is_concat=False, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adj_mat: torch.Tensor, docnum, secnum):
        x = x.squeeze(0)
        adj_mat = adj_mat.squeeze(0)
        adj_x = adj_mat.clone().sum(dim=1, keepdim=True).repeat(1, x.shape[1]).bool()
        adj_mat = adj_mat.unsqueeze(-1).bool()
        x = self.dropout(x)
        x = self.layer1(x, adj_mat, docnum, secnum)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.output(x, adj_mat, docnum, secnum).masked_fill(adj_x == 0, float(0))
        return x.unsqueeze(0)


class StepWiseGraphConvLayer(nn.Module):
    def __init__(self, in_dim, hid_dim, dropout_p=0.1, act=nn.LeakyReLU(), nheads=6, iter=1, final="att"):
        super().__init__()
        self.act = act
        self.dropout = nn.Dropout(dropout_p)
        self.iter = iter
        self.in_dim = in_dim
        self.gat = nn.ModuleList([GAT(in_features=in_dim, n_hidden=hid_dim, n_classes=in_dim,
                                      dropout=dropout_p, n_heads=nheads) for _ in range(iter)])
        self.gat2 = nn.ModuleList([GAT(in_features=in_dim, n_hidden=hid_dim, n_classes=in_dim,
                                       dropout=dropout_p, n_heads=nheads) for _ in range(iter)])
        self.gat3 = nn.ModuleList([GAT(in_features=in_dim, n_hidden=hid_dim, n_classes=in_dim,
                                       dropout=dropout_p, n_heads=nheads) for _ in range(iter)])

        self.out_ffn = MLP([in_dim * 3, hid_dim, hid_dim, in_dim], layers=3, dropout_p=dropout_p)

    def forward(self, feature, adj, docnum, secnum):
        sen_adj = adj.clone()
        sen_adj[:, -docnum - secnum - 1:, :] = sen_adj[:, :, -docnum - secnum - 1:] = 0
        sec_adj = adj.clone()
        sec_adj[:, :-docnum - secnum - 1, :] = sec_adj[:, -docnum - 1:, :] = sec_adj[:, :, -docnum - 1:] = 0
        doc_adj = adj.clone()
        doc_adj[:, :-docnum - 1, :] = 0

        feature_sen = feature.clone()
        feature_resi = feature

        feature_sen_re = feature_sen.clone()
        for i in range(0, self.iter):
            feature_sen = self.gat[i](feature_sen, sen_adj, docnum, secnum)
        feature_sen = F.layer_norm(feature_sen + feature_sen_re, [self.in_dim])

        feature_sec = feature_sen.clone()
        feature_sec_re = feature_sec.clone()
        for i in range(0, self.iter):
            feature_sec = self.gat2[i](feature_sec, sec_adj, docnum, secnum)
        feature_sec = F.layer_norm(feature_sec + feature_sec_re, [self.in_dim])

        feature_doc = feature_sec.clone()
        feature_doc_re = feature_doc.clone()
        for i in range(0, self.iter):
            feature_doc = self.gat3[i](feature_doc, doc_adj, docnum, secnum)
        feature_doc = F.layer_norm(feature_doc + feature_doc_re, [self.in_dim])

        feature_sec[:, :-docnum-secnum-1, :] = adj[:, :-docnum-secnum-1, -docnum-secnum-1:-docnum-1] @ feature_sec[:, -docnum-secnum-1:-docnum-1, :]
        feature_doc[:, -docnum-secnum-1:-docnum-1, :] = adj[:, -docnum-secnum-1:-docnum-1, -docnum-1:] @ feature_doc[:, -docnum-1:, :]
        feature_doc[:, :-docnum-secnum-1, :] = adj[:, :-docnum-secnum-1, -docnum-secnum-1:-docnum-1] @ feature_doc[:, -docnum-secnum-1:-docnum-1, :]
        feature = torch.concat([feature_doc, feature_sec, feature_sen], dim=-1)
        feature = F.layer_norm(self.out_ffn(feature) + feature_resi, [self.in_dim])
        return feature
