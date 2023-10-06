import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hid_dim, layers=2, act=nn.LeakyReLU(), dropout_p=0.3, keep_last_layer=False):
        super(MLP, self).__init__()
        self.layers = layers
        self.act = act
        self.dropout = nn.Dropout(dropout_p)
        self.keep_last = keep_last_layer

        self.mlp_layers = nn.ModuleList([])
        if layers == 1:
            self.mlp_layers.append(nn.Linear(in_dim, out_dim))
        else:
            self.mlp_layers.append(nn.Linear(in_dim, hid_dim))
            for i in range(self.layers - 2):
                self.mlp_layers.append(nn.Linear(hid_dim, hid_dim))
            self.mlp_layers.append(nn.Linear(hid_dim, out_dim))

    def forward(self, x):
        for i in range(len(self.mlp_layers) - 1):
            x = self.dropout(self.act(self.mlp_layers[i](x)))
        if self.keep_last:
            x = self.mlp_layers[-1](x)
        else:
            x = self.act(self.mlp_layers[-1](x))
        return x


# borrowed from labml.ai
class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, n_heads: int,
                 is_concat: bool = True,
                 dropout: float = 0.6,
                 leaky_relu_negative_slope: float = 0.2):
        """
        * `in_features`, $F$, is the number of input features per node
        * `out_features`, $F'$, is the number of output features per node
        * `n_heads`, $K$, is the number of attention heads
        * `is_concat` whether the multi-head results should be concatenated or averaged
        * `dropout` is the dropout probability
        * `leaky_relu_negative_slope` is the negative slope for leaky relu activation
        """
        super().__init__()

        self.is_concat = is_concat
        self.n_heads = n_heads

        # Calculate the number of dimensions per head
        if is_concat:
            assert out_features % n_heads == 0
            # If we are concatenating the multiple heads
            self.n_hidden = out_features // n_heads
        else:
            # If we are averaging the multiple heads
            self.n_hidden = out_features

        # Linear layer for initial transformation;
        # i.e. to transform the node embeddings before self-attention
        self.linear = nn.Linear(in_features, self.n_hidden * n_heads, bias=False)
        # Linear layer to compute attention score $e_{ij}$
        self.attn = nn.Linear(self.n_hidden * 2, 1, bias=False)
        # The activation for attention score $e_{ij}$
        self.activation = nn.LeakyReLU(negative_slope=leaky_relu_negative_slope)
        # Softmax to compute attention $\alpha_{ij}$
        self.softmax = nn.Softmax(dim=1)
        # Dropout layer to be applied for attention
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, adj_mat: torch.Tensor):
        """
        * `h`, $\mathbf{h}$ is the input node embeddings of shape `[n_nodes, in_features]`.
        * `adj_mat` is the adjacency matrix of shape `[n_nodes, n_nodes, n_heads]`.
        We use shape `[n_nodes, n_nodes, 1]` since the adjacency is the same for each head.

        Adjacency matrix represent the edges (or connections) among nodes.
        `adj_mat[i][j]` is `True` if there is an edge from node `i` to node `j`.
        """

        # Number of nodes
        n_nodes = h.shape[0]
        g = self.linear(h).view(n_nodes, self.n_heads, self.n_hidden)
        g_repeat = g.repeat(n_nodes, 1, 1)
        g_repeat_interleave = g.repeat_interleave(n_nodes, dim=0)
        g_concat = torch.cat([g_repeat_interleave, g_repeat], dim=-1)
        g_concat = g_concat.view(n_nodes, n_nodes, self.n_heads, 2 * self.n_hidden)
        e = self.activation(self.attn(g_concat))
        # del g_concat, g_repeat, g_repeat_interleave
        # torch.cuda.empty_cache()
        # Remove the last dimension of size `1`
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

    def forward(self, x: torch.Tensor, adj_mat: torch.Tensor):
        x = x.squeeze(0)
        adj_mat = adj_mat.squeeze(0)
        adj_mat = adj_mat.unsqueeze(-1).bool()
        x = self.dropout(x)
        #         print('inp', x)
        x = self.layer1(x, adj_mat)
        #         print(x)
        x = self.activation(x)
        x = self.dropout(x)
        return self.output(x, adj_mat).unsqueeze(0)


class StepWiseGraphConvLayer(nn.Module):
    def __init__(self, in_dim, out_dim, hid_dim, dropout_p=0.3, act=nn.LeakyReLU(), nheads=6, iter=1, final="att"):
        super().__init__()
        self.act = act
        self.dropout = nn.Dropout(dropout_p)
        self.iter = iter
        self.gat = nn.ModuleList([GAT(in_features=in_dim, n_hidden=hid_dim, n_classes=in_dim,
                                      dropout=dropout_p, n_heads=nheads) for _ in range(iter)])
        self.gat2 = nn.ModuleList([GAT(in_features=in_dim, n_hidden=hid_dim, n_classes=in_dim,
                                       dropout=dropout_p, n_heads=nheads) for _ in range(iter)])
        self.gat3 = nn.ModuleList([GAT(in_features=in_dim, n_hidden=hid_dim, n_classes=in_dim,
                                       dropout=dropout_p, n_heads=nheads) for _ in range(iter)])

        self.feature_fusion_layer = nn.Linear(in_dim * 3, in_dim)
        self.ffn = MLP(in_dim, in_dim, hid_dim, dropout_p=dropout_p, layers=3)
        self.out_ffn = MLP(in_dim, in_dim, hid_dim, dropout_p=dropout_p)

    def forward(self, feature, adj, docnum, secnum):
        sen_adj = adj.clone()
        sen_adj[:, :, -docnum - secnum - 1:] = 0
        sec_adj = adj.clone()
        sec_adj[:, :, :-docnum - secnum - 1] = sec_adj[:, :, -docnum - 1:] = 0
        doc_adj = adj.clone()
        doc_adj[:, :, :-docnum - 1] = 0

        feature_sen = feature.clone()
        feature_sec = feature.clone()
        feature_doc = feature.clone()
        feature_resi = feature

        feature_sen_re = feature_sen
        feature_sec_re = feature_sec
        feature_doc_re = feature_doc
        for i in range(0, self.iter):
            feature_sen = self.gat[i](feature_sen, sen_adj)
        feature_sen += feature_sen_re

        for i in range(0, self.iter):
            feature_sec = self.gat2[i](feature_sec, sec_adj)
        feature_sec += feature_sec_re

        for i in range(0, self.iter):
            feature_doc = self.gat3[i](feature_doc, doc_adj)
        feature_doc += feature_doc_re

        feature = torch.concat([feature_doc, feature_sec, feature_sen], dim=-1)
        feature = self.dropout(F.leaky_relu(self.feature_fusion_layer(feature)))
        feature = self.ffn(feature)
        feature = self.out_ffn(feature) + feature_resi
        return feature
