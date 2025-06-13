import torch
import torch.nn as nn
from graph_layers import MLP
from graph_models import Contrast_Encoder, End2End_Encoder
from load_data import getPositionEncoding

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PositionVec = torch.stack([torch.from_numpy(getPositionEncoding(i, d=768)) for i in range(200)], dim=0).float().to(device)
PositionVec_100 = torch.stack([torch.from_numpy(getPositionEncoding(i, d=768+100)) for i in range(200)], dim=0).float().to(device)
PositionVec_200 = torch.stack([torch.from_numpy(getPositionEncoding(i, d=768+200)) for i in range(200)], dim=0).float().to(device)
PositionVec_300 = torch.stack([torch.from_numpy(getPositionEncoding(i, d=768+300)) for i in range(200)], dim=0).float().to(device)
PositionVec_512 = torch.stack([torch.from_numpy(getPositionEncoding(i, d=512)) for i in range(200)], dim=0).float().to(device)
PositionVec_50 = torch.stack([torch.from_numpy(getPositionEncoding(i, d=768+50)) for i in range(800)], dim=0).float().to(device)

class BiLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=0, layers=1, act=nn.LeakyReLU(), dropout=0.1):
        super(BiLSTM, self).__init__()
        self.bi_lstm = nn.LSTM(input_dim, hidden_dim, proj_size=output_dim, num_layers=layers, batch_first=True, dropout=dropout, bidirectional=True)

    def forward(self, x, doc_lens, *arg):
        docs, cur_i = [], 0
        for le in doc_lens:
            docs.append(x[0, cur_i:cur_i + le])
            cur_i += le
        x = nn.utils.rnn.pack_sequence(docs, enforce_sorted=False)
        x, _ = self.bi_lstm(x)
        x = nn.utils.rnn.unpack_sequence(x)
        x = torch.cat(x, dim=0).unsqueeze(0)
        return x

class CNN(nn.Module):
    def __init__(self, input_dim, kernel_sizes: list, num_kernels: list, act=nn.LeakyReLU(), dropout=0.1):
        super(CNN, self).__init__()
        assert len(kernel_sizes) == len(num_kernels)
        self.conv1d_list = nn.ModuleList([
            nn.Conv1d(in_channels=input_dim, out_channels=num_kernels[i], kernel_size=kernel_sizes[i], padding='same')
            for i in range(len(kernel_sizes))
        ])
        self.act = act

    def forward(self, x, doc_lens, *args):
        x = x[:, :sum(doc_lens)]
        posVec = torch.cat([PositionVec_50[:l] for l in doc_lens], dim=0)
        x_doc = (x[0] + posVec).permute(1, 0)
        x_conv_list = [self.act(conv1d(x_doc)) for conv1d in self.conv1d_list]
        x = torch.cat(x_conv_list, dim=0).permute(1, 0).unsqueeze(0)
        return x

class LSTM_CNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_sizes: list, num_kernels: list, dilation, lstm_layers=1, act=nn.LeakyReLU(), dropout=0.1):
        super(LSTM_CNN, self).__init__()
        assert len(kernel_sizes) == len(num_kernels)
        self.bi_lstm = nn.LSTM(input_dim, hidden_dim, num_layers=lstm_layers, batch_first=True, dropout=dropout, bidirectional=True)
        self.conv1d_list = nn.ModuleList([
            nn.Conv1d(in_channels=hidden_dim * 2, out_channels=num_kernels[i], kernel_size=kernel_sizes[i], padding='same', dilation=dilation)
            for i in range(len(kernel_sizes))
        ])
        self.mlp = MLP([sum(num_kernels), 1], layers=1, act=act, dropout_p=dropout, keep_last_layer=True)
        self.act = act
        self.dilation = dilation

    def forward(self, x, doc_lens, *args):
        x = x[:, :sum(doc_lens)]
        x, _ = self.bi_lstm(x[0])
        posVec = torch.cat([PositionVec_512[:l] for l in doc_lens], dim=0)
        x = (x + posVec).permute(1, 0)
        x_conv_list = [self.act(conv1d(x)) for conv1d in self.conv1d_list]
        x = torch.cat(x_conv_list, dim=0).permute(1, 0).unsqueeze(0)
        return x

class CNN_LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_sizes: list, num_kernels: list, lstm_space, act=nn.LeakyReLU(), dropout=0.1):
        super(CNN_LSTM, self).__init__()
        assert len(kernel_sizes) == len(num_kernels)
        self.conv1d_list = nn.ModuleList([
            nn.Conv1d(in_channels=input_dim, out_channels=num_kernels[i], kernel_size=kernel_sizes[i], padding='same')
            for i in range(len(kernel_sizes))
        ])
        self.pooler_list = nn.ModuleList([nn.MaxPool1d(size, padding=int(size/2), stride=1) for size in kernel_sizes])
        self.lstm = nn.LSTMCell(sum(num_kernels), hidden_dim)
        self.lstm_rev = nn.LSTMCell(sum(num_kernels), hidden_dim)
        self.hidden = hidden_dim
        self.space = lstm_space # Window size
        self.mlp = MLP([hidden_dim * 2, 1], layers=1, act=act, dropout_p=dropout, keep_last_layer=True)
        self.act = act

    def getHiddenStates(self, x, lstm):
        h, c, start_x = torch.zeros((x.shape[0], self.hidden)).to(device), torch.zeros((x.shape[0], self.hidden)).to(device), 0
        hx, cx = torch.zeros((self.space, self.hidden)).to(device), torch.zeros((self.space, self.hidden)).to(device)
        while start_x < x.shape[0]:
            end_x = min(x.shape[0], start_x + self.space)
            hx, cx = lstm(x[start_x:end_x], (hx[:end_x-start_x], cx[:end_x-start_x]))
            h[start_x:end_x], c[start_x:end_x] = hx, cx
            start_x = end_x
        return h, c

    def forward(self, x, doc_lens, *args):
        x = x[:, :sum(doc_lens)]
        posVec = torch.cat([PositionVec_300[:l] for l in doc_lens], dim=0)
        x_doc = (x[0] + posVec).permute(1, 0)
        x_conv_list = [self.act(conv1d(x_doc)) for conv1d in self.conv1d_list]
        x = torch.cat(x_conv_list, dim=0).permute(1, 0)

        x_pooled = [pooler(x_conv) for pooler, x_conv in zip(self.pooler_list, x_conv_list)]
        x_pooled = torch.cat(x_pooled, dim=0).permute(1, 0)
        x_pooled_0, x_pooled_1 = x_pooled.clone(), x_pooled.clone()

        h, c = self.getHiddenStates(x_pooled_0, self.lstm)
        h_rev, c_rev = self.getHiddenStates(torch.flip(x_pooled_1, dims=[0]), self.lstm_rev)
        h_rev, c_rev = torch.flip(h_rev, dims=[0]), torch.flip(c_rev, dims=[0])
        min_dis = int(self.space/2) + 1

        x_bi = torch.zeros((x.shape[0], self.hidden * 2)).to(device)
        for i, x_i in enumerate(x):
            hi = h[i - min_dis] if i >= min_dis else torch.zeros(self.hidden).to(device)
            ci = c[i - min_dis] if i >= min_dis else torch.zeros(self.hidden).to(device)
            hi, _ = self.lstm(x_i, (hi, ci))
            hi_rev = h[i + min_dis] if i + min_dis < x.shape[0] else torch.zeros(self.hidden).to(device)
            ci_rev = c[i + min_dis] if i + min_dis < x.shape[0] else torch.zeros(self.hidden).to(device)
            hi_rev, _ = self.lstm_rev(x_i, (hi_rev, ci_rev))
            x_bi[i] = torch.cat([hi, hi_rev], dim=0)

        x = x_bi.unsqueeze(0)
#         x = self.mlp(x)
        return x

class LSTM_Cat_CNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_sizes: list, num_kernels: list, lstm_layers=1, act=nn.LeakyReLU(), dropout=0.1):
        super(LSTM_Cat_CNN, self).__init__()
        assert len(kernel_sizes) == len(num_kernels)
        self.bi_lstm = nn.LSTM(input_dim, hidden_dim, num_layers=lstm_layers, batch_first=True, dropout=dropout, bidirectional=True)
        self.conv1d_list = nn.ModuleList([
            nn.Conv1d(in_channels=input_dim, out_channels=num_kernels[i], kernel_size=kernel_sizes[i], padding='same')
            for i in range(len(kernel_sizes))
        ])
        self.mlp = MLP([sum(num_kernels) + hidden_dim*2, 1], layers=1, act=act, dropout_p=dropout, keep_last_layer=True)
        self.act = act

    def forward(self, x, doc_lens, *args):
        x = x[:, :sum(doc_lens)]
        x_org = x.clone()
        posVec = torch.cat([PositionVec_100[:l] for l in doc_lens], dim=0)
        x_doc = (x[0] + posVec).permute(1, 0)
        x_conv_list = [self.act(conv1d(x_doc)) for conv1d in self.conv1d_list]
        x_cnn = torch.cat(x_conv_list, dim=0).permute(1, 0)

        docs, cur_i = [], 0
        for le in doc_lens:
            docs.append(x_org.clone()[0, cur_i:cur_i + le])
            cur_i += le
        x = nn.utils.rnn.pack_sequence(docs, enforce_sorted=False)
        x, _ = self.bi_lstm(x)
        x = nn.utils.rnn.unpack_sequence(x)
        x_lstm = torch.cat(x, dim=0)

        x = torch.cat([x_cnn, x_lstm], dim=1).unsqueeze(0)
#         x = self.mlp(x)
        return x

class SelfAttention(nn.Module):
    def __init__(self, input_dim):
        super(SelfAttention, self).__init__()
        self.input_dim = input_dim
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
#         self.value = nn.Linear(input_dim, input_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor, adj_mat: torch.Tensor):
        queries = self.query(x)
        keys = self.key(x)
        values = x.clone() #self.value(x)
        scores = torch.bmm(queries, keys.transpose(1, 2)) / (self.input_dim ** 0.5)
        scores = scores.masked_fill(adj_mat == 0, float(-1e9))
        attention = self.softmax(scores)
        weighted = torch.bmm(attention, values)
        return weighted

class ATT_layer(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super(ATT_layer, self).__init__()
        self.attention = SelfAttention(input_dim)
        self.layer_norm = [nn.LayerNorm(input_dim, device=device) for i in range(2)]
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, input_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor, adj_mat: torch.Tensor):
        adj_x = adj_mat.clone().sum(dim=-1, keepdim=True).repeat(1, 1, x.shape[-1]).bool()
        x_re = x.clone()
        x = self.attention(x, adj_mat.bool())
        x = x_re + self.layer_norm[0](x)

        x_re = x.clone()
        x = self.relu(self.linear1(x))
        x = self.linear2(self.dropout(x))
        x = x_re + self.layer_norm[1](x)
        x = x.masked_fill(adj_x == 0, float(0))
        return x

class ATT(nn.Module):
    def __init__(self, input_dim, hidden_dim, iter, act=nn.LeakyReLU(), dropout=0.1):
        super(ATT, self).__init__()
        self.iter = iter
        self.gat = nn.ModuleList([ATT_layer(input_dim, hidden_dim) for _ in range(iter[0])])
#         self.mlp = MLP([input_dim, 1], layers=1, act=act, dropout_p=dropout, keep_last_layer=True)

    def forward(self, feature, doc_lens, adj, docnum, secnum):
        sen_adj = adj.clone()
        sen_adj[:, -docnum-secnum-1:, :] = sen_adj[:, :, -docnum-secnum-1:] = 0

        posVec = torch.cat([PositionVec_200[:l] for l in doc_lens] + [torch.zeros(secnum+docnum+1, feature.shape[-1]).float().to(device)], dim=0)
        feature = feature + posVec.unsqueeze(0)

        for i in range(self.iter[0]):
            feature = self.gat[i](feature.clone(), sen_adj)
        return feature[:, :-docnum-secnum-1]

class M1_M2(nn.Module):
    def __init__(self, model, input_dim, hidden_dim, reduce_dims, act=nn.LeakyReLU(0.1), dropout=0.1):
        super(M1_M2, self).__init__()
        self.dense_bert = nn.Linear(input_dim, reduce_dims[0])
        self.linear = nn.Linear(input_dim, reduce_dims[1])
        self.model = model
        self.mlp = MLP([hidden_dim + reduce_dims[0], 1], layers=1, act=act, dropout_p=dropout, keep_last_layer=True)

    def forward(self, x_bert, x_graph, doc_lens, adj, docnum, secnum, is_attn=False):
        x_org = self.dense_bert(x_bert.clone()[:, :-secnum-docnum-1])
        x = torch.cat([x_bert, self.linear(x_graph)], dim=-1)
        if not is_attn:
            x = x[:, :-secnum-docnum-1]
        x = self.model(x, doc_lens, adj, docnum, secnum)
        x = torch.cat([x, x_org], dim=-1)
        x = self.mlp(x)
        return x

class Frozen_Fusion(nn.Module):
    def __init__(self, input):
        super(Frozen_Fusion, self).__init__()
        self.graph_encoder_1 = Contrast_Encoder(input, 1024, 4)
        self.graph_encoder_2 = End2End_Encoder(input, 1024, 4)
        self.LSTM_module = M1_M2(BiLSTM(input+100, 256, act=nn.LeakyReLU(0.1)),
                                 input, 256*2, (100,100))
        self.CNN_module = M1_M2(CNN(input+50, [5,7,9,11], [150,150,150,150], act=nn.LeakyReLU(0.1)),
                                input, 600, (100,50))
        self.L2C_module = M1_M2(LSTM_CNN(input+50, 256, [3,5,7,9], [150,150,150,150], 9, act=nn.LeakyReLU(0.1)),
                                input, 600, (200,50))
        self.C2L_module = M1_M2(CNN_LSTM(input+300, 256, [3,5,7], [200,200,200], lstm_space=7, act=nn.LeakyReLU(0.1)),
                                input, 256*2, (200,300))
        self.LcatC_module = M1_M2(LSTM_Cat_CNN(input+100, 256, [5,7,9,11], [150,150,150,150], act=nn.LeakyReLU(0.1)),
                                input, 256*2+600, (200,100))
        self.Attn_module = M1_M2(ATT(input+200, 256, [1], act=nn.LeakyReLU(0.1)),
                                 input, 768+200, (200,200))

    def freeze_graph_encoder(self):
        for param in self.graph_encoder_1.parameters():
            param.requires_grad = False
        for param in self.graph_encoder_2.parameters():
            param.requires_grad = False

    def forward(self, x, doc_lens, adj, docnum, secnum):
        x_graph = self.graph_encoder_1(x, doc_lens, adj, docnum, secnum)
        x_graph, graph_logit = self.graph_encoder_2(x_graph, doc_lens, adj, docnum, secnum)
        x_L = self.LSTM_module(x, x_graph, doc_lens, adj, docnum, secnum)
        x_C = self.CNN_module(x, x_graph, doc_lens, adj, docnum, secnum)
        x_L2C = self.L2C_module(x, x_graph, doc_lens, adj, docnum, secnum)
        x_C2L = self.C2L_module(x, x_graph, doc_lens, adj, docnum, secnum)
        x_LcatC = self.LcatC_module(x, x_graph, doc_lens, adj, docnum, secnum)
        x_Attn = self.Attn_module(x, x_graph, doc_lens, adj, docnum, secnum, is_attn=True)
        return graph_logit, x_graph, x_L, x_C, x_L2C, x_C2L, x_LcatC, x_Attn
