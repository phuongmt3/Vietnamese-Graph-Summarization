import torch
import torch.nn as nn

from layers import StepWiseGraphConvLayer
from load_data import loadTestGraphs
from models import Contrast_Encoder, End2End_Encoder
from run_functions import val_e2e

args = {'seed': 42, 'batch_size': 1, 'input': 768, 'hidden': 2048, 'heads': 8,
        'epochs': 30, 'log_every': 20, 'lr': 0.0003, 'dropout': 0.1}

c_graph_encoder = StepWiseGraphConvLayer(in_dim=768, out_dim=args['hidden'], hid_dim=args['hidden'],
                                         dropout_p=args['dropout'], act=nn.LeakyReLU(), nheads=8, iter=1)
s_graph_encoder = StepWiseGraphConvLayer(in_dim=768, out_dim=args['hidden'], hid_dim=args['hidden'],
                                         dropout_p=args['dropout'], act=nn.LeakyReLU(), nheads=8, iter=1)
contrast_filter = Contrast_Encoder(c_graph_encoder, args['hidden'], dropout_p=args['dropout'])
summarization_encoder = End2End_Encoder(s_graph_encoder, 768, args['hidden'], args['dropout'])

# LOAD TRAINED MODELS
summarization_encoder.load_state_dict(torch.load('./models/e_35_1.358232855796814.mdl'), strict=False)
contrast_filter.load_state_dict(torch.load('./models/c_35_0.0015203384682536125.mdl'), strict=False)

testGraphs = loadTestGraphs()

model = [contrast_filter, summarization_encoder]
rouge2_score_mean, loss, c_loss, s_loss, summs, goldens, rouge2_score_list = val_e2e(testGraphs, model, mode='test', max_word_num=160)

submission = [summs[i].replace('_', ' ') for i in range(300)]
with open('./VLSP Dataset/results.txt', 'w', encoding='utf-8') as f:
    for line in submission:
        f.write(line + '\n')
