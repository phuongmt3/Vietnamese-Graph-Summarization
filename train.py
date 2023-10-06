import torch
import torch.nn as nn
import random
import os
import utils
from layers import StepWiseGraphConvLayer
from load_data import loadTrainGraphs, loadValGraphs
from models import Contrast_Encoder, End2End_Encoder
from run_functions import train_e2e, val_e2e

args = {'seed': 42, 'batch_size': 1, 'input': 768, 'hidden': 2048, 'heads': 8,
        'epochs': 30, 'log_every': 20, 'lr': 0.0003, 'dropout': 0.1}

utils.seed_everything(args['seed'])

c_graph_encoder = StepWiseGraphConvLayer(in_dim=768, out_dim=args['hidden'], hid_dim=args['hidden'],
                                         dropout_p=args['dropout'], act=nn.LeakyReLU(), nheads=8, iter=1)
s_graph_encoder = StepWiseGraphConvLayer(in_dim=768, out_dim=args['hidden'], hid_dim=args['hidden'],
                                         dropout_p=args['dropout'], act=nn.LeakyReLU(), nheads=8, iter=1)
contrast_filter = Contrast_Encoder(c_graph_encoder, args['hidden'], dropout_p=args['dropout'])
summarization_encoder = End2End_Encoder(s_graph_encoder, 768, args['hidden'], args['dropout'])

optimizer = torch.optim.Adam([ {'params': summarization_encoder.parameters()},
                            {'params': contrast_filter.parameters()}], lr=args['lr'], weight_decay=1e-5)

trainset, valset = loadTrainGraphs(), loadValGraphs()[:100]

model_save_root_path = './models'
c_patient = 30
best_r2, best_c_loss, best_s_loss = 0, 10000, 10000
history = {'loss': [], 'val_loss': []}

for i in range(args['epochs']):
    print("Epoch {}".format(i))
    random.shuffle(trainset)

    if c_patient < 0:
        for p in contrast_filter.parameters():
            p.requires_grad = False
        print("Stop Training Contrast")

    model = [contrast_filter, summarization_encoder]
    loss = train_e2e(trainset, model, optimizer)
    history['loss'].append(loss)
    print("At Epoch {}, Train Loss: {}".format(i, loss))
    #     torch.cuda.empty_cache()

    rouge2_score, loss, c_loss, s_loss = val_e2e(valset, model)
    #     torch.cuda.empty_cache()
    history['val_loss'].append(loss)
    print("At Epoch {}, Val Loss: {}, Val CLoss: {}, Val SLoss: {},Val R1: {}".format(i, loss, c_loss, s_loss,
                                                                                      rouge2_score))
    if rouge2_score > best_r2:
        model_save_path = os.path.join(model_save_root_path, "e_{}_{}.mdl".format(i, s_loss))
        torch.save(summarization_encoder.state_dict(), model_save_path)

        model_save_path = os.path.join(model_save_root_path, "c_{}_{}.mdl".format(i, c_loss))
        torch.save(contrast_filter.state_dict(), model_save_path)
        best_r2 = rouge2_score
        print("Epoch {} Has best R1 Score of {}, saved Model to {}".format(i, best_r2, model_save_path))
    if c_loss < best_c_loss and c_patient >= 0:
        best_c_loss = c_loss
        c_patient = 30
    else:
        c_patient -= 1
