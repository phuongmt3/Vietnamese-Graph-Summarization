import torch
import torch.nn as nn
import os
from torch.utils.data import DataLoader
from frozen_fusion_model import Frozen_Fusion
from run_functions import train_e2e, val_e2e, init_wandb, wandb_finish
from load_data import GraphDataset
import random
import numpy as np
import copy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

epochs = 50
batch_size = 1
seed = 42
log_every = 10
lr0 = 3e-4
input = 768
hidden = 1024
run_name = 'frozen_fusion'


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
seed_everything(seed)


def main():
    train_loader = DataLoader(
        GraphDataset(mode='train', limit=200),
        batch_size=batch_size,
        shuffle=False,
        num_workers=1,
    )

    val_loader = DataLoader(
        GraphDataset(mode='val', limit=100),
        batch_size=batch_size,
        shuffle=False,
        num_workers=1,
    )

    init_wandb('', epochs, lr0, batch_size, run_name, 'VNSumm')

    model = Frozen_Fusion(input).to(device)
    optimizer = torch.optim.AdamW([{'params': model.parameters()}, ], lr=lr0)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.75, patience=2, threshold=0.0001)
    loss_func = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(7).to(device))
    print(model)

    model_save_root_path = '/content/drive/MyDrive/Summarization/checkpoints'
    best_r2, best_s_loss, best_c_loss, c_loss_patience, s_loss_patience = 0, 10000, 10000, 2, 10
    model_state_dicts = []
    model_save_path = ''

    for i in range(epochs):
        train_e2e(train_loader, scheduler, model, optimizer, loss_func, log_every, i + 1, c_loss_patience > 0)
        torch.cuda.empty_cache()
        rouge2_score, loss = val_e2e(val_loader, scheduler, model, loss_func)
        torch.cuda.empty_cache()

        if c_loss_patience > 0:
            if loss > best_c_loss:
                c_loss_patience -= 1
                if c_loss_patience == 0:
                    model.freeze_graph_encoder()
                    print("Freezed Graph Encoder")
            else:
                c_loss_patience = 2
                best_c_loss = loss
        elif s_loss_patience <= 0 and best_c_loss > best_s_loss:
            c_loss_patience = 2

        model_state_dict = copy.deepcopy(model.state_dict())
        model_state_dicts.append(model_state_dict)
        if c_loss_patience <= 0:
            if rouge2_score > best_r2 or loss < best_s_loss:
                model_save_path = os.path.join(model_save_root_path,
                                               run_name + "_e_{}_{}.pt".format(i, round(rouge2_score, 6)))
                torch.save(model.state_dict(), model_save_path)
                best_r2 = max(best_r2, rouge2_score)
                best_s_loss = min(best_s_loss, loss)
                s_loss_patience = 10
                print("Epoch {} Has best loss of {}, saved Model to {}".format(i, best_s_loss, model_save_path))
            else:
                s_loss_patience -= 1
    wandb_finish()

    model.load_state_dict(torch.load(model_save_path, device), strict=True)
    preds, refs, macro_scores = val_e2e(val_loader, scheduler, model, loss_func, mode='test', max_word_num=200)
    print(macro_scores)
