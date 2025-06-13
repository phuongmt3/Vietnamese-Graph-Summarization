import torch
from graph_models import InfoNCE
from utils import getRouge2, cal_rouge
from tqdm import tqdm
import wandb


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def init_wandb(key, epochs, lr0, batch_size, run_name, project):
    wandb.login(key=key)
    wandb.init(
        project=project,
        config={
            "epochs": epochs,
            "optimizer": "AdamW",
            "lr": lr0,
            "loss": "MSE",
            'batch_size': batch_size,
        },
        # id="cc3twa0g", resume="must"
    )
    wandb.run.name = run_name

def wandb_finish():
    wandb.finish()

def train_e2e(train_loader, scheduler, model, optimizer, loss_func, log_every, epoch, train_graph:bool):
    model.train()
    epoch_loss, batch_num = [[], []], 0
    refs, preds = [], []

    for data_batch in tqdm(train_loader, desc=f'Epoch {epoch}', unit='batch'):
        optimizer.zero_grad()
        graph_logit, pg, x_L, x_C, x_L2C, x_C2L, x_LcatC, x_Attn = model(data_batch['feature'].to(device), data_batch['doc_lens'],
                                                            data_batch['adj'].to(device), data_batch['docnum'], data_batch['secnum'])
        if train_graph:
            s_loss = loss_func(graph_logit.squeeze(-1), data_batch['labels'].to(device))
            infonce = InfoNCE(tau=0.2)
            mask = torch.zeros(1, data_batch['feature'].shape[1]).float().to(device)
            mask[:, :-data_batch['docnum']-data_batch['secnum']-1] = data_batch['labels']
            mask[:, -data_batch['docnum']-data_batch['secnum']-1:] = 1.
            _, goldenVec, _, _, _, _, _, _ = model(data_batch['golden_feature'].to(device), data_batch['golden_doc_lens'],
                                                data_batch['golden_adj'].to(device), 1, 1)
            c_loss = infonce(goldenVec[0, :-3], pg.squeeze(0), mask)
        else:
            s_loss = sum([loss_func(x.squeeze(-1), data_batch['labels'].to(device)) for x in [x_L, x_C, x_L2C, x_C2L, x_LcatC, x_Attn]]) / 6.
            c_loss = torch.tensor(0., requires_grad=True).to(device)

        loss = s_loss + c_loss
        loss.backward()
        optimizer.step()
        model.zero_grad()
        epoch_loss[0] += [s_loss.item()]
        epoch_loss[1] += [c_loss.item()]
        batch_num += 1

        if batch_num % log_every == 0:
            wandb.log({'train/batch': batch_num, 'train/lr': scheduler.get_last_lr()[0],
                       'train/b_s_loss': s_loss.item(), 'train/b_c_loss': c_loss.item()})

    wandb.log({'train/epoch': epoch, 'train/e_s_loss': sum(epoch_loss[0]) / len(epoch_loss[0]),
               'train/e_c_loss': sum(epoch_loss[1]) / len(epoch_loss[1])})

def val_e2e(val_loader, scheduler, model, optimizer, loss_func, mode='val', max_word_num=200):
    model.eval()
    epoch_loss, batch_num = [[], []], 0
    refs, preds = [], []

    for data_batch in val_loader:
        with torch.no_grad():
            graph_logit, pg, x_L, x_C, x_L2C, x_C2L, x_LcatC, x_Attn = model(data_batch['feature'].to(device), data_batch['doc_lens'],
                                                            data_batch['adj'].to(device), data_batch['docnum'], data_batch['secnum'])
            s_loss = sum([loss_func(x.squeeze(-1), data_batch['labels'].to(device)) for x in [x_L, x_C, x_L2C, x_C2L, x_LcatC, x_Attn]]) / 6.
            infonce = InfoNCE(tau=0.2)
            mask = torch.zeros(1, data_batch['feature'].shape[1]).float().to(device)
            mask[:, :-data_batch['docnum']-data_batch['secnum']-1] = data_batch['labels']
            mask[:, -data_batch['docnum']-data_batch['secnum']-1:] = 1.
            _, goldenVec, _, _, _, _, _, _ = model(data_batch['golden_feature'].to(device), data_batch['golden_doc_lens'],
                                data_batch['golden_adj'].to(device), 1, 1)
            c_loss = infonce(goldenVec[0, :-3], pg.squeeze(0), mask)
            batch_num += 1

        epoch_loss[0] += [s_loss.item()]
        epoch_loss[1] += [c_loss.item()]
        # scores = torch.sigmoid(x_L.squeeze(-1)).detach().cpu()
        scores = [torch.sigmoid(x.squeeze(-1)).detach().cpu()[0] for x in [graph_logit, x_L, x_C, x_L2C, x_C2L, x_LcatC, x_Attn]]
        scores = torch.stack(scores).permute(1, 0).tolist()
        scores = [torch.tensor([s for s in sent if s >= 0.65] if len([s for s in sent if s >= 0.65]) > 0 else [0.]).mean() for sent in scores]
        sent_texts = [s[0] for s in data_batch['sent_text']]
        refs.append(data_batch['golden'][0])
        preds.append(get_summary(torch.stack(scores), sent_texts, max_word_num))
        # preds.append(get_summary(scores[0], sent_texts, max_word_num))
        # print('\n', len(scores[0]), len(sent_texts))
        # for sc, se in zip(scores[0], sent_texts):
        #     print(sc, se)
        # print(pg)

    ranking_scores = cal_rouge(refs, preds)

    if mode == 'val':
        scheduler.step(0.1/ranking_scores[5])
        wandb.log({'val/e_s_loss': sum(epoch_loss[0]) / len(epoch_loss[0]),
                    'val/e_c_loss': sum(epoch_loss[1]) / len(epoch_loss[1]),
                    'val/e_P': ranking_scores[3],
                    'val/e_R': ranking_scores[4],
                    'val/e_F': ranking_scores[5]})
        return ranking_scores[5], sum(epoch_loss[0]) / len(epoch_loss[0])
    return preds, refs, ranking_scores

def get_summary(scores, sents, max_word_num=200):
    ranked_score_idxs = torch.argsort(scores, dim=0, descending=True)
    wordCnt = 0
    summSentIDList = []
    for i in ranked_score_idxs:
        if wordCnt >= max_word_num: break
        s = sents[i]

        replicated = False
        for chosedID in summSentIDList:
            if getRouge2(sents[chosedID], s, 'p') >= 0.45:
                replicated = True
                break
        if replicated: continue

        wordCnt += len(s.split(' '))
        summSentIDList.append(i)
    summSentIDList = sorted(summSentIDList)
    return ' '.join([s for i, s in enumerate(sents) if i in summSentIDList])

def extractive_infer(model, data):
    model.eval()
    feature = data.feature.unsqueeze(0)
    doc_lens = data.doc_lens
    adj = data.adj.unsqueeze(0)
    docnum = data.docnum
    secnum = data.secnum

    with torch.no_grad():
        graph_logit, pg, x_L, x_C, x_L2C, x_C2L, x_LcatC, x_Attn = model(feature, doc_lens, adj, docnum, secnum)
        scores = [torch.sigmoid(x.squeeze(-1)).detach().cpu()[0] for x in [graph_logit, x_L, x_C, x_L2C, x_C2L, x_LcatC, x_Attn]]
        scores = torch.stack(scores).permute(1, 0).tolist()
        scores = [torch.tensor([s for s in sent if s >= 0.65] if len([s for s in sent if s >= 0.65]) > 0 else [0.]).mean() for sent in scores]

    return torch.stack(scores), data.sent_text
