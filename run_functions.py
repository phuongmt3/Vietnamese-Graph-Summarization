import numpy as np
import torch
import torch.nn.functional as F
from models import InfoNCE
from utils import getRouge2


def train_e2e(train_dataloader, model, optimizer):
    model[0].train()
    model[1].train()
    c_loss, s_loss, loss, batch_num = 0, 0, 0, 0
    print_epo = 20

    for i, data in enumerate(train_dataloader):
        batch_loss, bc_loss, bs_loss, batch_size = train_e2e_batch(data, model, optimizer)
        loss += batch_loss
        c_loss += bc_loss
        s_loss += bs_loss
        batch_num += 1

        if i % print_epo == 0:
            print("Batch {}, Loss: {}".format(i, loss / batch_num))
            print("Batch {}, C-Loss: {}".format(i, c_loss / batch_num))
            print("Batch {}, S-Loss: {}".format(i, s_loss / batch_num))

    return loss / batch_num


def train_e2e_batch(data_batch, model, optimizer):
    c_model = model[0]
    s_model = model[1]

    optimizer.zero_grad()
    feature = data_batch.feature.unsqueeze(0)
    adj = data_batch.adj.unsqueeze(0)
    labels = data_batch.score_onehot.unsqueeze(0)
    goldenVec = data_batch.goldenVec.unsqueeze(0)
    docnum = data_batch.docnum
    secnum = data_batch.secnum

    pg = c_model(feature, adj, docnum, secnum)
    x = s_model(pg, adj, docnum, secnum)

    s_loss = F.binary_cross_entropy_with_logits(x.squeeze(-1), labels, pos_weight=torch.tensor(10))
    pg = pg.squeeze(0)
    infonce = InfoNCE(tau=0.2)

    mask = torch.zeros(1, feature.shape[1])
    mask[:, :-docnum - secnum - 1] = labels
    mask[:, -docnum - secnum - 1:] = 1
    neg_mask = 1 - mask
    c_loss = infonce(goldenVec, pg, mask, neg_mask)

    loss = s_loss + 0.5 * c_loss
    loss.backward()
    optimizer.step()

    return loss.data, c_loss.data, s_loss.data, x.shape[0]


def val_e2e(val_dataloader, model, mode='val', max_word_num=160, sent_num=0):
    model[0].eval()
    model[1].eval()
    loss, c_loss, s_loss = 0, 0, 0
    batch_num = 0
    rouge2_score = []

    all_summaries = []
    all_gt = []
    for i, data in enumerate(val_dataloader):
        cur_loss, c_loss_b, s_loss_b, scores = val_e2e_batch(data, model)
        loss += cur_loss
        c_loss += c_loss_b
        s_loss += s_loss_b

        summary_text = get_summary(scores, data.sents, max_word_num, sent_num)
        all_gt.append(data.golden)
        all_summaries.append(summary_text)
        rouge2_score.append(getRouge2(data.golden, summary_text, 'f'))
        batch_num += 1

    rouge2_score_mean = np.mean(rouge2_score)
    loss = loss / batch_num
    c_loss /= batch_num
    s_loss /= batch_num

    if mode != 'val':
        return rouge2_score_mean, loss, c_loss, s_loss, all_summaries, all_gt, rouge2_score
    return rouge2_score_mean, loss, c_loss, s_loss


def val_e2e_batch(data_batch, model):
    c_model = model[0]
    s_model = model[1]
    feature = data_batch.feature.unsqueeze(0)
    adj = data_batch.adj.unsqueeze(0)
    labels = data_batch.score_onehot.unsqueeze(0)
    goldenVec = data_batch.goldenVec.unsqueeze(0)
    docnum = data_batch.docnum
    secnum = data_batch.secnum

    with torch.no_grad():
        pg = c_model(feature, adj, docnum, secnum)
        x = s_model(pg, adj, docnum, secnum)

        pg = pg.squeeze(0)
        infonce = InfoNCE(tau=0.2)

        mask = torch.zeros(1, feature.shape[1])
        mask[:, :-docnum - secnum - 1] = labels
        mask[:, -1] = 1
        neg_mask = 1 - mask
        c_loss = infonce(goldenVec, pg, mask, neg_mask)
        s_loss = F.binary_cross_entropy_with_logits(x.squeeze(-1), labels, pos_weight=torch.tensor(10))

        loss = c_loss * 0.5 + s_loss
        scores = torch.sigmoid(x.squeeze(-1))
    return loss.data, c_loss.data, s_loss.data, scores


def get_summary(scores, sents, max_word_num=160, sent_num=0):
    # scores : (batch_size, sen_len)
    ranked_score_idxs = torch.argsort(scores, dim=1, descending=True).squeeze(0)
    wordCnt = 0
    summSentIDList = []
    for i in ranked_score_idxs:
        if wordCnt >= max_word_num and sent_num == 0:
            break
        elif sent_num > 0 and len(summSentIDList) == sent_num:
            break
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