import random
import os
import numpy as np
import torch
from rouge import Rouge
from pyvi import ViTokenizer



def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def getRouge2(ref, pred, kind):  # tokenized input
    try:
        return round(Rouge().get_scores(pred.lower(), ref.lower())[0]['rouge-2'][kind], 4)
    except ValueError:
        return 0.0


def getRouge1(ref, pred, kind):  # tokenized input
    return round(Rouge().get_scores(pred.lower(), ref.lower())[0]['rouge-1'][kind], 4)


def getRougeL(ref, pred, kind):  # tokenized input
    return round(Rouge().get_scores(pred.lower(), ref.lower())[0]['rouge-l'][kind], 4)


def cal_rouge(goldens, predicts, avg=True):
    ppredicts = list(map(lambda x: ViTokenizer.tokenize(x.replace('_', ' ')).lower(), predicts))
    pgoldens = list(map(lambda x: ViTokenizer.tokenize(x.replace('_', ' ')).lower(), goldens))

    rouge = Rouge()
    scores = rouge.get_scores(ppredicts, pgoldens, avg=avg)

    if not avg:
        return scores

    return list(map(
        lambda x: round(x, 4),
        (
            scores['rouge-1']['p'], scores['rouge-1']['r'], scores['rouge-1']['f'],
            scores['rouge-2']['p'], scores['rouge-2']['r'], scores['rouge-2']['f'],
            scores['rouge-l']['p'], scores['rouge-l']['r'], scores['rouge-l']['f'],
        )
    ))
