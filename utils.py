import random
import os
import numpy as np
import torch
from rouge import Rouge


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def getRouge2(ref, pred, kind): # tokenized input
    try:
        return round(Rouge().get_scores(pred.lower(), ref.lower())[0]['rouge-2'][kind], 4)
    except ValueError:
        return 0.0
