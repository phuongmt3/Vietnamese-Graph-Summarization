import pickle
import json
import numpy as np
import pandas as pd
from rouge import Rouge
import string
import copy
import time
import csv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation


stop_w = ['...']
with open('./VLSP Dataset/vietnamese-stopwords-dash.txt', 'r', encoding='utf-8') as f:
    for w in f.readlines():
        stop_w.append(w.strip())
stop_w.extend([c for c in '!"#$%&\'()*+,./:;<=>?@[\\]^`{|}~…“”’‘'])


def removeRedundant(text):
    text = text.lower()
    words = [w for w in text.split(' ') if w not in stop_w]
    return ' '.join(words)


cate_clusID = {'Giáo dục':[], 'Giải trí - Thể thao':[], 'Khoa học - Công nghệ':[], 'Kinh tế':[], 'Pháp luật':[],
                 'Thế giới':[], 'Văn hóa - Xã hội':[], 'Đời sống':[]}
clusTrees = []
with open('./VLSP Dataset/train_data_new.jsonl', 'r', encoding='utf-8') as file:
    for clusterid, cluster in enumerate(file.readlines()):
        category = json.loads(cluster)['category']
        cate_clusID[category].append(clusterid)
        with open('./VLSP Dataset/wordVec/train/train_' + str(clusterid) + '.pkl', 'rb') as fp:
            clusTrees.append(pickle.load(fp))

# TRAIN LDA for each category
cate_models = {}
for key, ids in cate_clusID.items():
    paras = []
    for i in ids:
        clusTree = clusTrees[i]
        paraList = []
        for d, doc in enumerate(clusTree['docs']):
            paraList.append([])
            para, curOrgSecID = [], 0
            for s, sent in enumerate(doc['sents']):
                if sent['secid'] != curOrgSecID:
                    paraList[-1].append(removeRedundant(' '.join(para)))
                    para, curOrgSecID = [], sent['secid']
                para.append(sent['raw_sent'])
            if para is not []:
                paraList[-1].append(removeRedundant(' '.join(para)))

        for d, doc in enumerate(paraList):
            for s, sent in enumerate(doc):
                paras.append(sent)

    print(key, len(paras))
    tf = TfidfVectorizer(min_df=2, max_df=1.0, max_features=3000, sublinear_tf=True)
    X = tf.fit_transform(paras)
    lda_model = LatentDirichletAllocation(n_components=4, learning_method='online', random_state=42, max_iter=1)
    lda_model.fit(X)
    cate_models[key] = (tf, lda_model)

with open('./VLSP Dataset/LDA_models_4top.pkl', mode='wb') as fp:
    pickle.dump(cate_models, fp)
