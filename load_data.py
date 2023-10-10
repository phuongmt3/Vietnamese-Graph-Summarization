import json
import torch
import numpy as np
import pickle
from transformers import AutoModel, AutoTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from utils import getRouge2

phobert = AutoModel.from_pretrained("vinai/phobert-base-v2")
tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")

stop_w = ['...']
with open('./VLSP Dataset/vietnamese-stopwords-dash.txt', 'r', encoding='utf-8') as f:
    for w in f.readlines():
        stop_w.append(w.strip())
stop_w.extend([c for c in '!"#$%&\'()*+,./:;<=>?@[\\]^`{|}~…“”’‘'])

with open('./VLSP Dataset/LDA_models.pkl', mode='rb') as fp:
    cate_models = pickle.load(fp)


def removeRedundant(text):
    text = text.lower()
    words = [w for w in text.split(' ') if w not in stop_w]
    return ' '.join(words)


class Graph:
    def __init__(self, sents, sentVecs, scores, doc_sec_mask, sec_sen_mask, golden, threds=0.5):
        assert len(sentVecs) == len(scores) == len(sents)
        self.docnum = len(doc_sec_mask)
        self.secnum = len(sec_sen_mask)
        self.adj = torch.from_numpy(mask_to_adj(doc_sec_mask, sec_sen_mask)).float()
        self.feature = np.concatenate((np.array(sentVecs), np.zeros((self.secnum + self.docnum + 1, sentVecs[0].size))))
        self.score = torch.from_numpy(np.array(scores))
        self.score_onehot = (self.score >= threds).float()
        self.sents = np.array(sents)
        self.golden = golden
        self.goldenVec = get_phoBert_vec(golden)
        self.init_node_vec()
        self.feature = torch.from_numpy(self.feature).float()

    def init_node_vec(self):
        docnum, secnum = self.docnum, self.secnum
        for i in range(-secnum - docnum - 1, -docnum - 1):
            mask = self.adj[i].clone()
            mask[-secnum - docnum - 1:] = 0
            self.feature[i] = np.mean(self.feature[mask.bool()], axis=0)
        for i in range(-docnum - 1, -1):
            mask = self.adj[i].clone()
            mask[-docnum - 1:] = 0
            self.feature[i] = np.mean(self.feature[mask.bool()], axis=0)
        self.feature[-1] = np.mean(self.feature[-docnum - 1:-1], axis=0)


def get_phoBert_vec(text, limit_len=400):
    sent = text.lower()
    input_ids = torch.tensor([tokenizer.encode(sent)])
    if input_ids.shape[1] > 256:
        # print('DEVIDED')
        sents = sent.split(' . ')
        wcnt = [len(s.split(' ')) for s in sents]
        wcnt_all = sum(wcnt)

        while wcnt_all > limit_len:
            # print('DEL')
            wcnt_all -= wcnt[-1]
            sents.pop()
            wcnt.pop()

        part1, part2 = [], []
        for i, s in enumerate(sents):
            if sum(wcnt[:i]) <= wcnt_all / 2:
                part1.append(s)
            else:
                part2.append(s)

        sents = [' . '.join(part1), ' . '.join(part2)]
        input_ids = [torch.tensor([tokenizer.encode(sent)]) for sent in sents]
        with torch.no_grad():
            return torch.cat([phobert(input_ids[0])["pooler_output"], phobert(input_ids[1])["pooler_output"]],
                             dim=0)

    with torch.no_grad():
        features = phobert(input_ids)
    return features["pooler_output"]


def mask_to_adj(doc_sec_mask, sec_sen_mask):
    sen_num = sec_sen_mask.shape[1]
    sec_num = sec_sen_mask.shape[0]
    doc_num = doc_sec_mask.shape[0]
    adj = np.zeros((sen_num + sec_num + doc_num + 1, sen_num + sec_num + doc_num + 1))
    # section connection
    adj[-sec_num - doc_num - 1:-doc_num - 1, 0:-sec_num - doc_num - 1] = sec_sen_mask
    adj[0:-sec_num - doc_num - 1, -sec_num - doc_num - 1:-doc_num - 1] = sec_sen_mask.T
    for i in range(0, doc_num):
        doc_mask = doc_sec_mask[i]
        doc_mask = doc_mask.reshape((1, len(doc_mask)))
        adj[sen_num:-doc_num - 1, sen_num:-doc_num - 1] += doc_mask * doc_mask.T
    # doc connection
    adj[-doc_num - 1:-1, -sec_num - doc_num - 1:-doc_num - 1] = doc_sec_mask
    adj[-sec_num - doc_num - 1:-doc_num - 1, -doc_num - 1:-1] = doc_sec_mask.T
    adj[-doc_num - 1:, -doc_num - 1:] = 1

    # build sentence connection
    for i in range(0, sec_num):
        sec_mask = sec_sen_mask[i]
        sec_mask = sec_mask.reshape((1, len(sec_mask)))
        adj[:sen_num, :sen_num] += sec_mask * sec_mask.T
    return adj


def meanTokenVecs(sent):
    tokenVecList = [sp['vector'] for sp in sent['spans'] if sp['vector'] is not None]
    return np.mean(np.array(tokenVecList), axis=0)


def getPositionEncoding(pos, d=768, n=10000):
    P = np.zeros(d)
    for i in np.arange(int(d / 2)):
        denominator = np.power(n, 2 * i / d)
        P[2 * i] = np.sin(pos / denominator)
        P[2 * i + 1] = np.cos(pos / denominator)
    return P


def divideIntoSections():
    secnum, sentnum = 0, 0
    for d, doc in enumerate(clusTree['docs']):
        total_words = 0
        sentnum += len(doc['sents'])
        for s, sent in enumerate(doc['sents']):
            total_words += len(sent['raw_sent'].split(' '))
        if total_words <= 500:
            minSec = 150
        elif total_words <= 1000:
            minSec = 200
        elif total_words <= 2000:
            minSec = 300
        else:
            minSec = 400

        wcnt, curOrgSecID, startSec = 0, 0, secnum
        for s, sent in enumerate(doc['sents']):
            wcnt += len(sent['raw_sent'].split(' '))
            clusTree['docs'][d]['sents'][s]['section_new'] = secnum
            if wcnt >= minSec and sent['secid'] != curOrgSecID and s < len(doc['sents']) - 1:
                secnum += 1
                wcnt = 0
            if sent['secid'] != curOrgSecID:
                curOrgSecID = sent['secid']

        if 0 < wcnt < minSec and secnum > startSec:
            for s in range(len(doc['sents']) - 1, -1, -1):
                if clusTree['docs'][d]['sents'][s]['section_new'] < secnum:
                    break
                clusTree['docs'][d]['sents'][s]['section_new'] = secnum - 1
            secnum -= 1
        secnum += 1
    return secnum, sentnum


def divideIntoSections_lda():
    secnum, sentnum = 0, 0
    paraList, paras, ids, newSecID = [], [], [], {}
    for d, doc in enumerate(clusTree['docs']):
        sentnum += len(doc['sents'])
        paraList.append([])

        para, curOrgSecID = [], 0
        for s, sent in enumerate(doc['sents']):
            if sent['secid'] != curOrgSecID:
                paraList[-1].append(' '.join(para))
                para, curOrgSecID = [], sent['secid']
            para.append(sent['raw_sent'])
        if para is not []:
            paraList[-1].append(' '.join(para))

    for d, doc in enumerate(paraList):
        for p, para in enumerate(doc):
            paras.append(removeRedundant(para))
            ids.append((d, p))

    tf, lda_model = cate_models[clusTree['category']]
    X = tf.transform(paras)
    lda_top = lda_model.transform(X)

    secnum, groupset, wcnt = 0, {}, 0
    for p in range(len(paras)):
        idd = ids[p]  # (doc, para)
        if min(lda_top[p]) == max(lda_top[p]):
            newSecID[(idd[0], idd[1])] = 0
            continue
        name, score = -1, 0
        for i, topic in enumerate(lda_top[p]):
            if topic > score:
                score, name = topic, i
        if name not in groupset:
            groupset[name] = len(groupset)
        newSecID[(idd[0], idd[1])] = groupset[name]

    prevSecnum, doc_endsec = 0, []
    for d, doc in enumerate(clusTree['docs']):
        groupset = {}
        for s, sent in enumerate(doc['sents']):
            if newSecID[(d, sent['secid'])] not in groupset:
                groupset[newSecID[(d, sent['secid'])]] = len(groupset) + prevSecnum
            clusTree['docs'][d]['sents'][s]['section_new'] = groupset[newSecID[(d, sent['secid'])]]
        prevSecnum = max(groupset.values()) + 1
        doc_endsec.append(max(groupset.values()))
    return doc_endsec, max(groupset.values()) + 1, sentnum


clusTree = None
""" each cluster into a single clusTree
a cluster includes multiple documents
a document includes multiple sentences
a sentence contains a dependency tree of this sentence in form of a spans list
each span is a single word with corresponding BERT embedding
"""


def loadTrainGraphs():
    global clusTree
    trainGraphs = []
    with open('./VLSP Dataset/train_segmentedSumm.txt', 'r', encoding='utf-8') as f:
        goldenList = json.load(f)
    for cluster in range(200):
        print(cluster)
        sents, sentVecs, scores, secIDs = [], [], [], []
        with open('./VLSP Dataset/wordVec/train/train_' + str(cluster) + '.pkl', 'rb') as fp:
            clusTree = pickle.load(fp)
        doc_endsec, secnum, sentnum = divideIntoSections_lda()
        doc_sec_mask = np.zeros((len(clusTree['docs']), secnum))
        sec_sen_mask = np.zeros((secnum, sentnum))
        cursec, cursent = 0, 0
        for d, doc in enumerate(clusTree['docs']):
            doc_sec_mask[d][cursec:doc_endsec[d] + 1] = 1
            cursec = doc_endsec[d] + 1
            for s, sent in enumerate(doc['sents']):
                sents.append(sent['raw_sent'])
                sentVecs.append(meanTokenVecs(sent) + getPositionEncoding(d) + getPositionEncoding(s))
                scores.append(getRouge2(goldenList[cluster], sent['raw_sent'], 'p'))
                sec_sen_mask[sent['section_new'], cursent] = 1
                cursent += 1

        trainGraphs.append(Graph(sents, sentVecs, scores, doc_sec_mask, sec_sen_mask, goldenList[cluster]))
    return trainGraphs


def loadValGraphs():
    global clusTree
    testGraphs = []
    with open('./VLSP Dataset/val_segmentedSumm.pkl', 'rb') as f:
        goldenList = pickle.load(f)
    for cluster in range(100):
        print(cluster)
        sents, sentVecs, scores, secIDs = [], [], [], []
        with open('./VLSP Dataset/wordVec/val/val_' + str(cluster) + '.pkl', 'rb') as fp:
            clusTree = pickle.load(fp)
        doc_endsec, secnum, sentnum = divideIntoSections_lda()
        doc_sec_mask = np.zeros((len(clusTree['docs']), secnum))
        sec_sen_mask = np.zeros((secnum, sentnum))
        cursec, cursent = 0, 0
        for d, doc in enumerate(clusTree['docs']):
            doc_sec_mask[d][cursec:doc_endsec[d] + 1] = 1
            cursec = doc_endsec[d] + 1
            for s, sent in enumerate(doc['sents']):
                sents.append(sent['raw_sent'])
                sentVecs.append(meanTokenVecs(sent) + getPositionEncoding(d) + getPositionEncoding(s))
                scores.append(getRouge2(goldenList[cluster], sent['raw_sent'], 'p'))
                sec_sen_mask[sent['section_new'], cursent] = 1
                cursent += 1

        testGraphs.append(Graph(sents, sentVecs, scores, doc_sec_mask, sec_sen_mask, goldenList[cluster]))
    return testGraphs


def loadTestGraphs():
    testGraphs = []
    with open('./VLSP Dataset/test_segmentedSumm.pkl', 'rb') as f:
        goldenList = pickle.load(f)
    for cluster in range(300):
        print(cluster)
        sents, sentVecs, scores, secIDs = [], [], [], []
        with open('./VLSP Dataset/wordVec/test/test_' + str(cluster) + '.pkl', 'rb') as fp:
            clusTree = pickle.load(fp)
        doc_endsec, secnum, sentnum = divideIntoSections_lda()
        doc_sec_mask = np.zeros((len(clusTree['docs']), secnum))
        sec_sen_mask = np.zeros((secnum, sentnum))
        cursec, cursent = 0, 0
        for d, doc in enumerate(clusTree['docs']):
            doc_sec_mask[d][cursec:doc_endsec[d] + 1] = 1
            cursec = doc_endsec[d] + 1
            for s, sent in enumerate(doc['sents']):
                sents.append(sent['raw_sent'])
                sentVecs.append(meanTokenVecs(sent) + getPositionEncoding(d) + getPositionEncoding(s))
                scores.append(getRouge2(goldenList[cluster], sent['raw_sent'], 'p'))
                sec_sen_mask[sent['section_new'], cursent] = 1
                cursent += 1

        testGraphs.append(Graph(sents, sentVecs, scores, doc_sec_mask, sec_sen_mask, goldenList[cluster]))
    return testGraphs
