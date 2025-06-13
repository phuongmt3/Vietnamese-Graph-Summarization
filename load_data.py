import json
import torch
import numpy as np
import pickle
from transformers import AutoModel, AutoTokenizer
from torch.utils.data import Dataset
from tqdm import tqdm
from utils import getRouge2
import re
import string
import pandas as pd
from underthesea import sent_tokenize, word_tokenize

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
phobert = AutoModel.from_pretrained("vinai/phobert-base-v2")
tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")
tokenizer_bartpho = AutoTokenizer.from_pretrained("vinai/bartpho-syllable-base")

stop_w = ['...']
with open('./VLSP Dataset/vietnamese-stopwords-dash.txt', 'r', encoding='utf-8') as f:
    for w in f.readlines():
        stop_w.append(w.strip())
stop_w.extend([c for c in '!"#$%&\'()*+,./:;<=>?@[\\]^`{|}~…“”’‘'])

with open('./VLSP Dataset/LDA_models.pkl', mode='rb') as fp:
    cate_models = pickle.load(fp)
vlsp_dataset_path = './VLSP Dataset'


class Cluster:
    def __init__(self, cluster, sent_texts, sent_vecs, spanIds, doc_lens, doc_sec_mask, sec_sen_mask, golden, threds=0.5):
        assert len(sent_vecs) == len(sent_texts)
        self.docnum = len(doc_sec_mask)
        self.secnum = len(sec_sen_mask)
        self.feature = torch.cat((torch.stack(sent_vecs, dim=0), torch.zeros((self.secnum+self.docnum+1, sent_vecs[0].shape[0]))), dim=0)
        self.adj = torch.from_numpy(mask_to_adj(doc_sec_mask, sec_sen_mask)).float()
        self.sent_text = sent_texts
        self.spanIds = spanIds
        self.doc_lens = doc_lens
        self.cluster = cluster
        self.golden = golden
        # self.goldenVec = get_cluster_vec(golden)
        self.score = torch.Tensor([getRouge2(golden, s, 'p') for s in sent_texts]).float()
        self.score_onehot = (self.score >= threds).float()
        self.init_node_vec()
        self.feature = self.feature.float()

    def init_node_vec(self):
        docnum, secnum = self.docnum, self.secnum
        for i in range(-secnum-docnum-1, -docnum-1):
            mask = self.adj[i].clone()
            mask[-secnum-docnum-1:] = 0
            self.feature[i] = torch.mean(self.feature[mask.bool()], dim=0)
        for i in range(-docnum-1, -1):
            mask = self.adj[i].clone()
            mask[-docnum-1:] = 0
            self.feature[i] = torch.mean(self.feature[mask.bool()], dim=0)
        self.feature[-1] = torch.mean(self.feature[-docnum-1:-1], dim=0)

def mask_to_adj(doc_sec_mask, sec_sen_mask):
    sen_num = sec_sen_mask.shape[1]
    sec_num = sec_sen_mask.shape[0]
    doc_num = doc_sec_mask.shape[0]
    adj = np.zeros((sen_num+sec_num+doc_num+1, sen_num+sec_num+doc_num+1))
    # section connection
    adj[-sec_num-doc_num-1:-doc_num-1, 0:-sec_num-doc_num-1] = sec_sen_mask
    adj[0:-sec_num-doc_num-1, -sec_num-doc_num-1:-doc_num-1] = sec_sen_mask.T
    for i in range(0, doc_num):
        doc_mask = doc_sec_mask[i]
        doc_mask = doc_mask.reshape((1, len(doc_mask)))
        adj[sen_num:-doc_num-1, sen_num:-doc_num-1] += doc_mask * doc_mask.T
    # doc connection
    adj[-doc_num-1:-1, -sec_num-doc_num-1:-doc_num-1] = doc_sec_mask
    adj[-sec_num-doc_num-1:-doc_num-1, -doc_num-1:-1] = doc_sec_mask.T
    adj[-doc_num-1:, -doc_num-1:] = 1

    #build sentence connection
    for i in range(0, sec_num):
        sec_mask = sec_sen_mask[i]
        sec_mask = sec_mask.reshape((1, len(sec_mask)))
        adj[:sen_num, :sen_num] += sec_mask * sec_mask.T
    return adj

def get_cluster_vec(text):
    sent = text.lower()
    input_ids = torch.tensor([tokenizer.encode(sent)])
    sents = sent.split(' . ')
    wcnt = [len(s.split(' ')) for s in sents]
    wcnt_all = sum(wcnt)
    part_cnt = (wcnt_all - 1) // 200 + 1
    word_per_part = wcnt_all // part_cnt + 1

    part_sents, part_id = [], 0
    for i, s in enumerate(sents):
        if len(part_sents) < part_id:
            part_sents.append([])
        if sum(wcnt[:i]) <= (part_id+1) * word_per_part:
            part_sents[part_id].append(s)
        else:
            part_sents[part_id].append(s)
            part_id += 1
    part_sents = [' . '.join(sents) for sents in part_sents]
    input_ids = [torch.tensor([tokenizer.encode(sent)]) for sent in part_sents]
    with torch.no_grad():
        features = [phobert(input_id.to(device))["pooler_output"] for input_id in input_ids]
    return torch.cat(features, dim=0)

class GraphDataset(Dataset):
    def __init__(self, mode='train', limit=200):
        self.clusters, self.goldens = [], []
        self.prepare_data(mode, limit)

    def prepare_data(self, mode, limit):
        if mode == 'train':
            with open(f'{vlsp_dataset_path}/{mode}_segmentedSumm.txt', 'r', encoding='utf-8') as f:
                goldenList = json.load(f)
        elif mode == 'val':
            with open(f'{vlsp_dataset_path}/validation_segmentedSumm.txt', 'r', encoding='utf-8') as f:
                goldenList = [g.strip() for g in f.readlines()]
        else:
            goldenList = ['No summary.'] * limit

        with open(f'{vlsp_dataset_path}/{mode}_tree.pkl', 'rb') as fp:
            clusTrees = pickle.load(fp)
        with open(f'{vlsp_dataset_path}/{mode}_secnum.pkl', 'rb') as fp:
            seclist = pickle.load(fp)

        for cluster in tqdm(range(0, limit)):
            sents, sentVecs, secIDs, spanIds, doc_lens = [], [], [], [], []
            with open(f'{vlsp_dataset_path}/{mode}_vec/{mode}_' + str(cluster) + '.pkl', 'rb') as fp:
                vec_list = pickle.load(fp)
            clusTree = clusTrees[cluster]

            secnum = max(seclist[cluster][len(seclist[cluster]) - 1].values()) + 1
            sentnum = sum([len(doc.values()) for doc in seclist[cluster].values()])
            doc_sec_mask = np.zeros((len(clusTree['docs']), secnum))
            sec_sen_mask = np.zeros((secnum, sentnum))
            cursec, cursent = 0, 0
            for d, doc in enumerate(clusTree['docs']):
                doc_lens.append(len(doc['sents']))
                doc_endsec = max(seclist[cluster][d].values())
                doc_sec_mask[d][cursec:doc_endsec + 1] = 1
                cursec = doc_endsec + 1
                for s, sent in enumerate(doc['sents']):
                    sents.append(sent['raw_sent'])
                    sentVecs.append(torch.from_numpy(meanTokenVecs(d, s, vec_list, clusTree)))
                    spanIds.append((d, s))
                    sec_sen_mask[seclist[cluster][d][s], cursent] = 1
                    cursent += 1

            self.clusters.append(Cluster(cluster, sents, sentVecs, spanIds, doc_lens, doc_sec_mask, sec_sen_mask, goldenList[cluster]))

            sents = goldenList[cluster].split(' . ')
            sentVecs = [self.getPhoBERT_sent_vectors(sent) for sent in sents]
            self.goldens.append(Cluster(cluster, sents, sentVecs, [], [len(sents)], np.ones((1, 1)), np.ones((1, len(sents))), 'abc'))

    def getPhoBERT_sent_vectors(self, sent):
        sent = sent.lower()
        input_ids = torch.tensor([tokenizer.encode(sent, truncation=True)])
        with torch.no_grad():
            features = phobert(input_ids.to(device)).last_hidden_state[0, 1:-1, :].mean(dim=0)
        return features.detach().cpu()

    def __getitem__(self, index):
        data, golden_graph = self.clusters[index], self.goldens[index]
        return {
            'feature': data.feature,
            'labels': data.score_onehot,
            'adj': data.adj,
            'doc_lens': data.doc_lens,
            'docnum': data.docnum,
            'secnum': data.secnum,
            'golden': data.golden,
            'sent_text': data.sent_text,
            'spanIds': data.spanIds,
            'golden_feature': golden_graph.feature,
            'golden_adj': golden_graph.adj,
            'golden_doc_lens': golden_graph.doc_lens,
            'cluster': data.cluster
        }

    def __len__(self):
        return len(self.clusters)

def meanTokenVecs(d, s, vec_list, clusTree):
    tokenVecList, sp = [], 0
    while (d, s, sp) in vec_list:
        span = clusTree['docs'][d]['sents'][s]['spans'][sp]
        if (span['wtype'] == 'CH' or span['word'] in string.punctuation) and not span['children']:
            sp += 1
            continue
        tokenVecList.append(vec_list[(d, s, sp)])
        sp += 1
    sent_vec = np.mean(np.array(tokenVecList), axis=0) if len(tokenVecList) else np.zeros(768)
    return sent_vec

def getPositionEncoding(pos, d=768, n=10000):
    P = np.zeros(d)
    for i in np.arange(int(d/2)):
        denominator = np.power(n, 2*i/d)
        P[2*i] = np.sin(pos/denominator)
        P[2*i+1] = np.cos(pos/denominator)
    return P

def removeRedundant(text):
    text = text.lower()
    words = [w for w in text.split(' ') if w not in stop_w]
    return ' '.join(words)


def divideSection(doc_text, category='Giáo dục'):
    sent_para, para_sec, sent_sec = {}, {}, {}

    paras = [para for para in doc_text.split('\n') if para != '']
    all_sents = []
    # prepare sent_Para
    sentcnt = 0
    for i, para in enumerate(paras):
        sents = [word_tokenize(sent, format="text") for sent in sent_tokenize(para) if sent != '' and len(sent) > 4]
        all_sents.extend(sents)
        for ii, sent in enumerate(sents):
            sent_para[sentcnt + ii] = i
            sent = removeRedundant(sent)
        sentcnt += len(sents)

    # prepare para_sec
    paras = [removeRedundant(para) for para in paras]
    tf, lda_model = cate_models[category]
    X = tf.transform(paras)
    lda_top = lda_model.transform(X)
    for i, para_top in enumerate(lda_top):
        para_sec[i] = para_top.argmax()

    # output sent_sec
    for k, v in sent_para.items():
        sent_sec[k] = para_sec[v]
    return sent_sec, all_sents


def loadClusterData(docs_org, category='Giáo dục'):  # docs_org: list of text for each document
    seclist, docs = {}, []
    for d, doc in enumerate(docs_org):
        seclist[d], sentTexts = divideSection(doc, category)
        docs.append(sentTexts)

    secnum = 0
    for k, val_dict in seclist.items():
        vals = set(val_dict.values())
        for ki, vi in val_dict.items():
            for i, v in enumerate(vals):
                if vi == v:
                    val_dict[ki] = i + secnum
                    break
        seclist[k] = val_dict
        secnum += len(vals)

    sents, sentVecs, secIDs, doc_lens = [], [], [], []
    sentnum = sum([len(doc.values()) for doc in seclist.values()])
    doc_sec_mask = np.zeros((len(docs), secnum))
    sec_sen_mask = np.zeros((secnum, sentnum))
    cursec, cursent = 0, 0

    for d, doc in enumerate(docs):
        doc_lens.append(len(doc))
        doc_endsec = max(seclist[d].values())
        doc_sec_mask[d][cursec:doc_endsec + 1] = 1
        cursec = doc_endsec + 1
        for s, sent in enumerate(doc):
            sents.append(sent)
            sentVecs.append(meanTokenVecs(sent))
            sec_sen_mask[seclist[d][s], cursent] = 1
            cursent += 1

    return Cluster(sents, sentVecs, doc_lens, doc_sec_mask, sec_sen_mask)


class SummDataset(Dataset):
    def __init__(self, folder, mode='train', limit=200):
        self.clusters = []
        self.prepare_data(folder, mode, limit)

    def prepare_data(self, folder, mode, limit, word_cnt_max=220, increase_thres=0.1):
        if mode == 'train':
            with open(f'{vlsp_dataset_path}/{mode}_segmentedSumm.txt', 'r', encoding='utf-8') as f:
                goldenList = json.load(f)
        elif mode == 'val':
            with open(f'{vlsp_dataset_path}/validation_segmentedSumm.txt', 'r', encoding='utf-8') as f:
                goldenList = [g.strip() for g in f.readlines()]
        with open(f'{vlsp_dataset_path}/{mode}_tree.pkl', 'rb') as fp:
            clusTrees = pickle.load(fp)

        for cluster in tqdm(range(0, limit)):
            golden_sents = [s.strip() for s in goldenList[cluster].split(' . ')]
            clusTree = clusTrees[cluster]
            sents = []
            for d, doc in enumerate(clusTree['docs']):
                for s, sent in enumerate(doc['sents']):
                    sents.append(sent['raw_sent'])
            df = pd.DataFrame.from_dict({'text': sents})

            for gold_sent in golden_sents:
                if len(gold_sent) < 5: continue
                df.loc[:, 'scores'] = df['text'].apply(lambda x: getRouge2(gold_sent, x, 'r'))
                scores = torch.tensor(df['scores'].tolist())
                ranked_score_idxs = torch.argsort(scores, dim=0, descending=True)
                texts = df['text'].tolist()

                cur_score, cur_text, summSentIDList = scores[ranked_score_idxs[0]], texts[ranked_score_idxs[0]], [ranked_score_idxs[0]]
                for i in ranked_score_idxs[1:]:
                    if scores[i] <= 0.2: break
                    new_text = cur_text + ' ' + texts[i]
                    new_score = getRouge2(gold_sent, new_text, 'r')
                    if new_score >= cur_score + increase_thres:
                        cur_score, cur_text = new_score, new_text
                        summSentIDList.append(i)

                summSentIDList = sorted(summSentIDList)
                select_by_top_num_df = df.iloc[summSentIDList]
                select_by_top_num_df.loc[:, 'text'] = select_by_top_num_df['text'].apply(lambda x: self.normalize_text(x))
                gold_sent = gold_sent + '.' if gold_sent[-1] != '.' else gold_sent
                self.clusters.append((' '.join(select_by_top_num_df['text']), self.normalize_text(gold_sent)))

    def normalize_text(self, text):
        text = str(text).replace('_', ' ')
        text = re.sub(r'\s+([.,;:"?)/!?”])', r'\1', text)
        text = re.sub(r'([(“])\s+', r'\1', text)
        return text

    def __getitem__(self, index):
        input, output = self.clusters[index]
        text_tokenized = tokenizer_bartpho(input, padding=True, max_length=1024, truncation=True, return_tensors="pt")
        label_tokenized = tokenizer_bartpho(output, padding=True, max_length=1024, truncation=True, return_tensors="pt")
        label_tokenized = label_tokenized['input_ids']
        # label_tokenized[label_tokenized == tokenizer_bartpho.pad_token_id] = -100
        return {'input_ids': text_tokenized['input_ids'][0],
                'attention_mask': text_tokenized['attention_mask'][0],
                'labels': label_tokenized[0]}

    def __len__(self):
        return len(self.clusters)

