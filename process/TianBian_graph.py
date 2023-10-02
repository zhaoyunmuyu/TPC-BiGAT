import os
import gc
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
# from config.config import Config

class Node_tweet(object):
    def __init__(self, idx=None):
        self.children = []
        self.idx = idx
        self.word = []
        self.index = []
        self.parent = None

def str2matrix(Str):
    wordFreq, wordIndex = [], []
    for pair in Str.split(' '):
        freq=float(pair.split(':')[1])
        index=int(pair.split(':')[0])
        if index<=5000:
            wordFreq.append(freq)
            wordIndex.append(index)
    return wordFreq, wordIndex

def constructMat(tree):
    index2node = {}
    for i in tree:
        node = Node_tweet(idx=i)
        index2node[i] = node
    for j in tree:
        indexC = j
        indexP = tree[j]['parent']
        nodeC = index2node[indexC]
        wordFreq, wordIndex = str2matrix(tree[j]['vec'])
        nodeC.index = wordIndex
        nodeC.word = wordFreq
        ## not root node ##
        if not indexP == 'None':
            nodeP = index2node[int(indexP)]
            nodeC.parent = nodeP
            nodeP.children.append(nodeC)
        ## root node ##
        else:
            rootindex=indexC-1
            root_index=nodeC.index
            root_word=nodeC.word
    rootfeat = np.zeros([1, 5000])
    if len(root_index)>0:
        rootfeat[0, np.array(root_index)] = np.array(root_word)
    matrix=np.zeros([len(index2node),len(index2node)])
    row=[]
    col=[]
    x_word=[]
    x_index=[]
    for index_i in range(len(index2node)):
        for index_j in range(len(index2node)):
            if index_i+1 in index2node.keys() and index_j+1 in index2node.keys():
                if index2node[index_i+1].children != None and index2node[index_j+1] in index2node[index_i+1].children:
                    matrix[index_i][index_j]=1
                    row.append(index_i)
                    col.append(index_j)
        if index_i+1 in index2node.keys():
            x_word.append(index2node[index_i+1].word)
            x_index.append(index2node[index_i+1].index)
    edgematrix=[row,col]
    x = np.zeros([len(x_index), 5000])
    for i in range(len(x_index)):
        if len(x_index[i])>0:
            x[i, np.array(x_index[i])] = np.array(x_word[i])
    return x, edgematrix, rootfeat, rootindex

def loadEid(path, treeDic, y, logger):
    out = path.split('.')[0].split('/')
    if os.path.exists(path):
        logger.info(f"{out[-1]}:Pass")
        return None
    if treeDic is None: return None
    try:
        x_features, edgematrix, root_feat, root_index = constructMat(treeDic)
    except:
        return None
    features = np.array(x_features)
    tree = np.array(edgematrix)
    # if len(tree) == 0: return None
    rootfeat = np.array(root_feat)
    rootindex = np.array(root_index)
    label = np.array(y)
    del x_features, y, edgematrix, root_feat, root_index
    gc.collect()
    np.savez(path,x=features,y=label,root=rootfeat,rootindex=rootindex,tree=tree)
    del features, label, rootfeat, rootindex, tree
    gc.collect()
    out = path.split('.')[0].split('/')
    # logger.info(f"{out[-1]}:{len(treeDic)}")
    return None

def extract_TianBian_dataset(config, obj):
    logger = config.logger
    post_delay = config.post_delay
    treeDic = {}
    labelDic = {}
    event = []
    if config.dataset == 'TianBian-Weibo':
        treePath = os.path.join(config.dataset_dir, 'weibotree.txt')
        labelPath = os.path.join(config.dataset_dir, 'weibo_id_label.txt')
        for line in open(treePath):
            line = line.rstrip()
            eid, indexP, indexC, Vec = line.split('\t')[0], line.split('\t')[1], int(line.split('\t')[2]), line.split('\t')[3]
            if not treeDic.__contains__(eid):
                treeDic[eid] = {}
            if len(treeDic[eid]) <= post_delay:
                treeDic[eid][indexC] = {'parent': indexP, 'vec': Vec}
        # label
        l1 = l2 = 0
        for line in open(labelPath):
            line = line.rstrip()
            eid, label = line.split(' ')[0], line.split(' ')[1]
            labelDic[eid] = int(label)
            event.append(eid)
            if labelDic[eid]==0: l1 += 1
            if labelDic[eid]==1: l2 += 1
    else:
        treePath = os.path.join(config.dataset_dir, 'data.TD_RvNN.vol_5000.txt')
        labelPath = os.path.join(config.dataset_dir, obj+'_label_All.txt')
        for line in open(treePath):
            line = line.rstrip()
            eid, indexP, indexC = line.split('\t')[0], line.split('\t')[1], int(line.split('\t')[2])
            max_degree, maxL, Vec = int(line.split('\t')[3]), int(line.split('\t')[4]), line.split('\t')[5]
            if not treeDic.__contains__(eid):
                treeDic[eid] = {}
            if len(treeDic[eid]) <= post_delay:
                treeDic[eid][indexC] = {'parent': indexP, 'max_degree': max_degree, 'maxL': maxL, 'vec': Vec}
        # label
        label_n, label_f, label_t, label_u = ['news', 'non-rumor'], ['false'], ['true'], ['unverified']
        l1 = l2 = l3 = l4 = 0
        for line in open(labelPath):
            line = line.rstrip()
            label, eid = line.split('\t')[0], line.split('\t')[2]
            label=label.lower()
            event.append(eid)
            if label in label_n: labelDic[eid]=0 ; l1 += 1
            if label in label_f: labelDic[eid]=1 ; l2 += 1
            if label in label_t: labelDic[eid]=2 ; l3 += 1
            if label in label_u: labelDic[eid]=3 ; l4 += 1
    if not os.path.exists(config.result_graph_dir):  
        os.makedirs(config.result_graph_dir)
    Parallel(n_jobs=3, backend='threading')(delayed(loadEid)(os.path.join(config.result_graph_dir, eid+'.npz'), treeDic[eid] if eid in treeDic else None, labelDic[eid], logger) for eid in tqdm(event))
    return

if __name__ == '__main__':
    path = '/home/code/2_Paper/BEARD/graph/TianBian-Twitter15/100'
    print(len(os.listdir(path)))