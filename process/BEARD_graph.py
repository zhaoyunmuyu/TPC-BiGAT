'''
 Filename:  make_graph.py
 Description:  根据原始data构建图数据并保存
 Created:  2022年11月2日 16时49分
 Author:  Li Shao
'''

import os
import gc
import json
import time
import torch
import warnings
import numpy as np
from tqdm import tqdm
from datetime import datetime
from joblib import Parallel, delayed
from config.config import Config
from utils.text_filter import text_filter
from transformers import RobertaTokenizer, RobertaModel, AlbertTokenizer, AlbertModel, logging
logging.set_verbosity_error()
warnings.filterwarnings('ignore', category=FutureWarning)

tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
token_model = RobertaModel.from_pretrained('roberta-base')
# tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
# token_model = AlbertModel.from_pretrained('albert-base-v2')


class Node_tweet(object):
    def __init__(self):
        self.index = 0
        self.children = []
        self.parent = None
        self.inputs_features=[]
        self.sen_len=0


def timestamp(timestr):
    dateFormatter = "%a %b %d %H:%M:%S %z %Y"
    temp = datetime.strptime(timestr, dateFormatter)
    struct_time = time.strptime(str(temp),'%Y-%m-%d %H:%M:%S%z')
    return time.mktime(struct_time)


def str2vec(str):
    with torch.no_grad():
        inputs = tokenizer(str, return_tensors="pt")
        outputs = token_model(**inputs)
        last_hidden_states = outputs.last_hidden_state   #torch.Size([1, sen_len, 768])
        sen_len=int(last_hidden_states.size()[-2])-2
        # word_vec= last_hidden_states.squeeze(0)[1:-1]  #drop out the [CLS] and [SEP]
        word_vec = last_hidden_states.squeeze(0)[0]      #sentece_vector
    return word_vec, sen_len


def BERD_topic_text(path, topic_id):
    with open(path+'/BEARD_info.json','r',encoding='utf-8') as f :
        data = json.load(f)
        text = data[topic_id]['claim']
    return text


def constructPost(treeDic):
    index2node = {}
    for j in treeDic:
        index = treeDic[j]['index']
        print(index,'/',len(treeDic))
        indexP = treeDic[j]['parent']
        indexC = j
        index2node[indexC] = Node_tweet()
        nodeC = index2node[indexC]
        nodeC.index = index
        # 删除URL
        text = treeDic[indexC]['text']
        token_features, sen_len = str2vec(text)
        nodeC.inputs_features.append(token_features)
        nodeC.sen_len = sen_len
        if not indexP == None:
            nodeP = index2node[indexP]
            nodeC.parent = nodeP
            nodeP.children.append(nodeC)
        else:
            rootindex=nodeC.index
            rootfeat=nodeC.inputs_features[0].detach()
    row=[]
    col=[]
    x_features=[]
    x_senlen=[]
    for id in index2node:
        index_r = index2node[id].index
        if index2node[id].children != None:
            for child in index2node[id].children:
                index_c = child.index
                row.append(index_r)
                col.append(index_c)
        x_features.append(index2node[id].inputs_features[0].detach())
        x_senlen.append(index2node[id].sen_len)
    edgematrix=[row,col]
    return x_features, x_senlen, edgematrix, rootfeat, rootindex


def save_data(path, y, treeDic, logger, Level='Topic'):
    if treeDic is None: return None
    x_features, x_senlen, edgematrix, root_feat, root_index = constructPost(treeDic)
    features = np.array(x_features, dtype=object)
    senlen = np.array(x_senlen)
    tree = np.array(edgematrix)
    # if len(tree) == 0: return None
    rootfeat = root_feat
    rootindex = np.array(root_index)
    label = np.array(y)
    del x_features, y, x_senlen, edgematrix, root_feat, root_index
    gc.collect()
    np.savez(path,x=features,y=label,root=rootfeat,rootindex=rootindex,tree=tree,seqlen=senlen)
    del features, label, rootfeat, rootindex, tree, senlen
    gc.collect()
    out = path.split('.')[0].split('/')
    if Level == 'Post':
        logger.info(f"{out[-2]}-{out[-1]}:{len(treeDic)}")
    elif Level == 'Topic':
        logger.info(f"{out[-1]}:{len(treeDic)}")
    return None


def extract_post(post_path, graph_dir, time_delay, logger):
    index = 0
    treeDic = {}
    post_id = post_path.split('/')[-1]
    event_id = post_path.split('/')[-2]
    if not os.path.exists(os.path.join(graph_dir, event_id)):  
        os.makedirs(os.path.join(graph_dir, event_id))
    if os.path.exists(os.path.join(graph_dir, event_id, post_id+'.npz')):
        logger.info(f"{event_id}-{post_id}:Pass")
        return None
    with open(post_path+'/original.jsonl') as f:
        data = json.load(f)
        text = data['full_text'].replace("\n","")
        ori_time = timestamp(data['created_at'])
        treeDic[post_id] = {'index':index, 'parent': None, 'time':ori_time, 'text': text}
        index += 1
    for _, sub_dirs, _ in os.walk(post_path):
        if sub_dirs == []:
            break
        for dir in sorted(sub_dirs):
            sub_path = os.path.join(post_path,dir)
            sub_list = os.listdir(sub_path)
            for comment in sub_list:
                comment_path = os.path.join(sub_path,comment)
                with open(comment_path) as f:
                    data = json.load(f)
                    time = timestamp(data['created_at'])
                    if (time - ori_time) < time_delay:
                        indexP = comment.split('-')[0]
                        indexC = comment.split('-')[1].split('.')[0]
                        text = data['full_text'].replace("\n","")
                        text = text_filter(text)
                        treeDic[indexC] = {'index':index, 'parent': indexP, 'time':time, 'text': text}
                        index += 1
        break
    return treeDic


def extract_event_allP(event_path, event_text, graph_dir, time_delay, logger):
    index = 0
    treeDic = {}
    ori_times = []
    event_id = event_path.split('/')[-1]
    if not os.path.exists(graph_dir):  
        os.makedirs(graph_dir)
    if os.path.exists(os.path.join(graph_dir, event_id+'.npz')):
        logger.info(f"{event_id}:Pass")
        return None
    treeDic[event_id] = {'index':index, 'parent': None, 'text': event_text}
    index += 1

    for _, dirs, _ in os.walk(event_path):
        post_list = dirs
        break
    for post_id in post_list:
        post_path = os.path.join(event_path, post_id)
        with open(post_path+'/original.jsonl') as f:
            # print(event_id,post_id)
            data = json.load(f)
            ori_times.append(timestamp(data['created_at']))
    ori_times.sort()
    ori_time = ori_times[0]

    for post_id in post_list:
        post_path = os.path.join(event_path, post_id)
        with open(post_path+'/original.jsonl') as f:
            data = json.load(f)
            text = data['full_text'].replace("\n","")
            treeDic[post_id] = {'index':index, 'parent': event_id, 'text': text}
            index += 1
        for _, sub_dirs, _ in os.walk(post_path):
            if sub_dirs == []:
                break
            for dir in sorted(sub_dirs):
                sub_path = os.path.join(post_path,dir)
                sub_list = os.listdir(sub_path)
                for comment in sub_list:
                    comment_path = os.path.join(sub_path,comment)
                    with open(comment_path) as f:
                        print(comment_path)
                        data = json.load(f)
                        time = timestamp(data['created_at']) 
                        if (time - ori_time) < time_delay:
                            indexP = comment.split('-')[0]
                            indexC = comment.split('-')[1].split('.')[0]
                            text = data['full_text'].replace("\n","")
                            text = text_filter(text)
                            treeDic[indexC] = {'index':index, 'parent': indexP, 'text': text}
                            index += 1
            break
    path = os.path.join(graph_dir, event_id+'.npz')
    label = 1 if event_id[0] == 'S' else 0
    treeDic_len = save_data(path, label, treeDic, logger)
    return treeDic_len


def extract_event_eachP(event_path, event_text, graph_dir, time_delay, logger):
    index = 0
    treeDic = {}
    event_id = event_path.split('/')[-1]
    if not os.path.exists(graph_dir):  
        os.makedirs(graph_dir)
    if os.path.exists(os.path.join(graph_dir, event_id+'.npz')):
        logger.info(f"{event_id}:Pass")
        return None
    treeDic[event_id] = {'index':index, 'parent': None, 'text': event_text}
    index += 1

    for _, dirs, _ in os.walk(event_path):
        post_list = dirs
        break
    for post_id in post_list:
        post_path = os.path.join(event_path, post_id)
        with open(post_path+'/original.jsonl') as f:
            data = json.load(f)
            text = data['full_text'].replace("\n","")
            ori_time = timestamp(data['created_at'])
            treeDic[post_id] = {'index':index, 'parent': event_id, 'text': text}
            index += 1
        for _, sub_dirs, _ in os.walk(post_path):
            if sub_dirs == []:
                break
            for dir in sorted(sub_dirs):
                sub_path = os.path.join(post_path,dir)
                sub_list = os.listdir(sub_path)
                for comment in sub_list:
                    comment_path = os.path.join(sub_path,comment)
                    with open(comment_path) as f:
                        data = json.load(f)
                        time = timestamp(data['created_at']) 
                        if (time - ori_time) < time_delay:
                            indexP = comment.split('-')[0]
                            indexC = comment.split('-')[1].split('.')[0]
                            text = data['full_text'].replace("\n","")
                            text = text_filter(text)
                            treeDic[indexC] = {'index':index, 'parent': indexP, 'text': text}
                            index += 1
            break
    path = os.path.join(graph_dir, event_id+'.npz')
    label = 1 if event_id[0] == 'S' else 0
    treeDic_len = save_data(path, label, treeDic, logger)
    return treeDic_len


def extract_BEARD_dataset(config):
    dataset_dir = config.dataset_dir
    result_graph_dir = config.result_graph_dir
    time_delay = config.time_delay
    logger = config.logger
    level = config.level

    event_list = os.listdir(dataset_dir)
    for event_id in event_list:
        event_path = os.path.join(dataset_dir, event_id)
        for _, dirs, _ in os.walk(event_path):
            post_list = dirs
            break
        if post_list == []:
            logger.info(f"{event_id}:None structure")
        # Post-Level
        elif level == 'Post':
            logger.info(f"Start Post-Level Graph")
            post_dict = {}
            for post_id in dirs:
                post_dict[post_id] = {}
                post_path = os.path.join(event_path, post_id)
                post_dict[post_id] = extract_post(post_path, result_graph_dir, time_delay, logger)
                # label = 1 if event_id[0] == 'S' else 0
                # path = os.path.join(graph_dir, event_id, post_id+'.npz')
                # save_data(path, label, post_dict[post_id], logger, 'Post')
            label = 1 if event_id[0] == 'S' else 0
            Parallel(n_jobs=3, backend='threading')(delayed(save_data)(os.path.join(result_graph_dir, event_id, post_id+'.npz'), label, post_dict[post_id], logger, 'Post') for post_id in post_dict)
        # Topic-Level
        elif level == 'Topic-allP':
            event_text = BERD_topic_text(dataset_dir+'/../', event_id)
            print(event_id)
            extract_event_allP(event_path, event_text, result_graph_dir, time_delay, logger)
        elif level == 'Topic-eachP':
            event_text = BERD_topic_text(dataset_dir+'/../', event_id)
            extract_event_eachP(event_path, event_text, result_graph_dir, time_delay, logger)

if __name__ == '__main__':
    config = Config()
    extract_BEARD_dataset(config, 'structure0_25')
