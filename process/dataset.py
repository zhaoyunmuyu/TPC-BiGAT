'''
 Filename:  dataset.py
 Description:  通过文件和id_list构建Dataset
 Created:  2022年11月2日 16时49分
 Author:  Li Shao
'''

import os
import sys
sys.path.append(sys.path[0]+'/..')
import torch
import random
import numpy as np
from torch.utils.data import Dataset, TensorDataset
from torch_geometric.data import Data
from config.config import Config

# TD图结构
class TDGraphDataset(Dataset):
    def __init__(self, graph, data_path, droprate=0):
        self.graph = graph
        self.data_path = data_path
        self.droprate = droprate

    def __len__(self):
        return len(self.graph)

    def __getitem__(self, index):
        id =self.graph[index]
        data=np.load(os.path.join(self.data_path, id), allow_pickle=True)
        edgeindex = data['tree']
        if self.droprate > 0:
            row = list(edgeindex[0])
            col = list(edgeindex[1])
            length = len(row)
            poslist = random.sample(range(length), int(length * (1 - self.droprate)))
            poslist = sorted(poslist)
            rows = list(np.array(row)[poslist])
            cols = list(np.array(col)[poslist])
            new_edgeindex = [rows, cols]
        else:
            new_edgeindex = edgeindex
        try:
            x_features = torch.as_tensor(torch.stack(list(data['x'])),dtype=torch.float32)
        except:
            x_features = torch.tensor(data['x'],dtype=torch.float32)
        x_data = Data(x=x_features,
                    edge_index=torch.LongTensor(new_edgeindex),
                    root=torch.from_numpy(data['root']),
                    rootindex=torch.LongTensor([int(data['rootindex'])]))
        y_data = Data(y=torch.LongTensor([int(data['y'])]))
        return [x_data,y_data]

# BU图结构
class BUGraphDataset(Dataset):
    def __init__(self, graph, data_path, droprate=0):
        self.graph = graph
        self.data_path = data_path
        self.droprate = droprate

    def __len__(self):
        return len(self.graph)

    def __getitem__(self, index):
        id =self.graph[index]
        data=np.load(os.path.join(self.data_path, id), allow_pickle=True)
        edgeindex = data['tree']
        if self.droprate > 0:
            row = list(edgeindex[1])
            col = list(edgeindex[0])
            length = len(row)
            poslist = random.sample(range(length), int(length * (1 - self.droprate)))
            poslist = sorted(poslist)
            rows = list(np.array(row)[poslist])
            cols = list(np.array(col)[poslist])
            new_edgeindex = [rows, cols]
        else:
            new_edgeindex = edgeindex
        try:
            x_features = torch.as_tensor(torch.stack(list(data['x'])),dtype=torch.float32)
        except:
            x_features = torch.tensor(data['x'],dtype=torch.float32)
        x_data = Data(x=x_features,
                    edge_index=torch.LongTensor(new_edgeindex),
                    root=torch.from_numpy(data['root']),
                    rootindex=torch.LongTensor([int(data['rootindex'])]))
        y_data = Data(y=torch.LongTensor([int(data['y'])]))
        return [x_data,y_data]

# TD,BU双层图结构
class BiGraphDataset(Dataset):
    def __init__(self, graph, data_path, tddroprate=0, budroprate=0):
        self.graph = graph
        self.data_path = data_path
        self.tddroprate = tddroprate
        self.budroprate = budroprate

    def __len__(self):
        return len(self.graph)

    def __getitem__(self, index):
        id =self.graph[index]
        data=np.load(os.path.join(self.data_path, id), allow_pickle=True)
        # data=np.load(os.path.join(self.data_path, id + ".npz"), allow_pickle=True)
        edgeindex = data['tree']
        # ToptoDown删边
        if self.tddroprate > 0:
            tdrow = list(edgeindex[0])
            tdcol = list(edgeindex[1])
            length = len(tdrow)
            poslist = random.sample(range(length), int(length * (1 - self.tddroprate)))
            poslist = sorted(poslist)
            row = list(np.array(tdrow)[poslist])
            col = list(np.array(tdcol)[poslist])
            td_edgeindex = [row, col]
        else:
            td_edgeindex = edgeindex
        # BottomtoUp删边(行列交换)
        burow = list(edgeindex[1])
        bucol = list(edgeindex[0])
        if self.budroprate > 0:
            length = len(burow)
            poslist = random.sample(range(length), int(length * (1 - self.budroprate)))
            poslist = sorted(poslist)
            row = list(np.array(burow)[poslist])
            col = list(np.array(bucol)[poslist])
            bu_edgeindex = [row, col]
        else:
            bu_edgeindex = [burow,bucol]
        # 返回数据结构
        # seqlen = torch.LongTensor(np.array([(data['seqlen'])])).squeeze()
        try:
            x_features = torch.as_tensor(torch.stack(list(data['x'])),dtype=torch.float32)
        except:
            x_features = torch.tensor(data['x'],dtype=torch.float32)
        x_data = Data(x=x_features,
                    TD_edge_index=torch.LongTensor(td_edgeindex),
                    BU_edge_index=torch.LongTensor(bu_edgeindex),
                    root=torch.from_numpy(data['root']),
                    rootindex=torch.LongTensor([int(data['rootindex'])])
                    # ,seqlen=seqlen
                    )
        y_data = Data(y=torch.LongTensor([int(data['y'])]))
        # train_ids = TensorDataset(x_data, y_data) 
        return [x_data,y_data]

# 无向图结构(合并两个矩阵)
class UdGraphDataset(Dataset):
    def __init__(self, graph, data_path, droprate=0):
        self.graph = graph
        self.data_path = data_path
        self.droprate = droprate

    def __len__(self):
        return len(self.graph)

    def __getitem__(self, index):
        id =self.graph[index]
        data=np.load(os.path.join(self.data_path, id), allow_pickle=True)
        edgeindex = data['tree']
        row = list(edgeindex[0])
        col = list(edgeindex[1])
        burow = list(edgeindex[1])
        bucol = list(edgeindex[0])
        row.extend(burow)
        col.extend(bucol)
        if self.droprate > 0:
            length = len(row)
            poslist = random.sample(range(length), int(length * (1 - self.droprate)))
            poslist = sorted(poslist)
            rows = list(np.array(row)[poslist])
            cols = list(np.array(col)[poslist])
            new_edgeindex = [rows, cols]
        else:
            new_edgeindex = [row, col]
        # 返回数据结构
        try:
            x_features = torch.as_tensor(torch.stack(list(data['x'])),dtype=torch.float32)
        except:
            x_features = torch.tensor(data['x'],dtype=torch.float32)
        x_data = Data(x=x_features,
                    edge_index=torch.LongTensor(new_edgeindex),
                    root=torch.from_numpy(data['root']),
                    rootindex=torch.LongTensor([int(data['rootindex'])]))
        y_data = Data(y=torch.LongTensor([int(data['y'])]))
        return [x_data,y_data]

# 读取数据集
def loadTdData(train_id, test_id, graph_dir, logger, droprate=0):
    traindata_list = TDGraphDataset(train_id, graph_dir, droprate=droprate)
    testdata_list = TDGraphDataset(test_id, graph_dir, droprate=droprate)
    logger.info(f"Loading BiGraph train set:{len(traindata_list)}, test set:{len(testdata_list)}")
    return traindata_list, testdata_list

def loadBuData(train_id, test_id, graph_dir, logger, droprate=0):
    traindata_list = BUGraphDataset(train_id, graph_dir, droprate=droprate)
    testdata_list = BUGraphDataset(test_id, graph_dir, droprate=droprate)
    logger.info(f"Loading BiGraph train set:{len(traindata_list)}, test set:{len(testdata_list)}")
    return traindata_list, testdata_list

def loadUdData(train_id, test_id, graph_dir, logger, droprate=0):
    traindata_list = UdGraphDataset(train_id, graph_dir, droprate=droprate)
    testdata_list = UdGraphDataset(test_id, graph_dir, droprate=droprate)
    logger.info(f"Loading BiGraph train set:{len(traindata_list)}, test set:{len(testdata_list)}")
    return traindata_list, testdata_list

def loadBiData(train_id, test_id, graph_dir, logger, TDdroprate=0, BUdroprate=0):
    traindata_list = BiGraphDataset(train_id, graph_dir, tddroprate=TDdroprate, budroprate=BUdroprate)
    testdata_list = BiGraphDataset(test_id, graph_dir, tddroprate=TDdroprate, budroprate=BUdroprate)
    logger.info(f"Loading BiGraph train set:{len(traindata_list)}, test set:{len(testdata_list)}")
    return traindata_list, testdata_list

if __name__ == '__main__':
    config = Config()
    data_path = config.result_graph_dir
    graph = ['766299560415223808.npz']
    dataset = BiGraphDataset(graph, data_path)
    for item in dataset:
        print(type(item[0]))
