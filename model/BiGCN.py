'''
 Filename:  BiGCN.py
 Description:  双向图卷积网络模型结构
 Created:  2022年11月2日 16时49分
 Author:  Li Shao
'''

import copy
import torch
from config.config import Config
import torch.nn.functional as F
from torch_scatter import scatter_mean
from torch_geometric.nn import GCNConv

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Top2Down
class TDGCN(torch.nn.Module):
    def __init__(self,in_feats,hid_feats,out_feats):
        super(TDGCN, self).__init__()
        self.conv1 = GCNConv(in_feats, hid_feats)
        # 加强了根源节点，所以为hid+in
        self.conv2 = GCNConv(hid_feats+in_feats, out_feats)

    # 输入节点特征矩阵x和邻接关系edge_index
    def forward(self, data):
        x, edge_index = data.x, data.TD_edge_index
        x1=copy.copy(x.float())
        x = self.conv1(x, edge_index)
        # 根源节点加强
        x2=copy.copy(x)
        rootindex = data.rootindex
        root_extend = torch.zeros(len(data.batch), x1.size(1)).to(device)
        batch_size = max(data.batch) + 1
        for num_batch in range(batch_size):
            index = (torch.eq(data.batch, num_batch))
            root_extend[index] = x1[rootindex[num_batch]]
        x = torch.cat((x,root_extend), 1)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        # 根源节点加强
        root_extend = torch.zeros(len(data.batch), x2.size(1)).to(device)
        for num_batch in range(batch_size):
            index = (torch.eq(data.batch, num_batch))
            root_extend[index] = x2[rootindex[num_batch]]
        x = torch.cat((x,root_extend), 1)
        # x = F.relu(x)
        # x = F.dropout(x, training=self.training)
        # 进行mean运算
        x= scatter_mean(x, data.batch, dim=0)
        return x

# Bottom2Up
class BUGCN(torch.nn.Module):
    def __init__(self,in_feats,hid_feats,out_feats):
        super(BUGCN, self).__init__()
        self.conv1 = GCNConv(in_feats, hid_feats)
        self.conv2 = GCNConv(hid_feats+in_feats, out_feats)

    def forward(self, data):
        x = data.x
        edge_index = data.BU_edge_index
        x1 = copy.copy(x.float())
        x = self.conv1(x, edge_index)
        x2 = copy.copy(x)
        rootindex = data.rootindex
        root_extend = torch.zeros(len(data.batch), x1.size(1)).to(device)
        batch_size = max(data.batch) + 1
        for num_batch in range(batch_size):
            index = (torch.eq(data.batch, num_batch))
            root_extend[index] = x1[rootindex[num_batch]]
        x = torch.cat((x,root_extend), 1)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        root_extend = torch.zeros(len(data.batch), x2.size(1)).to(device)
        for num_batch in range(batch_size):
            index = (torch.eq(data.batch, num_batch))
            root_extend[index] = x2[rootindex[num_batch]]
        x = torch.cat((x,root_extend), 1)
        # x = F.relu(x)
        # x = F.dropout(x, training=self.training)
        x= scatter_mean(x, data.batch, dim=0)
        return x

# 网络结构
class BiGCN(torch.nn.Module):
    def __init__(self,in_feats,hid_feats,out_feats,num_class,dropout):
        super(BiGCN, self).__init__()
        self.TDGCN = TDGCN(in_feats, hid_feats, out_feats)
        self.BUGCN = BUGCN(in_feats, hid_feats, out_feats)
        self.fc=torch.nn.Linear((out_feats+hid_feats)*2,num_class)

    def forward(self, x_data):
        TD_x = self.TDGCN(x_data)
        BU_x = self.BUGCN(x_data)
        x = torch.cat((BU_x, TD_x), 1)
        x = self.fc(x)
        x = F.log_softmax(x, dim=1)
        return x
    
    def get_name(self):
        return 'BiGCN'

if __name__ == '__main__':
    config = Config()