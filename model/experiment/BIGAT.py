import copy
import torch
from config.config import Config
import torch.nn.functional as F
from torch_scatter import scatter_mean
from torch_geometric.nn import GATConv

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Top2Down
class TDGAT(torch.nn.Module):
    def __init__(self,in_feats,hid_feats,out_feats, dropout):
        super(TDGAT, self).__init__()
        self.dropout = dropout
        self.gat1 = GATConv(in_feats, hid_feats)
        self.gat2 = GATConv(hid_feats, out_feats)

    def forward(self, data):
        x = data.x
        edge_index = data.TD_edge_index
        x = self.gat1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gat2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = scatter_mean(x, data.batch, dim=0)
        return x

# Bottom2Up
class BUGAT(torch.nn.Module):
    def __init__(self,in_feats,hid_feats,out_feats,dropout):
        super(BUGAT, self).__init__()
        self.dropout = dropout
        self.gat1 = GATConv(in_feats, hid_feats)
        self.gat2 = GATConv(hid_feats, out_feats)

    def forward(self, data):
        x = data.x
        edge_index = data.BU_edge_index
        x = self.gat1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gat2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = scatter_mean(x, data.batch, dim=0)
        return x

# 网络结构
class BiGAT(torch.nn.Module):
    def __init__(self,in_feats,hid_feats,out_feats,num_class,dropout):
        super(BiGAT, self).__init__()
        self.TDGAT = TDGAT(in_feats, hid_feats, out_feats, dropout)
        self.BUGAT = BUGAT(in_feats, hid_feats, out_feats, dropout)
        self.fc=torch.nn.Linear(out_feats * 2, num_class)

    def forward(self, x_data):
        TD_x = self.TDGAT(x_data)
        BU_x = self.BUGAT(x_data)
        x = torch.cat((BU_x, TD_x), 1)
        x = self.fc(x)
        x = F.log_softmax(x, dim=1)
        return x
    
    def get_name(self):
        return 'bigat'

if __name__ == '__main__':
    config = Config()