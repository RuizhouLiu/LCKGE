import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .graph_utils import to_cuda

class GCNLayer(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=False, batch_norm=False):
        super(GCNLayer, self).__init__()
        self.weight = torch.Tensor(in_features, out_features)
        self.weight = nn.Parameter(nn.init.xavier_uniform_(self.weight))
        if bias:
            self.bias = torch.Tensor(out_features)
            self.bias = nn.Parameter(nn.init.xavier_uniform_(self.bias))
        else:
            self.register_parameter('bias', None)

        self.bn = nn.BatchNorm1d(out_features) if batch_norm else None

    def forward(self, input, adj, batch_norm=True):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support)

        if self.bias is not None:
            output = output + self.bias

        if self.bn is not None and batch_norm:
            output = self.compute_bn(output)

        return output

    def compute_bn(self, x):
        if len(x.shape) == 2:
            return self.bn(x)
        else:
            return self.bn(x.view(-1, x.size(-1))).view(x.size())


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, ofeat, graph_hops, dropout, batch_norm=False):
        super(GCN, self).__init__()
        self.dropout = dropout

        self.graph_encoders = nn.ModuleList()
        self.graph_encoders.append(GCNLayer(nfeat, nhid, batch_norm=batch_norm))

        for _ in range(graph_hops - 2):
            self.graph_encoders.append(GCNLayer(nhid, nhid, batch_norm=batch_norm))

        self.graph_encoders.append(GCNLayer(nhid, ofeat, batch_norm=False))


    def forward(self, x, node_anchor_adj):
        for i, encoder in enumerate(self.graph_encoders[:-1]):
            x = F.relu(encoder(x, node_anchor_adj))
            x = F.dropout(x, self.dropout, training=self.training)

        x = self.graph_encoders[-1](x, node_anchor_adj)
        return x
