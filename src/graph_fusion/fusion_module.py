import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .graph_learn import GraphLearner
from .gcn import GCN
from .graph_utils import normalize_adj, to_cuda

VERY_SMALL_NUMBER = 1e-12


class GraphFusion(nn.Module):
    def __init__(self, input_dim, hidden_dim, gnn_nhid, gnn_ofeat,
                 topk=None, epsilon=None, num_pers=4, metric_type='weighted_cosine', graph_include_self=False,
                 gnn_hops=2, gnn_dropout=0.5, gnn_batch_norm=False, feat_adj_dropout=0, graph_skip_conn=0.2,
                 device=None) -> None:
        super(GraphFusion, self).__init__()

        self.device = device
        self.graph_metric_type = metric_type
        self.feat_adj_dropout = feat_adj_dropout
        self.graph_skip_conn = graph_skip_conn
        self.graph_include_self = graph_include_self

        self.graph_learner =  GraphLearner(input_size=input_dim, hidden_size=hidden_dim, 
                                          topk=topk, epsilon=epsilon, num_pers=num_pers,
                                          metric_type=metric_type, device=device)
        self.graph_learner2 = GraphLearner(input_size=input_dim, hidden_size=hidden_dim, 
                                          topk=topk, epsilon=epsilon, num_pers=num_pers,
                                          metric_type=metric_type, device=device)
        
        self.gcn = GCN(input_dim, gnn_nhid, gnn_ofeat, 
                       graph_hops=gnn_hops, dropout=gnn_dropout, batch_norm=gnn_batch_norm)
    
        self.param_init()

        
    def param_init(self):
        for name, param in self.named_parameters():
            if str.endswith(name, 'weight'):
                nn.init.kaiming_normal_(param)
            elif str.endswith(name, 'bias'):
                nn.init.zeros_(param)
        

    def forward(self, stru_embeds, mm_embeds):
        """
        stru_embeds:    [nums x input_dim]
        mm_embeds:      [nums x input_dim]
        """
        struc_shape, mm_shape = stru_embeds.shape, mm_embeds.shape
        embeddings = torch.cat([stru_embeds, mm_embeds], dim=0)

        cur_raw_adj, cur_adj = self.learn_graph(self.graph_learner, embeddings)
        cur_raw_adj = F.dropout(cur_raw_adj, self.feat_adj_dropout, training=self.training)
        cur_adj = F.dropout(cur_adj, self.feat_adj_dropout, training=self.training)

        # GNN message propagation
        node_vec = torch.relu(self.gcn.graph_encoders[0](embeddings, cur_adj))
        node_vec = F.dropout(node_vec, self.gcn.dropout, training=self.training)
        # Add mid GNN layers
        for encoder in self.gcn.graph_encoders[1:-1]:
            node_vec = torch.relu(encoder(node_vec, cur_adj))
            node_vec = F.dropout(node_vec, self.gcn.dropout, training=self.training)
        # BP to update weights
        output = self.gcn.graph_encoders[-1](node_vec, cur_adj)
        output = F.log_softmax(output, dim=-1)
        

    def learn_graph(self, graph_learner, node_features, graph_skip_conn=None, graph_include_self=False, init_adj=None):
       
        raw_adj = graph_learner(node_features)

        if self.graph_metric_type in ('kernel', 'weighted_cosine'):
            assert raw_adj.min().item() >= 0
            adj = raw_adj / torch.clamp(torch.sum(raw_adj, dim=-1, keepdim=True), min=VERY_SMALL_NUMBER)

        elif self.graph_metric_type == 'cosine':
            adj = (raw_adj > 0).float()
            adj = normalize_adj(adj)

        else:
            adj = torch.softmax(raw_adj, dim=-1)


        if graph_skip_conn in (0, None):
            if graph_include_self:
                adj = adj + to_cuda(torch.eye(adj.size(0)), self.device)
        else:
            adj = graph_skip_conn * init_adj.to_dense() + (1 - graph_skip_conn) * adj

        return raw_adj, adj

    

