import torch
import numpy as np

VERY_SMALL_NUMBER = 1e-12

def to_cuda(x, device=None):
    if device:
        x = x.to(device)
    return x

def normalize_adj(mx):
    """Row-normalize matrix: symmetric normalized Laplacian"""
    rowsum = mx.sum(1)
    r_inv_sqrt = torch.pow(rowsum, -0.5).flatten()
    r_inv_sqrt[torch.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = torch.diag(r_inv_sqrt)
    return torch.mm(torch.mm(mx, r_mat_inv_sqrt).transpose(-1, -2), r_mat_inv_sqrt)

def add_graph_loss(out_adj, features, smooth_ratio, degree_ratio, sparsity_ratio, device):
    # Graph regularization
    graph_loss = 0
    L = torch.diagflat(torch.sum(out_adj, dim=-1)) - out_adj
    graph_loss += smooth_ratio * torch.trace(torch.mm(features.transpose(-1, -2), torch.mm(L, features))) / int(np.prod(out_adj.shape))
    ones_vec = to_cuda(torch.ones(out_adj.size(-1)), device)
    graph_loss += -degree_ratio * torch.mm(ones_vec.unsqueeze(0), torch.log(torch.mm(out_adj, ones_vec.unsqueeze(-1)) + VERY_SMALL_NUMBER)).squeeze() / out_adj.shape[-1]
    graph_loss += sparsity_ratio * torch.sum(torch.pow(out_adj, 2)) / int(np.prod(out_adj.shape))
    return graph_loss

def diff(X, Y, Z):
    assert X.shape == Y.shape
    diff_ = torch.sum(torch.pow(X - Y, 2))
    norm_ = torch.sum(torch.pow(Z, 2))
    diff_ = diff_ / torch.clamp(norm_, min=VERY_SMALL_NUMBER)
    return diff_