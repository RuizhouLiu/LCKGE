from typing import Dict
import datetime
import torch
import os
import pickle
import numpy as np


def avg_both(mrs: Dict[str, float], mrrs: Dict[str, float], hits: Dict[str, torch.FloatTensor]):
    """
    aggregate metrics for missing lhs and rhs
    :param mrrs: d
    :param hits:
    :return:
    """
    mr = (mrs['lhs'] + mrs['rhs']) / 2
    mrr = (mrrs['lhs'] + mrrs['rhs']) / 2.
    h = (hits['lhs'] + hits['rhs']) / 2.
    return {'MR': mr, 'MRR': mrr, 'hits@[1,3,10]': h.tolist()}


def read_tab_file(tabfile):
    ents = []
    others = []
    for line in open(tabfile, 'r'):
        ent, other = line.strip().split('\t')
        ents.append(ent)
        others.append(other)
    return ents, others


def get_savedir(model, dataset):
    """Get unique saving directory name."""
    dt = datetime.datetime.now()
    date = dt.strftime("%m_%d")
    save_dir = os.path.join(
        "/data/liuruizhou/MMKGE_model/MoSE_MCL/logs", date, dataset,
        model + dt.strftime('_%H_%M_%S')
    )
    os.makedirs(save_dir, exist_ok=True)
    return save_dir

def format_metrics(metrics, split):
    """Format metrics for logging."""
    result = ""
    result += "MRR: {:.3f} | ".format(metrics['MRR'])
    result += "H@1: {:.3f} | ".format(metrics['hits@[1,3,10]'][0])
    result += "H@3: {:.3f} | ".format(metrics['hits@[1,3,10]'][1])
    result += "H@10: {:.3f}".format(metrics['hits@[1,3,10]'][2])
    return result

def load_mmdata(path, requires_grad=True):
    data = pickle.load(open(path, 'rb'))
    if isinstance(data, np.ndarray):
        mmdata = torch.from_numpy(data)
    elif isinstance(data, list):
        if isinstance(data[0], np.ndarray):
            mmdata = np.concatenate(data, dim=0)
        elif isinstance(data[0], torch.Tensor):
            mmdata = torch.cat(data, dim=0)
    elif isinstance(data, torch.Tensor):
        mmdata = data
        
    mmdata = mmdata / torch.norm(mmdata, dim=-1, keepdim=True)
    return torch.nn.Parameter(torch.Tensor(mmdata.float()), requires_grad=requires_grad)
    