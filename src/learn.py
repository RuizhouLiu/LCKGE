import argparse
import os

import logging
import shutil
import torch
from torch import optim

torch.cuda.empty_cache()
from torch.optim.lr_scheduler import ReduceLROnPlateau
from datasets import Dataset, DATA_PATH
from models import ComplEx, LorentzKG, MMKGE_CLorentz_v1, LorentzKG2, LorentzKG3, MMLorentzKG, MMKGE_CL
from regularizers import F2, N3
from optimizers import KBCOptimizer, MMKBCOptimizer
from datetime import datetime

import json
import numpy as np
import time
import ast
from utils import avg_both, get_savedir, format_metrics


seed = 2022
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

big_datasets = ['WN9', 'FB15K-237', 'FB15K', 'FB', 'DB15K', 'YAGO15K', 'FB15K_new']
datasets = big_datasets

parser = argparse.ArgumentParser(
    description="Relational learning contraption"
)

parser.add_argument(
    '--dataset', choices=datasets,
    default='FB15K-237',
    help="Dataset in {}".format(datasets)
)

models = ['ComplEx', 'LorentzKG', 'MMKGE_CLorentz_v1', 'LorentzKG2', 'LorentzKG3', 'MMLorentzKG', 'MMKGE_CL']
parser.add_argument(
    '--model', choices=models,
    default='ComplExMDR',
    help="Model in {}".format(models)
)

parser.add_argument(
    '--alpha', default=1, type=float,
    help="Modality embedding ratio in modality_structure fusion. Default=1 means dscp/img emb does not fuse structure emb."
)

parser.add_argument(
    '--modality_split', default=True, type=ast.literal_eval,
    help="Whether split modalities."
)

parser.add_argument(
    '--fusion_img', default=True, type=ast.literal_eval,
    help="Whether fusion img modality graph."
)

parser.add_argument(
    '--fusion_dscp', default=True, type=ast.literal_eval,
    help="Whether fusion description modality graph."
)

parser.add_argument(
    '--scale', default=20, type=float,
    help="temp parameter"
)

regularizers = ['N3', 'F2']
parser.add_argument(
    '--regularizer', choices=regularizers, default='N3',
    help="Regularizer in {}".format(regularizers)
)

optimizers = ['Adagrad', 'Adam', 'SGD']
parser.add_argument(
    '--optimizer', choices=optimizers, default='Adagrad',
    help="Optimizer in {}".format(optimizers)
)

parser.add_argument(
    '--max_epochs', default=200, type=int,
    help="Number of epochs."
)
parser.add_argument(
    '--valid', default=3, type=float,
    help="Number of epochs before valid."
)
parser.add_argument(
    '--rank', default=2000, type=int,
    help="Factorization rank."
)
parser.add_argument(
    '--batch_size', default=1000, type=int,
    help="Factorization rank."
)
parser.add_argument(
    '--reg', default=0.01, type=float,
    help="Regularization weight"
)
parser.add_argument(
    '--init', default=1e-3, type=float,
    help="Initial scale"
)
parser.add_argument(
    '--learning_rate', default=1e-1, type=float,
    help="Learning rate"
)
parser.add_argument(
    '--decay1', default=0.9, type=float,
    help="decay rate for the first moment estimate in Adam"
)
parser.add_argument(
    '--decay2', default=0.999, type=float,
    help="decay rate for second moment estimate in Adam"
)
parser.add_argument(
    '--note', default=None,
    help="model setting or ablation note for ckpt save"
)
parser.add_argument(
    '--early_stopping', default=10, type=int,
    help="stop training until MRR stop increasing after early stopping epoches"
)
parser.add_argument(
    '--ckpt_dir', default='../ckpt/'
)
parser.add_argument(
    '--rand_ratio', default=1, type=float
)
parser.add_argument(
    '--img_info', default='../data/FB15K-237/img_vec.pickle'
)
parser.add_argument(
    '--dscp_info', default='../data/FB15K-237/dscp_vec.pickle'
)

args = parser.parse_args()

################################################

print("running setting args: ", args)

save_dir = get_savedir(args.model, args.dataset)
# file logger
logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
    filename=os.path.join(save_dir, "train.log")
)
device = torch.device('cuda:0')

console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s %(levelname)-8s %(message)s")
console.setFormatter(formatter)
logging.getLogger("").addHandler(console)
logging.info("Saving logs in: {}".format(save_dir))

with open(os.path.join(save_dir, "config.json"), "w") as fjson:
    json.dump(vars(args), fjson, indent=4)

# copy source codes to log dirs
folder_path = os.path.split(os.path.abspath(__file__))[0]
shutil.copytree(os.path.join(folder_path), os.path.join(save_dir, os.path.split(folder_path)[-1]))

# load dataset
dataset = Dataset(args.dataset, device=device)
examples = torch.from_numpy(dataset.get_train().astype('int64'))

logging.info(dataset.get_shape())

img_info  = os.path.join(DATA_PATH, args.dataset, "vis_embeddings.pkl") if args.fusion_img  else None
dscp_info = os.path.join(DATA_PATH, args.dataset, "txt_embeddings.pkl") if args.fusion_dscp else None

model = {
    'ComplEx': lambda: ComplEx(dataset.get_shape(), args, device=device),
    'LorentzKG': lambda: LorentzKG(dataset.get_shape(), args=args, device=device),
    'MMKGE_CLorentz_v1': lambda: MMKGE_CLorentz_v1(dataset.get_shape(), rank=args.rank, args=args,
                                                   visu_path=img_info, ling_path=dscp_info, device=device),
    'LorentzKG2': lambda: LorentzKG2(dataset.get_shape(), args=args, device=device),
    'LorentzKG3': lambda: LorentzKG3(dataset.get_shape(), args=args, device=device),
    'MMLorentzKG': lambda: MMLorentzKG(dataset.get_shape(), rank=args.rank, args=args,
                                       visu_path=img_info, ling_path=dscp_info, device=device),
    'MMKGE_CL': lambda: MMKGE_CL(dataset.get_shape(), rank=args.rank, args=args,
                                 visu_path=img_info, ling_path=dscp_info, device=device)
}[args.model]()

regularizer = {
    'F2': F2(args.reg),
    'N3': N3(args.reg),
}[args.regularizer]


model.to(device)

optim_method = {
    'Adagrad': lambda: optim.Adagrad(model.parameters(), lr=args.learning_rate),
    'Adam': lambda: optim.Adam(model.parameters(), lr=args.learning_rate, betas=(args.decay1, args.decay2)),
    'SGD': lambda: optim.SGD(model.parameters(), lr=args.learning_rate)
}[args.optimizer]()

optimizer = KBCOptimizer(model, regularizer, optim_method, args.batch_size, 
                         modality_split=args.modality_split, 
                         fusion_dscp=args.fusion_dscp,
                         fusion_img=args.fusion_img)
scheduler = ReduceLROnPlateau(optim_method, 'min', factor=0.5, verbose=True, patience=10, threshold=1e-3)



cur_loss = 0
best_epoch = None
best_mrr = None

since = time.time()
for e in range(args.max_epochs):
    optimizer.model.train()
    cur_loss = optimizer.epoch(examples).tolist()
    logging.info(f"\t Epoch {e} | average train loss: {cur_loss:.4f}")

    if (e + 1) % args.valid == 0:
        optimizer.model.eval()
        valid, test, train = [
            avg_both(*dataset.eval(model, split, -1 if split != 'train' else 5000))
            for split in ['valid', 'test', 'train']
        ]

        logging.info('\tTrain: ' + format_metrics(train, split="train"))
        logging.info('\tValid: ' + format_metrics(valid, split="valid"))
        logging.info('\tTest: '  + format_metrics(test, split="test"))

        valid_mrr = test["MRR"]
        if not best_mrr or valid_mrr > best_mrr:
            best_mrr = valid_mrr
            counter = 0
            best_epoch = e
            logging.info(f"\tSaving model at epoch {e} in {save_dir}")
            torch.save(model.cpu().state_dict(), os.path.join(save_dir, "model.pt"))
            model.to(device)
        else:
            counter += 1
            if counter == args.early_stopping:
                logging.info("\t Early stopping")
                break
            elif counter == args.early_stopping // 2:
                pass

        scheduler.step(valid['MRR'])
        print("Learning rate at epoch {}: {}".format(e + 1, scheduler._last_lr))

logging.info("\t Optimization finished")

time_elapsed = time.time() - since
sec_per_epoch = time_elapsed / float(e + 1)
logging.info('\tTime consuming: {:.3f}s, average sec per epoch: {:.3f}s'.format(time_elapsed, sec_per_epoch))
logging.info(f'\tlast_lr: {scheduler._last_lr[0]:.5f}')

if not best_mrr:
    torch.save(model.cpu().state_dict(), os.path.join(save_dir, "model.pt"))
else:
    logging.info("\t Loading best model saved at epoch {}".format(best_epoch))
    model.load_state_dict(torch.load(os.path.join(save_dir, "model.pt")))
model.to(device)
model.eval()

# Test metrics
test = avg_both(*dataset.eval(model, 'test', 50000))
logging.info(format_metrics(test, split="test"))



