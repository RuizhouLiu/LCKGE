import pickle
from abc import ABC, abstractmethod
from typing import Tuple, List, Dict

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm
  
from utils import load_mmdata
from mm_fusion.fusion import FusionModule_v6, GraphFusion, CLFusion, CLFusion2, LoretzFusion, LoretzFusion2
from manifolds.lorentz import Lorentz



def minmax_norm(x):
    x_min, x_max = x.min(), x.max()
    return (x - x_min) / (x_max - x_min)


class KBCModel(nn.Module, ABC):
    @abstractmethod
    def get_rhs(self, chunk_begin: int, chunk_size: int):
        pass

    @abstractmethod
    def get_queries(self, queries: torch.Tensor):
        pass

    @abstractmethod
    def score(self, x: torch.Tensor):
        pass

    


class MMKGE_CLorentz_v1(KBCModel):
    def __init__(self, sizes: Tuple[int, int, int], rank: int, args,
                init_size: float = 1e-3,
                visu_path: str = None, ling_path: str = None,
                device: torch.DeviceObjType = None,) -> None:
        super(MMKGE_CLorentz_v1, self).__init__()

        self.args = args

        self.sizes = sizes
        self.rank = rank
        self.device = device
        self.init_size = init_size
        self.manifold = Lorentz(max_norm=5.0)
        

        if args.optimizer in ["adgrad", "Adagrad"]:
            self.ent_embeds = nn.Embedding(sizes[0], 5 * rank, sparse=True)
            # self.rel_embeds = nn.Embedding(sizes[1], 5 * rank, sparse=True)
            self.rel_rotate = nn.Embedding(sizes[1], 4 * rank, sparse=True)
            self.rel_boosts = nn.Embedding(sizes[1], 4 * rank, sparse=True)
            
        elif args.optimizer in ["adam", "Adam"]:
            self.ent_embeds = nn.Embedding(sizes[0], 2 * rank, sparse=False)
            # self.rel_embeds = nn.Embedding(sizes[1], 2 * rank, sparse=False)
            self.rel_rotate = nn.Embedding(sizes[1], 4 * rank, sparse=False)
            self.rel_boosts = nn.Embedding(sizes[1], 4 * rank, sparse=True)


        if visu_path is not None:
            self.visu_embeds = load_mmdata(visu_path, requires_grad=False)
            self.visu_num, self.visu_dim = self.visu_embeds.shape
            if args.optimizer in ["adgrad", "Adagrad"]:
                # self.visu_rhs_embeds = nn.Embedding(sizes[0], 2 * rank, sparse=True)
                self.visu_rel_rotate = nn.Embedding(sizes[1], 4 * rank, sparse=True)
                self.visu_rel_boosts = nn.Embedding(sizes[1], 4 * rank, sparse=True)
            elif args.optimizer in ["adam", "Adam"]:
                # self.visu_rhs_embeds = nn.Embedding(sizes[0], 2 * rank, sparse=False)
                self.visu_rel_rotate = nn.Embedding(sizes[1], 4 * rank, sparse=False)
                self.visu_rel_boosts = nn.Embedding(sizes[1], 4 * rank, sparse=False)
            
        
        if ling_path is not None:
            self.ling_embeds = load_mmdata(ling_path, requires_grad=False)
            self.ling_num, self.ling_dim = self.ling_embeds.shape
            if args.optimizer in ["adgrad", "Adagrad"]:
                # self.ling_rhs_embeds = nn.Embedding(sizes[0], 2 * rank, sparse=True)
                self.ling_rel_rotate = nn.Embedding(sizes[1], 4 * rank, sparse=True)
                self.ling_rel_boosts = nn.Embedding(sizes[1], 4 * rank, sparse=True)
            elif args.optimizer in ["adam", "Adam"]:
                # self.ling_rhs_embeds = nn.Embedding(sizes[0], 2 * rank, sparse=False)
                self.ling_rel_rotate = nn.Embedding(sizes[1], 4 * rank, sparse=False)
                self.ling_rel_boosts = nn.Embedding(sizes[1], 4 * rank, sparse=False)
        
        self.visu_fusion = CLFusion(self.visu_embeds.shape, self.ent_embeds.weight.shape)
        self.ling_fusion = CLFusion(self.ling_embeds.shape, self.ent_embeds.weight.shape)

        self.param_init()

    def param_init(self):
        for name, param in self.named_parameters():
            if str.endswith(name, 'weight') and 'embeds' in name:
                param.data *= self.init_size
            elif str.endswith(name, 'weight') and 'linear' in name:
                nn.init.kaiming_normal_(param)
            elif str.endswith(name, 'bias') and 'linear' in name:
                nn.init.zeros_(param)
            elif str.endswith(name, 'weight') and 'fusion' in name:
                if param.ndim == 2:
                    nn.init.kaiming_normal_(param)
                else:
                    nn.init.normal_(param)
            elif str.endswith(name, 'bias') and 'fusion' in name:
                nn.init.zeros_(param)
            elif name in ['visu_embeds', 'ling_embeds']:
                param.data *= self.init_size
    
    def forward(self, x):

        visu_feats, visu_cl_loss = self.visu_fusion(self.ent_embeds.weight.clone().detach(), self.visu_embeds)
        ling_feats, ling_cl_loss = self.ling_fusion(self.ent_embeds.weight.clone().detach(), self.ling_embeds)
        cl_loss = visu_cl_loss + ling_cl_loss

        stru_feats, visu_feats, ling_feats = self.ent_embeds.weight.view(-1, self.rank, 5), \
                                                         visu_feats.view(-1, self.rank, 5), \
                                                         ling_feats.view(-1, self.rank, 5)

        stru_feats, visu_feats, ling_feats = self.manifold.expmap0(stru_feats), \
                                             self.manifold.expmap0(visu_feats), \
                                             self.manifold.expmap0(ling_feats)

        
        slhs, srel, srhs = self.ent_embeds(x[:, 0]),   self.rel_embeds(x[:, 1]),     self.ent_embeds(x[:, 2])
        vlhs, vrel, vrhs = visu_feats[x[:, 0]], self.visu_rhs_embeds.weight[x[:, 1]], visu_feats[x[:, 2]]
        llhs, lrel, lrhs = ling_feats[x[:, 0]], self.ling_rhs_embeds.weight[x[:, 1]], ling_feats[x[:, 2]]

        # # struct embeddings
        # slhs = slhs[:, :self.rank], slhs[:, self.rank:]
        # srel = srel[:, :self.rank], srel[:, self.rank:]
        # srhs = srhs[:, :self.rank], srhs[:, self.rank:]
        # # visual embeddings
        # vlhs = vlhs[:, :self.rank], vlhs[:, self.rank:]
        # vrel = vrel[:, :self.rank], vrel[:, self.rank:]
        # vrhs = vrhs[:, :self.rank], vrhs[:, self.rank:]
        # # linguistic embeddings
        # llhs = llhs[:, :self.rank], llhs[:, self.rank:]
        # lrel = lrel[:, :self.rank], lrel[:, self.rank:]
        # lrhs = lrhs[:, :self.rank], lrhs[:, self.rank:]
        # # to_score
        # to_score_s = self.ent_embeds.weight[:, :self.rank], self.ent_embeds.weight[:, self.rank:]
        # to_score_v = visu_feats[:, :self.rank], visu_feats[:, self.rank:]
        # to_score_l = ling_feats[:, :self.rank], ling_feats[:, self.rank:]


        # score_stru = (
        #         (slhs[0] * srel[0] - slhs[1] * srel[1]) @ to_score_s[0].transpose(0, 1) +
        #         (slhs[0] * srel[1] + slhs[1] * srel[0]) @ to_score_s[1].transpose(0, 1)
        # )
        # factors_stru = (
        #     torch.sqrt(slhs[0] ** 2 + slhs[1] ** 2),
        #     torch.sqrt(srel[0] ** 2 + srel[1] ** 2),
        #     torch.sqrt(srhs[0] ** 2 + srhs[1] ** 2)
        # )

        # score_visu = (
        #         (vlhs[0] * vrel[0] - vlhs[1] * vrel[1]) @ to_score_v[0].transpose(0, 1) +
        #         (vlhs[0] * vrel[1] + vlhs[1] * vrel[0]) @ to_score_v[1].transpose(0, 1)
        # )
        # factors_visu = (
        #     torch.sqrt(vlhs[0] ** 2 + vlhs[1] ** 2),
        #     torch.sqrt(vrel[0] ** 2 + vrel[1] ** 2),
        #     torch.sqrt(vrhs[0] ** 2 + vrhs[1] ** 2)
        # )
        # score_ling = (
        #         (llhs[0] * lrel[0] - llhs[1] * lrel[1]) @ to_score_l[0].transpose(0, 1) +
        #         (llhs[0] * lrel[1] + llhs[1] * lrel[0]) @ to_score_l[1].transpose(0, 1)
        # )
        # factors_ling = (
        #     torch.sqrt(llhs[0] ** 2 + llhs[1] ** 2),
        #     torch.sqrt(lrel[0] ** 2 + lrel[1] ** 2),
        #     torch.sqrt(lrhs[0] ** 2 + lrhs[1] ** 2)
        # )

        # return score_stru, factors_stru, score_visu, factors_visu, score_ling, factors_ling, cl_loss

    
    def get_ranking(
            self, queries: torch.Tensor,
            filters: Dict[Tuple[int, int], List[int]],
            batch_size: int = 1000, chunk_size: int = -1
    ):
        ranks = torch.ones(len(queries))
        ranks_stru = torch.ones(len(queries))
        ranks_visu = torch.ones(len(queries))
        ranks_ling = torch.ones(len(queries))

        with tqdm(total=queries.shape[0], unit='ex') as bar:
            bar.set_description(f'Evaluation')
            with torch.no_grad():
                b_begin = 0
                while b_begin < len(queries):
                    these_queries = queries[b_begin:b_begin + batch_size].cpu().to(self.device)
                    target_idxs = these_queries[:, 2].cpu().tolist()
                    score_stru, _, score_visu, _, score_ling, _, _ = self.forward(these_queries)
                    targets_stru = torch.stack([score_stru[row, col] for row, col in enumerate(target_idxs)]).unsqueeze(-1)
                    targets_visu = torch.stack([score_visu[row, col] for row, col in enumerate(target_idxs)]).unsqueeze(-1)
                    targets_ling = torch.stack([score_ling[row, col] for row, col in enumerate(target_idxs)]).unsqueeze(-1)

                    for i, query in enumerate(these_queries):
                        filter_out = filters[(query[0].item(), query[1].item())]
                        filter_out += [queries[b_begin + i, 2].item()]  
                        score_stru[i, torch.LongTensor(filter_out)] = -1e6
                        score_visu[i, torch.LongTensor(filter_out)] = -1e6
                        score_ling[i, torch.LongTensor(filter_out)] = -1e6

                    ranks_stru[b_begin:b_begin + batch_size] += torch.sum(
                        (score_stru >= targets_stru).float(), dim=1
                    ).cpu()
                    ranks_visu[b_begin:b_begin + batch_size] += torch.sum(
                        (score_visu >= targets_visu).float(), dim=1
                    ).cpu()
                    ranks_ling[b_begin:b_begin + batch_size] += torch.sum(
                        (score_ling >= targets_ling).float(), dim=1
                    ).cpu()

                    b_begin += batch_size
                    bar.update(batch_size)
        
        ranks_fusion = torch.min(ranks_stru, torch.min(ranks_visu, ranks_ling))
        print("ranks_str: {:.4f}, ranks_img: {:.4f}, ranks_dscp: {:.4f}".format(
                sum(ranks_fusion == ranks_stru) / ranks.shape[0],
                sum(ranks_fusion == ranks_visu) / ranks.shape[0],
                sum(ranks_fusion == ranks_ling) / ranks.shape[0]))

        return ranks_fusion
    
    def get_queries(self, queries: torch.Tensor):
        pass

    def get_rhs(self, chunk_begin: int, chunk_size: int):
        pass

    def score(self, x: torch.Tensor):
        pass


class LorentzKG(KBCModel):
    def __init__(self, sizes: Tuple[int, int, int], args, device) -> None:
        super().__init__()

        self.sizes = sizes
        self.rank = args.rank
        self.init_size = args.init
        self.device = device
        self.manifold = Lorentz(max_norm=2.0)
        self.reg_noise_ratio = 0
        self.margin = 1.08

        self.ent_embedding = nn.Parameter(torch.Tensor(torch.empty((sizes[0], self.rank, 5))), requires_grad=True)
        self.rel_rotate = nn.Parameter(torch.Tensor(torch.empty((sizes[1], self.rank, 4))), requires_grad=True)
        self.rel_boosts = nn.Parameter(torch.Tensor(torch.empty((sizes[1], self.rank, 4))), requires_grad=True)

        self.rel_rotate2 = nn.Parameter(torch.Tensor(torch.empty((sizes[1], self.rank, 4))), requires_grad=True)
        self.rel_boosts2 = nn.Parameter(torch.Tensor(torch.empty((sizes[1], self.rank, 4))), requires_grad=True)

        self.a = nn.Parameter(torch.zeros(sizes[0], 1), requires_grad=True)
        self.b = nn.Parameter(torch.zeros(sizes[0], 1), requires_grad=True)
        # self.rel_rotate = nn.Embedding(sizes[1], 4 * self.rank)
        # self.rel_boosts = nn.Embedding(sizes[1], 4 * self.rank)
    
        # self.ent_embedding.weight.data *= args.init
        # self.rel_rotate.weight.data *= args.init
        # self.rel_boosts.weight.data *= args.init
        self.eye = torch.eye(4).to(device)

        self.param_init()
    
    def param_init(self):
        ent_data= torch.randn(self.ent_embedding.data.shape, device=self.device) * self.init_size
        ent_data = ent_data / ent_data.norm(dim=-1, keepdim=True)
        self.ent_embedding.data = ent_data

        nn.init.kaiming_normal_(self.rel_rotate.data)
        self.rel_rotate.data *= self.init_size
        nn.init.kaiming_normal_(self.rel_boosts.data)
        self.rel_boosts.data *= self.init_size

        nn.init.kaiming_normal_(self.rel_rotate2.data)
        self.rel_rotate2.data *= self.init_size
        nn.init.kaiming_normal_(self.rel_boosts2.data)
        self.rel_boosts2.data *= self.init_size
        

    def forward(self, queries):
        # lhs = self.ent_embedding(queries[:, 0])
        # rhs = self.ent_embedding(queries[:, 2])
        # rel = self.rel_embedding(queries[:, 1])
        
        # lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        # rhs = rhs[:, :self.rank], rhs[:, self.rank:]
        # rel = rel[:, :self.rank], rel[:, self.rank:]

        # to_score = self.ent_embedding.weight[:, :self.rank], self.ent_embedding.weight[:, self.rank:]

        # return (
        #             (lhs[0] * rel[0] - lhs[1] * rel[1]) @ to_score[0].transpose(0, 1) + \
        #             (lhs[0] * rel[1] + lhs[1] * rel[0]) @ to_score[1].transpose(0, 1)
        #     ), (
        #             torch.sqrt(lhs[0] ** 2 + lhs[1] ** 2),
        #             torch.sqrt(rhs[0] ** 2 + rhs[1] ** 2),
        #             torch.sqrt(rel[0] ** 2 + rel[1] ** 2)
        #     ), torch.tensor(0.0)

        ent_embedding = self.manifold.expmap0(self.ent_embedding, dim=-1)
        ent_embedding = self.ent_embedding
        lhs, rhs = ent_embedding[queries[:, 0]], ent_embedding[queries[:, 0]]
        rel_rotate, rel_boost = self.rel_rotate[queries[:, 1]], self.rel_boosts[queries[:, 1]]
        rel_rotate2, rel_boost2 = self.rel_rotate2[queries[:, 1]], self.rel_boosts2[queries[:, 1]]
        
        lhs = self.lorentz_rotate(lhs, rel_rotate)
        lhs = self.lorentz_boost2(lhs, rel_boost)

        # lhs = self.lorentz_rotate(lhs, rel_rotate2)
        # lhs = self.lorentz_boost(lhs, rel_boost2)

        scores = -2.0 - 2.0 * self.cinner(lhs, ent_embedding)

        # scores = -torch.matmul(lhs[:,:,1:].clone().view(lhs.shape[0], -1), ent_embedding[:,:,1:].clone().view(self.sizes[0], -1).T)
        # scores = -torch.einsum('idj,kdj->ikd', lhs, ent_embedding).mean(-1)


        if self.training:
            reg_noise = self.reg_noise_ratio * torch.randn((scores.shape[0], 1), device=self.device, requires_grad=False)
            # scores = self.margin - scores + self.a[queries[:, 0]] + self.b[queries[:, 2]] + reg_noise
            scores = -scores
        else:
            reg_noise = self.reg_noise_ratio * torch.zeros((scores.shape[0], 1), device=self.device, requires_grad=False)
            scores =  -scores
        
        

        reg_lhs, reg_rhs = torch.sqrt((lhs ** 2).sum(-1)), torch.sqrt((rhs ** 2).sum(-1))
        reg_rel = torch.sqrt((rel_rotate ** 2).sum(-1)) + torch.sqrt((rel_boost ** 2).sum(-1)) 
       
        # return scores, (reg_lhs, reg_rel, reg_rhs)
        return scores, (reg_rel)



    def lorentz_rotate(self, lhs, rot, mode='lhs'):
        if mode == 'lhs':
            rot = torch.nn.functional.gelu(rot)
            lhs_t, lhs_r, lhs_i, lhs_j, lhs_k = torch.chunk(lhs, 5, dim=-1)
            rot_r, rot_i, rot_j, rot_k = torch.chunk(rot, 4, dim=-1)
            # dominator_q = torch.sqrt(rot_r ** 2 + rot_i ** 2 + rot_j ** 2 + rot_k ** 2)
            # rot_r = rot_r / dominator_q
            # rot_i = rot_i / dominator_q
            # rot_j = rot_j / dominator_q
            # rot_k = rot_k / dominator_q

            A = lhs_r * rot_r - lhs_i * rot_i - lhs_j * rot_j - lhs_k * rot_k
            B = lhs_r * rot_i + rot_r * lhs_i + lhs_j * rot_k - rot_j * lhs_k
            C = lhs_r * rot_j + rot_r * lhs_j + lhs_k * rot_i - rot_k * lhs_i
            D = lhs_r * rot_k + rot_r * lhs_k + lhs_i * rot_j - rot_i * lhs_j

            return torch.cat([lhs_t, A, B, C, D], dim=-1)
        else:
            
            rot = torch.nn.functional.gelu(rot).unsqueeze(1)
            lhs_t, lhs_r, lhs_i, lhs_j, lhs_k = torch.chunk(lhs, 5, dim=-1)
            rot_r, rot_i, rot_j, rot_k = torch.chunk(rot, 4, dim=-1)
            dominator_q = torch.sqrt(rot_r ** 2 + rot_i ** 2 + rot_j ** 2 + rot_k ** 2)
            rot_r = rot_r / dominator_q
            rot_i = rot_i / dominator_q
            rot_j = rot_j / dominator_q
            rot_k = rot_k / dominator_q

            A = lhs_r * rot_r - lhs_i * rot_i - lhs_j * rot_j - lhs_k * rot_k
            B = lhs_r * rot_i + rot_r * lhs_i + lhs_j * rot_k - rot_j * lhs_k
            C = lhs_r * rot_j + rot_r * lhs_j + lhs_k * rot_i - rot_k * lhs_i
            D = lhs_r * rot_k + rot_r * lhs_k + lhs_i * rot_j - rot_i * lhs_j

            return torch.cat([lhs_t, A, B, C, D], dim=-1)
    
    def lorentz_boost(self, lhs, boost, mode='lhs'):
        if mode == 'lhs':
            boost = torch.tanh(boost)
            boost = boost / np.power(4, 1)
            lhs_t, lhs_s = lhs.narrow(-1, 0, 1), lhs.narrow(-1, 1, lhs.shape[-1] - 1)

            boost2 = torch.sum(boost * boost, dim=-1, keepdim=True)
            boost2boost = torch.einsum('bdi,bdj->bdij', boost, boost)
            zeta = 1 / (torch.sqrt(1 - boost2) + 1e-8)


            x_t = zeta * lhs_t - zeta * torch.sum(boost * lhs_s, dim=-1, keepdim=True)
            x_s = -1 * zeta * lhs_t * boost + lhs_s + ((zeta - 1) / (boost2 + 1e-9)) * torch.einsum('bdij, bdj->bdi', boost2boost, lhs_s)
            
            return torch.cat([x_t, x_s], dim=-1)
        else:
            
            boost = torch.tanh(boost).unsqueeze(1)
            boost = boost / np.power(4, 1)
            lhs_t, lhs_s = lhs.narrow(-1, 0, 1), lhs.narrow(-1, 1, lhs.shape[-1] - 1)

            boost2 = torch.sum(boost * boost, dim=-1, keepdim=True)
            boost2boost = torch.einsum('nbdi,nbdj->nbdij', boost, boost)
            zeta = 1 / (torch.sqrt(1 - boost2) + 1e-8)


            x_t = zeta * lhs_t - zeta * torch.sum(boost * lhs_s, dim=-1, keepdim=True)
            x_s = -1 * zeta * lhs_t * boost + lhs_s + ((zeta - 1) / (boost2 + 1e-9)) * torch.einsum('bndij, bndj->bndi', boost2boost.expand(-1, self.sizes[0], -1, -1, -1), lhs_s)
            
            return torch.cat([x_t, x_s], dim=-1)
        
    def lorentz_boost2(self, lhs, boost):
        boost = torch.sigmoid(boost)
        boost = boost / np.power(4, 1)
        lhs_t, lhs_s = lhs.narrow(-1, 0, 1), lhs.narrow(-1, 1, lhs.shape[-1] - 1)

        eye = self.eye.view(1, 1, 4, 4).expand((lhs.shape[0], self.rank, -1, -1))

        boost2 = torch.matmul(boost.unsqueeze(-2), boost.unsqueeze(-1)).squeeze(-1)
        boost2boost = torch.sqrt(eye + torch.matmul(boost.unsqueeze(-1), boost.unsqueeze(-2)))

        xt = lhs_t * boost2 + torch.matmul(boost.unsqueeze(-2), lhs_s.unsqueeze(-1)).squeeze(-1)
        xs = lhs_t * boost + torch.matmul(boost2boost, lhs_s.unsqueeze(-1)).squeeze(-1)

        return torch.cat([xt, xs], dim=-1)
    
    def cinner(self, x: torch.Tensor, y: torch.Tensor, mode='lhs'):
        """
        x: [batch, dim, 5]
        y: [#ent,  dim, 5]
        """
        x = x.clone()
        x.narrow(-1, 0, 1).mul_(-1)
        if mode == 'lhs':
            return torch.matmul(x.permute((1,0,2)), y.permute((1,2,0))).mean(0)
        else:
            return (x * y).sum(-1).mean(-1)
    
    def cinner2(self, x, y):
        pass
            


    def get_ranking(
            self, queries: torch.Tensor,
            filters: Dict[Tuple[int, int], List[int]],
            batch_size: int = 1000, chunk_size: int = -1
    ):
        ranks = torch.ones(len(queries))
        # ranks_stru = torch.ones(len(queries))
        # ranks_visu = torch.ones(len(queries))
        # ranks_ling = torch.ones(len(queries))

        with tqdm(total=queries.shape[0], unit='ex') as bar:
            bar.set_description(f'Evaluation')
            with torch.no_grad():
                b_begin = 0
                while b_begin < len(queries):
                    these_queries = queries[b_begin:b_begin + batch_size].cpu().to(self.device)
                    target_idxs = these_queries[:, 2].cpu().tolist()
                    score, _ = self.forward(these_queries)
                    targets = torch.stack([score[row, col] for row, col in enumerate(target_idxs)]).unsqueeze(-1)
                    # targets_visu = torch.stack([score_visu[row, col] for row, col in enumerate(target_idxs)]).unsqueeze(-1)
                    # targets_ling = torch.stack([score_ling[row, col] for row, col in enumerate(target_idxs)]).unsqueeze(-1)

                    for i, query in enumerate(these_queries):
                        filter_out = filters[(query[0].item(), query[1].item())]
                        filter_out += [queries[b_begin + i, 2].item()]  
                        score[i, torch.LongTensor(filter_out)] = -1e6
                        # score_visu[i, torch.LongTensor(filter_out)] = -1e6
                        # score_ling[i, torch.LongTensor(filter_out)] = -1e6

                    ranks[b_begin:b_begin + batch_size] += torch.sum(
                        (score >= targets).float(), dim=1
                    ).cpu()
                    # ranks_visu[b_begin:b_begin + batch_size] += torch.sum(
                    #     (score_visu >= targets_visu).float(), dim=1
                    # ).cpu()
                    # ranks_ling[b_begin:b_begin + batch_size] += torch.sum(
                    #     (score_ling >= targets_ling).float(), dim=1
                    # ).cpu()

                    b_begin += batch_size
                    bar.update(batch_size)
        
        # ranks_fusion = torch.min(ranks_stru, torch.min(ranks_visu, ranks_ling))
        # print("ranks_str: {:.4f}, ranks_img: {:.4f}, ranks_dscp: {:.4f}".format(
        #         sum(ranks_fusion == ranks_stru) / ranks.shape[0],
        #         sum(ranks_fusion == ranks_visu) / ranks.shape[0],
        #         sum(ranks_fusion == ranks_ling) / ranks.shape[0]))

        return ranks
    
    def get_queries(self, queries: torch.Tensor):
        pass

    def get_rhs(self, chunk_begin: int, chunk_size: int):
        pass

    def score(self, x: torch.Tensor):
        pass


class LorentzKG2(KBCModel):
    def __init__(self, sizes: Tuple[int, int, int], args, device) -> None:
        super().__init__()

        self.sizes = sizes
        self.rank = args.rank
        self.init_size = args.init
        self.device = device
        self.manifold = Lorentz(max_norm=2.0)
        self.reg_noise_ratio = 0.05

        self.ent_embeddings = nn.Parameter(torch.Tensor(torch.empty(sizes[0], self.rank)), requires_grad=True)
        self.rel_rotate = nn.Parameter(torch.Tensor(torch.empty(sizes[1], self.rank-1, self.rank-1)), requires_grad=True)   
        self.rel_boosts = nn.Parameter(torch.Tensor(torch.empty(sizes[1], self.rank-1)), requires_grad=True)

        self.rel_rotate2 = nn.Parameter(torch.Tensor(torch.empty(sizes[1], self.rank-1, self.rank-1)), requires_grad=True)
        self.rel_boosts2 = nn.Parameter(torch.Tensor(torch.empty(sizes[1], self.rank-1)), requires_grad=True)

        self.register_buffer('I3', torch.eye(self.rank - 1,).view(1, 1, self.rank - 1, self.rank - 1).expand(
            [sizes[1], self.rank - 1, -1, -1]))
        self.register_buffer('Iw', torch.eye(self.rank - 1,).view(1, self.rank - 1, self.rank - 1).expand(
            [sizes[1], -1, -1]))
        
        self.param_init()


    def param_init(self):
        ent_data= torch.randn(self.ent_embeddings.data.shape, device=self.device) * self.init_size
        ent_data = ent_data / ent_data.norm(dim=-1, keepdim=True)
        self.ent_embeddings.data = ent_data

        nn.init.kaiming_normal_(self.rel_rotate.data)
        self.rel_rotate.data *= self.init_size
        nn.init.kaiming_normal_(self.rel_boosts.data)
        self.rel_boosts.data *= self.init_size

        nn.init.kaiming_normal_(self.rel_rotate2.data)
        self.rel_rotate2.data *= self.init_size
        nn.init.kaiming_normal_(self.rel_boosts2.data)
        self.rel_boosts2.data *= self.init_size

    def forward(self, queries):
        ent_embedding = self.manifold.expmap0(self.ent_embeddings, dim=-1)
        lhs, rhs = ent_embedding[queries[:, 0]], ent_embedding[queries[:, 2]]
        rel_rotate, rel_boosts = self.rel_rotate[queries[:, 1]], self.rel_boosts[queries[:, 1]]

        lhs = self.lorentz_rotate(lhs, rel_rotate, queries[:, 1])
        lhs = self.lorentz_boosts(lhs, rel_boosts)

        # lhs = self.lorentz_rotate(lhs, queries[:, 1], rel_id=1)
        # lhs = self.lorentz_boosts(lhs, queries[:, 1], rel_id=1)

        scroes = 2.0 - 2.0 * self.manifold.cinner(lhs, ent_embedding)

        if self.training:
            reg_noise = self.reg_noise_ratio * torch.randn((scroes.shape[0], 1), device=self.device, requires_grad=False)
        else:
            reg_noise = self.reg_noise_ratio * torch.zeros((scroes.shape[0], 1), device=self.device, requires_grad=False)
        
        scroes += reg_noise

        reg_rel = torch.sqrt((self.rel_rotate.view(self.sizes[1], -1) ** 2).sum(-1)) + torch.sqrt((self.rel_boosts ** 2).sum(-1, keepdim=True)) 
        
        return scroes, reg_rel

    
    def lorentz_rotate(self, lhs, rot, rel_idx):

        rot = torch.nn.functional.gelu(rot)
        lhs_t, lhs_s = lhs.narrow(-1,0,1), lhs.narrow(-1,1,lhs.shape[-1]-1)
        rot = self.othogonal_matrix(rot, self.I3[rel_idx], self.Iw[rel_idx])
        lhs_s = torch.einsum('bi,bij->bj',lhs_s, rot)
        return torch.cat([lhs_t, lhs_s], dim=-1)
        
    

    def lorentz_boosts(self, lhs, boosts):

        boosts = torch.tanh(boosts)
        lhs_t, lhs_s = lhs.narrow(-1,0,1), lhs.narrow(-1,1,lhs.shape[-1]-1)
        boosts = boosts / np.power(self.rank, 1)

        boost2 = torch.sum(boosts * boosts, dim=-1, keepdim=True)
        boost2boost = torch.einsum('bi,bj->bij', boosts, boosts)
        zeta = 1 / (torch.sqrt(1 - boost2) + 1e-8)

        x_0 = zeta * lhs_t - zeta * torch.sum(boosts * lhs_s, dim=-1, keepdim=True)
        x_r = -1 * zeta * lhs_t * boosts + lhs_s + \
            ((zeta - 1) / (boost2 + 1e-9)) * torch.einsum('bij, bj -> bi', boost2boost, lhs_s)
        
        return torch.cat([x_0, x_r], dim=-1)

    
    def othogonal_matrix(self, vv, I3, Iw):  # vv tensor of [#batch, dim-1, dim-1]
        bvv = torch.einsum('bwv, bwk -> bwvk', vv, vv)
        nbvv = torch.einsum('bwlv, bwvi -> bwli', vv.unsqueeze(-2), vv.unsqueeze(-1))
        qbvvt = (I3 - 2 * bvv / nbvv).permute([1, 0, 2, 3])
        for i in range(qbvvt.shape[0]):
            Iw = Iw @ qbvvt[i]
        return Iw  # [batch, dim-1, dim-1] othogonal matrix

    

    def get_ranking(
            self, queries: torch.Tensor,
            filters: Dict[Tuple[int, int], List[int]],
            batch_size: int = 1000, chunk_size: int = -1
    ):
        ranks = torch.ones(len(queries))
        # ranks_stru = torch.ones(len(queries))
        # ranks_visu = torch.ones(len(queries))
        # ranks_ling = torch.ones(len(queries))

        with tqdm(total=queries.shape[0], unit='ex') as bar:
            bar.set_description(f'Evaluation')
            with torch.no_grad():
                b_begin = 0
                while b_begin < len(queries):
                    these_queries = queries[b_begin:b_begin + batch_size].cpu().to(self.device)
                    target_idxs = these_queries[:, 2].cpu().tolist()
                    score, _ = self.forward(these_queries)
                    targets = torch.stack([score[row, col] for row, col in enumerate(target_idxs)]).unsqueeze(-1)
                    # targets_visu = torch.stack([score_visu[row, col] for row, col in enumerate(target_idxs)]).unsqueeze(-1)
                    # targets_ling = torch.stack([score_ling[row, col] for row, col in enumerate(target_idxs)]).unsqueeze(-1)

                    for i, query in enumerate(these_queries):
                        filter_out = filters[(query[0].item(), query[1].item())]
                        filter_out += [queries[b_begin + i, 2].item()]  
                        score[i, torch.LongTensor(filter_out)] = -1e6
                        # score_visu[i, torch.LongTensor(filter_out)] = -1e6
                        # score_ling[i, torch.LongTensor(filter_out)] = -1e6

                    ranks[b_begin:b_begin + batch_size] += torch.sum(
                        (score >= targets).float(), dim=1
                    ).cpu()
                    # ranks_visu[b_begin:b_begin + batch_size] += torch.sum(
                    #     (score_visu >= targets_visu).float(), dim=1
                    # ).cpu()
                    # ranks_ling[b_begin:b_begin + batch_size] += torch.sum(
                    #     (score_ling >= targets_ling).float(), dim=1
                    # ).cpu()

                    b_begin += batch_size
                    bar.update(batch_size)
        
        # ranks_fusion = torch.min(ranks_stru, torch.min(ranks_visu, ranks_ling))
        # print("ranks_str: {:.4f}, ranks_img: {:.4f}, ranks_dscp: {:.4f}".format(
        #         sum(ranks_fusion == ranks_stru) / ranks.shape[0],
        #         sum(ranks_fusion == ranks_visu) / ranks.shape[0],
        #         sum(ranks_fusion == ranks_ling) / ranks.shape[0]))

        return ranks
    
    def get_queries(self, queries: torch.Tensor):
        pass

    def get_rhs(self, chunk_begin: int, chunk_size: int):
        pass

    def score(self, x: torch.Tensor):
        pass


class LorentzKG3(KBCModel):
    def __init__(self, sizes: Tuple[int, int, int], args, device) -> None:
        super().__init__()

        self.sizes = sizes
        self.rank = args.rank
        self.init_size = args.init
        self.device = device
        self.manifold = Lorentz(max_norm=2.0)
        self.reg_noise_ratio = 0
        self.margin = 1.08

        self.ent_comps = 5
        self.rel_comps = 4

        self.scale = torch.Tensor([1. / np.sqrt(self.rank)]).to(device)

        self.ent_embedding = nn.Parameter(torch.Tensor(torch.empty((sizes[0], self.rank, self.ent_comps))), requires_grad=True)
        self.ent_embedding1 = nn.Parameter(torch.Tensor(torch.empty((sizes[0], self.ent_comps * self.rank))), requires_grad=True)

        self.rel_rotate  = nn.Parameter(torch.Tensor(torch.empty((sizes[1], self.rank, self.rel_comps))), requires_grad=True)
        self.rel_boosts  = nn.Parameter(torch.Tensor(torch.empty((sizes[1], self.rank, self.rel_comps))), requires_grad=True)
        self.rel_rotate2 = nn.Parameter(torch.Tensor(torch.empty((sizes[1], self.rank, self.rel_comps))), requires_grad=True)
        self.rel_boosts2 = nn.Parameter(torch.Tensor(torch.empty((sizes[1], self.rank, self.rel_comps))), requires_grad=True)

        self.rel_aux = nn.Parameter(torch.Tensor(torch.empty((sizes[1], self.ent_comps * self.rank))), requires_grad=True)
        self.rel_aux1 = nn.Parameter(torch.Tensor(torch.empty((sizes[1], self.ent_comps * self.rank))), requires_grad=True)

        self.atten = nn.Parameter(torch.Tensor(torch.empty((sizes[1], 1, self.ent_comps * self.rank))), requires_grad=True)

        self.softmax = nn.Softmax(dim=-1)
        
        self.eye = nn.Parameter(torch.eye(self.rel_comps), requires_grad=False)

        self.param_init()
    

    def param_init(self):
        ent_data= torch.randn(self.ent_embedding.data.shape, device=self.device) * self.init_size
        ent_data = ent_data / ent_data.norm(dim=-1, keepdim=True)
        self.ent_embedding.data = ent_data

        nn.init.kaiming_normal_(self.ent_embedding1.data)
        self.ent_embedding1.data *= self.init_size

        nn.init.kaiming_normal_(self.rel_rotate.data)
        self.rel_rotate.data *= self.init_size
        nn.init.kaiming_normal_(self.rel_boosts.data)
        self.rel_boosts.data *= self.init_size
        nn.init.kaiming_normal_(self.rel_rotate2.data)
        self.rel_rotate2.data *= self.init_size
        nn.init.kaiming_normal_(self.rel_boosts2.data)
        self.rel_boosts2.data *= self.init_size
        nn.init.kaiming_normal_(self.atten.data)
        self.atten.data *= self.init_size

        nn.init.kaiming_normal_(self.rel_aux.data)
        self.rel_aux.data *= self.init_size
        nn.init.kaiming_normal_(self.rel_aux1.data)
        self.rel_aux1.data *= self.init_size

    def forward(self, queries):

        lhs_t = self.ent_embedding[queries[:, 0]], self.ent_embedding1[queries[:, 0]].view(queries.shape[0], -1)
        rhs_t = self.ent_embedding[queries[:, 2]].view(queries.shape[0], -1), self.ent_embedding1[queries[:, 2]].view(queries.shape[0], -1)
        rel = self.rel_aux[queries[:, 1]], self.rel_aux1[queries[:, 1]]
        rel_rotate = self.rel_rotate[queries[:, 1]]
        rel_boosts = self.rel_boosts[queries[:, 1]]
        rel_atten = self.atten[queries[:, 1]]
        entity1 = self.ent_embedding.view(self.sizes[0], -1)
        entity2 = self.ent_embedding1.view(self.sizes[0], -1)
        
        lhs = lhs_t[0]

        # lhs = self.manifold.expmap0(lhs)

        lhs_rot = self.lorentz_rotate(lhs, rel_rotate).view(lhs.shape[0], -1).unsqueeze(-1)
        lhs_boost = self.lorentz_boost2(lhs, rel_boosts).view(lhs.shape[0], -1).unsqueeze(-1)

        cands = torch.cat([lhs_rot, lhs_boost], dim=-1)
        att_weights = self.softmax(torch.matmul(rel_atten * self.scale, cands))
        lhs = (cands * att_weights).sum(-1)

        # lhs = self.lorentz_boost2(lhs, rel_boosts).view(lhs.shape[0], -1)
        # lhs = self.manifold.logmap0(lhs)

        scores = (lhs * rel[0] - lhs_t[1] * rel[1]) @ entity1.T + \
                 (lhs * rel[1] + lhs_t[1] * rel[0]) @ entity2.T

        regs = (
            torch.sqrt(lhs_t[0].view(queries.shape[0], -1) ** 2 + lhs_t[1] ** 2),
            torch.sqrt(rel[0] ** 2 + rel[1] ** 2),
            torch.sqrt(rhs_t[0] ** 2 + rhs_t[1] ** 2),
            torch.sqrt((rel_rotate ** 2).sum(-1)),
            torch.sqrt((rel_boosts ** 2).sum(-1)),
        )

        return scores, regs


    def lorentz_rotate(self, lhs, rot):
        rot = torch.nn.functional.gelu(rot)
        lhs_t, lhs_r, lhs_i, lhs_j, lhs_k = torch.chunk(lhs, self.ent_comps, dim=-1)
        rot_r, rot_i, rot_j, rot_k = torch.chunk(rot, self.rel_comps, dim=-1)
        # dominator_q = torch.sqrt(rot_r ** 2 + rot_i ** 2 + rot_j ** 2 + rot_k ** 2)
        # rot_r = rot_r / dominator_q
        # rot_i = rot_i / dominator_q
        # rot_j = rot_j / dominator_q
        # rot_k = rot_k / dominator_q

        A = lhs_r * rot_r - lhs_i * rot_i - lhs_j * rot_j - lhs_k * rot_k
        B = lhs_r * rot_i + rot_r * lhs_i + lhs_j * rot_k - rot_j * lhs_k
        C = lhs_r * rot_j + rot_r * lhs_j + lhs_k * rot_i - rot_k * lhs_i
        D = lhs_r * rot_k + rot_r * lhs_k + lhs_i * rot_j - rot_i * lhs_j

        return torch.cat([lhs_t, A, B, C, D], dim=-1)

    def lorentz_boost(self, lhs, boost):
        boost = torch.tanh(boost)
        boost = boost / np.power(4, 1)
        lhs_t, lhs_s = lhs.narrow(-1, 0, 1), lhs.narrow(-1, 1, lhs.shape[-1] - 1)

        boost2 = torch.sum(boost * boost, dim=-1, keepdim=True)
        boost2boost = torch.einsum('bdi,bdj->bdij', boost, boost)
        zeta = 1 / (torch.sqrt(1 - boost2) + 1e-8)


        x_t = zeta * lhs_t - zeta * torch.sum(boost * lhs_s, dim=-1, keepdim=True)
        x_s = -1 * zeta * lhs_t * boost + lhs_s + ((zeta - 1) / (boost2 + 1e-9)) * torch.einsum('bdij, bdj->bdi', boost2boost, lhs_s)
        
        return torch.cat([x_t, x_s], dim=-1)
    
    def lorentz_boost2(self, lhs, boost):
        boost = torch.sigmoid(boost)
        boost = boost / np.power(4, 1)
        lhs_t, lhs_s = lhs.narrow(-1, 0, 1), lhs.narrow(-1, 1, lhs.shape[-1] - 1)

        eye = self.eye.view(1, 1, self.rel_comps, self.rel_comps).expand((lhs.shape[0], self.rank, -1, -1))

        boost2 = torch.matmul(boost.unsqueeze(-2), boost.unsqueeze(-1)).squeeze(-1)
        boost2boost = torch.sqrt(eye + torch.matmul(boost.unsqueeze(-1), boost.unsqueeze(-2)))

        xt = lhs_t * boost2 + torch.matmul(boost.unsqueeze(-2), lhs_s.unsqueeze(-1)).squeeze(-1)
        xs = lhs_t * boost + torch.matmul(boost2boost, lhs_s.unsqueeze(-1)).squeeze(-1)

        return torch.cat([xt, xs], dim=-1)


    
    def get_queries(self, queries: torch.Tensor):
        pass

    def get_rhs(self, chunk_begin: int, chunk_size: int):
        pass

    def score(self, x: torch.Tensor):
        pass

    def get_ranking(
            self, queries: torch.Tensor,
            filters: Dict[Tuple[int, int], List[int]],
            batch_size: int = 1000, chunk_size: int = -1
    ):
        ranks = torch.ones(len(queries))

        with tqdm(total=queries.shape[0], unit='ex') as bar:
            bar.set_description(f'Evaluation')
            with torch.no_grad():
                b_begin = 0
                while b_begin < len(queries):
                    these_queries = queries[b_begin:b_begin + batch_size].cpu().to(self.device)
                    target_idxs = these_queries[:, 2].cpu().tolist()
                    score, _ = self.forward(these_queries)
                    targets = torch.stack([score[row, col] for row, col in enumerate(target_idxs)]).unsqueeze(-1)

                    for i, query in enumerate(these_queries):
                        filter_out = filters[(query[0].item(), query[1].item())]
                        filter_out += [queries[b_begin + i, 2].item()]  
                        score[i, torch.LongTensor(filter_out)] = -1e6

                    ranks[b_begin:b_begin + batch_size] += torch.sum(
                        (score >= targets).float(), dim=1
                    ).cpu()

                    b_begin += batch_size
                    bar.update(batch_size)

        return ranks


        
class MMLorentzKG(KBCModel):
    def __init__(self, sizes: Tuple[int, int, int], rank: int, args,
                init_size: float = 1e-3,
                visu_path: str = None, ling_path: str = None,
                device: torch.DeviceObjType = None,) -> None:
        super().__init__()

        self.args = args

        self.sizes = sizes
        self.rank = rank
        self.device = device
        self.init_size = init_size
        self.manifold = Lorentz(max_norm=5.0)
        self.margin = 1.08
        self.visu_path = visu_path
        self.ling_path = ling_path

        self.ent_comps = 5
        self.rel_comps = 4

        self.scale = torch.Tensor([1. / np.sqrt(self.rank)]).to(device)
        self.softmax = nn.Softmax(dim=-1)

        # structure embeddings
        self.stru_entities = nn.Parameter(torch.Tensor(torch.empty((sizes[0], 2 * self.rank, self.ent_comps))), requires_grad=True)
        self.stru_rel_rotate  = nn.Parameter(torch.Tensor(torch.empty((sizes[1], self.rank, self.rel_comps))), requires_grad=True)
        self.stru_rel_boosts  = nn.Parameter(torch.Tensor(torch.empty((sizes[1], self.rank, self.rel_comps))), requires_grad=True)
        self.stru_atten = nn.Parameter(torch.Tensor(torch.empty((sizes[1], 1, self.ent_comps * self.rank))), requires_grad=True)
        self.stru_rel_aux = nn.Parameter(torch.Tensor(torch.empty((sizes[1], 2, self.ent_comps * self.rank))), requires_grad=True)

        if visu_path is not None:
            self.visu_embeds = load_mmdata(visu_path, requires_grad=False)
            self.visu_num, self.visu_dim = self.visu_embeds.shape
            self.visu_fusion = LoretzFusion2(self.visu_embeds.shape, self.stru_entities.shape, rand_ratio=args.rand_ratio)
            self.visu_rel_rotate = nn.Parameter(torch.Tensor(torch.empty((sizes[1], self.rank, self.rel_comps))), requires_grad=True)
            self.visu_rel_boosts = nn.Parameter(torch.Tensor(torch.empty((sizes[1], self.rank, self.rel_comps))), requires_grad=True)
            self.visu_atten = nn.Parameter(torch.Tensor(torch.empty((sizes[1], 1, self.ent_comps * self.rank))), requires_grad=True)
            self.visu_rel_aux = nn.Parameter(torch.Tensor(torch.empty((sizes[1], 2, self.ent_comps * self.rank))), requires_grad=True)

        if ling_path is not None:
            self.ling_embeds = load_mmdata(ling_path, requires_grad=False)
            self.ling_num, self.ling_dim = self.ling_embeds.shape
            self.ling_fusion = LoretzFusion2(self.ling_embeds.shape, self.stru_entities.shape, rand_ratio=args.rand_ratio)
            self.ling_rel_rotate = nn.Parameter(torch.Tensor(torch.empty((sizes[1], self.rank, self.rel_comps))), requires_grad=True)
            self.ling_rel_boosts = nn.Parameter(torch.Tensor(torch.empty((sizes[1], self.rank, self.rel_comps))), requires_grad=True)
            self.ling_atten = nn.Parameter(torch.Tensor(torch.empty((sizes[1], 1, self.ent_comps * self.rank))), requires_grad=True)
            self.ling_rel_aux = nn.Parameter(torch.Tensor(torch.empty((sizes[1], 2, self.ent_comps * self.rank))), requires_grad=True)

        self.params_init()

    def params_init(self):
        nn.init.normal_(self.stru_entities.data)
        self.stru_entities.data *= self.init_size
        nn.init.normal_(self.stru_rel_rotate.data)
        self.stru_rel_rotate.data *= self.init_size
        nn.init.normal_(self.stru_rel_boosts.data)
        self.stru_rel_boosts.data *= self.init_size
        nn.init.normal_(self.stru_atten.data)
        self.stru_atten.data *= self.init_size
        nn.init.normal_(self.stru_rel_aux.data)
        self.stru_rel_aux.data *= self.init_size

        if self.visu_path is not None:
            nn.init.normal_(self.visu_rel_rotate.data)
            self.visu_rel_rotate.data *= self.init_size
            nn.init.normal_(self.visu_rel_boosts.data)
            self.visu_rel_boosts.data *= self.init_size
            nn.init.normal_(self.visu_atten.data)
            self.visu_atten.data *= self.init_size
            nn.init.normal_(self.visu_rel_aux.data)
            self.visu_rel_aux.data *= self.init_size

        if self.ling_path is not None:
            nn.init.normal_(self.ling_rel_rotate.data)
            self.ling_rel_rotate.data *= self.init_size
            nn.init.normal_(self.ling_rel_boosts.data)
            self.ling_rel_boosts.data *= self.init_size
            nn.init.normal_(self.ling_atten.data)
            self.ling_atten.data *= self.init_size
            nn.init.normal_(self.ling_rel_aux.data)
            self.ling_rel_aux.data *= self.init_size
            
    
    def forward(self, queries):
        slhs, srhs = self.stru_entities[queries[:, 0]], self.stru_entities[queries[:, 2]]
        slhs_t, srhs_t = (slhs[:, :self.rank], slhs[:, self.rank:].view(queries.shape[0], -1)), (srhs[:, :self.rank].view(queries.shape[0], -1), srhs[:, self.rank:].view(queries.shape[0], -1))
        srel_rotate, srel_boosts = self.stru_rel_rotate[queries[:, 1]], self.stru_rel_boosts[queries[:, 1]]
        stru_atten = self.stru_atten[queries[:, 1]]
        stru_rel_aux = self.stru_rel_aux[queries[:, 1]][:, 0], self.stru_rel_aux[queries[:, 1]][:, 1]
        stru_to_scores = self.stru_entities[:, :self.rank].view(self.sizes[0], -1), self.stru_entities[:, self.rank:].view(self.sizes[0], -1)

        slhs = slhs_t[0]
        slhs_rotate = self.lorentz_rotate(slhs, srel_rotate).view(queries.shape[0], -1).unsqueeze(-1)
        slhs_boosts = self.lorentz_boost(slhs, srel_boosts).view(queries.shape[0], -1).unsqueeze(-1)
        stru_cands = torch.cat([slhs_rotate, slhs_boosts], dim=-1)
        stru_weights = self.softmax(torch.matmul(stru_atten * self.scale, stru_cands))
        slhs = (stru_cands * stru_weights).sum(-1)

        stru_scores = (slhs * stru_rel_aux[0] - slhs_t[1] * stru_rel_aux[1]) @ stru_to_scores[0].T + \
                      (slhs * stru_rel_aux[1] + slhs_t[1] * stru_rel_aux[0]) @ stru_to_scores[1].T
        
        stru_regs = (
            torch.sqrt(slhs_t[0].view(queries.shape[0], -1) ** 2 + slhs_t[1] ** 2),
            torch.sqrt(stru_rel_aux[0] ** 2 + stru_rel_aux[1] ** 2),
            torch.sqrt(srhs_t[0] ** 2 + srhs_t[1] ** 2) * (1./3.),
            torch.sqrt((srel_rotate ** 2).sum(-1)),
            torch.sqrt((srel_boosts ** 2).sum(-1))
        )

        return_value = []
        fusion_loss = 0.0
        return_value.append(stru_scores)
        return_value.append(stru_regs)

        if self.args.modality_split:
            if self.visu_path is not None:
                visu_entities, visu_loss = self.visu_fusion(self.stru_entities.clone().detach(), self.visu_embeds)
                vlhs, vrhs =      visu_entities[queries[:, 0]],      visu_entities[queries[:, 2]]
                vlhs_t, vrhs_t = (vlhs[:, :self.rank], vlhs[:, self.rank:].view(queries.shape[0], -1)), (vrhs[:, :self.rank].view(queries.shape[0], -1), vrhs[:, self.rank:].view(queries.shape[0], -1))
                vrel_rotate, vrel_boosts = self.visu_rel_rotate[queries[:, 1]], self.visu_rel_boosts[queries[:, 1]]
                visu_atten = self.visu_atten[queries[:, 1]]
                visu_rel_aux = self.visu_rel_aux[queries[:, 1]][:, 0], self.visu_rel_aux[queries[:, 1]][:, 1]
                visu_to_scores = visu_entities[:, :self.rank].view(self.sizes[0], -1), visu_entities[:, self.rank:].view(self.sizes[0], -1)

                vlhs = vlhs_t[0]
                vlhs_rotate = self.lorentz_rotate(vlhs, vrel_rotate).view(queries.shape[0], -1).unsqueeze(-1)
                vlhs_boosts = self.lorentz_boost(vlhs, vrel_boosts).view(queries.shape[0], -1).unsqueeze(-1)
                visu_cands = torch.cat([vlhs_rotate, vlhs_boosts], dim=-1)
                visu_weights = self.softmax(torch.matmul(visu_atten * self.scale, visu_cands))
                vlhs = (visu_cands * visu_weights).sum(-1)

                visu_scores = (vlhs * visu_rel_aux[0] - vlhs_t[1] * visu_rel_aux[1]) @ visu_to_scores[0].T + \
                              (vlhs * visu_rel_aux[1] + vlhs_t[1] * visu_rel_aux[0]) @ visu_to_scores[1].T

                visu_regs = (
                    torch.sqrt(vlhs_t[0].view(queries.shape[0], -1) ** 2 + vlhs_t[1] ** 2),
                    torch.sqrt(visu_rel_aux[0] ** 2 + visu_rel_aux[1] ** 2),
                    torch.sqrt(vrhs_t[0] ** 2 + vrhs_t[1] ** 2) * (1./3.),
                    torch.sqrt((vrel_rotate ** 2).sum(-1)),
                    torch.sqrt((vrel_boosts ** 2).sum(-1))
                )

                fusion_loss += visu_loss
                return_value.append(visu_scores)
                return_value.append(visu_regs)
            
            if self.ling_path is not None:
                ling_entities, ling_loss = self.ling_fusion(self.stru_entities.clone().detach(), self.ling_embeds)
                llhs, lrhs =      ling_entities[queries[:, 0]],      ling_entities[queries[:, 2]]
                llhs_t, lrhs_t = (llhs[:, :self.rank], llhs[:, self.rank:].view(queries.shape[0], -1)), (lrhs[:, :self.rank].view(queries.shape[0], -1), lrhs[:, self.rank:].view(queries.shape[0], -1))
                lrel_rotate, lrel_boosts = self.ling_rel_rotate[queries[:, 1]], self.ling_rel_boosts[queries[:, 1]]
                ling_rel_aux = self.ling_rel_aux[queries[:, 1]][:, 0], self.ling_rel_aux[queries[:, 1]][:, 1]
                ling_to_scores = ling_entities[:, :self.rank].view(self.sizes[0], -1), ling_entities[:, self.rank:].view(self.sizes[0], -1)
                ling_atten = self.ling_atten[queries[:, 1]]
                
                llhs = llhs_t[0]
                llhs_rotate = self.lorentz_rotate(llhs, lrel_rotate).view(queries.shape[0], -1).unsqueeze(-1)
                llhs_boosts = self.lorentz_boost(llhs, lrel_boosts).view(queries.shape[0], -1).unsqueeze(-1)
                ling_cands = torch.cat([llhs_rotate, llhs_boosts], dim=-1)
                ling_weights = self.softmax(torch.matmul(ling_atten * self.scale, ling_cands))
                llhs = (ling_cands * ling_weights).sum(-1)

                ling_scores = (llhs * ling_rel_aux[0] - llhs_t[1] * ling_rel_aux[1]) @ ling_to_scores[0].T + \
                              (llhs * ling_rel_aux[1] + llhs_t[1] * ling_rel_aux[0]) @ ling_to_scores[1].T

                ling_regs = (
                    torch.sqrt(llhs_t[0].view(queries.shape[0], -1) ** 2 + llhs_t[1] ** 2),
                    torch.sqrt(ling_rel_aux[0] ** 2 + ling_rel_aux[1] ** 2),
                    torch.sqrt(lrhs_t[0] ** 2 + lrhs_t[1] ** 2) * (1./3.),
                    torch.sqrt((lrel_rotate ** 2).sum(-1)),
                    torch.sqrt((lrel_boosts ** 2).sum(-1))
                )

                fusion_loss += ling_loss
                return_value.append(ling_scores)
                return_value.append(ling_regs)
        
        return_value.append(fusion_loss)
        return tuple(return_value)




    def lorentz_rotate(self, lhs, rot):
        rot = torch.nn.functional.gelu(rot)
        lhs_t, lhs_r, lhs_i, lhs_j, lhs_k = torch.chunk(lhs, self.ent_comps, dim=-1)
        rot_r, rot_i, rot_j, rot_k = torch.chunk(rot, self.rel_comps, dim=-1)

        A = lhs_r * rot_r - lhs_i * rot_i - lhs_j * rot_j - lhs_k * rot_k
        B = lhs_r * rot_i + rot_r * lhs_i + lhs_j * rot_k - rot_j * lhs_k
        C = lhs_r * rot_j + rot_r * lhs_j + lhs_k * rot_i - rot_k * lhs_i
        D = lhs_r * rot_k + rot_r * lhs_k + lhs_i * rot_j - rot_i * lhs_j

        return torch.cat([lhs_t, A, B, C, D], dim=-1)

    def lorentz_boost(self, lhs, boost):
        boost = torch.tanh(boost)
        boost = boost / np.power(4, 1)
        lhs_t, lhs_s = lhs.narrow(-1, 0, 1), lhs.narrow(-1, 1, lhs.shape[-1] - 1)

        boost2 = torch.sum(boost * boost, dim=-1, keepdim=True)
        boost2boost = torch.einsum('bdi,bdj->bdij', boost, boost)
        zeta = 1 / (torch.sqrt(1 - boost2) + 1e-8)


        x_t = zeta * lhs_t - zeta * torch.sum(boost * lhs_s, dim=-1, keepdim=True)
        x_s = -1 * zeta * lhs_t * boost + lhs_s + ((zeta - 1) / (boost2 + 1e-9)) * torch.einsum('bdij, bdj->bdi', boost2boost, lhs_s)
        
        return torch.cat([x_t, x_s], dim=-1)

    def get_ranking(
            self, queries: torch.Tensor,
            filters: Dict[Tuple[int, int], List[int]],
            batch_size: int = 1000, chunk_size: int = -1
    ):
        ranks = torch.ones(len(queries))
        ranks_stru = torch.ones(len(queries))
        ranks_visu = torch.ones(len(queries))
        ranks_ling = torch.ones(len(queries))

        with tqdm(total=queries.shape[0], unit='ex') as bar:
            bar.set_description(f'Evaluation')
            with torch.no_grad():
                b_begin = 0
                while b_begin < len(queries):
                    these_queries = queries[b_begin:b_begin + batch_size].cpu().to(self.device)
                    target_idxs = these_queries[:, 2].cpu().tolist()
                    if self.args.modality_split:
                        if (self.visu_path is not None) and (self.ling_path is not None):
                            score_stru, _, score_visu, _, score_ling, _, _ = self.forward(these_queries)
                            targets_stru = torch.stack([score_stru[row, col] for row, col in enumerate(target_idxs)]).unsqueeze(-1)
                            targets_visu = torch.stack([score_visu[row, col] for row, col in enumerate(target_idxs)]).unsqueeze(-1)
                            targets_ling = torch.stack([score_ling[row, col] for row, col in enumerate(target_idxs)]).unsqueeze(-1)

                            for i, query in enumerate(these_queries):
                                filter_out = filters[(query[0].item(), query[1].item())]
                                filter_out += [queries[b_begin + i, 2].item()]  
                                score_stru[i, torch.LongTensor(filter_out)] = -1e6
                                score_visu[i, torch.LongTensor(filter_out)] = -1e6
                                score_ling[i, torch.LongTensor(filter_out)] = -1e6

                            ranks_stru[b_begin:b_begin + batch_size] += torch.sum(
                                (score_stru >= targets_stru).float(), dim=1
                            ).cpu()
                            ranks_visu[b_begin:b_begin + batch_size] += torch.sum(
                                (score_visu >= targets_visu).float(), dim=1
                            ).cpu()
                            ranks_ling[b_begin:b_begin + batch_size] += torch.sum(
                                (score_ling >= targets_ling).float(), dim=1
                            ).cpu()
                        
                        elif (self.visu_path is None) and (self.ling_path is not None):
                            score_stru, _, score_ling, _, _ = self.forward(these_queries)
                            targets_stru = torch.stack([score_stru[row, col] for row, col in enumerate(target_idxs)]).unsqueeze(-1)
                            targets_ling = torch.stack([score_ling[row, col] for row, col in enumerate(target_idxs)]).unsqueeze(-1)

                            for i, query in enumerate(these_queries):
                                filter_out = filters[(query[0].item(), query[1].item())]
                                filter_out += [queries[b_begin + i, 2].item()]  
                                score_stru[i, torch.LongTensor(filter_out)] = -1e6
                                score_ling[i, torch.LongTensor(filter_out)] = -1e6

                            ranks_stru[b_begin:b_begin + batch_size] += torch.sum(
                                (score_stru >= targets_stru).float(), dim=1
                            ).cpu()
                            ranks_ling[b_begin:b_begin + batch_size] += torch.sum(
                                (score_ling >= targets_ling).float(), dim=1
                            ).cpu()
                        elif (self.visu_path is not None) and (self.ling_path is None):
                            score_stru, _, score_visu, _, _ = self.forward(these_queries)
                            targets_stru = torch.stack([score_stru[row, col] for row, col in enumerate(target_idxs)]).unsqueeze(-1)
                            targets_visu = torch.stack([score_visu[row, col] for row, col in enumerate(target_idxs)]).unsqueeze(-1)

                            for i, query in enumerate(these_queries):
                                filter_out = filters[(query[0].item(), query[1].item())]
                                filter_out += [queries[b_begin + i, 2].item()]  
                                score_stru[i, torch.LongTensor(filter_out)] = -1e6
                                score_visu[i, torch.LongTensor(filter_out)] = -1e6

                            ranks_stru[b_begin:b_begin + batch_size] += torch.sum(
                                (score_stru >= targets_stru).float(), dim=1
                            ).cpu()
                            ranks_visu[b_begin:b_begin + batch_size] += torch.sum(
                                (score_visu >= targets_visu).float(), dim=1
                            ).cpu()
                    else:
                        score_stru, _, _ = self.forward(these_queries)
                        targets_stru = torch.stack([score_stru[row, col] for row, col in enumerate(target_idxs)]).unsqueeze(-1)

                        for i, query in enumerate(these_queries):
                            filter_out = filters[(query[0].item(), query[1].item())]
                            filter_out += [queries[b_begin + i, 2].item()]  
                            score_stru[i, torch.LongTensor(filter_out)] = -1e6

                        ranks_stru[b_begin:b_begin + batch_size] += torch.sum(
                            (score_stru >= targets_stru).float(), dim=1
                        ).cpu()

                    b_begin += batch_size
                    bar.update(batch_size)
                    
        if self.args.modality_split:
            if (self.visu_path is not None) and (self.ling_path is not None):
                ranks_fusion = torch.min(ranks_stru, torch.min(ranks_visu, ranks_ling))
                print("ranks_str: {:.4f}, ranks_img: {:.4f}, ranks_dscp: {:.4f}".format(
                    sum(ranks_fusion == ranks_stru) / ranks.shape[0],
                    sum(ranks_fusion == ranks_visu) / ranks.shape[0],
                    sum(ranks_fusion == ranks_ling) / ranks.shape[0]))
            elif (self.visu_path is None) and (self.ling_path is not None):
                ranks_fusion = torch.min(ranks_stru, ranks_ling)
                print("ranks_str: {:.4f}, ranks_dscp: {:.4f}".format(
                    sum(ranks_fusion == ranks_stru) / ranks.shape[0],
                    sum(ranks_fusion == ranks_ling) / ranks.shape[0]))
            elif (self.visu_path is not None) and (self.ling_path is None):
                ranks_fusion = torch.min(ranks_stru, ranks_visu)
                print("ranks_str: {:.4f}, ranks_img: {:.4f}".format(
                    sum(ranks_fusion == ranks_stru) / ranks.shape[0],
                    sum(ranks_fusion == ranks_visu) / ranks.shape[0]))
        else:
            ranks_fusion = ranks_stru
            print("ranks_str: {:.4f}".format(sum(ranks_fusion == ranks_stru) / ranks.shape[0]))

        
        return ranks_fusion

    def get_queries(self, queries: torch.Tensor):
        pass

    def get_rhs(self, chunk_begin: int, chunk_size: int):
        pass

    def score(self, x: torch.Tensor):
        pass

class ComplEx(KBCModel):
    def __init__(self, sizes: Tuple[int, int, int], args, device) -> None:
        super().__init__()

        self.sizes = sizes
        self.rank = args.rank
        self.init_size = args.init
        self.device = device

        self.ent_embedding = nn.Embedding(sizes[0], 2 * self.rank)
        self.rel_embedding = nn.Embedding(sizes[1], 2 * self.rank)
    
        self.ent_embedding.weight.data *= args.init
        self.rel_embedding.weight.data *= args.init
    
    def forward(self, queries):
        lhs = self.ent_embedding(queries[:, 0])
        rhs = self.ent_embedding(queries[:, 2])
        rel = self.rel_embedding(queries[:, 1])
        
        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rhs = rhs[:, :self.rank], rhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]

        to_score = self.ent_embedding.weight[:, :self.rank], self.ent_embedding.weight[:, self.rank:]

        return (
                    (lhs[0] * rel[0] - lhs[1] * rel[1]) @ to_score[0].transpose(0, 1) + \
                    (lhs[0] * rel[1] + lhs[1] * rel[0]) @ to_score[1].transpose(0, 1)
            ), (
                    torch.sqrt(lhs[0] ** 2 + lhs[1] ** 2),
                    torch.sqrt(rhs[0] ** 2 + rhs[1] ** 2),
                    torch.sqrt(rel[0] ** 2 + rel[1] ** 2)
            ), 0.0

    def get_ranking(
            self, queries: torch.Tensor,
            filters: Dict[Tuple[int, int], List[int]],
            batch_size: int = 1000, chunk_size: int = -1
    ):
        ranks = torch.ones(len(queries))
        # ranks_stru = torch.ones(len(queries))
        # ranks_visu = torch.ones(len(queries))
        # ranks_ling = torch.ones(len(queries))

        with tqdm(total=queries.shape[0], unit='ex') as bar:
            bar.set_description(f'Evaluation')
            with torch.no_grad():
                b_begin = 0
                while b_begin < len(queries):
                    these_queries = queries[b_begin:b_begin + batch_size].cpu().to(self.device)
                    target_idxs = these_queries[:, 2].cpu().tolist()
                    score, _, _ = self.forward(these_queries)
                    targets = torch.stack([score[row, col] for row, col in enumerate(target_idxs)]).unsqueeze(-1)
                    # targets_visu = torch.stack([score_visu[row, col] for row, col in enumerate(target_idxs)]).unsqueeze(-1)
                    # targets_ling = torch.stack([score_ling[row, col] for row, col in enumerate(target_idxs)]).unsqueeze(-1)

                    for i, query in enumerate(these_queries):
                        filter_out = filters[(query[0].item(), query[1].item())]
                        filter_out += [queries[b_begin + i, 2].item()]  
                        score[i, torch.LongTensor(filter_out)] = -1e6
                        # score_visu[i, torch.LongTensor(filter_out)] = -1e6
                        # score_ling[i, torch.LongTensor(filter_out)] = -1e6

                    ranks[b_begin:b_begin + batch_size] += torch.sum(
                        (score >= targets).float(), dim=1
                    ).cpu()
                    # ranks_visu[b_begin:b_begin + batch_size] += torch.sum(
                    #     (score_visu >= targets_visu).float(), dim=1
                    # ).cpu()
                    # ranks_ling[b_begin:b_begin + batch_size] += torch.sum(
                    #     (score_ling >= targets_ling).float(), dim=1
                    # ).cpu()

                    b_begin += batch_size
                    bar.update(batch_size)
        
        # ranks_fusion = torch.min(ranks_stru, torch.min(ranks_visu, ranks_ling))
        # print("ranks_str: {:.4f}, ranks_img: {:.4f}, ranks_dscp: {:.4f}".format(
        #         sum(ranks_fusion == ranks_stru) / ranks.shape[0],
        #         sum(ranks_fusion == ranks_visu) / ranks.shape[0],
        #         sum(ranks_fusion == ranks_ling) / ranks.shape[0]))

        return ranks
    
    def get_queries(self, queries: torch.Tensor):
        pass

    def get_rhs(self, chunk_begin: int, chunk_size: int):
        pass

    def score(self, x: torch.Tensor):
        pass

class MMKGE_CL(KBCModel):
    def __init__(self, sizes: Tuple[int, int, int], rank: int, args,
                init_size: float = 1e-3,
                visu_path: str = None, ling_path: str = None,
                device: torch.DeviceObjType = None,) -> None:
        super(MMKGE_CL, self).__init__()

        self.args = args

        self.sizes = sizes
        self.rank = rank
        self.device = device
        self.init_size = init_size
        

        if args.optimizer in ["adgrad", "Adagrad"]:
            self.ent_embeds = nn.Embedding(sizes[0], 2 * rank, sparse=True)
            self.rel_embeds = nn.Embedding(sizes[1], 2 * rank, sparse=True)
        elif args.optimizer in ["adam", "Adam"]:
            self.ent_embeds = nn.Embedding(sizes[0], 2 * rank, sparse=False)
            self.rel_embeds = nn.Embedding(sizes[1], 2 * rank, sparse=False)


        if visu_path is not None:
            self.visu_embeds = load_mmdata(visu_path, requires_grad=False)
            self.visu_num, self.visu_dim = self.visu_embeds.shape
            if args.optimizer in ["adgrad", "Adagrad"]:
                self.visu_rhs_embeds = nn.Embedding(sizes[0], 2 * rank, sparse=True)
            elif args.optimizer in ["adam", "Adam"]:
                self.visu_rhs_embeds = nn.Embedding(sizes[0], 2 * rank, sparse=False)
            
        
        if ling_path is not None:
            self.ling_embeds = load_mmdata(ling_path, requires_grad=False)
            self.ling_num, self.ling_dim = self.ling_embeds.shape
            if args.optimizer in ["adgrad", "Adagrad"]:
                self.ling_rhs_embeds = nn.Embedding(sizes[0], 2 * rank, sparse=True)
            elif args.optimizer in ["adam", "Adam"]:
                self.ling_rhs_embeds = nn.Embedding(sizes[0], 2 * rank, sparse=False)
        

        # self.fusion_module = FusionModule_v6(self.ent_embeds.weight.shape, self.visu_embeds.shape, self.ling_embeds.shape)
        # self.visu_fusion = GraphFusion(self.visu_embeds.shape, self.ent_embeds.weight.shape)
        # self.ling_fusion = GraphFusion(self.ling_embeds.shape, self.ent_embeds.weight.shape)

        self.visu_fusion = CLFusion(self.visu_embeds.shape, self.ent_embeds.weight.shape)
        self.ling_fusion = CLFusion(self.ling_embeds.shape, self.ent_embeds.weight.shape)

        self.param_init()

    def param_init(self):
        for name, param in self.named_parameters():
            if str.endswith(name, 'weight') and 'embeds' in name:
                param.data *= self.init_size
            elif str.endswith(name, 'weight') and 'linear' in name:
                nn.init.kaiming_normal_(param)
            elif str.endswith(name, 'bias') and 'linear' in name:
                nn.init.zeros_(param)
            elif str.endswith(name, 'weight') and 'fusion' in name:
                if param.ndim == 2:
                    nn.init.kaiming_normal_(param)
                else:
                    nn.init.normal_(param)
            elif str.endswith(name, 'bias') and 'fusion' in name:
                nn.init.zeros_(param)
            elif name in ['visu_embeds', 'ling_embeds']:
                param.data *= self.init_size
    
    def forward(self, x):
        # sam_indices = torch.randperm(self.visu_embeds.shape[0])

        # cl_loss, ling_feats, visu_feats = self.fusion_module(self.ent_embeds.weight.clone().detach(), self.visu_embeds, self.ling_embeds)
        visu_feats, visu_cl_loss = self.visu_fusion(self.ent_embeds.weight.clone().detach(), self.visu_embeds)
        ling_feats, ling_cl_loss = self.ling_fusion(self.ent_embeds.weight.clone().detach(), self.ling_embeds)
        cl_loss = visu_cl_loss + ling_cl_loss
        
        slhs, srel, srhs = self.ent_embeds(x[:, 0]),   self.rel_embeds(x[:, 1]),     self.ent_embeds(x[:, 2])
        vlhs, vrel, vrhs = visu_feats[x[:, 0]], self.visu_rhs_embeds.weight[x[:, 1]], visu_feats[x[:, 2]]
        llhs, lrel, lrhs = ling_feats[x[:, 0]], self.ling_rhs_embeds.weight[x[:, 1]], ling_feats[x[:, 2]]

        # struct embeddings
        slhs = slhs[:, :self.rank], slhs[:, self.rank:]
        srel = srel[:, :self.rank], srel[:, self.rank:]
        srhs = srhs[:, :self.rank], srhs[:, self.rank:]
        # visual embeddings
        vlhs = vlhs[:, :self.rank], vlhs[:, self.rank:]
        vrel = vrel[:, :self.rank], vrel[:, self.rank:]
        vrhs = vrhs[:, :self.rank], vrhs[:, self.rank:]
        # linguistic embeddings
        llhs = llhs[:, :self.rank], llhs[:, self.rank:]
        lrel = lrel[:, :self.rank], lrel[:, self.rank:]
        lrhs = lrhs[:, :self.rank], lrhs[:, self.rank:]
        # to_score
        to_score_s = self.ent_embeds.weight[:, :self.rank], self.ent_embeds.weight[:, self.rank:]
        to_score_v = visu_feats[:, :self.rank], visu_feats[:, self.rank:]
        to_score_l = ling_feats[:, :self.rank], ling_feats[:, self.rank:]


        score_stru = (
                (slhs[0] * srel[0] - slhs[1] * srel[1]) @ to_score_s[0].transpose(0, 1) +
                (slhs[0] * srel[1] + slhs[1] * srel[0]) @ to_score_s[1].transpose(0, 1)
        )
        factors_stru = (
            torch.sqrt(slhs[0] ** 2 + slhs[1] ** 2),
            torch.sqrt(srel[0] ** 2 + srel[1] ** 2),
            torch.sqrt(srhs[0] ** 2 + srhs[1] ** 2)
        )

        score_visu = (
                (vlhs[0] * vrel[0] - vlhs[1] * vrel[1]) @ to_score_v[0].transpose(0, 1) +
                (vlhs[0] * vrel[1] + vlhs[1] * vrel[0]) @ to_score_v[1].transpose(0, 1)
        )
        factors_visu = (
            torch.sqrt(vlhs[0] ** 2 + vlhs[1] ** 2),
            torch.sqrt(vrel[0] ** 2 + vrel[1] ** 2),
            torch.sqrt(vrhs[0] ** 2 + vrhs[1] ** 2)
        )
        score_ling = (
                (llhs[0] * lrel[0] - llhs[1] * lrel[1]) @ to_score_l[0].transpose(0, 1) +
                (llhs[0] * lrel[1] + llhs[1] * lrel[0]) @ to_score_l[1].transpose(0, 1)
        )
        factors_ling = (
            torch.sqrt(llhs[0] ** 2 + llhs[1] ** 2),
            torch.sqrt(lrel[0] ** 2 + lrel[1] ** 2),
            torch.sqrt(lrhs[0] ** 2 + lrhs[1] ** 2)
        )

        return score_stru, factors_stru, score_visu, factors_visu, score_ling, factors_ling, cl_loss
    
    def get_ranking(
            self, queries: torch.Tensor,
            filters: Dict[Tuple[int, int], List[int]],
            batch_size: int = 1000, chunk_size: int = -1
    ):
        ranks = torch.ones(len(queries))
        ranks_stru = torch.ones(len(queries))
        ranks_visu = torch.ones(len(queries))
        ranks_ling = torch.ones(len(queries))

        with tqdm(total=queries.shape[0], unit='ex') as bar:
            bar.set_description(f'Evaluation')
            with torch.no_grad():
                b_begin = 0
                while b_begin < len(queries):
                    these_queries = queries[b_begin:b_begin + batch_size].cpu().to(self.device)
                    target_idxs = these_queries[:, 2].cpu().tolist()
                    score_stru, _, score_visu, _, score_ling, _, _ = self.forward(these_queries)
                    targets_stru = torch.stack([score_stru[row, col] for row, col in enumerate(target_idxs)]).unsqueeze(-1)
                    targets_visu = torch.stack([score_visu[row, col] for row, col in enumerate(target_idxs)]).unsqueeze(-1)
                    targets_ling = torch.stack([score_ling[row, col] for row, col in enumerate(target_idxs)]).unsqueeze(-1)

                    for i, query in enumerate(these_queries):
                        filter_out = filters[(query[0].item(), query[1].item())]
                        filter_out += [queries[b_begin + i, 2].item()]  
                        score_stru[i, torch.LongTensor(filter_out)] = -1e6
                        score_visu[i, torch.LongTensor(filter_out)] = -1e6
                        score_ling[i, torch.LongTensor(filter_out)] = -1e6

                    ranks_stru[b_begin:b_begin + batch_size] += torch.sum(
                        (score_stru >= targets_stru).float(), dim=1
                    ).cpu()
                    ranks_visu[b_begin:b_begin + batch_size] += torch.sum(
                        (score_visu >= targets_visu).float(), dim=1
                    ).cpu()
                    ranks_ling[b_begin:b_begin + batch_size] += torch.sum(
                        (score_ling >= targets_ling).float(), dim=1
                    ).cpu()

                    b_begin += batch_size
                    bar.update(batch_size)
        
        ranks_fusion = torch.min(ranks_stru, torch.min(ranks_visu, ranks_ling))
        print("ranks_str: {:.4f}, ranks_img: {:.4f}, ranks_dscp: {:.4f}".format(
                sum(ranks_fusion == ranks_stru) / ranks.shape[0],
                sum(ranks_fusion == ranks_visu) / ranks.shape[0],
                sum(ranks_fusion == ranks_ling) / ranks.shape[0]))

        return ranks_fusion
    
    def get_queries(self, queries: torch.Tensor):
        pass

    def get_rhs(self, chunk_begin: int, chunk_size: int):
        pass

    def score(self, x: torch.Tensor):
        pass