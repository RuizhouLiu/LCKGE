import torch
import torch.nn as nn
import torch.nn.functional as F

from graph_fusion.gcn import GCN, GCNLayer


def to_cuda(x, device=None):
    if device:
        x = x.to(device)
    return x


def cossim(x, y):
    def normalize(x, axis=-1):
        x = 1. * x / torch.max(torch.norm(x, 2, axis, keepdim=True).expand_as(x), torch.tensor(1e-12))
        return x

    return torch.matmul(normalize(x), normalize(y).transpose(-1, -2))


def build_epsilon_neighbourhood(attention, epsilon, markoff_value):
    mask = (attention > epsilon).detach().float()
    weighted_adjacency_matrix = attention * mask + markoff_value * (1 - mask)
    return weighted_adjacency_matrix


class FusionModule_v5(nn.Module):
    def __init__(self, stru_dim, vis_dim, ling_dim, tau = 0.01, topk=6) -> None:
        super().__init__()
        self.stru_dim = stru_dim
        self.vis_dim = vis_dim
        self.ling_dim = ling_dim
        self.tau = tau
        self.topk = topk

        self.func_l_vl = nn.Linear(ling_dim, stru_dim)
        self.func_v_vl = nn.Linear(vis_dim, stru_dim)
        self.func_s_svl = nn.Linear(stru_dim, stru_dim)
        self.func_v_vl_svl = nn.Sequential(*[
            nn.LeakyReLU(inplace=True),
            nn.Linear(stru_dim, stru_dim),
            nn.BatchNorm1d(stru_dim)
        ])
        self.func_l_vl_svl = nn.Sequential(*[
            nn.LeakyReLU(inplace=True),
            nn.Linear(stru_dim, stru_dim),
            nn.BatchNorm1d(stru_dim)
        ])

        self.atten_mat = nn.Linear(stru_dim, stru_dim)

    def forward(self, stru_feats, vis_feats, ling_feats):
        z_l_vl = self.func_l_vl(ling_feats)
        z_v_vl = self.func_v_vl(vis_feats)

        z_s_svl = self.func_s_svl(stru_feats)
        
        z_l_vl_svl = self.func_l_vl_svl(z_l_vl)
        z_v_vl_svl = self.func_v_vl_svl(z_v_vl)

        loss_sl_svl = self.neibor_mnce_v2(z_s_svl, z_l_vl_svl)
        loss_vl_svl = self.neibor_mnce_v2(z_s_svl, z_v_vl_svl)
        
        loss = loss_sl_svl + loss_vl_svl

        return loss, z_s_svl, z_v_vl_svl, z_l_vl_svl
    
    def neibor_mnce(self, stru, mmFeat):
        def cossim(m1, m2):
            """
            m1: [#ent, dim]
            m2: [#ent, dim]
            """
            def normalize(x, axis=-1):
                x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
                return x

            return normalize(m1) @ normalize(m2).T

        mmFeats_dist_map = self.atten_mat(mmFeat) @ self.atten_mat(mmFeat).T
        _, index = torch.topk(mmFeats_dist_map, k=self.topk, dim=-1)
        index = (torch.arange(mmFeats_dist_map.shape[0]).unsqueeze(-1).expand((-1, self.topk)).to(index.device), index)
        
        dist_map = torch.exp(cossim(stru, mmFeat) / self.tau)
        pos_score = dist_map[index]

        return (-torch.log(pos_score.sum(dim=-1) / dist_map.sum(dim=-1))).mean()
    

    def neibor_mnce_v2(self, stru, mmFeat):
        def cossim(m1, m2):
            """
            m1: [#ent, dim]
            m2: [#ent, dim]
            """
            def normalize(x, axis=-1):
                x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
                return x

            return normalize(m1) @ normalize(m2).T
        
        dist_map = torch.exp(cossim(stru, mmFeat) / self.tau)
        atten_weights = F.softmax(cossim(self.atten_mat(stru), self.atten_mat(mmFeat)), dim=-1)

        pos_scores = atten_weights * dist_map
        return (-torch.log(pos_scores.sum(dim=-1) / dist_map.sum(dim=-1))).mean()
    

class FusionModule_v6(nn.Module):
    def __init__(self, stru_shape, vis_shape, ling_shape, tau = 0.5, topk=6) -> None:
        super(FusionModule_v6, self).__init__()

        self.stru_num, self.stru_dim = stru_shape
        self.visu_num, self.visu_dim = vis_shape
        self.ling_num, self.ling_dim = ling_shape
        self.tau = tau
        self.topk = 6
        self.topk2 = 6

        self.fusion_v_vl = nn.Sequential(*[
            nn.Linear(self.visu_dim, self.stru_dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(self.stru_dim, self.stru_dim),
            nn.BatchNorm1d(self.stru_dim)
        ])

        self.fusion_l_vl = nn.Sequential(*[
            nn.Linear(self.ling_dim, self.stru_dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(self.stru_dim, self.stru_dim),
            nn.BatchNorm1d(self.stru_dim)
        ])

    def forward(self, stru_feats, visu_feats, ling_feats):
        ling_feats = self.fusion_l_vl(ling_feats)
        visu_feats = self.fusion_v_vl(visu_feats) 

        loss_sl = self.neibor_mnce_v2(stru_feats, ling_feats)
        loss_sv = self.neibor_mnce_v2(stru_feats, visu_feats)

        loss = loss_sl + loss_sv

        return loss, ling_feats, visu_feats

    def neibor_mnce(self, stru, mmFeat):
        def inner(x, y):
            """
            args:
            ===
            x: [n_samples, dim]
            y: [n_samples, dim]
            """
            return x @ y.T
        
        def cossim(x, y):
            def normalize(x, axis=-1):
                x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
                return x

            return normalize(x) @ normalize(y).T
        
        def l2dist(x, y):
            return torch.cdist(x, y)
        
        sm_distMap = torch.exp(cossim(stru, mmFeat) / self.tau)
        mm_distMap = cossim(mmFeat, mmFeat)
        maxSM_idx = torch.argmax(sm_distMap, dim=-1)
        _, maxMM_idx = torch.topk(mm_distMap[maxSM_idx], k=self.topk, dim=-1)
        maxMM_idx = (torch.arange(mm_distMap[maxSM_idx].shape[0]).unsqueeze(-1).expand((-1, self.topk)).to(maxMM_idx.device), maxMM_idx)
        pos_socres = sm_distMap[maxMM_idx]
        
        return (-torch.log(pos_socres.sum(dim=-1) / sm_distMap.sum(dim=-1))).mean()

    def neibor_mnce_v2(self, stru, mmFeat):
        def inner(x, y):
            """
            args:
            ===
            x: [n_samples, dim]
            y: [n_samples, dim]
            """
            return x @ y.T
        
        def cossim(x, y):
            def normalize(x, axis=-1):
                x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
                return x

            return normalize(x) @ normalize(y).T
        
        def l2dist(x, y):
            return torch.cdist(x, y)
        
        sm_distMap = torch.exp(cossim(stru, mmFeat) / self.tau)
        mm_distMap = cossim(mmFeat, mmFeat)
        _, maxSM_idx = torch.topk(sm_distMap, k=self.topk, dim=-1)
        pos_socres = []
        for maxSM_idx_i in maxSM_idx.T:
            _, maxMM_idx_i = torch.topk(mm_distMap[maxSM_idx_i], k=self.topk2, dim=-1)
            maxMM_idx = (torch.arange(mm_distMap[maxSM_idx_i].shape[0]).unsqueeze(-1).expand((-1, self.topk2)).to(maxMM_idx_i.device), maxMM_idx_i)
            pos_socres.append(sm_distMap[maxMM_idx])
        pos_socres = torch.cat(pos_socres, dim=-1)
        return (-torch.log(pos_socres.sum(dim=-1) / sm_distMap.sum(dim=-1))).mean()
    

class GraphFusion(nn.Module):
    def __init__(self, mm_shape, st_shape, num_pers=6, hidden_dim=32, tau = 0.5, topk=6) -> None:
        super().__init__()

        self.st_num, self.st_dim = st_shape
        self.mm_num, self.mm_dim = mm_shape

        self.num_pers = num_pers
        self.hidden_dim = hidden_dim
        self.tau = tau
        self.topk = 6
        self.topk2 = 16

        self.self_weights  = torch.Tensor(num_pers, self.mm_dim)    # all graph weights
        self.self_weights  = nn.Parameter(nn.init.kaiming_normal_(self.self_weights))
        self.inter_weights = torch.Tensor(num_pers, self.st_dim)    # bi-graph weights
        self.inter_weights = nn.Parameter(nn.init.kaiming_normal_(self.inter_weights))

        self.mm_encoder = GCNLayer(in_features=self.mm_dim, out_features=self.st_dim)
        self.gcn = nn.ModuleList([
            GCNLayer(in_features=self.st_dim, out_features=self.st_dim),
            GCNLayer(in_features=self.st_dim, out_features=self.st_dim),
        ])
        self.act_func = nn.LeakyReLU(inplace=True)
    
    def forward(self, mm_feats, st_feats):
        mm_adj = self.cal_mm_adj(mm_feats)
        mm_feats_hid = self.act_func(self.mm_encoder(mm_feats, mm_adj))
        stmm_adj = self.cal_stmm_adj(st_feats, mm_feats_hid)
        fused_adj = torch.matmul(stmm_adj, mm_adj)
        mm_feats_hid = self.gcn[0](mm_feats_hid, fused_adj)
        
        return mm_feats_hid


    def cal_mm_adj(self, mm_feats):
        """
        Args:
        ---
        mm_feats: [mm_num, mm_dim]
        """

        expand_weights = self.self_weights.unsqueeze(1)
        weighted_feats = mm_feats.unsqueeze(0) * expand_weights
        attens = cossim(weighted_feats, weighted_feats).mean(0)
        adj = build_epsilon_neighbourhood(attens, 0, 0)
        adj = adj / torch.clamp(torch.sum(adj, dim=-1, keepdim=True), min=1e-12)
        return adj
    
    def cal_stmm_adj(self, st_feats, mm_feats):
        """
        Args:
        ---
        st_feats: [st_num, st_dim]
        mm_feats: [mm_num, mm_dim]
        """
        expand_weights = self.inter_weights.unsqueeze(1)
        weighted_feats = st_feats.unsqueeze(0) * expand_weights
        attens = cossim(weighted_feats, mm_feats).mean(0)
        adj = build_epsilon_neighbourhood(attens, 0, 0)
        adj = adj / torch.clamp(torch.sum(adj, dim=-1, keepdim=True), min=1e-12)
        return adj
    

class CLFusion(nn.Module):
    def __init__(self, mm_shape, st_shape, num_pers=6, hidden_dim=32, tau = 0.5, topk=6) -> None:
        super().__init__()

        self.st_num, self.st_dim = st_shape
        self.mm_num, self.mm_dim = mm_shape

        self.num_pers = num_pers
        self.hidden_dim = hidden_dim
        self.tau = tau
        self.topk = topk
        self.topk2 = 16

        self.mm_encoder = nn.Sequential(*[
            nn.Linear(self.mm_dim, self.st_dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(self.st_dim, self.st_dim),
            nn.BatchNorm1d(self.st_dim)
        ])

        self.weights = torch.Tensor(num_pers, self.st_dim)    # bi-graph weights
        self.weights = nn.Parameter(nn.init.kaiming_normal_(self.weights))

    def forward(self, st_feats, mm_feats):
        mm_feats = self.mm_encoder(mm_feats)
        loss = self.neibor_mnce(st_feats, mm_feats)
        return mm_feats, loss


    def weighted_neibor_mnce(self, st_feats, mm_feats):
        expand_weights = self.weights.unsqueeze(1)
        weighted_mm_feats = mm_feats.unsqueeze(0) * expand_weights 
        sm_distMap = torch.exp(cossim(st_feats, weighted_mm_feats).mean(0) / self.tau)
        mm_distMap = cossim(weighted_mm_feats, weighted_mm_feats).mean(0)

        maxSM_idx = torch.argmax(sm_distMap, dim=-1)
        _, maxMM_idx = torch.topk(mm_distMap[maxSM_idx], k=self.topk, dim=-1)
        maxMM_idx = (torch.arange(mm_distMap[maxSM_idx].shape[0]).unsqueeze(-1).expand((-1, self.topk)).to(maxMM_idx.device), maxMM_idx)
        pos_socres = sm_distMap[maxMM_idx]
        
        return (-torch.log(pos_socres.sum(dim=-1) / sm_distMap.sum(dim=-1))).mean()
    

    def neibor_mnce(self, stru, mmFeat):
        def inner(x, y):
            """
            args:
            ===
            x: [n_samples, dim]
            y: [n_samples, dim]
            """
            return x @ y.T
        
        def cossim(x, y):
            def normalize(x, axis=-1):
                x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
                return x

            return normalize(x) @ normalize(y).T
        
        def l2dist(x, y):
            return torch.cdist(x, y)
        
        sm_distMap = torch.exp(cossim(stru, mmFeat) / self.tau)
        mm_distMap = cossim(mmFeat, mmFeat)
        maxSM_idx = torch.argmax(sm_distMap, dim=-1)
        _, maxMM_idx = torch.topk(mm_distMap[maxSM_idx], k=self.topk, dim=-1)
        maxMM_idx = (torch.arange(mm_distMap[maxSM_idx].shape[0]).unsqueeze(-1).expand((-1, self.topk)).to(maxMM_idx.device), maxMM_idx)
        pos_socres = sm_distMap[maxMM_idx]
        
        return (-torch.log(pos_socres.sum(dim=-1) / sm_distMap.sum(dim=-1))).mean()



class LoretzFusion(nn.Module):
    def __init__(self, mm_shape, st_shape, num_pers=6, hidden_dim=32, tau = 0.5, topk=6) -> None:
        super().__init__()  

        """
        mm_shape: [#ent, mm_dim]
        st_shape: [#ent, rank, #comps]
        """
        
        self.mm_num, self.mm_dim = mm_shape
        self.st_num, self.st_dim, self.comps = st_shape

        self.num_pers = num_pers
        self.hidden_dim = hidden_dim
        self.tau = tau
        self.topk = topk
        self.topk2 = 16

        self.mm_encoder_t = nn.Sequential(*[
            nn.Linear(self.mm_dim, self.st_dim),
            nn.GELU(),
            nn.Linear(self.st_dim, self.st_dim),
            nn.BatchNorm1d(self.st_dim)
        ])

        self.mm_encoder_r = nn.Sequential(*[
            nn.Linear(self.mm_dim, self.st_dim),
            nn.GELU(),
            nn.Linear(self.st_dim, self.st_dim),
            nn.BatchNorm1d(self.st_dim)
        ])

        self.mm_encoder_i = nn.Sequential(*[
            nn.Linear(self.mm_dim, self.st_dim),
            nn.GELU(),
            nn.Linear(self.st_dim, self.st_dim),
            nn.BatchNorm1d(self.st_dim)
        ])

        self.mm_encoder_j = nn.Sequential(*[
            nn.Linear(self.mm_dim, self.st_dim),
            nn.GELU(),
            nn.Linear(self.st_dim, self.st_dim),
            nn.BatchNorm1d(self.st_dim)
        ])

        self.mm_encoder_k = nn.Sequential(*[
            nn.Linear(self.mm_dim, self.st_dim),
            nn.GELU(),
            nn.Linear(self.st_dim, self.st_dim),
            nn.BatchNorm1d(self.st_dim)
        ])
    
        self.params_init()


    def params_init(self):
        for name, param in self.named_parameters():
            if str.endswith(name, 'weight'):
                if not '3' in name:
                    nn.init.kaiming_normal_(param)
                else:
                    nn.init.normal_(param)
            elif str.endswith(name, 'bias'):
                nn.init.zeros_(param)


    def forward(self, st_feats, mm_feats):
        """
        st_feats: [#ent, rank, comps]
        mm_feats: [#ent, mm_dim]
        """
        st_feats_t, st_feats_r, st_feats_i, st_feats_j, st_feats_k = torch.chunk(st_feats, self.comps, dim=-1)
        mm_feats_t = self.mm_encoder_t(mm_feats)
        mm_feats_r = self.mm_encoder_r(mm_feats)
        mm_feats_i = self.mm_encoder_i(mm_feats)
        mm_feats_j = self.mm_encoder_j(mm_feats)
        mm_feats_k = self.mm_encoder_k(mm_feats)

        loss_t = self.neibor_mnce(st_feats_t.squeeze(), mm_feats_t)
        loss_r = self.neibor_mnce(st_feats_r.squeeze(), mm_feats_r)
        loss_i = self.neibor_mnce(st_feats_i.squeeze(), mm_feats_i)
        loss_j = self.neibor_mnce(st_feats_j.squeeze(), mm_feats_j)
        loss_k = self.neibor_mnce(st_feats_k.squeeze(), mm_feats_k)

        mm_feats = torch.stack([mm_feats_t, mm_feats_r, mm_feats_i, mm_feats_j, mm_feats_k], dim=-1)
        loss = (loss_t + loss_r + loss_i + loss_j + loss_k) / 5.0
        return mm_feats, loss
    
    def neibor_mnce(self, stru, mmFeat):
        def inner(x, y):
            """
            args:
            ===
            x: [n_samples, dim]
            y: [n_samples, dim]
            """
            return x @ y.T
        
        def cossim(x, y):
            def normalize(x, axis=-1):
                x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
                return x

            return normalize(x) @ normalize(y).T
        
        def l2dist(x, y):
            return torch.cdist(x, y)
        
        mm_distMap = cossim(mmFeat, mmFeat)
        sm_distMap = torch.exp(cossim(stru, mmFeat) / self.tau)
       
        maxSM_idx = torch.argmax(sm_distMap, dim=-1)
        _, maxMM_idx = torch.topk(mm_distMap[maxSM_idx], k=self.topk, dim=-1)
        maxMM_idx = (torch.arange(mm_distMap[maxSM_idx].shape[0]).unsqueeze(-1).expand((-1, self.topk)).to(maxMM_idx.device), maxMM_idx)
        pos_socres = sm_distMap[maxMM_idx]
        
        return (-torch.log(pos_socres.sum(dim=-1) / sm_distMap.sum(dim=-1))).mean()
    
    def nbatchs_neibor_mnce(self, stru, mmFeat, nbatchs=10):

        def cossim(x, y):
            def normalize(x, axis=-1):
                x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
                return x

            return normalize(x) @ normalize(y).T
        
        batch_size = int(self.st_num / nbatchs) + 1
        loss = 0
        ids = torch.arange(self.st_num).to(stru.device)
        b_begin = 0
        while b_begin < stru.shape[0]:
            batch_ids = ids[b_begin:b_begin+batch_size]
            batch_stru = stru[batch_ids]

            sm_distMap = torch.exp(cossim(batch_stru, mmFeat) / self.tau)
            maxSM_idx = torch.argmax(sm_distMap, dim=-1)
            mm_distMap = cossim(mmFeat[maxSM_idx], mmFeat)
            _, maxMM_idx = torch.topk(mm_distMap, k=self.topk, dim=-1)

            maxMM_idx = (torch.arange(batch_ids.shape[0]).unsqueeze(-1).expand((-1, self.topk)).to(maxMM_idx.device), maxMM_idx)
            pos_socres = sm_distMap[maxMM_idx]

            loss += (-torch.log(pos_socres.sum(dim=-1) / sm_distMap.sum(dim=-1))).mean()
            b_begin += batch_size
        
        return loss / float(nbatchs)

class CLFusion2(nn.Module):
    def __init__(self, mm_shape, st_shape, num_pers=6, hidden_dim=32, tau = 0.5, topk=6) -> None:
        super().__init__()

        self.mm_num, self.mm_dim = mm_shape
        self.st_num, self.st_dim, self.comps = st_shape

        self.num_pers = num_pers
        self.hidden_dim = hidden_dim
        self.tau = tau
        self.topk = topk
        self.topk2 = 16

        self.mm_encoder = nn.Sequential(*[
            nn.Linear(self.mm_dim, self.comps * self.st_dim),
            nn.GELU(),
            nn.Linear(self.comps * self.st_dim, self.comps * self.st_dim),
            nn.BatchNorm1d(self.comps * self.st_dim)
        ])

        # self.weights = torch.Tensor(num_pers, self.st_dim)    # bi-graph weights
        # self.weights = nn.Parameter(nn.init.kaiming_normal_(self.weights))

    def forward(self, st_feats, mm_feats):
        mm_feats = self.mm_encoder(mm_feats)
        loss = self.neibor_mnce(st_feats.view(self.st_num, -1), mm_feats)
        return mm_feats.view(self.st_num, -1, self.comps), loss


    def weighted_neibor_mnce(self, st_feats, mm_feats):
        expand_weights = self.weights.unsqueeze(1)
        weighted_mm_feats = mm_feats.unsqueeze(0) * expand_weights 
        sm_distMap = torch.exp(cossim(st_feats, weighted_mm_feats).mean(0) / self.tau)
        mm_distMap = cossim(weighted_mm_feats, weighted_mm_feats).mean(0)

        maxSM_idx = torch.argmax(sm_distMap, dim=-1)
        _, maxMM_idx = torch.topk(mm_distMap[maxSM_idx], k=self.topk, dim=-1)
        maxMM_idx = (torch.arange(mm_distMap[maxSM_idx].shape[0]).unsqueeze(-1).expand((-1, self.topk)).to(maxMM_idx.device), maxMM_idx)
        pos_socres = sm_distMap[maxMM_idx]
        
        return (-torch.log(pos_socres.sum(dim=-1) / sm_distMap.sum(dim=-1))).mean()
    

    def neibor_mnce(self, stru, mmFeat):
        def inner(x, y):
            """
            args:
            ===
            x: [n_samples, dim]
            y: [n_samples, dim]
            """
            return x @ y.T
        
        def cossim(x, y):
            def normalize(x, axis=-1):
                x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
                return x

            return normalize(x) @ normalize(y).T
        
        def l2dist(x, y):
            return torch.cdist(x, y)
        
        sm_distMap = torch.exp(cossim(stru, mmFeat) / self.tau)
        mm_distMap = cossim(mmFeat, mmFeat)
        maxSM_idx = torch.argmax(sm_distMap, dim=-1)
        _, maxMM_idx = torch.topk(mm_distMap[maxSM_idx], k=self.topk, dim=-1)
        maxMM_idx = (torch.arange(mm_distMap[maxSM_idx].shape[0]).unsqueeze(-1).expand((-1, self.topk)).to(maxMM_idx.device), maxMM_idx)
        pos_socres = sm_distMap[maxMM_idx]
        
        return (-torch.log(pos_socres.sum(dim=-1) / sm_distMap.sum(dim=-1))).mean()


class LoretzFusion2(nn.Module):
    def __init__(self, mm_shape, st_shape, num_pers=6, hidden_dim=32, tau = 0.1, topk=6, rand_ratio=1, is_skip=False) -> None:
        super().__init__()  

        """
        mm_shape: [#ent, mm_dim]
        st_shape: [#ent, rank, #comps]
        """
        
        self.mm_num, self.mm_dim = mm_shape
        self.st_num, self.st_dim, self.comps = st_shape

        self.num_pers = num_pers
        self.hidden_dim = hidden_dim
        self.tau = tau
        self.topk = topk
        self.topk2 = 16
        self.rand_ratio = rand_ratio
        self.is_skip = is_skip

        self.mm_encoder_t = nn.Sequential(*[
            nn.Linear(self.mm_dim, self.st_dim),
            nn.GELU(),
            nn.Linear(self.st_dim, self.st_dim),
            nn.BatchNorm1d(self.st_dim)
        ])

        self.mm_encoder_r = nn.Sequential(*[
            nn.Linear(self.mm_dim, self.st_dim),
            nn.GELU(),
            nn.Linear(self.st_dim, self.st_dim),
            nn.BatchNorm1d(self.st_dim)
        ])

        self.mm_encoder_i = nn.Sequential(*[
            nn.Linear(self.mm_dim, self.st_dim),
            nn.GELU(),
            nn.Linear(self.st_dim, self.st_dim),
            nn.BatchNorm1d(self.st_dim)
        ])

        self.mm_encoder_j = nn.Sequential(*[
            nn.Linear(self.mm_dim, self.st_dim),
            nn.GELU(),
            nn.Linear(self.st_dim, self.st_dim),
            nn.BatchNorm1d(self.st_dim)
        ])

        self.mm_encoder_k = nn.Sequential(*[
            nn.Linear(self.mm_dim, self.st_dim),
            nn.GELU(),
            nn.Linear(self.st_dim, self.st_dim),
            nn.BatchNorm1d(self.st_dim)
        ])
    
        self.params_init()


    def params_init(self):
        for name, param in self.named_parameters():
            if str.endswith(name, 'weight'):
                if not '3' in name:
                    nn.init.kaiming_normal_(param)
                else:
                    nn.init.normal_(param)
            elif str.endswith(name, 'bias'):
                nn.init.zeros_(param)


    def forward(self, st_feats, mm_feats):
        """
        st_feats: [#ent, rank, comps]
        mm_feats: [#ent, mm_dim]
        """
        st_feats_t, st_feats_r, st_feats_i, st_feats_j, st_feats_k = torch.chunk(st_feats, self.comps, dim=-1)
        mm_feats_t = self.mm_encoder_t(mm_feats)
        mm_feats_r = self.mm_encoder_r(mm_feats)
        mm_feats_i = self.mm_encoder_i(mm_feats)
        mm_feats_j = self.mm_encoder_j(mm_feats)
        mm_feats_k = self.mm_encoder_k(mm_feats)

        if self.is_skip:
            mm_feats = torch.stack([mm_feats_t, mm_feats_r, mm_feats_i, mm_feats_j, mm_feats_k], dim=-1)
            return mm_feats, 0.0

        seq_cands = (torch.rand(size=(self.mm_num,), device=mm_feats.device) <= self.rand_ratio)

        loss_t = self.neibor_mnce(st_feats_t.squeeze(), mm_feats_t[seq_cands])
        loss_r = self.neibor_mnce(st_feats_r.squeeze(), mm_feats_r[seq_cands])
        loss_i = self.neibor_mnce(st_feats_i.squeeze(), mm_feats_i[seq_cands])
        loss_j = self.neibor_mnce(st_feats_j.squeeze(), mm_feats_j[seq_cands])
        loss_k = self.neibor_mnce(st_feats_k.squeeze(), mm_feats_k[seq_cands])

        mm_feats = torch.stack([mm_feats_t, mm_feats_r, mm_feats_i, mm_feats_j, mm_feats_k], dim=-1)
        loss = (loss_t + loss_r + loss_i + loss_j + loss_k) / 5.0
        return mm_feats, loss
    
    def neibor_mnce(self, stru, mmFeat):
        def inner(x, y):
            """
            args:
            ===
            x: [n_samples, dim]
            y: [n_samples, dim]
            """
            return x @ y.T
        
        def cossim(x, y):
            def normalize(x, axis=-1):
                x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
                return x

            return normalize(x) @ normalize(y).T
        
        def l2dist(x, y):
            def normalize(x, axis=-1):
                x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
                return x
            return torch.cdist(normalize(x), normalize(y))
        
        mm_distMap = cossim(mmFeat, mmFeat)
        sm_distMap = torch.exp(cossim(stru, mmFeat) / self.tau)
       
        maxSM_idx = torch.argmax(sm_distMap, dim=-1)
        _, maxMM_idx = torch.topk(mm_distMap[maxSM_idx], k=self.topk, dim=-1)
        maxMM_idx = (torch.arange(mm_distMap[maxSM_idx].shape[0]).unsqueeze(-1).expand((-1, self.topk)).to(maxMM_idx.device), maxMM_idx)
        pos_socres = sm_distMap[maxMM_idx]
        
        return (-torch.log(pos_socres.sum(dim=-1) / sm_distMap.sum(dim=-1))).mean()