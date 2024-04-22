import torch
import tqdm
from torch import nn
from torch import optim
import torch.nn.functional as F

from models import KBCModel
from regularizers import Regularizer
from graph_fusion.fusion_module import GraphFusion
from graph_fusion.graph_learn import get_binarized_kneighbors_graph
from graph_fusion.graph_utils import add_graph_loss, diff


# os.environ['CUDA_VISIBLE_DEVICES'] = device


class KBCOptimizer(object):
    def __init__(
            self, model: KBCModel, regularizer: Regularizer, optimizer: optim.Optimizer, batch_size: int = 256,
            modality_split=True, fusion_img=True, fusion_label=True, fusion_dscp=True,
            verbose: bool = True
    ):
        self.model = model
        self.regularizer = regularizer
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.verbose = verbose
        self.modality_split = modality_split
        self.fusion_img = fusion_img
        self.fusion_label = fusion_label
        self.fusion_dscp = fusion_dscp

    def epoch(self, examples: torch.LongTensor):
        actual_examples = examples[torch.randperm(examples.shape[0]), :]  # 随机数
        # examples shape: torch.Size([966284, 3])
        # actual_examples 是examples的乱序版
        loss = nn.CrossEntropyLoss(reduction='mean')
        with tqdm.tqdm(total=examples.shape[0], unit='ex', disable=not self.verbose, ncols=80) as bar:
            bar.set_description(f'train loss')
            b_begin = 0
            while b_begin < examples.shape[0]:
                input_batch = actual_examples[
                              b_begin:b_begin + self.batch_size
                              ].to(self.model.device)  # [batch, 3]
                truth = input_batch[:, 2]
                # truth shape: 1000
                if self.modality_split:
                    if self.fusion_img and self.fusion_dscp:
                        preds_str, fac_str, \
                        preds_img, fac_img, \
                        preds_dscp, fac_dscp, \
                        cl_loss = self.model.forward(input_batch)
                        # preds shape: 1000 * 14951, N = 1000 batch size, C = 14951 class number
                        l_str_fit = loss(preds_str, truth)
                        l_str_reg = self.regularizer.forward(fac_str)
                        l_img_fit = loss(preds_img, truth)
                        l_img_reg = self.regularizer.forward(fac_img)
                        l_dscp_fit = loss(preds_dscp, truth)
                        l_dscp_reg = self.regularizer.forward(fac_dscp)
                        l = l_str_fit + l_str_reg + l_img_fit + l_img_reg + l_dscp_fit + l_dscp_reg + cl_loss
                    elif self.fusion_img:
                        preds_str, fac_str, preds_img, fac_img, cl_loss = self.model.forward(input_batch)
                        l_str_fit = loss(preds_str, truth)
                        l_str_reg = self.regularizer.forward(fac_str)
                        l_img_fit = loss(preds_img, truth)
                        l_img_reg = self.regularizer.forward(fac_img)
                        l = l_str_fit + l_str_reg + l_img_fit + l_img_reg + cl_loss
                    elif self.fusion_dscp:
                        preds_str, fac_str, preds_dscp, fac_dscp, cl_loss = self.model.forward(input_batch)
                        l_str_fit = loss(preds_str, truth)
                        l_str_reg = self.regularizer.forward(fac_str)
                        l_dscp_fit = loss(preds_dscp, truth)
                        l_dscp_reg = self.regularizer.forward(fac_dscp)
                        l = l_str_fit + l_str_reg + l_dscp_fit + l_dscp_reg + cl_loss
                    else:
                        preds_str, fac_str, cl_loss = self.model.forward(input_batch)
                        l_str_fit = loss(preds_str, truth)
                        l_str_reg = self.regularizer.forward(fac_str)
                        l = l_str_fit + l_str_reg + cl_loss
                else:
                    preds_str, fac_str, cl_loss = self.model.forward(input_batch)
                    l_str_fit = loss(preds_str, truth)
                    l_str_reg = self.regularizer.forward(fac_str)
                    l = l_str_fit + l_str_reg + cl_loss

                self.optimizer.zero_grad()
                l.backward()
                # try:
                #     for name, weight in self.model.named_parameters():
                #         if weight.requires_grad:
                #             print(weight.grad.mean(), weight.grad.min(), weight.grad.max())
                #             print(name)
                #             if torch.isnan(weight.grad.mean()):
                #                 print(input_batch)
                #
                # except Exception as e:
                #     pass

                self.optimizer.step()
                b_begin += self.batch_size
                bar.update(input_batch.shape[0])
                bar.set_postfix(loss=f'{l.item():.0f}')
        return l


class MMKBCOptimizer(object):
    def __init__(self, model, regularizer: Regularizer, optimizer: optim.Optimizer, 
                 batch_size: int = 256, verbose: bool = True
    ) -> None:
        self.model = model
        self.regularizer = regularizer
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.verbose = verbose
        self.loss = nn.CrossEntropyLoss(reduction='mean')

    
    def epoch(self, examples: torch.LongTensor):
        # 每一个mini-batch，先 graph learner 去学习一轮，然后 kge算一个loss出来

        actual_examples = examples[torch.randperm(examples.shape[0]), :]  # 随机数
        # examples shape: torch.Size([966284, 3])
        # actual_examples 是examples的乱序版
        
        with tqdm.tqdm(total=examples.shape[0], unit='ex', disable=not self.verbose, ncols=80) as bar:
            bar.set_description(f'train loss')
            b_begin = 0
            while b_begin < examples.shape[0]:
                input_batch = actual_examples[
                              b_begin:b_begin + self.batch_size
                              ].cuda()  # [batch, 3]
                truth = input_batch[:, 2]
                # truth shape: 1000
                ####################################### forward ####################################
                # create init embeddings and adj
                visu_shape, ling_shape = self.model.visu_embeds.shape, self.model.ling_embeds
                vs_embeddings = torch.cat([
                    self.model.visu_linear(self.model.visu_embeds), 
                    self.model.ent_embeds.weight
                ], dim=0) # [v; s]
                ls_embeddings = torch.cat([
                    self.model.ling_linear(self.model.ling_embeds),
                    self.model.ent_embeds.weight
                ]) # [l; s]

                vs_init_adj = get_binarized_kneighbors_graph(vs_embeddings, topk=self.model.vs_topk)
                ls_init_adj = get_binarized_kneighbors_graph(ls_embeddings, self.model.ls_topk)

                init_loss = 0
                #### Init VS GL & GNN
                vs_embeddings = F.dropout(vs_embeddings, self.model.visu_graph_fusion.feat_adj_dropout, training=self.model.training)
                vs_init_node_vec = vs_embeddings
                vs_cur_raw_adj, vs_cur_adj = self.model.visu_graph_fusion.learn_graph(self.model.visu_graph_fusion.graph_learner, vs_init_node_vec, 
                                                                                      graph_skip_conn = self.model.visu_graph_fusion.graph_skip_conn,
                                                                                      graph_include_self=self.model.visu_graph_fusion.graph_include_self,
                                                                                      init_adj=vs_init_adj)
                vs_cur_raw_adj = F.dropout(vs_cur_raw_adj, self.model.visu_graph_fusion.feat_adj_dropout, training=self.model.training)
                vs_cur_adj = F.dropout(vs_cur_adj, self.model.visu_graph_fusion.feat_adj_dropout, training=self.model.training)

                vs_node_vec = torch.relu(self.model.visu_graph_fusion.gcn.graph_encoders[0](vs_init_node_vec, vs_cur_adj))
                vs_node_vec = F.dropout(vs_node_vec, self.model.visu_graph_fusion.gcn.dropout, training=self.model.training)
                # Add mid GNN layers
                for encoder in self.model.visu_graph_fusion.gcn.graph_encoders[1:-1]:
                    vs_node_vec = torch.relu(encoder(vs_node_vec, vs_cur_adj))
                    vs_node_vec = F.dropout(vs_node_vec, self.model.visu_graph_fusion.gcn.dropout, training=self.model.training)
                # BP to update weights
                vs_fusion_embeds = self.model.visu_graph_fusion.gcn.graph_encoders[-1](vs_node_vec, vs_cur_adj)
                init_loss += add_graph_loss(vs_cur_raw_adj, vs_init_node_vec, 
                                            smooth_ratio=self.model.vs_smooth_ratio, 
                                            degree_ratio=self.model.vs_degree_ratio,
                                            sparsity_ratio=self.model.vs_sparsity_ratio)

                #### Init LS GL & GNN
                ls_embeddings = F.dropout(ls_embeddings, self.model.ling_graph_fusion.feat_adj_dropout, training=self.model.training)
                ls_init_node_vec = ls_embeddings
                ls_cur_raw_adj, ls_cur_adj = self.model.ling_graph_fusion.learn_graph(self.model.ling_graph_fusion.graph_learner, ls_init_node_vec,
                                                                                      graph_skip_conn = self.model.ling_graph_fusion.graph_skip_conn,
                                                                                      graph_include_self=self.model.ling_graph_fusion.graph_include_self,
                                                                                      init_adj=ls_init_adj)
                ls_cur_raw_adj = F.dropout(ls_cur_raw_adj, self.model.ling_graph_fusion.feat_adj_dropout, training=self.model.training)
                ls_cur_adj = F.dropout(ls_cur_adj, self.model.ling_graph_fusion.feat_adj_dropout, training=self.model.training)

                ls_node_vec = torch.relu(self.model.ling_graph_fusion.gcn.graph_encoders[0](ls_init_node_vec, ls_cur_adj))
                ls_node_vec = F.dropout(ls_node_vec, self.model.ling_graph_fusion.gcn.dropout, training=self.model.training)
                # Add mid GNN layers
                for encoder in self.model.ling_graph_fusion.gcn.graph_encoders[1:-1]:
                    ls_node_vec = torch.relu(encoder(ls_node_vec, ls_cur_adj))
                    ls_node_vec = F.dropout(ls_node_vec, self.model.ling_graph_fusion.gcn.dropout, training=self.model. training)
                # BP to update weights
                ls_fusion_embeds = self.model.ling_graph_fusion.gcn.graph_encoders[-1](ls_node_vec, ls_cur_adj)
                init_loss += add_graph_loss(ls_cur_raw_adj, ls_init_node_vec)


                init_loss += self.cal_kge_loss(input_batch, self.model.ent_embeds.weight, vs_fusion_embeds[:visu_shape[0]], ls_fusion_embeds[:ling_shape[0]])

                # While loop optim for VS and LS
                vs_pre_raw_adj = vs_cur_raw_adj
                ls_pre_raw_adj = ls_cur_raw_adj
                vs_first_raw_adj, vs_first_adj = vs_cur_raw_adj, vs_cur_adj
                ls_first_raw_adj, ls_first_adj = ls_cur_raw_adj, ls_cur_adj

                iter_ = 0
                eps_adj = 1e-7
                loop_loss = 0
                while (iter_ == 0 or \
                      (diff(vs_cur_raw_adj, vs_pre_raw_adj, vs_first_raw_adj).item() > eps_adj)  and \
                      (diff(ls_cur_raw_adj, ls_pre_raw_adj, ls_first_raw_adj).item() > eps_adj)) and \
                      iter_ < self.model.max_iter_:
                    iter_ += 1
                    #### VS part ####
                    vs_pre_adj, vs_pre_raw_adj = vs_cur_adj, vs_cur_raw_adj
                    vs_cur_raw_adj, vs_cur_adj = self.model.visu_graph_fusion.learn_graph(self.model.visu_graph_fusion.graph_learner2, vs_node_vec, 
                                                                                          graph_skip_conn = self.model.visu_graph_fusion.graph_skip_conn,
                                                                                          graph_include_self=self.model.visu_graph_fusion.graph_include_self,
                                                                                          init_adj=vs_init_adj)
                    
                    vs_cur_adj = self.model.vs_update_adj_ratio * vs_cur_adj + (1 - self.model.vs_update_adj_ratio) * vs_first_adj
                    vs_node_vec = torch.relu(self.model.visu_graph_fusion.gcn.graph_encoders[0](vs_init_node_vec, vs_cur_adj))
                    vs_node_vec = F.dropout(vs_node_vec, self.model.vs_gl_dropout, training=self.model.training)
                    # Add mid GNN layers
                    for encoder in self.model.visu_graph_fusion.gcn.graph_encoders[1:-1]:
                        vs_node_vec = torch.relu(encoder(vs_node_vec, vs_cur_adj))
                        vs_node_vec = F.dropout(vs_node_vec, self.model.vs_gl_dropout, training=self.model.training)
                    # BP to update weights
                    vs_fusion_embeds = self.model.visu_graph_fusion.gcn.graph_encoders[-1](vs_node_vec, vs_cur_adj)
                    loop_loss += add_graph_loss(vs_cur_raw_adj, vs_init_node_vec)

                    #### LS part ####
                    ls_pre_adj, ls_pre_raw_adj = ls_cur_adj, ls_cur_raw_adj
                    ls_cur_raw_adj, ls_cur_adj = self.model.ling_graph_fusion.learn_graph(self.model.ling_graph_fusion.graph_learner2, ls_node_vec,
                                                                                          graph_skip_conn=self.model.ling_graph_fusion.graph_skip_conn,
                                                                                          graph_include_self=self.model.ling_graph_fusion.graph_include_self,
                                                                                          init_adj=ls_init_adj)
                    ls_cur_adj = self.model.ls_update_adj_ratio * ls_cur_adj + (1 - self.model.ls_update_adj_ratio) * ls_first_adj
                    ls_node_vec = torch.relu(self.model.ling_graph_fusion.gcn.graph_encoders[0](ls_init_node_vec, ls_cur_adj))
                    ls_node_vec = F.dropout(ls_node_vec, self.model.ls_gl_dropout, training=self.model.training)
                    # Add mid GNN layers
                    for encoder in self.model.ling_graph_fusion.gcn.graph_encoders[1:-1]:
                        ls_node_vec = torch.relu(encoder(ls_node_vec, ls_cur_adj))
                        ls_node_vec = F.dropout(ls_node_vec, self.model.ls_gl_dropout, training=self.model.training)
                    # BP to update weights
                    ls_fusion_embeds = self.model.ling_graph_fusion.gcn.graph_encoders[-1](ls_node_vec, ls_cur_adj)
                    loop_loss += add_graph_loss(ls_cur_raw_adj, ls_init_node_vec)

                    loop_loss += self.cal_kge_loss(input_batch, self.model.ent_embeds.weight, vs_fusion_embeds[:visu_shape[0]], ls_fusion_embeds[:ling_shape[0]])

                if iter_ > 0:
                    loss = loop_loss / iter_ + init_loss
                else:
                    loss = init_loss

                self.optimizer.zero_grad()
                loss.backward()    # 这个是反传，前面都是在做前向传播，计算loss
                self.optimizer.step()

                b_begin += self.batch_size
                bar.update(input_batch.shape[0])
                bar.set_postfix(loss=f'{loss.item():.0f}')
        return loss

    def cal_kge_loss(self, input_batch, stru_embeddings, vs_embeddings, ls_embeddings):
        truth = input_batch[:, 2]

        score_stru, factors_stru, \
        score_visu, factors_visu, \
        score_ling, factors_ling = self.model.KGE_predict(input_batch, stru_embeddings, vs_embeddings, ls_embeddings)
        l_stru_fit = self.loss(score_stru, truth)
        l_stru_reg = self.regularizer.forward(factors_stru)
        l_visu_fit = self.loss(score_visu, truth)
        l_visu_reg = self.regularizer.forward(factors_visu)
        l_ling_fit = self.loss(score_ling, truth)
        l_ling_reg = self.regularizer.forward(factors_ling)

        return l_stru_fit + l_stru_reg + l_visu_fit + l_visu_reg + l_ling_fit + l_ling_reg
