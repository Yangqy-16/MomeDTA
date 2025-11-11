import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.utils.weight_norm import weight_norm
from torch_geometric.data import Data
from omegaconf import DictConfig

from .module import *
from .fusion import BANLayer, CAN_Layer, MAN, MANnew, BidirectionalCrossAttention, MultiModalAttentionalPooler
from .basics import MultiModalFusion, MLP, Projector, CNN_1D, Downsample_CNN
from .mgraph import GraphNet
from ..utils.metrics import get_cindex, get_rm2
from coach_pl.configuration import configurable
from coach_pl.model import MODEL_REGISTRY


class FusePair(nn.Module):
    """ Fuse a single pair of drug and prot embedding (same modal) """
    def __init__(self, cfg: DictConfig, drug_dim: int, prot_dim: int, predict: bool = False):#
        super().__init__()
        self.cfg = cfg
        self.predict = predict

        common_dim = cfg.MODEL.PROJECTION_DIM
        if cfg.MODEL.ENCODER == 'linear':
            self.drug_projector = Projector(input_dim=drug_dim, out_dim=common_dim, activation_fn=cfg.MODEL.ACTIVATION_FN)
            self.prot_projector = Projector(input_dim=prot_dim, out_dim=common_dim, activation_fn=cfg.MODEL.ACTIVATION_FN)
        elif cfg.MODEL.ENCODER == '1d-cnn':
            self.drug_projector = CNN_1D(input_dim=drug_dim, hidden_dim=common_dim, max_len=cfg.DATASET.DRUG_MAX_LEN)
            self.prot_projector = CNN_1D(input_dim=prot_dim, hidden_dim=common_dim, max_len=cfg.DATASET.PROT_MAX_LEN)
        elif cfg.MODEL.ENCODER == 'ds-cnn':
            self.drug_projector = Downsample_CNN(input_dim=drug_dim, output_dim=common_dim)
            self.prot_projector = Downsample_CNN(input_dim=prot_dim, output_dim=common_dim)

        if cfg.MODEL.FUSION == 'BAN':
            self.att_fuse = weight_norm(BANLayer(v_dim=common_dim, q_dim=common_dim, h_dim=common_dim*2, h_out=2), name='h_mat', dim=None)
        elif cfg.MODEL.FUSION == 'CAN':
            self.att_fuse = CAN_Layer(hidden_dim=common_dim, num_heads=8)
        elif cfg.MODEL.FUSION == 'MAN':
            self.att_fuse = MAN(drug_hidden_dim=common_dim, protein_hidden_dim=common_dim)
        elif cfg.MODEL.FUSION == 'MANnew':
            self.att_fuse = MANnew(drug_hidden_dim=common_dim, protein_hidden_dim=common_dim)
        elif cfg.MODEL.FUSION == 'BCA':
            self.att_fuse = BidirectionalCrossAttention(d_model=common_dim, n_heads=4, dim_head=common_dim//4, dropout=0.1, talking_heads=True)
        
        if cfg.MODEL.ADD_POOL:
            self.bn = nn.LayerNorm(1024)
            self.linear_pre = nn.Sequential(nn.Linear(common_dim*2, 1024), nn.GELU())
            self.linear_post = nn.Sequential(nn.Linear(common_dim*2, 1024), nn.GELU())
            # self.mlp_pred = nn.Sequential(nn.Linear(1024, 512), nn.ELU(), nn.Linear(512, common_dim))
            self.mlp_pred = nn.Sequential(nn.Linear(1024, common_dim*2)) #, nn.ELU()
        
        if predict:
            self.head = MLP(input_dim=common_dim*2)
    
    def forward(self, drug_embed: Tensor, drug_mask: Tensor, prot_embed: Tensor, prot_mask: Tensor) -> Tensor:
        """
        Args:
            Embeddings (last hidden state) of frozen LLMs, all of size [batch_size, seq_len, embed_dim]
            Mask: all of size [batch_size, seq_len]

        Returns:
            Fused feature of size [batch_size, 1, projection_dim*2]
        """
        # batch_size = drug_embed.shape[0]

        if self.cfg.MODEL.ENCODER == '1d-cnn':
            drug_embed, drug_pool = self.drug_projector(drug_embed)
            prot_embed, prot_pool = self.prot_projector(prot_embed)
        else:
            drug_embed = self.drug_projector(drug_embed)
            prot_embed = self.prot_projector(prot_embed)
        
        att_map = None
        if self.cfg.MODEL.FUSION == 'BAN':
            fused_feat, att_map = self.att_fuse(drug_embed, prot_embed)  #, softmax=True
        elif self.cfg.MODEL.FUSION == 'CAN':
            fused_feat = self.att_fuse(prot_embed, drug_embed, prot_mask, drug_mask)
        elif self.cfg.MODEL.FUSION == 'MAN':
            c, p = self.att_fuse(drug_embed, prot_embed)
            fused_feat = torch.cat([c, p], dim=-1)
        elif self.cfg.MODEL.FUSION == 'MANnew':
            c, p, att_map = self.att_fuse(drug_embed, prot_embed, drug_mask, prot_mask) #
            fused_feat = torch.cat([c, p], dim=-1)
        elif self.cfg.MODEL.FUSION == 'BCA':
            fused_feat = self.att_fuse(drug_embed, prot_embed, drug_mask, prot_mask)
        
        if self.cfg.MODEL.ADD_POOL:
            h_pre = self.bn(self.linear_pre(torch.cat([drug_pool, prot_pool], dim=-1)))
            h_post = self.linear_post(fused_feat)
            fused_feat = self.mlp_pred(h_pre + h_post) #

        # fused_feat = fused_feat.view(batch_size, 1, -1)
        if self.predict:
            logits = self.head(fused_feat).squeeze(-1)
            return fused_feat, logits
        else:
            return fused_feat, att_map


class Wrapper(nn.Module):
    @classmethod
    def from_config(cls, cfg: DictConfig):
        if cfg.MODEL.USE_1D and not cfg.MODEL.USE_3D:
            drug_dim = cfg.MODEL.DRUG_1D_DIM
            prot_dim = cfg.MODEL.PROT_1D_DIM
        elif cfg.MODEL.USE_3D and not cfg.MODEL.USE_1D:
            drug_dim = cfg.MODEL.DRUG_3D_DIM
            prot_dim = cfg.MODEL.PROT_3D_DIM
        else:
            raise ValueError('Incorrect situation to construct this class from config!')
    
        return {
            "cfg": cfg,
            "drug_dim": drug_dim,
            "prot_dim": prot_dim,
            "predict": True,
        }


@MODEL_REGISTRY.register()
class MomeDTA(nn.Module):
    @configurable
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg

        if cfg.MODEL.USE_1D:
            self.fusion1d = FusePair(cfg, cfg.MODEL.DRUG_1D_DIM, cfg.MODEL.PROT_1D_DIM)
        if cfg.MODEL.USE_2D:
            self.fusion2d = FusePair(cfg, cfg.MODEL.DRUG_2D_DIM, cfg.MODEL.PROT_2D_DIM)
        if cfg.MODEL.USE_3D:
            self.fusion3d = FusePair(cfg, cfg.MODEL.DRUG_3D_DIM, cfg.MODEL.PROT_3D_DIM)

        self.modal_fusion = MultiModalFusion(modal_num=cfg.MODEL.USE_1D+cfg.MODEL.USE_2D+cfg.MODEL.USE_3D)

        common_dim = cfg.MODEL.PROJECTION_DIM
        self.classifier = MLP(input_dim=common_dim*2) #, hidden_dim=common_dim, output_dim=common_dim

    @classmethod
    def from_config(cls, cfg: DictConfig):
        return {
            "cfg": cfg,
        }

    def forward(self, data: dict[str, Tensor]) -> Tensor:
        batch_size = len(list(data.values())[0])

        multimodal_feats = []  #emb1d.clone().detach(), emb2d.clone().detach(), emb3d.clone().detach()
        if self.cfg.MODEL.USE_1D:
            multimodal_feats.append(self.fusion1d(data['drug1d_embed'], data['drug1d_mask'], data['prot1d_embed'], data['prot1d_mask']).view(batch_size, 1, -1))
        if self.cfg.MODEL.USE_2D:
            multimodal_feats.append(self.fusion2d(data['drug2d_embed'], data['drug2d_mask'], data['prot2d_embed'], data['prot2d_mask']).view(batch_size, 1, -1))
        if self.cfg.MODEL.USE_3D:
            multimodal_feats.append(self.fusion3d(data['drug3d_embed'], data['drug3d_mask'], data['prot3d_embed'], data['prot3d_mask']).view(batch_size, 1, -1))
        
        cloned_mm_feats = [i.clone().detach() for i in multimodal_feats]

        dti_features = torch.cat(multimodal_feats, dim=1)
        fusion_features = self.modal_fusion(dti_features)

        output = self.classifier(fusion_features)
        output = output.squeeze(-1)
        return output, cloned_mm_feats


def cal_metric(results: Tensor, truths: Tensor):
    test_preds = results.view(-1).cpu().detach().numpy()
    test_truth = truths.view(-1).cpu().detach().numpy()

    acc = get_cindex(test_truth, test_preds)
    return acc


class ClassifierGuided(nn.Module):
    def __init__(self, proj_dim, num_mod):
        super(ClassifierGuided, self).__init__()
        # Classifiers
        self.num_mod = num_mod
        self.classifers = nn.ModuleList([
            MLP(input_dim=proj_dim)
            for _ in range(self.num_mod)
        ])

    def cal_coeff(self, y: Tensor, cls_res: list[Tensor]) -> list[float]:
        acc_list = list()
        for r in cls_res:
            acc = cal_metric(r, y)
            acc_list.append(acc)
        return acc_list

    def forward(self, x: list[Tensor]) -> list[Tensor]:
        self.cls_res = list()
        for i in range(len(x)):
            self.cls_res.append(self.classifers[i](x[i].squeeze(1)))
        return self.cls_res


def masked_mean(embedding: Tensor, mask: Tensor) -> Tensor:
    # embedding形状: [batch_size, seq_len, embed_dim]
    # mask形状: [batch_size, seq_len]，有效部分为1，无效部分为0
    
    # 将mask扩展到与embedding相同的维度
    mask = mask.unsqueeze(-1).expand_as(embedding)  # [batch_size, seq_len, embed_dim]
    
    # 计算有效元素的总和
    sum_embedding = torch.sum(embedding * mask, dim=1)  # [batch_size, embed_dim]
    
    # 计算每个样本的有效元素数量
    valid_counts = torch.sum(mask, dim=1)  # [batch_size, embed_dim]
    
    # 计算平均值，避免除以0
    mean_embedding = sum_embedding / (valid_counts + 1e-10)  # [batch_size, embed_dim]
    
    return mean_embedding


class Gating(nn.Module):
    def __init__(self, num_experts=3, embed_dim=256):
        super().__init__()
        self.num_experts = num_experts
        self.embed_dim = embed_dim
        
        # 1. 融合3个专家的嵌入特征（用于生成权重）
        self.embedding_fusion = nn.Sequential(
            # 输入为3个嵌入的拼接：[batch_size, 3*embed_dim]
            nn.Linear(num_experts * embed_dim, embed_dim),
            nn.ReLU(),
            nn.LayerNorm(embed_dim)  # 稳定训练
        )
        
        # 2. 生成维度级权重：[batch_size, embed_dim, num_experts]
        self.weight_generator = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * num_experts),
            nn.Unflatten(1, (embed_dim, num_experts))  # 拆分维度
        )
    
    def forward(self, expert_embeddings: list[Tensor]):
        """
        Args:
            expert_embeddings: 列表, 包含3个专家的嵌入, 每个形状为[batch_size, embed_dim]
        Returns:
            weights: 维度级权重，形状为[batch_size, embed_dim, num_experts]
        """
        # 拼接3个专家的嵌入：[batch_size, 3*embed_dim]
        # h1, h2, h3 = expert_embeddings
        fused_emb = torch.cat(expert_embeddings, dim=1)  # [B, 3*E]
        
        # 融合嵌入特征
        gate_feat = self.embedding_fusion(fused_emb)  # [B, E]
        
        # 生成并归一化权重
        weights = self.weight_generator(gate_feat)  # [B, E, 3]
        return torch.softmax(weights, dim=2)  # 每个维度上权重和为1


@MODEL_REGISTRY.register()
class MoE(nn.Module):
    @configurable
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.cfg = cfg

        if cfg.MODEL.USE_1D:
            self.expert1d = FusePair(cfg, cfg.MODEL.DRUG_1D_DIM, cfg.MODEL.PROT_1D_DIM, predict=False)#
        if cfg.MODEL.USE_2D:
            self.expert2d = GraphNet(cfg)
        if cfg.MODEL.USE_3D:
            self.expert3d = FusePair(cfg, cfg.MODEL.DRUG_3D_DIM, cfg.MODEL.PROT_3D_DIM, predict=False)#

        embed_dim = cfg.MODEL.PROJECTION_DIM
        self.gating = Gating(num_experts=(cfg.MODEL.USE_1D+cfg.MODEL.USE_2D+cfg.MODEL.USE_3D), embed_dim=embed_dim*2)

        self.predict_head = MLP(input_dim=embed_dim*2) #, hidden_dims=[embed_dim]
    
    @classmethod
    def from_config(cls, cfg: DictConfig):
        return {
            "cfg": cfg,
        }
    
    def forward(self, data: dict[str, Tensor | Data]) -> Tensor:
        # 1. 专家输出嵌入
        expert_embeddings = []
        if self.cfg.MODEL.USE_1D:
            f1, att1 = self.expert1d(data['drug1d_embed'], data['drug1d_mask'], data['prot1d_embed'], data['prot1d_mask'])
            expert_embeddings.append(f1)
        if self.cfg.MODEL.USE_2D:
            f2 = self.expert2d(data)  #data['drug2d_embed'], data['prot2d_embed']
            expert_embeddings.append(f2)
        if self.cfg.MODEL.USE_3D:
            f3, att3 = self.expert3d(data['drug3d_embed'], data['drug3d_mask'], data['prot3d_embed'], data['prot3d_mask'])
            expert_embeddings.append(f3)
        # expert_embeddings = [f1, f2, f3]
        
        # 2. 门控生成维度级权重（基于专家嵌入）
        W = self.gating(expert_embeddings)  # [B, E, 3]
        
        # 3. 堆叠专家嵌入：[batch_size, num_experts, embed_dim]
        H = torch.stack(expert_embeddings, dim=1)  # [B, 3, E]
        
        # 4. 融合特征：[B, 3, E] × [B, E, 3] → [B, E]
        fused_feat = torch.einsum('bme, bem -> be', H, W)  # [B, E]
        
        # 5. 最终预测
        logit = self.predict_head(fused_feat).squeeze(-1)  # [B, 1] -> [B]
        return logit, W.clone().detach(), att1.clone().detach(), att3.clone().detach() #.clone().detach()


@MODEL_REGISTRY.register()
class Ablation(nn.Module):
    @configurable
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.cfg = cfg

        if cfg.MODEL.USE_1D:
            self.model = FusePair(cfg, cfg.MODEL.DRUG_1D_DIM, cfg.MODEL.PROT_1D_DIM, predict=False)#
        elif cfg.MODEL.USE_2D:
            self.model = GraphNet(cfg)
        elif cfg.MODEL.USE_3D:
            self.model = FusePair(cfg, cfg.MODEL.DRUG_3D_DIM, cfg.MODEL.PROT_3D_DIM, predict=False)#

        embed_dim = cfg.MODEL.PROJECTION_DIM
        self.predict_head = MLP(input_dim=embed_dim*2) #, hidden_dims=[embed_dim]
    
    @classmethod
    def from_config(cls, cfg: DictConfig):
        return {
            "cfg": cfg,
        }
    
    def forward(self, data: dict[str, Tensor | Data]) -> Tensor:
        if self.cfg.MODEL.USE_1D:
            f = self.model(data['drug1d_embed'], data['drug1d_mask'], data['prot1d_embed'], data['prot1d_mask'])
        elif self.cfg.MODEL.USE_2D:
            f = self.model(data)  #data['drug2d_embed'], data['prot2d_embed']
        elif self.cfg.MODEL.USE_3D:
            f = self.model(data['drug3d_embed'], data['drug3d_mask'], data['prot3d_embed'], data['prot3d_mask'])

        logit = self.predict_head(f).squeeze(-1)  # [B, 1] -> [B]
        return logit
