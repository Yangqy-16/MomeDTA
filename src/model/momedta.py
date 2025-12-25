import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.utils.weight_norm import weight_norm
from torch_geometric.data import Data
from omegaconf import DictConfig

from .module import *
from .fusion import BANLayer, CAN_Layer, MAN, MANnew
from .basics import MLP, Projector, CNN_1D, Downsample_CNN
from .mgraph import GraphNet
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
        # elif cfg.MODEL.FUSION == 'BCA':
        #     self.att_fuse = BidirectionalCrossAttention(d_model=common_dim, n_heads=4, dim_head=common_dim//4, dropout=0.1, talking_heads=True)
        
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
        # elif self.cfg.MODEL.FUSION == 'BCA':
        #     fused_feat = self.att_fuse(drug_embed, prot_embed, drug_mask, prot_mask)
        
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


class Gating(nn.Module):
    """
    Gating network to generate weights for each expert.
    Written by Doubao AI and checked by authors.
    """
    def __init__(self, num_experts=3, embed_dim=256):
        super().__init__()
        self.num_experts = num_experts
        self.embed_dim = embed_dim
        
        self.embedding_fusion = nn.Sequential(
            nn.Linear(num_experts * embed_dim, embed_dim),
            nn.ReLU(),
            nn.LayerNorm(embed_dim)
        )
        
        self.weight_generator = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * num_experts),
            nn.Unflatten(1, (embed_dim, num_experts))
        )
    
    def forward(self, expert_embeddings: list[Tensor]):
        """
        Args:
            expert_embeddings: list, each element of size [batch_size, embed_dim]
        Returns:
            weights: [batch_size, embed_dim, num_experts]
        """
        fused_emb = torch.cat(expert_embeddings, dim=1)  # [B, 3*E]
        gate_feat = self.embedding_fusion(fused_emb)  # [B, E]
        weights = self.weight_generator(gate_feat)  # [B, E, 3]
        return torch.softmax(weights, dim=2)  # sum of weights = 1


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
        if self.cfg.MODEL.USE_1D:
            f1, att1 = self.expert1d(data['drug1d_embed'], data['drug1d_mask'], data['prot1d_embed'], data['prot1d_mask'])
        if self.cfg.MODEL.USE_2D:
            f2 = self.expert2d(data)  #data['drug2d_embed'], data['prot2d_embed']
        if self.cfg.MODEL.USE_3D:
            f3, att3 = self.expert3d(data['drug3d_embed'], data['drug3d_mask'], data['prot3d_embed'], data['prot3d_mask'])

        expert_embeddings = [f1, f2, f3]
        W = self.gating(expert_embeddings)  # [B, E, 3]
        H = torch.stack(expert_embeddings, dim=1)  # [B, 3, E]
        fused_feat = torch.einsum('bme, bem -> be', H, W)  # [B, E]
        
        logit = self.predict_head(fused_feat).squeeze(-1)  # [B, 1] -> [B]
        return logit, W.clone().detach(), att1.clone().detach(), att3.clone().detach()


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
