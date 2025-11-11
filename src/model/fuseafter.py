import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.utils.weight_norm import weight_norm
from omegaconf import DictConfig

from .module import *
from .fusion import BANLayer, CAN_Layer, MAN, MANnew
from .basics import Classifier, MultiModalFusion, MLP
from coach_pl.configuration import configurable
from coach_pl.model import MODEL_REGISTRY


@MODEL_REGISTRY.register()
class FusionAfterNet(nn.Module):
    """
    First fuse 1d+2d+3d features, then communicate between drug and prot.
    Can be used only when all drug LLMs' outputs have equal length!
    """
    @configurable
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.project = ProjectNet(cfg)
        self.drug_fusion = MultiModalFusion(modal_num=cfg.MODEL.MODAL_NUM, with_weight=cfg.MODEL.FUSION_WEIGHT)
        self.prot_fusion = MultiModalFusion(modal_num=cfg.MODEL.MODAL_NUM, with_weight=cfg.MODEL.FUSION_WEIGHT)

        if cfg.MODEL.FUSION == 'BAN':
            self.fusion = weight_norm(BANLayer(v_dim=cfg.MODEL.PROJECTION_DIM, q_dim=cfg.MODEL.PROJECTION_DIM, h_dim=cfg.MODEL.PROJECTION_DIM//2, h_out=2), name='h_mat', dim=None)
            self.classifier = Classifier(input_dim=cfg.MODEL.PROJECTION_DIM//2, hidden_dim=cfg.MODEL.PROJECTION_DIM, output_dim=cfg.MODEL.PROJECTION_DIM//2) #
        elif cfg.MODEL.FUSION == 'CAN':
            self.fusion = CAN_Layer(hidden_dim=cfg.MODEL.PROJECTION_DIM, num_heads=8)
            self.classifier = Classifier(input_dim=cfg.MODEL.PROJECTION_DIM*2, hidden_dim=cfg.MODEL.PROJECTION_DIM*2, output_dim=cfg.MODEL.PROJECTION_DIM) #
        elif cfg.MODEL.FUSION == 'MAN':
            self.fusion = MAN(drug_hidden_dim=cfg.MODEL.PROJECTION_DIM, protein_hidden_dim=cfg.MODEL.PROJECTION_DIM)
            self.classifier = Classifier(input_dim=cfg.MODEL.PROJECTION_DIM*2, hidden_dim=cfg.MODEL.PROJECTION_DIM, output_dim=cfg.MODEL.PROJECTION_DIM) #
        elif cfg.MODEL.FUSION == 'MANnew':
            self.fusion = MANnew(drug_hidden_dim=cfg.MODEL.PROJECTION_DIM, protein_hidden_dim=cfg.MODEL.PROJECTION_DIM)
            self.classifier = Classifier(input_dim=cfg.MODEL.PROJECTION_DIM*2, hidden_dim=cfg.MODEL.PROJECTION_DIM, output_dim=cfg.MODEL.PROJECTION_DIM) #

    @classmethod
    def from_config(cls, cfg: DictConfig):
        return {
            "cfg": cfg,
        }

    def forward(self, 
        drug1d_embed: Tensor, drug2d_embed: Tensor, drug3d_embed: Tensor, drug_mask: Tensor,
        prot1d_embed: Tensor, prot2d_embed: Tensor, prot3d_embed: Tensor, prot_mask: Tensor
    ) -> Tensor:
        """
        Args:
            Embeddings (last hidden state) of frozen LLMs, all of size [batch_size, seq_len, embed_dim]
            Mask: all of size [batch_size, seq_len]

        Returns:
            Logits for interaction prediction
        """
        batch_size = drug1d_embed.shape[0]
        drug1d_embed, drug2d_embed, drug3d_embed, prot1d_embed, prot2d_embed, prot3d_embed = self.project(drug1d_embed, drug2d_embed, drug3d_embed, prot1d_embed, prot2d_embed, prot3d_embed)
        
        drug1d_embed = drug1d_embed.view(batch_size, 1, -1)
        drug2d_embed = drug2d_embed.view(batch_size, 1, -1)
        drug3d_embed = drug3d_embed.view(batch_size, 1, -1)
        drug_feat = torch.cat([drug1d_embed, drug2d_embed, drug3d_embed], dim=1)
        drug_feat = self.drug_fusion(drug_feat)

        prot1d_embed = prot1d_embed.view(batch_size, 1, -1)
        prot2d_embed = prot2d_embed.view(batch_size, 1, -1)
        prot3d_embed = prot3d_embed.view(batch_size, 1, -1)
        prot_feat = torch.cat([prot1d_embed, prot2d_embed, prot3d_embed], dim=1)
        prot_feat = self.prot_fusion(prot_feat)
        
        if self.cfg.MODEL.FUSION == 'BAN':
            fusion_features, att = self.fusion(prot_feat, drug_feat)
        elif self.cfg.MODEL.FUSION == 'CAN':
            fusion_features = self.fusion(prot_feat, drug_feat, prot_mask, drug_mask)
        elif self.cfg.MODEL.FUSION == 'MAN':
            c, p = self.fusion(drug_feat.permute(0, 2, 1), prot_feat)
            fusion_features = torch.cat([c, p], dim=-1)
        elif self.cfg.MODEL.FUSION == 'MANnew':
            c, p = self.fusion(drug_feat, prot_feat, drug_mask, prot_mask)
            fusion_features = torch.cat([c, p], dim=-1)

        output = self.classifier(fusion_features)
        output = output.squeeze(-1)
        return output
