import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.utils.weight_norm import weight_norm
from omegaconf import DictConfig

from .module import *
from .fusion import BANLayer, CAN_Layer, MAN, MANnew
from .basics import MultiModalFusion, MLP
from coach_pl.configuration import configurable
from coach_pl.model import MODEL_REGISTRY


@MODEL_REGISTRY.register()
class Frozen4Downstream(nn.Module):
    """
    This model is specially for Frozen LLMs in downstream prediction.
    """
    @configurable
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.project = ProjectNet(cfg)
        self.downstream = DownstreamNet(cfg)

    @classmethod
    def from_config(cls, cfg: DictConfig):
        return {
            "cfg": cfg,
        }
    
    def forward(self, drug1d_embed: Tensor, drug3d_embed: Tensor, prot1d_embed: Tensor, prot3d_embed: Tensor) -> Tensor:
        """
        Args:
            Embeddings (last hidden state) of frozen LLMs

        Returns:
            Logits for interaction prediction
        """
        drug1d_embed, drug3d_embed, prot1d_embed, prot3d_embed = self.project(drug1d_embed, drug3d_embed, prot1d_embed, prot3d_embed)
        output = self.downstream(drug1d_embed, drug3d_embed, prot1d_embed, prot3d_embed)
        return output
