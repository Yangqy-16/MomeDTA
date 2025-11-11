import torch
import torch.nn as nn
from torch import Tensor
from omegaconf import DictConfig

from .module import *
from .pretrain import PretrainModel
from coach_pl.configuration import configurable
from coach_pl.model import MODEL_REGISTRY
from src.utils.typedef import *


@MODEL_REGISTRY.register()
class WholePipeline(nn.Module):
    """
    The whole pipeline including LLM, Projector, and DownstreamNet.
    """
    @configurable
    def __init__(self, cfg: DictConfig):
        """
        Similar as PretrainModel
        """
        super().__init__()
        self.pretrain = PretrainModel(cfg)
        self.downstream = DownstreamNet(cfg)

    @classmethod
    def from_config(cls, cfg: DictConfig):
        return {
            "cfg": cfg,
        }
    
    def forward(self, mol1d_batch: ValidType, mol3d_batch: ValidType, prot1d_batch: ValidType, prot3d_batch: ValidType) -> Tensor:
        """
        Args:
            Same as PretrainModel

        Returns:
            Logits for interaction prediction
        """
        mol1d_embed, mol3d_embed, prot1d_embed, prot3d_embed = self.pretrain(mol1d_batch, mol3d_batch, prot1d_batch, prot3d_batch)        
        output = self.downstream(mol1d_embed, mol3d_embed, prot1d_embed, prot3d_embed)
        return output
