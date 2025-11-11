from typing import Any

import torch
import torch.nn as nn
from torch import Tensor
from omegaconf import DictConfig

from .module import LLM, ProjectNet
from coach_pl.configuration import configurable
from coach_pl.model import MODEL_REGISTRY
from src.utils.typedef import ValidType


@MODEL_REGISTRY.register()
class PretrainModel(nn.Module):
    """
    A neural network module for drug-target interaction prediction using both 1D and 3D features.
    
    This model processes both molecule and protein data through separate encoders and projectors
    to generate combined features for interaction prediction.
    """
    @configurable
    def __init__(self, cfg: DictConfig):
        """
        Initializes the Model with LLM encoders and projectors for both molecules and proteins.

        Args:
            cfg: Configuration object containing model parameters and paths
        """
        super().__init__()

        self.LLM = LLM(cfg)
        self.projector = ProjectNet(cfg)

    @classmethod
    def from_config(cls, cfg: DictConfig) -> dict[str, Any]:
        return {
            "cfg": cfg,
        }

    def forward(self, mol1d_batch: ValidType, mol3d_batch: ValidType, prot1d_batch: ValidType, prot3d_batch: ValidType) -> tuple[Tensor, Tensor]:
        """
        Forward pass of the model processing both molecule and protein data.

        Args:
            mol1d_batch: Batch of 1D molecular data
            mol3d_batch: Batch of 3D molecular data
            prot1d_batch: Batch of 1D protein data
            prot3d_batch: Batch of 3D protein data

        Returns:
            tuple: (dti1d_features, dti3d_features) Combined features for contrastive learning
        """
        # encode and project
        mol1d_embed, mol3d_embed, prot1d_embed, prot3d_embed = self.LLM(mol1d_batch, mol3d_batch, prot1d_batch, prot3d_batch)
        mol1d_embed, mol3d_embed, prot1d_embed, prot3d_embed = self.projector(mol1d_embed, mol3d_embed, prot1d_embed, prot3d_embed)

        # combine features
        # dti1d_features = torch.cat([mol1d_embed, prot1d_embed], dim=-1)
        # dti3d_features = torch.cat([mol3d_embed, prot3d_embed], dim=-1)

        return mol1d_embed, mol3d_embed, prot1d_embed, prot3d_embed
