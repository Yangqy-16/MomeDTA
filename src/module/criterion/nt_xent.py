from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig

from coach_pl.configuration import configurable
from coach_pl.criterion import CRITERION_REGISTRY


@CRITERION_REGISTRY.register()
class NTXent(nn.Module):
    """
    A neural network module for computing the normalized temperature-scaled cross-entropy loss (NT-Xent Loss).
    
    This loss is used for contrastive learning, where the goal is to maximize the similarity between positive pairs
    and minimize the similarity between negative pairs.

    Attributes:
        - batch_size: Number of samples in a batch
        - temperature: Temperature scaling factor
        - device: Device to run the computations on (e.g., 'cpu' or 'cuda')
        - softmax: Softmax function for normalizing logits
        - mask_samples_from_same_repr: Mask to filter out positive samples from the similarity matrix
        - similarity_function: Function to compute similarity between representations
        - criterion: Cross-entropy loss function
    """
    @configurable
    def __init__(self, device, batch_size, temperature, use_cosine_similarity):
        """
        Initializes the NTXentLoss module.

        Args:
            device: Device to run the computations on (e.g., 'cpu' or 'cuda')
            batch_size: Number of samples in a batch
            temperature: Temperature scaling factor
            use_cosine_similarity: Whether to use cosine similarity (True) or dot product similarity (False)
        """
        super().__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device
        self.softmax = nn.Softmax(dim=-1)
        self.mask_samples_from_same_repr = self._get_correlated_mask().type(torch.bool)
        self.similarity_function = self._get_similarity_function(use_cosine_similarity)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
    
    @classmethod
    def from_config(cls, cfg: DictConfig) -> dict[str, Any]:
        return {
            "device": cfg.DEVICE,
            "batch_size": cfg.DATALOADER.TRAIN.BATCH_SIZE,
            "temperature": cfg.CRITERION.TEMPERATURE,
            "use_cosine_similarity": cfg.CRITERION.USE_COSINE_SIMILARITY,
        }

    def _get_similarity_function(self, use_cosine_similarity):
        """
        Returns the similarity function based on the use_cosine_similarity flag.

        Args:
            use_cosine_similarity: Whether to use cosine similarity (True) or dot product similarity (False)

        Returns:
            Function: Similarity function
        """
        if use_cosine_similarity:
            self._cosine_similarity = nn.CosineSimilarity(dim=-1)
            return self._cosine_simililarity
        else:
            return self._dot_simililarity

    def _get_correlated_mask(self):
        """
        Creates a mask to filter out positive samples from the similarity matrix.

        Returns:
            torch.Tensor: Mask tensor
        """
        diag = np.eye(2 * self.batch_size)
        l1 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=-self.batch_size)
        l2 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=self.batch_size)
        mask = torch.from_numpy((diag + l1 + l2))
        mask = (1 - mask).type(torch.bool)
        return mask.to(self.device)

    @staticmethod
    def _dot_simililarity(x, y):
        """
        Computes the dot product similarity between two sets of representations.

        Args:
            x: Tensor of shape (N, C)
            y: Tensor of shape (2N, C)

        Returns:
            torch.Tensor: Similarity matrix of shape (N, 2N)
        """
        v = torch.tensordot(x.unsqueeze(1), y.T.unsqueeze(0), dims=2)
        return v

    def _cosine_simililarity(self, x, y):
        """
        Computes the cosine similarity between two sets of representations.

        Args:
            x: Tensor of shape (N, C)
            y: Tensor of shape (2N, C)

        Returns:
            torch.Tensor: Similarity matrix of shape (N, 2N)
        """
        v = self._cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v
    
    def _normalize(self, x, dim=1):
        """
        Normalizes the input tensor along a given dimension.

        Args:
            x: Input tensor
            dim: Dimension to normalize along

        Returns:
            torch.Tensor: Normalized tensor
        """
        return F.normalize(x, dim=dim)

    def forward(self, zis, zjs):
        """
        Forward pass of the NTXentLoss.

        Args:
            zis: Tensor of shape (batch_size, embedding_dim) containing embeddings from one view
            zjs: Tensor of shape (batch_size, embedding_dim) containing embeddings from another view

        Returns:
            torch.Tensor: Computed NT-Xent loss
        """
        zjs = self._normalize(zjs)
        zis = self._normalize(zis)

        representations = torch.cat([zjs, zis], dim=0)

        similarity_matrix = self.similarity_function(representations, representations)

        # filter out the scores from the positive samples
        l_pos = torch.diag(similarity_matrix, self.batch_size)
        r_pos = torch.diag(similarity_matrix, -self.batch_size)
        positives = torch.cat([l_pos, r_pos]).view(2 * self.batch_size, 1)

        negatives = similarity_matrix[self.mask_samples_from_same_repr].view(2 * self.batch_size, -1)

        logits = torch.cat((positives, negatives), dim=1)
        logits /= self.temperature

        labels = torch.zeros(2 * self.batch_size).to(self.device).long()
        loss = self.criterion(logits, labels)

        return loss / (2 * self.batch_size)
