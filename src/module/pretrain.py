from collections import defaultdict
from typing import Any

import time
import torch
import torch.nn as nn
from torch import Tensor
from torch.optim.optimizer import Optimizer
from torch.optim import lr_scheduler, Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau, MultiStepLR
import pytorch_lightning as pl
from pytorch_lightning.trainer.states import RunningStage
from omegaconf import DictConfig

from coach_pl.configuration import configurable
from coach_pl.model import build_model
from coach_pl.criterion import build_criterion
from coach_pl.module import MODULE_REGISTRY
from .criterion import *
from src.utils.scheduler import LinearWarmupCosineAnnealingLR


@MODULE_REGISTRY.register()
class PretrainModule(pl.LightningModule):
    @configurable
    def __init__(self, model: nn.Module, criterion: torch.nn.Module, cfg: DictConfig) -> None:
        super().__init__()

        self.cfg = cfg
        self.model = torch.compile(model) if cfg.MODULE.COMPILE else model
        self.criterion = torch.compile(criterion) if cfg.MODULE.COMPILE else criterion
        self.monitor = cfg.TRAINER.CHECKPOINT.MONITOR
        self.save_hyperparameters(cfg)

    @classmethod
    def from_config(cls, cfg: DictConfig) -> dict[str, Any]:
        return {
            "model": build_model(cfg),
            "criterion": build_criterion(cfg),
            "cfg": cfg,
        }

    def configure_gradient_clipping(self, optimizer: Optimizer, gradient_clip_val: int | float | None = None, gradient_clip_algorithm: str | None = None) -> None:
        if gradient_clip_algorithm == "value":
            nn.utils.clip_grad.clip_grad_value_(self.model.parameters(), gradient_clip_val)
        elif gradient_clip_algorithm == "norm":
            nn.utils.clip_grad.clip_grad_norm_(self.model.parameters(), max_norm=gradient_clip_val)

    def configure_optimizers(self) -> dict[str, Optimizer | dict[str, str | Optimizer]]:
        optimizer = Adam(
            self.model.parameters(), 
            lr=self.cfg.MODULE.OPTIMIZER.LR, 
            betas=(self.cfg.MODULE.OPTIMIZER.BETA1, 
                   self.cfg.MODULE.OPTIMIZER.BETA2), 
            weight_decay=self.cfg.MODULE.OPTIMIZER.WEIGHT_DECAY
        )

        if self.cfg.MODULE.SCHEDULER.NAME == 'LinearWarmupCosineAnnealingLR':
            scheduler = LinearWarmupCosineAnnealingLR(
                optimizer, 
                warmup_epochs=self.cfg.MODULE.SCHEDULER.WARMUP_EPOCH, 
                max_epochs=self.cfg.TRAINER.MAX_EPOCHS, 
                warmup_start_lr=self.cfg.MODULE.SCHEDULER.WARMUP_START_LR, 
                eta_min=self.cfg.MODULE.SCHEDULER.MIN_LR, 
                last_epoch=-1
            )
        elif self.cfg.MODULE.SCHEDULER.NAME == 'ReduceLROnPlateau':
            scheduler = ReduceLROnPlateau(
                optimizer,
                mode=self.cfg.TRAINER.CHECKPOINT.MONITOR_MODE,
                factor=self.cfg.MODULE.SCHEDULER.FACTOR,
                patience=self.cfg.MODULE.SCHEDULER.PATIENCE
            )
        else:
            return {"optimizer": optimizer}

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
                "monitor": self.cfg.TRAINER.CHECKPOINT.MONITOR,  # required when scheduler = ReduceLROnPlateau
            }
        }

    def forward(self, 
        batch: tuple[Tensor, dict[str, Tensor], Tensor, Tensor, Tensor, list[Any], list[Any]]
    ) -> Tensor:
        mol1d_emb, mol3d_emb, protein1d_emb, protein3d_emb, _, _, _ = batch
        mol1d_emb, protein1d_emb, protein3d_emb = \
            mol1d_emb.to('cuda'), protein1d_emb.to('cuda'), protein3d_emb.to('cuda')
        mol3d_emb = {k: v.to('cuda') for k, v in mol3d_emb.items()}

        mol1d_embed, mol3d_embed, prot1d_embed, prot3d_embed = self.model(mol1d_emb, mol3d_emb, protein1d_emb, protein3d_emb)
        dti1d_features = torch.cat([mol1d_embed, prot1d_embed], dim=-1)
        dti3d_features = torch.cat([mol3d_embed, prot3d_embed], dim=-1)

        loss = 0.5 * self.criterion(dti1d_features, dti3d_features) + 0.5 * self.criterion(dti3d_features, dti1d_features)
        return loss

    def step_common(self, batch: Any, stage: RunningStage, stage_str: str) -> dict[str, Any]: #, flag: bool
        loss = self.forward(batch)

        self.log(f"{stage_str}_loss", loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=len(batch[-1]), sync_dist=True, rank_zero_only=True)
        return loss
    
    def on_train_epoch_start(self):
        self.epoch_start_time = time.time()
    
    def on_validation_epoch_start(self):
        self.epoch_start_time = time.time()
    
    def on_test_epoch_start(self) -> None:
        self.epoch_start_time = time.time()

    def training_step(self, batch: Any, batch_idx: int) -> dict[str, Any]:
        return self.step_common(batch, RunningStage.TRAINING, 'train') #, flag

    def validation_step(self, batch: Any, batch_idx: int) -> dict[str, Any]:
        return self.step_common(batch, RunningStage.VALIDATING, 'val') #, flag
    
    def test_step(self, batch: Any, batch_idx: int) -> None:
        return self.step_common(batch, RunningStage.TESTING, 'test')
