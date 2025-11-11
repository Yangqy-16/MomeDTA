from collections import defaultdict
from typing import Any

import time
import os
import pickle
from tqdm import tqdm
import pandas as pd
import torch
import torch.nn as nn
from torch import Tensor
from torch.optim.optimizer import Optimizer
from torch.optim import lr_scheduler, Adam
import pytorch_lightning as pl
from pytorch_lightning import LightningModule
from pytorch_lightning.trainer.states import RunningStage
from omegaconf import DictConfig

from coach_pl.configuration import configurable
from coach_pl.model import build_model
from coach_pl.module import MODULE_REGISTRY
from src.utils.metrics import compute_metrics, get_cindex, get_rm2
from src.utils.scheduler import LinearWarmupCosineAnnealingLR
from src.utils.typedef import ValidType
from .pretrain import PretrainModule
from src.module.criterion.infonce import InfoNCE


@MODULE_REGISTRY.register()
class DownstreamModule(PretrainModule, LightningModule):
    @configurable
    def __init__(self, model: nn.Module, cfg: DictConfig) -> None:
        LightningModule.__init__(self)

        self.cfg = cfg
        self.model = torch.compile(model) if cfg.MODULE.COMPILE else model
        self.monitor = cfg.TRAINER.CHECKPOINT.MONITOR
        self.criterion = nn.MSELoss()
        # self.infonce = InfoNCE(temperature=0.5)
        self.save_hyperparameters(cfg)

        self.losses = defaultdict(list)
        self.logits = defaultdict(list)
        self.labels = defaultdict(list)
        self.drugs = defaultdict(list)
        self.proteins = defaultdict(list)
        self.weights = defaultdict(list)
        self.att1s = defaultdict(list)
        self.att3s = defaultdict(list)

    @classmethod
    def from_config(cls, cfg: DictConfig) -> dict[str, Any]:
        return {
            "model": build_model(cfg),
            "cfg": cfg,
        }

    def forward(self, 
        batch: tuple[Tensor, ValidType, Tensor, Tensor, Tensor, list[str], list[str]]
    ) -> tuple[Tensor, Tensor, Tensor, list[str], list[str]]:
        mol1d_emb, mol3d_emb, protein1d_emb, protein3d_emb, label, drug, protein = batch
        mol1d_emb, protein1d_emb, protein3d_emb, label = \
            mol1d_emb.to('cuda'), protein1d_emb.to('cuda'), protein3d_emb.to('cuda'), label.to('cuda')
        if self.cfg.MODEL.MOL3D.FINETUNE:
            mol3d_emb = {k: v.to('cuda') for k, v in mol3d_emb.items()}
        else:
            mol3d_emb = mol3d_emb.to('cuda')

        logit = self.model(mol1d_emb, mol3d_emb, protein1d_emb, protein3d_emb)
        loss = self.criterion(logit, label)
        return loss, logit, label, drug, protein

    def step_common(self, batch: Any, stage: RunningStage, stage_str: str) -> Tensor:
        loss, logit, label, drug, protein, w, att1, att3 = self.forward(batch) #

        if stage == RunningStage.TRAINING:
            self.log(f"{stage_str}/loss", loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=len(label), sync_dist=True, rank_zero_only=True)
        
        self.losses[stage].append((loss.detach().item(), len(label)))
        self.logits[stage].append(logit.detach())#.cpu()
        self.labels[stage].append(label.detach())#.cpu()
        self.drugs[stage].extend(drug)
        self.proteins[stage].extend(protein)
        self.weights[stage].append(w.cpu())
        self.att1s[stage].append(att1.cpu())
        self.att3s[stage].append(att3.cpu())

        return loss

    def end_common(self, stage: RunningStage, stage_str: str) -> None:
        labels = torch.cat(self.labels[stage]).cpu().numpy()  # [N]
        logits = torch.cat(self.logits[stage]).cpu().numpy()  # [N]
        metrics = compute_metrics(labels, logits)

        for metric in ['mse', 'ci', 'r2', 'rm2', 'pearsonr']:
            self.log(f"{stage_str}/{metric}", metrics[metric], on_step=False, on_epoch=True, sync_dist=True, rank_zero_only=True)
        
        epoch_time = time.time() - self.epoch_start_time  # 计算当前epoch时长
        self.log(f"{stage_str}/time", epoch_time)

        if stage == RunningStage.TRAINING:
            weights = torch.cat(self.weights[stage]).cpu()
            outputs = list(zip(self.drugs[stage], self.proteins[stage], weights))

            count_dir = f'{self.cfg.OUTPUT_DIR}/count'
            for drug, prot, weight in outputs:
                if drug == '0f2b142c0370e165793f9ecabe1d706b' and prot == '3b93a11bc1683dc849a09618fb5af54b':
                    this_dir = f'{count_dir}/{drug}_{prot}'
                    os.makedirs(this_dir, exist_ok=True)
                    torch.save(weight, f'{this_dir}/weight_epoch{self.current_epoch}.pt')
        
        if stage == RunningStage.VALIDATING:
            self.log(self.monitor, metrics[self.monitor.lower()], logger=False, on_step=False, on_epoch=True, sync_dist=True, rank_zero_only=True)
        
        if stage == RunningStage.TESTING:
            for metric in ['spearmanr', 'mae']:
                self.log(f"{stage_str}/{metric}", metrics[metric], on_step=False, on_epoch=True, sync_dist=True, rank_zero_only=True)

            df = pd.DataFrame({'drug_id': self.drugs[stage], 'prot_id': self.proteins[stage], 'logit': logits})
            output_dir = f'{self.cfg.OUTPUT_DIR}/csv_log'
            df.to_csv(f'{output_dir}/version_{len(os.listdir(output_dir)) - 1}/{stage_str}_logits.csv', index=False)

            # weights = torch.cat(self.weights[stage]).cpu()
            # att1s = torch.cat(self.att1s[stage]).cpu()
            # att3s = torch.cat(self.att3s[stage]).cpu()
            # itp_dir = f'{self.cfg.OUTPUT_DIR}/itp'

            # outputs = list(zip(self.drugs[stage], self.proteins[stage], weights, att1s, att3s))
            # os.makedirs(itp_dir, exist_ok=True)
            # for drug, prot, weight, att1, att3 in tqdm(outputs):
            #     # if drug == '054820ee6232d2a73bd63964484c6ac0' and prot == '09f453c1d3cb772c44c5334072c276e5':
            #     this_dir = f'{itp_dir}/{drug}_{prot}'
            #     os.makedirs(this_dir, exist_ok=True)
            #     torch.save(weight, f'{this_dir}/weight.pt')
            #         # torch.save(att1, f'{this_dir}/att1.pt')
            #         # torch.save(att3, f'{this_dir}/att3.pt')
            #         # break

        self.losses[stage].clear()
        self.logits[stage].clear()
        self.labels[stage].clear()
        self.drugs[stage].clear()
        self.proteins[stage].clear()
        self.weights[stage].clear()
        self.att1s[stage].clear()
        self.att3s[stage].clear()

    def on_train_epoch_end(self) -> None:
        self.end_common(RunningStage.TRAINING, 'train')

    def on_validation_epoch_end(self) -> None:
        self.end_common(RunningStage.VALIDATING, 'val')

    def on_test_epoch_end(self) -> None:
        self.end_common(RunningStage.TESTING, 'test')
