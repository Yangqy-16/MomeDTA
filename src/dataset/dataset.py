from typing import Any

import pickle
import pandas as pd
from omegaconf import DictConfig
import torch
from torch import Tensor
from torch.utils.data import Dataset
from pytorch_lightning.trainer.states import RunningStage

from coach_pl.configuration import configurable
from coach_pl.dataset import DATASET_REGISTRY
from src.utils.typedef import *


@DATASET_REGISTRY.register()
class DTADataset(Dataset):
    """
    Able to deal with data of the forms:
        1) raw inputs before LLMs;
        2) embeddings right after LLMs and before Projectors;
        3) embeddings after Projectors.
    """
    @configurable
    def __init__(self, cfg: DictConfig, stage: RunningStage | str):
        self.cfg = cfg
        root, ds, setting, fold = cfg.DATASET.ROOT, cfg.DATASET.DS, cfg.DATASET.SETTING, cfg.DATASET.FOLD
        self.data_path = f'{root}/{ds}' #cfg.DATASET.DATA_PATH

        if stage == RunningStage.TRAINING or stage == 'train':
            indices = list(pd.read_csv(f'{root}/{ds}/splits/{setting}/fold_{fold}_train.csv')['index'])  #cfg.DATASET.TRAIN_DF
        elif stage == RunningStage.VALIDATING or stage == 'val':
            indices = list(pd.read_csv(f'{root}/{ds}/splits/{setting}/fold_{fold}_valid.csv')['index'])  #cfg.DATASET.VAL_DF
        elif stage == RunningStage.TESTING or stage == 'test':
            indices = list(pd.read_csv(f'{root}/{ds}/splits/{setting}/fold_{fold}_test.csv')['index'])  #'/data/qingyuyang/dta_ours/data/davis/temp.csv'   cfg.DATASET.TEST_DF
        else:
            raise ValueError('Undefined stage!')

        self.pairs = pd.read_csv(f'{self.data_path}/pairs.csv')
        self.df = self.pairs[self.pairs['index'].isin(indices)]

        self.drugs = pd.read_csv(f'{self.data_path}/drugs.csv')
        self.prots = pd.read_csv(f'{self.data_path}/prots.csv')

        if self.cfg.MODEL.USE_2D and not self.cfg.MODEL.DRUG_2D_MODEL:
            with open(f'{self.data_path}/drug_graphs.pkl', 'rb') as f:
                self.drug_graphs = pickle.load(f)
        if self.cfg.MODEL.USE_2D and not self.cfg.MODEL.PROT_2D_MODEL:
            with open(f'{self.data_path}/prot_graphs.pkl', 'rb') as f:
                self.prot_graphs = pickle.load(f)

    @classmethod
    def from_config(cls, cfg: DictConfig, stage: RunningStage) -> dict[str, Any]:
        return {
            "cfg": cfg,
            "stage": stage,
        }

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        """ Subclass MUST overwrite this method! """
        pass

    @property
    def collate_fn(self):
        return None

    @property
    def sampler(self):
        return None
