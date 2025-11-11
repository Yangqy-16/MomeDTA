from typing import Any

import torch
import torch.nn as nn
from torch import Tensor
from transformers import EsmForMaskedLM, AutoModel, EsmModel
from peft import LoraConfig, get_peft_model
from omegaconf import DictConfig

from .unimol.unimol import UniMolModel
from .unimol_lora.unimol import UniMolModelwithLoRA
from .basics import Projector, MultiModalFusion, MLP, CNN_1D
from coach_pl.configuration import configurable
from coach_pl.model import MODEL_REGISTRY
from src.utils.typedef import ValidType

__all__ = ['LLM', 'ProjectNet', 'DownstreamNet']


def print_num_params(name: str, model: nn.Module):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'{name}: {total_params} parameters and {trainable_params} trainable parameters')


@MODEL_REGISTRY.register()
class LLM(nn.Module):
    """
    This model processes both molecule and protein data through separate LLM encoders
    to generate embeddings.

    Attributes:
        mol1d_encoder: Encoder for 1D molecular features
        mol3d_encoder: Encoder for 3D molecular features
        prot1d_encoder: Encoder for 1D protein features
        prot3d_encoder: Encoder for 3D protein features
    """
    @configurable
    def __init__(self, cfg: DictConfig):
        super().__init__()

        # molecule 1D encoder: SELFormer
        self.mol1d_encoder = AutoModel.from_pretrained(cfg.MODEL.MOL1D.PATH)

        if cfg.MODEL.MOL1D.FINETUNE:
            assert cfg.MODEL.MOL1D.NUM_LORA_LAYERS > 0
            target_modules = []
            start_layer_idx = 12 - cfg.MODEL.MOL1D.NUM_LORA_LAYERS
            for idx in range(start_layer_idx, 12):
                for layer_name in ["attention.self.query", "attention.self.key", "attention.self.value", "attention.output.dense"]:
                    target_modules.append(f"encoder.layer.{idx}.{layer_name}")
            config = LoraConfig(r=cfg.MODEL.MOL1D.LORA_R,
                                lora_alpha=cfg.MODEL.MOL1D.LORA_ALPHA,
                                target_modules=target_modules,
                                lora_dropout=cfg.MODEL.MOL1D.LORA_DROPOUT)
            self.mol1d_encoder = get_peft_model(self.mol1d_encoder, config)
        else:
            for param in self.mol1d_encoder.parameters():
                param.requires_grad = False
        
        print_num_params('SELFormer', self.mol1d_encoder)

        # molecule 3D encoder: UniMol
        if cfg.MODEL.MOL3D.FINETUNE:
            self.mol3d_encoder = UniMolModelwithLoRA(cfg.MODEL.MOL3D.PATH)
            # NOTE: 仅训练LoRA参数
            for param in self.mol3d_encoder.parameters():
                param.requires_grad = False
            for name, param in self.mol3d_encoder.named_parameters():
                if any([f"layers.{layer}." in name for layer in self.mol3d_encoder.args.lora_layers]) and 'lora' in name:
                    param.requires_grad = True
        else:
            self.mol3d_encoder = UniMolModel(cfg.MODEL.MOL3D.PATH)
        
        print_num_params('UniMol', self.mol3d_encoder)

        # prot 1D encoder: ESM2
        self.prot1d_encoder = EsmModel.from_pretrained(cfg.MODEL.PROT1D.PATH)

        if cfg.MODEL.PROT1D.FINETUNE:
            assert cfg.MODEL.PROT1D.NUM_LORA_LAYERS > 0
            target_modules = []
            start_layer_idx = 33 - cfg.MODEL.PROT1D.NUM_LORA_LAYERS
            for idx in range(start_layer_idx, 33):
                for layer_name in ["attention.self.query", "attention.self.key", "attention.self.value", "attention.output.dense"]:
                    target_modules.append(f"encoder.layer.{idx}.{layer_name}")
            config = LoraConfig(r=cfg.MODEL.PROT1D.LORA_R,
                                lora_alpha=cfg.MODEL.PROT1D.LORA_ALPHA,
                                target_modules=target_modules,
                                lora_dropout=cfg.MODEL.PROT1D.LORA_DROPOUT)
            self.prot1d_encoder = get_peft_model(self.prot1d_encoder, config)
        else:
            for param in self.prot1d_encoder.parameters():
                param.requires_grad = False
        
        print_num_params('ESM2', self.prot1d_encoder)

        # prot 3D encoder: SaProt
        self.prot3d_encoder = EsmModel.from_pretrained(cfg.MODEL.PROT3D.PATH)

        if cfg.MODEL.PROT3D.FINETUNE:
            assert cfg.MODEL.PROT3D.NUM_LORA_LAYERS > 0
            target_modules = []
            start_layer_idx = 33 - cfg.MODEL.PROT3D.NUM_LORA_LAYERS
            for idx in range(start_layer_idx, 33):
                for layer_name in ["attention.self.query", "attention.self.key", "attention.self.value", "attention.output.dense"]:
                    target_modules.append(f"encoder.layer.{idx}.{layer_name}")
            config = LoraConfig(r=cfg.MODEL.PROT3D.LORA_R,
                                lora_alpha=cfg.MODEL.PROT3D.LORA_ALPHA,
                                target_modules=target_modules,
                                lora_dropout=cfg.MODEL.PROT3D.LORA_DROPOUT)
            self.prot3d_encoder = get_peft_model(self.prot3d_encoder, config)
        else:
            for param in self.prot3d_encoder.parameters():
                param.requires_grad = False

        print_num_params('SaProt', self.prot3d_encoder)

    @classmethod
    def from_config(cls, cfg: DictConfig) -> dict[str, Any]:
        return {
            "cfg": cfg,
        }

    def encode_mol1d(self, batch: dict[str, Tensor]):
        mol_output = self.mol1d_encoder(**batch)
        mol_emb = mol_output.last_hidden_state
        return torch.mean(mol_emb, dim=1)

    def encode_mol3d(self, batch: dict[str, Tensor]):
        return self.mol3d_encoder(**batch)

    def encode_prot1d(self, batch: dict[str, Tensor]):
        prot_output = self.prot1d_encoder(**batch)
        prot_emb = prot_output.last_hidden_state
        return torch.mean(prot_emb, dim=1)

    def encode_prot3d(self, batch: dict[str, Tensor]):
        prot_output = self.prot3d_encoder(**batch)
        prot_emb = prot_output.last_hidden_state
        return torch.mean(prot_emb, dim=1)

    def forward(self,
        mol1d_embed: ValidType, mol3d_embed: ValidType,
        prot1d_embed: ValidType, prot3d_embed: ValidType
    ) -> tuple[Tensor, ...]:
        if type(mol1d_embed) != Tensor:
            mol1d_embed = self.encode_mol1d(mol1d_embed)
        if type(mol3d_embed) != Tensor:
            mol3d_embed = self.encode_mol3d(mol3d_embed)

        if type(prot1d_embed) != Tensor:
            prot1d_embed = self.encode_prot1d(prot1d_embed)
        if type(prot3d_embed) != Tensor:
            prot3d_embed = self.encode_prot3d(prot3d_embed)

        return mol1d_embed, mol3d_embed, prot1d_embed, prot3d_embed


@MODEL_REGISTRY.register()
class ProjectNet(nn.Module):
    """
    This model processes both molecule and protein embeddings through separate projectors
    to generate projected embeddings.
    """
    @configurable
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg

        # projectors
        self.mol1d_projector = Projector(
            input_dim=768,
            out_dim=cfg.MODEL.PROJECTION_DIM,
            activation_fn=cfg.MODEL.ACTIVATION_FN,
        )
        # self.mol1d_projector = CNN_Project(768, cfg.MODEL.PROJECTION_DIM)
        # self.mol1d_projector = MLP(input_dim=768, hidden_dims=[768//2], output_dim=cfg.MODEL.PROJECTION_DIM, dropout_rate=0)
        
        self.mol2d_projector = Projector(
            input_dim=300,
            out_dim=cfg.MODEL.PROJECTION_DIM,
            activation_fn=cfg.MODEL.ACTIVATION_FN,
        )
        # self.mol2d_projector = CNN_Project(300, cfg.MODEL.PROJECTION_DIM)
        
        self.mol3d_projector = Projector(
            input_dim=512,
            out_dim=cfg.MODEL.PROJECTION_DIM,
            activation_fn=cfg.MODEL.ACTIVATION_FN,
        )
        # self.mol3d_projector = CNN_Project(512, cfg.MODEL.PROJECTION_DIM)
        # self.mol3d_projector = MLP(input_dim=512, hidden_dims=[512//2], output_dim=cfg.MODEL.PROJECTION_DIM, dropout_rate=0)

        self.prot1d_projector = Projector(
            input_dim=1280,
            out_dim=cfg.MODEL.PROJECTION_DIM,
            activation_fn=cfg.MODEL.ACTIVATION_FN,
        )
        # self.prot1d_projector = CNN_Project(1280, cfg.MODEL.PROJECTION_DIM)
        # self.prot1d_projector = MLP(input_dim=1280, hidden_dims=[1280//2], output_dim=cfg.MODEL.PROJECTION_DIM, dropout_rate=0)

        self.prot2d_projector = Projector(
            input_dim=3072,
            out_dim=cfg.MODEL.PROJECTION_DIM,
            activation_fn=cfg.MODEL.ACTIVATION_FN,
        )
        # self.prot2d_projector = CNN_Project(3072, cfg.MODEL.PROJECTION_DIM)

        self.prot3d_projector = Projector(
            input_dim=1280,
            out_dim=cfg.MODEL.PROJECTION_DIM,
            activation_fn=cfg.MODEL.ACTIVATION_FN,
        )
        # self.prot3d_projector = CNN_Project(1280, cfg.MODEL.PROJECTION_DIM)
        # self.prot3d_projector = MLP(input_dim=1280, hidden_dims=[1280//2], output_dim=cfg.MODEL.PROJECTION_DIM)

    @classmethod
    def from_config(cls, cfg: DictConfig):
        return {
            "cfg": cfg,
        }

    def forward(self,
        mol1d_embed: Tensor, mol2d_embed: Tensor, mol3d_embed: Tensor,
        prot1d_embed: Tensor, prot2d_embed: Tensor, prot3d_embed: Tensor
    ) -> tuple[Tensor]:
        mol1d_embed = self.mol1d_projector(mol1d_embed)
        mol2d_embed = self.mol2d_projector(mol2d_embed)
        mol3d_embed = self.mol3d_projector(mol3d_embed)

        prot1d_embed = self.prot1d_projector(prot1d_embed)
        prot2d_embed = self.prot2d_projector(prot2d_embed)
        prot3d_embed = self.prot3d_projector(prot3d_embed)

        return mol1d_embed, mol2d_embed, mol3d_embed, prot1d_embed, prot2d_embed, prot3d_embed


@MODEL_REGISTRY.register()
class DownstreamNet(nn.Module):
    """
    This model only performs downstream DTA prediction.
    """
    @configurable
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg

        # fusion
        self.modal_fusion = MultiModalFusion(modal_num=cfg.MODEL.MODAL_NUM, with_weight=cfg.MODEL.FUSION_WEIGHT)

        # classifier
        self.classifier = MLP(
            input_dim=cfg.MODEL.PROJECTION_DIM * 2, 
            hidden_dim=cfg.MODEL.PROJECTION_DIM * 4,
            output_dim=cfg.MODEL.PROJECTION_DIM,
            binary=1,
        )

    @classmethod
    def from_config(cls, cfg: DictConfig):
        return {
            "cfg": cfg,
        }
    
    def forward(self, mol1d_embed: Tensor, mol3d_embed: Tensor, prot1d_embed: Tensor, prot3d_embed: Tensor) -> Tensor:
        """
        Inputs:
            All of size [512]
        
        Returns:
            Logits for interaction prediction
        """
        batch_size = mol1d_embed.shape[0]

        # 根据配置决定是否使用 1D 模型
        if self.cfg.MODEL.USE_1D_MODELS:
            dti1d_features = torch.cat([mol1d_embed, prot1d_embed], dim=-1)
            dti1d_features = dti1d_features.view(batch_size, 1, -1)

        dti3d_features = torch.cat([mol3d_embed, prot3d_embed], dim=-1)

        # 合并 1D 和 3D 特征
        if dti1d_features is not None:
            dti3d_features = dti3d_features.view(batch_size, 1, -1)
            dti_features = torch.cat([dti1d_features, dti3d_features], dim=1)
            # 融合特征
            fusion_features = self.modal_fusion(dti_features)
        else:
            fusion_features = dti3d_features

        # 分类
        output = self.classifier(fusion_features)
        output = output.squeeze(-1)
        # output = torch.sigmoid(output)
        return output
