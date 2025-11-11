from .dataset import *
import numpy as np
from torch_geometric import data as DATA
from torch_geometric.data import Batch


def padding(samples: list[Tensor], max_len: int) -> tuple[Tensor, Tensor]:
    """
    Optimized version of padding function.
    NOTE: In the mask, 1 is valid position and 0 is masked position!
    """
    # Pre-compute lengths and determine max_len
    # count_len = torch.tensor([g.shape[0] for g in samples], dtype=torch.long)
    # max_len = min(int(count_len.max()), max_set_len)
    
    # Batch dimensions
    batch_size = len(samples)
    embed_dim = samples[0].shape[1]
    
    # Initialize batch and mask with zeros (more efficient than ones + overwrite)
    batched_sample = torch.zeros((batch_size, max_len, embed_dim), dtype=torch.float32)
    mask = torch.zeros((batch_size, max_len), dtype=torch.bool)
    
    # Fill batch and mask in parallel
    for i, sample in enumerate(samples):
        seq_len = min(sample.shape[0], max_len)
        batched_sample[i, :seq_len] = sample[:seq_len]  # Automatic broadcasting
        mask[i, :seq_len] = True
    
    return batched_sample, mask


class CollateBatch():
    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        self.mol_max_len = cfg.DATASET.DRUG_MAX_LEN
        self.prot_max_len = cfg.DATASET.PROT_MAX_LEN

    def __call__(self,
        batch: list[tuple[SeqType, ValidType, SeqType, SeqType, float, str, str]]
    ) -> tuple[dict[str, Tensor], Tensor, list[str], list[str]]:
        drug1d, drug2d, drug3d, prot1d, prot2d, prot3d, y, drug_id, prot_id = zip(*batch)

        data_dict = {}
        
        if self.cfg.MODEL.USE_1D:
            data_dict['drug1d_embed'], data_dict['drug1d_mask'] = padding(drug1d, self.mol_max_len)
            data_dict['prot1d_embed'], data_dict['prot1d_mask'] = padding(prot1d, self.prot_max_len)
        
        if self.cfg.MODEL.USE_2D:
            if self.cfg.MODEL.DRUG_2D_MODEL:
                data_dict['drug2d_embed'], data_dict['drug2d_mask'] = padding(drug2d, self.mol_max_len)
            else:
                data_dict['drug2d_embed'], data_dict['drug2d_mask'] = Batch.from_data_list(drug2d), None

            if self.cfg.MODEL.PROT_2D_MODEL:
                data_dict['prot2d_embed'], data_dict['prot2d_mask'] = padding(prot2d, self.prot_max_len)
            else:
                data_dict['prot2d_embed'], data_dict['prot2d_mask'] = Batch.from_data_list(prot2d), None
        
        if self.cfg.MODEL.USE_3D:
            data_dict['drug3d_embed'], data_dict['drug3d_mask'] = padding(drug3d, self.mol_max_len)
            data_dict['prot3d_embed'], data_dict['prot3d_mask'] = padding(prot3d, self.prot_max_len)

        y = torch.tensor(y).float()
        return data_dict, y, drug_id, prot_id


@DATASET_REGISTRY.register()
class PerAtomEmbedDataset(DTADataset):
    """
    When the input is the embedding after the LLM and before the Projector,
    load the offline-generated embeddings.
    """
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        y, prot_id, drug_id = row['affinity'], row['prot_id'], row['drug_id']

        drug1d, drug2d, drug3d, prot1d, prot2d, prot3d = None, None, None, None, None, None
        
        if self.cfg.MODEL.USE_1D:
            drug1d = torch.load(f"{self.data_path}/embed/{self.cfg.MODEL.DRUG_1D_MODEL}/{drug_id}.pt", weights_only=True)  # [seq_len, 768]
            prot1d = torch.load(f"{self.data_path}/embed/{self.cfg.MODEL.PROT_1D_MODEL}/{prot_id}.pt", weights_only=True)  # [seq_len, 1280]
        
        if self.cfg.MODEL.USE_2D:
            if self.cfg.MODEL.DRUG_2D_MODEL:
                drug2d = torch.load(f"{self.data_path}/embed/{self.cfg.MODEL.DRUG_2D_MODEL}/{drug_id}.pt", weights_only=True)  # [seq_len, 300]
            else:
                smiles = self.drugs[self.drugs['drug_id'] == drug_id]['iso_smiles'].item()

                x, edge_index, edge_attr = self.drug_graphs[smiles]
                x = (x - x.min()) / (x.max() - x.min())
                drug2d = DATA.Data(
                    x=x,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                )
            
            if self.cfg.MODEL.PROT_2D_MODEL:
                prot2d = torch.load(f"{self.data_path}/embed/{self.cfg.MODEL.PROT_2D_MODEL}/{prot_id}.pt", weights_only=True)  # [seq_len, 3072]
            else:
                prot2d = self.prot_graphs[prot_id]
        
        if self.cfg.MODEL.USE_3D:
            drug3d = torch.load(f"{self.data_path}/embed/{self.cfg.MODEL.DRUG_3D_MODEL}/{drug_id}.pt", weights_only=True)  # [seq_len, 512]
            prot3d = torch.load(f"{self.data_path}/embed/{self.cfg.MODEL.PROT_3D_MODEL}/{prot_id}.pt", weights_only=True)  # [seq_len, 1280]

        return drug1d, drug2d, drug3d, prot1d, prot2d, prot3d, y, drug_id, prot_id

    @property
    def collate_fn(self):
        return CollateBatch(self.cfg)
