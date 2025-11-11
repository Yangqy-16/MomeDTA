from .dataset import *
from .perseqemb import collate


@DATASET_REGISTRY.register()
class FinetuneDataset(DTADataset):
    """
    Used only when the model is already pretrained and being finetuned.
    """
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        y, prot_id, mol_id = row['affinity'], row['prot_id'], row['mol_id']

        selformer = torch.load(f"{self.data_path}/project/selformer/{mol_id}.pt")  # [512]
        unimol = torch.load(f"{self.data_path}/project/unimol/{mol_id}.pt")  # [512]
        esm2 = torch.load(f"{self.data_path}/project/esm2/{prot_id}.pt")  # [512]
        saprot = torch.load(f"{self.data_path}/project/saprot/{prot_id}.pt")  # [512]

        return selformer, unimol, esm2, saprot, y, mol_id, prot_id

    @property
    def collate_fn(self):
        return collate
