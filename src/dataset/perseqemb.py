from .dataset import *


def collate(
    batch: list[tuple[SeqType, ValidType, SeqType, SeqType, float, str, str]]
) -> tuple[ValidType, ValidType, ValidType, ValidType, Tensor, list[str], list[str]]:
    """
    Works when any of the LLMs are frozen or unfrozen.

    When an LLM is frozen, its input track of this function should be
    a Tensor which is its last hidden state (output embedding).

    When an LLM is unfrozen, its input track of this function should be
    either a dictionary (UniMol) or a sequence (others).
    """
    selformer, unimol, esm2, saprot, y, mol_id, prot_id = zip(*batch)

    selformer = torch.stack(selformer)
    unimol = torch.stack(unimol)
    esm2 = torch.stack(esm2)
    saprot = torch.stack(saprot)

    y = torch.tensor(y).float()
    return selformer, unimol, esm2, saprot, y, mol_id, prot_id


@DATASET_REGISTRY.register()
class PerSeqEmbedDataset(DTADataset):
    """
    When the input is the embedding after the LLM and before the Projector,
    load the offline-generated embeddings.
    """
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        y, prot_id, mol_id = row['affinity'], row['prot_id'], row['mol_id']

        selformer = torch.mean(torch.load(f"{self.data_path}/embed/selformer/{mol_id}.pt"), dim=0)  # [768]
        unimol = torch.mean(torch.load(f"{self.data_path}/embed/unimol/{mol_id}.pt"), dim=0)  # [512]
        esm2 = torch.mean(torch.load(f"{self.data_path}/embed/esm2/{prot_id}.pt"), dim=0)  # [1280]
        saprot = torch.mean(torch.load(f"{self.data_path}/embed/saprot/{prot_id}.pt"), dim=0)  # [1280]

        return selformer, unimol, esm2, saprot, y, mol_id, prot_id

    @property
    def collate_fn(self):
        return collate
