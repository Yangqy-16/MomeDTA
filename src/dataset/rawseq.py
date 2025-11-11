from transformers import EsmTokenizer, AutoTokenizer
from .dataset import *
from src.utils.padding import batch_collate_fn


class CollateBatch():
    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        self.selformer_tokenizer = AutoTokenizer.from_pretrained(cfg.MODEL.MOL1D.PATH)
        self.esm2_tokenizer: EsmTokenizer = EsmTokenizer.from_pretrained(cfg.MODEL.PROT1D.PATH)
        self.saprot_tokenizer: EsmTokenizer = EsmTokenizer.from_pretrained(cfg.MODEL.PROT3D.PATH)

        self.mol_max_len = cfg.DATASET.MOL_MAX_LEN
        self.prot_max_len = cfg.DATASET.PROT_MAX_LEN

    def __call__(self,
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

        selformer = self.selformer_tokenizer.batch_encode_plus(
            list(selformer),
            add_special_tokens=True,
            padding=True,
            truncation=True,
            max_length=self.mol_max_len,
            return_tensors="pt",
            return_attention_mask=True,
        )

        unimol = batch_collate_fn(unimol)

        esm2 = self.esm2_tokenizer.batch_encode_plus(
            list(esm2),
            add_special_tokens=True,
            padding=True,
            truncation=True,
            max_length=self.prot_max_len,
            return_tensors="pt",
            return_attention_mask=True,
        )

        saprot = self.saprot_tokenizer.batch_encode_plus(
            list(saprot),
            add_special_tokens=True,
            padding=True,
            truncation=True,
            max_length=self.prot_max_len,
            return_tensors="pt",
            return_attention_mask=True,
        )

        y = torch.tensor(y).float()
        return selformer, unimol, esm2, saprot, y, mol_id, prot_id


@DATASET_REGISTRY.register()
class RawSeqDataset(DTADataset):
    """
    When the input is a raw input before the LLM,
    load the processed data (UniMol) or the raw sequence in the dataframe (others).
    """
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        y, prot_id, mol_id = row['affinity'], row['prot_id'], row['mol_id']

        selformer = row['selfies']  # SELFIES string
        unimol = torch.load(f"{self.data_path}/token/unimol/{mol_id}.pt")  # UniMol input dict
        esm2 = row['protein seq']  # amino acid sequence
        saprot = row['sa seq']  # structure-aware sequence
        
        return selformer, unimol, esm2, saprot, y, mol_id, prot_id

    @property
    def collate_fn(self):
        return CollateBatch(self.cfg)
