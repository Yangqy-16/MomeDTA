import os
from tqdm import tqdm
import torch
from torch import nn
import pandas as pd
from transformers import AutoTokenizer, EsmTokenizer
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from src.model.pretrain import PretrainModel
from src.utils.padding import batch_collate_fn
from coach_pl.module import build_module
from coach_pl.tool.train import arg_parser
from coach_pl.tool.trainer import setup_cfg

# NOTE: You can manually set these since these may be differ from those in cfg file
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
df_path = '/home/qingyuyang/test/data/kiba/all.csv'
unimol_conf_path = '/home/qingyuyang/test/data/kiba/token/unimol'
batch_size = 8
output_root = '/home/qingyuyang/test/data/kiba/project'
os.makedirs(output_root, exist_ok=True)


if __name__ == "__main__":
    args = arg_parser().parse_args()
    cfg = setup_cfg(args)

    MyModule = build_module(cfg)
    print('Finish building model')
    MyModule = MyModule.load_from_checkpoint(
        '/home/qingyuyang/test/output/pretrain/temp/best_ckpts/epoch=1-loss_epoch=0.ckpt',
        map_location="cuda:0"
    )
    MyModule.eval()
    MyModule.freeze()
    print('Finish loading model')

    selformer_tokenizer = AutoTokenizer.from_pretrained(cfg.MODEL.MOL1D.PATH)
    esm2_tokenizer: EsmTokenizer = EsmTokenizer.from_pretrained(cfg.MODEL.PROT1D.PATH)
    saprot_tokenizer: EsmTokenizer = EsmTokenizer.from_pretrained(cfg.MODEL.PROT3D.PATH)

    df = pd.read_csv(df_path)

    mol_set = list(set(df['mol_id']))
    mol_selfies = dict(zip(list(df['mol_id']), list(df['selfies'])))
    assert sorted(list(mol_selfies.keys())) == sorted(mol_set)
    mol_selfies = [(key, value) for key, value in mol_selfies.items()]
    mol_conf = [(mol, torch.load(f'{unimol_conf_path}/{mol}.pt')) for mol in mol_set]

    prot_seqs = dict(zip(list(df['prot_id']), list(df['protein seq'])))
    sa_seqs = dict(zip(list(df['prot_id']), list(df['sa seq'])))
    assert sorted(list(prot_seqs.keys())) == sorted(list(sa_seqs.keys()))
    prot_seqs = [(key, value) for key, value in prot_seqs.items()]
    sa_seqs = [(key, value) for key, value in sa_seqs.items()]
    print('Finish loading data')

    with torch.no_grad():
        ######################## Inference for drugs ########################
        ## SELFormer
        print('SELFormer inference ...')
        output_dir = f'{output_root}/selformer'
        os.makedirs(output_dir, exist_ok=True)

        for idx in tqdm(range(0, len(mol_selfies), batch_size)):
            batch = mol_selfies[idx : min(idx + batch_size, len(mol_selfies))]
            name_list = [i[0] for i in batch]
            selfies = [i[1] for i in batch]

            selformer_embed = selformer_tokenizer.batch_encode_plus(
                selfies,
                add_special_tokens=True,
                padding=True,
                truncation=True,
                max_length=cfg.DATASET.MOL_MAX_LEN,
                return_tensors="pt",
                return_attention_mask=True,
            )
            selformer_embed = MyModule.model.LLM.encode_mol1d(selformer_embed)
            selformer_embed = MyModule.model.projector.mol1d_projector(selformer_embed)

            for name, embed in zip(name_list, selformer_embed):
                torch.save(embed, f'{output_dir}/{name}.pt')

        ## UniMol
        print('UniMol inference ...')
        output_dir = f'{output_root}/unimol'
        os.makedirs(output_dir, exist_ok=True)

        for idx in tqdm(range(0, len(mol_conf), batch_size)):
            batch = mol_conf[idx : min(idx + batch_size, len(mol_conf))]
            name_list = [i[0] for i in batch]
            conf = [i[1] for i in batch]

            unimol_embed = batch_collate_fn(conf)
            unimol_embed = MyModule.model.LLM.encode_mol3d(unimol_embed)
            unimol_embed = MyModule.model.projector.mol3d_projector(unimol_embed)

            for name, embed in zip(name_list, unimol_embed):
                torch.save(embed, f'{output_dir}/{name}.pt')

        ######################## Inference for proteins ########################
        ## ESM2
        print('ESM2 inference ...')
        output_dir = f'{output_root}/esm2'
        os.makedirs(output_dir, exist_ok=True)

        for idx in tqdm(range(0, len(prot_seqs), batch_size)):
            batch = prot_seqs[idx : min(idx + batch_size, len(prot_seqs))]
            name_list = [i[0] for i in batch]
            seqs = [i[1] for i in batch]

            esm_embed = esm2_tokenizer.batch_encode_plus(
                seqs,
                add_special_tokens=True,
                padding=True,
                truncation=True,
                max_length=cfg.DATASET.PROT_MAX_LEN,
                return_tensors="pt",
                return_attention_mask=True,
            )
            esm_embed = MyModule.model.LLM.encode_prot1d(esm_embed)
            esm_embed = MyModule.model.projector.prot1d_projector(esm_embed)

            for name, embed in zip(name_list, esm_embed):
                torch.save(embed, f'{output_dir}/{name}.pt')
        
        ## SaProt
        print('SaProt inference ...')
        output_dir = f'{output_root}/saprot'
        os.makedirs(output_dir, exist_ok=True)

        for idx in tqdm(range(0, len(sa_seqs), batch_size)):
            batch = sa_seqs[idx : min(idx + batch_size, len(sa_seqs))]
            name_list = [i[0] for i in batch]
            seqs = [i[1] for i in batch]

            saprot_embed = saprot_tokenizer.batch_encode_plus(
                seqs,
                add_special_tokens=True,
                padding=True,
                truncation=True,
                max_length=cfg.DATASET.PROT_MAX_LEN,
                return_tensors="pt",
                return_attention_mask=True,
            )
            saprot_embed = MyModule.model.LLM.encode_prot3d(saprot_embed)
            saprot_embed = MyModule.model.projector.prot3d_projector(saprot_embed)

            for name, embed in zip(name_list, saprot_embed):
                torch.save(embed, f'{output_dir}/{name}.pt')
