"""
NOTE: If your dataset contains drug structure (e.g. sdf), set if_struct = True;
      Otherwise, set if_struct = False.
"""
from pathlib import Path
import argparse
from tqdm import tqdm
import pandas as pd
import torch

from parse_sdf import *
from unimol_pre import *

def parse_arguments() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--if_struct", type=bool, default=False)
    parser.add_argument("--root", type=Path, default='/data/qingyuyang/dta_ours/data/metz')
    return parser.parse_args()

args = parse_arguments()
output_dir = f'{args.root}/token/unimol'
os.makedirs(output_dir, exist_ok=True)

if args.if_struct:
    df = pd.read_csv("...")  # NOTE: fill in your df path
    drug_ids = set(df['drug_id'])

    for drug in tqdm(drug_ids):
        atoms, coordinates = parse_sdf_text(f'...{drug}....sdf')  # NOTE: fill in your sdf path
        final_output = coords2unimol(atoms, coordinates, dictionary)
        torch.save(final_output, f"{output_dir}/{drug}.pt")
else:
    df = pd.read_csv(f"{args.root}/drugs.csv")
    smi_list = [(row['drug_id'], row['iso_smiles']) for _, row in df.iterrows()]
    smi_list = list(set(smi_list))

    job_name = 'get_mol_repr'   # replace to your custom name
    write_results(smi_list, job_name=job_name, outpath=output_dir)