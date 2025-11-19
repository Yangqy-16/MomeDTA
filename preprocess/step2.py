from pathlib import Path
import pandas as pd
import pickle
from tqdm import tqdm
from pandarallel import pandarallel
from rdkit import Chem
import selfies as sf
from foldseek_util import get_struc_seq
import argparse
import warnings
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()
parser.add_argument("--root", type=Path)
args = parser.parse_args()
input_dir = args.root  # NOTE: Change to your dir

def to_selfies(smiles):
    try:
        return sf.encoder(smiles, strict=False)
    except sf.EncoderError:
        return None

def iso2can(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(mol, isomericSmiles=False)

prots = pd.read_csv(f'{input_dir}/prots.csv')
prots['sa_seq'] = None

for i in tqdm(range(len(prots))):
    info = prots.iloc[i]
    pdb_fn = info['pdb'].split()[0] #f'{pdb_root}/{i}.pdb'
    seq = info['prot_seq']
    sa_seqs = get_struc_seq("/home/qingyuyang/MomeDTA/preprocess/bin/foldseek", pdb_fn)

    # assert len(sa_seqs) == 1  # NOTE: all proteins are single chain
    if len(sa_seqs) == 0:
        print(f'{i} has no chains')
    for chain, (_, _, combined_seq) in sa_seqs.items():
        assert combined_seq[0:-1:2] == seq, f'{seq} has problems: {combined_seq[0:-1:2]}'
        prots.loc[i, 'sa_seq'] = combined_seq
        break

original_len = len(prots)
prots.dropna(inplace=True)
new_len = len(prots)
assert original_len == new_len
prots.to_csv(f'{input_dir}/prots.csv', index=False)

drugs = pd.read_csv(f'{input_dir}/drugs.csv')
pandarallel.initialize()
drugs["selfies"] = drugs["iso_smiles"].copy()
drugs["selfies"] = drugs["selfies"].parallel_apply(to_selfies)
drugs["can_smiles"] = drugs["iso_smiles"].copy()
drugs["can_smiles"] = drugs["can_smiles"].parallel_apply(iso2can)
original_len = len(drugs)
drugs.dropna(inplace=True)
new_len = len(drugs)
assert original_len == new_len
drugs.to_csv(f'{input_dir}/drugs.csv', index=False)
