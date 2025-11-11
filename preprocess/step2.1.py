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
        print("Error")
        return None

def iso2can(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    return Chem.MolToSmiles(mol, isomericSmiles=False)

if False:

    prots = pd.read_csv(f'{input_dir}/prots.csv')

    original_len = len(prots)
    new_prots = prots.dropna(inplace=False)
    new_len = len(new_prots)
    assert original_len == new_len

    prots['sa_seq'] = None

    for i in tqdm(range(len(prots))):

        info = prots.iloc[i]
        pdb_fn = info['pdb'].split()[0] #f'{pdb_root}/{i}.pdb'
        seqs = info['prot_seq'].split(".")
        sa_seqs = get_struc_seq("/home/qingyuyang/test/preprocess/bin/foldseek", pdb_fn)
        
        assert len(sa_seqs) == len(seqs)  # NOTE: all proteins are single chain
        
        if len(sa_seqs) == 0:
            print(f'{i} has no chains')
        elif len(seqs) == 1:
            for chain, (_, _, combined_seq) in sa_seqs.items():
                assert combined_seq[0:-1:2] == seqs[0], f'index_{i}: {seqs[0]} has problems: {combined_seq[0:-1:2]}'
                break
            prots.loc[i, 'sa_seq'] = combined_seq
        else:
            sa_seq = []
            check_seq = []

            for chain, (_, _, combined_seq) in sa_seqs.items():
                sa_seq.append(combined_seq)
                check_seq.append(combined_seq[0:-1:2])
            
            assert ".".join(seqs) == ".".join(check_seq), f'index_{i}: {".".join(seqs)} ==? {".".join(check_seq)}'
            prots.loc[i, 'sa_seq'] = ".".join(sa_seq)

    original_len = len(prots)
    prots.dropna(inplace=True)
    new_len = len(prots)
    assert original_len == new_len
    prots.to_csv(f'{input_dir}/prots.csv', index=False)

drugs = pd.read_csv(f'{input_dir}/drugs.csv')

drugs["selfies"] = drugs["iso_smiles"].copy()
drugs["can_smiles"] = drugs["iso_smiles"].copy()

selfies = []
can = []
for index, row in tqdm(drugs.iterrows()):
    a = to_selfies(row["iso_smiles"])
    selfies.append(a)
    can.append(iso2can(row["iso_smiles"]))
    if (a == None):
        print(row['drug_id'])

drugs['selfies'] = selfies
drugs['can_smiles'] = can

original_len = len(drugs)
#drugs.dropna(inplace=True)
new_len = len(drugs)
#assert original_len == new_len
drugs.to_csv(f'{input_dir}/drugs.csv', index=False)
