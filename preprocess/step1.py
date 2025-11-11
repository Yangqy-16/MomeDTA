from pathlib import Path
import pandas as pd
import os
import pickle
import json
import shutil
import hashlib
import argparse

### NOTE: Change to your own dataset path ###
parser = argparse.ArgumentParser()
parser.add_argument("--input", type=Path)
parser.add_argument("--root", type=Path)
args = parser.parse_args()

input_df_path = args.input  # raw database
output_dir = args.root
#############################################

df = pd.read_csv(input_df_path, sep='\t')  # for .tsv
os.makedirs(output_dir, exist_ok=True)

### NOTE: This 2 lines only apply to our dataset; you can change the codes to suit yours ###
df = df[['protein seq', 'compound SMILES', 'affinity']]
df.rename({'protein seq': 'prot_seq',
           'compound SMILES': 'iso_smiles'}, axis=1, inplace=True)
############################################################################################

def seq2md5(seq: str) -> str:
    md5_seq = hashlib.md5(seq.encode('utf-8')).hexdigest()
    return md5_seq

# def removeX(seq: str) -> str:
#     """ AF2's prediction results don't contain X's coords """
#     return seq.replace('X', '')

"""
Here, you should fetch AF2-predicted structures of all proteins. 

For our datasets, you can find the corresponding data in the provided link in README.md.

For other datasets, we recommend you that:
1. Use set(df['prot_seq']) to get all unique protein sequences;
2. Search them in AFDB, and directly fetch the structure if the sequence is found;
3. For the rest of them, use AF2 to infer them;
4. Construct a DataFrame storing every sequence's PDB file path on your device.
"""

################# NOTE: Change to your own df for PDB path #################
index1 = pd.read_csv('/data/yueteng/AFDB/index.tsv', sep='\t')
index2 = pd.read_csv('/data/qingyuyang/gra_design/dta_prots/index.csv')
index = pd.concat([index1, index2], axis=0, ignore_index=True)
seq2pdb = dict(zip(list(index['protein_seq']), list(index['path'])))
print('Verify lengths:', len(index1), len(index2), len(index), len(seq2pdb))
with open('./seq2pdb.pkl', 'wb') as f:
    pickle.dump(seq2pdb, f)
# with open('/home/qingyuyang/test/preprocess/seq2pdb.pkl', 'rb') as f:
#     seq2pdb = pickle.load(f)
############################################################################

df['pdb'] = df['prot_seq'].map(seq2pdb)
print('All len:', len(df))
df.dropna(inplace=True, ignore_index=True)
print('Have PDB len:', len(df))

df['index'] = df.index

# dirty_prot_seqs = [s for s in set(df['prot_seq']) if 'X' in s]
# print(len(dirty_prot_seqs), 'protein sequences have X!')

# df['prot_seq'] = df['prot_seq'].map(removeX)
df['prot_id'] = df['prot_seq'].map(seq2md5)
df['drug_id'] = df['iso_smiles'].map(seq2md5)

prots = df[['prot_id', 'prot_seq', 'pdb']].drop_duplicates()
assert len(prots['prot_id']) == len(set(prots['prot_id'])) == len(set(df['prot_seq']))
assert len(prots['prot_seq']) == len(set(prots['prot_seq'])) == len(set(df['prot_seq']))
assert len(prots['pdb']) == len(set(prots['pdb'])) == len(set(df['prot_seq']))
prots.to_csv(f'{output_dir}/prots.csv', index=False)

drugs = df[['drug_id', 'iso_smiles']].drop_duplicates()
assert len(drugs['drug_id']) == len(set(drugs['drug_id'])) == len(set(df['iso_smiles']))
assert len(drugs['iso_smiles']) == len(set(drugs['iso_smiles'])) == len(set(df['iso_smiles']))
drugs.to_csv(f'{output_dir}/drugs.csv', index=False)

df_new = df[['index', 'drug_id', 'prot_id', 'affinity']]
assert len(df_new) == len(df)
df_new.to_csv(f'{output_dir}/pairs.csv', index=False)
