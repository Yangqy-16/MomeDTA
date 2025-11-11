from pathlib import Path
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import random
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--root", type=Path)
args = parser.parse_args()
cur_data = args.root  # NOTE: Change to your dir

df_drugs = pd.read_csv(f'{cur_data}/drugs.csv').sample(frac=1, replace=False, random_state=42)
df_prots = pd.read_csv(f'{cur_data}/prots.csv').sample(frac=1, replace=False, random_state=42)
df_pairs = pd.read_csv(f'{cur_data}/pairs.csv')

print('############################## warm ##############################')
path = f'{cur_data}/splits/warm'
os.makedirs(path, exist_ok=True)

k = 5
fold_size = len(df_pairs) // k
for i in range(k):
    test_start = i * fold_size
    if i != k - 1 and i != 0:
        test_end = (i + 1) * fold_size
        testset = df_pairs[test_start:test_end]
        tvset = pd.concat([df_pairs[0:test_start], df_pairs[test_end:]])
    elif i == 0:
        test_end = fold_size
        testset = df_pairs[test_start:test_end]
        tvset = df_pairs[test_end:]
    else:
        testset = df_pairs[test_start:]
        tvset = df_pairs[0:test_start]
    
    # split training-set and valid-set
    trainset, validset = train_test_split(tvset, test_size=0.2, random_state=0)
    print(f'train:{len(trainset)}, valid:{len(validset)}, test:{len(testset)}')
    trainset[['index']].to_csv(f'{path}/fold_{i}_train.csv', index=False, header=True) 
    validset[['index']].to_csv(f'{path}/fold_{i}_valid.csv', index=False, header=True)
    testset[['index']].to_csv(f'{path}/fold_{i}_test.csv', index=False, header=True)

print('############################## novel drug ##############################')
path = f'{cur_data}/splits/novel_drug'
os.makedirs(path, exist_ok=True)

for fold in range(5):
    filtered_list = [x for x in [0, 1, 2, 3, 4] if x != fold]
    random.seed(fold + 1000)
    val_split = random.choice(filtered_list)

    test_drug = list(df_drugs[df_drugs['split'] == fold]['drug_id'])
    val_drug = list(df_drugs[df_drugs['split'] == val_split]['drug_id'])
    train_drug = list(df_drugs[(df_drugs['split'] != fold) & (df_drugs['split'] != val_split)]['drug_id'])

    train_set = df_pairs[df_pairs['drug_id'].isin(train_drug)]
    val_set = df_pairs[df_pairs['drug_id'].isin(val_drug)]
    test_set = df_pairs[df_pairs['drug_id'].isin(test_drug)]

    print(f'train:{len(train_set)}, valid:{len(val_set)}, test:{len(test_set)}')
    train_set[['index']].to_csv(f'{path}/fold_{fold}_train.csv', index=False, header=True) 
    val_set[['index']].to_csv(f'{path}/fold_{fold}_valid.csv', index=False, header=True)
    test_set[['index']].to_csv(f'{path}/fold_{fold}_test.csv', index=False, header=True)

print('############################## novel prot ##############################')
path = f'{cur_data}/splits/novel_prot'
os.makedirs(path, exist_ok=True)

for fold in range(5):
    filtered_list = [x for x in [0, 1, 2, 3, 4] if x != fold]
    random.seed(fold + 1000)
    val_split = random.choice(filtered_list)

    test_prot = list(df_prots[df_prots['split'] == fold]['prot_id'])
    val_prot = list(df_prots[df_prots['split'] == val_split]['prot_id'])
    train_prot = list(df_prots[(df_prots['split'] != fold) & (df_prots['split'] != val_split)]['prot_id'])

    train_set = df_pairs[df_pairs['prot_id'].isin(train_prot)]
    val_set = df_pairs[df_pairs['prot_id'].isin(val_prot)]
    test_set = df_pairs[df_pairs['prot_id'].isin(test_prot)]

    print(f'train:{len(train_set)}, valid:{len(val_set)}, test:{len(test_set)}')
    train_set[['index']].to_csv(f'{path}/fold_{fold}_train.csv', index=False, header=True) 
    val_set[['index']].to_csv(f'{path}/fold_{fold}_valid.csv', index=False, header=True)
    test_set[['index']].to_csv(f'{path}/fold_{fold}_test.csv', index=False, header=True)

print('############################## novel pair ##############################')
path = f'{cur_data}/splits/novel_pair'
os.makedirs(path, exist_ok=True)

for fold in range(5):
    filtered_list = [x for x in [0, 1, 2, 3, 4] if x != fold]
    random.seed(fold + 1000)
    val_split = random.choice(filtered_list)

    test_drug = list(df_drugs[df_drugs['split'] == fold]['drug_id'])
    val_drug = list(df_drugs[df_drugs['split'] == val_split]['drug_id'])
    train_drug = list(df_drugs[(df_drugs['split'] != fold) & (df_drugs['split'] != val_split)]['drug_id'])

    test_prot = list(df_prots[df_prots['split'] == fold]['prot_id'])
    val_prot = list(df_prots[df_prots['split'] == val_split]['prot_id'])
    train_prot = list(df_prots[(df_prots['split'] != fold) & (df_prots['split'] != val_split)]['prot_id'])

    train_set = df_pairs[df_pairs['drug_id'].isin(train_drug) & df_pairs['prot_id'].isin(train_prot)]
    val_set = df_pairs[df_pairs['drug_id'].isin(val_drug) & df_pairs['prot_id'].isin(val_prot)]
    test_set = df_pairs[df_pairs['drug_id'].isin(test_drug) & df_pairs['prot_id'].isin(test_prot)]

    print(f'train:{len(train_set)}, valid:{len(val_set)}, test:{len(test_set)}')
    train_set[['index']].to_csv(f'{path}/fold_{fold}_train.csv', index=False, header=True) 
    val_set[['index']].to_csv(f'{path}/fold_{fold}_valid.csv', index=False, header=True)
    test_set[['index']].to_csv(f'{path}/fold_{fold}_test.csv', index=False, header=True)
