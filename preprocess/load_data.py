import os
import pandas as pd
import argparse

DATA_PATH = "/data/qingyuyang/dta_ours/data"
SAVE_PATH = "/data/qingyuyang/MMSG-DTA/dataset"


def load_prot_drug(dataset: str) -> tuple[dict[str, str], dict[str, str]]:
    """
    Func:
        load prot and drug data of a certain dataset
    Args:
        dataset: dataset name such as "davis"
    Return:
        prot_dict: {md5: prot_seq}
        drug_dict: {md5: iso_smiles}
    """
    prot_dict = {}
    for index, row in pd.read_csv(f"{DATA_PATH}/{dataset}/prots.csv").iterrows():
        prot_dict[row['prot_id']] = row['prot_seq']

    drug_dict = {}
    for index, row in pd.read_csv(f"{DATA_PATH}/{dataset}/drugs.csv").iterrows():
        drug_dict[row['drug_id']] = row['iso_smiles']

    return prot_dict, drug_dict


def load_split_idx(dataset: str, novel_type: str, split_num: int) -> tuple[list[int], list[int], list[int]]:
    """
    Func:
        load split index of a certain dataset
    Args:
        dataset: dataset name such as "davis"
        novel_type: warm / novel_pair / novel_drug / novel_prot
        split_num: 0 ~ 4
    Return:
        train_idx_list
        valid_idx_list
        test_idx_list
    """
    train_idx = pd.read_csv(f"{DATA_PATH}/{dataset}/splits/{novel_type}/fold_{split_num}_train.csv")['index']
    valid_idx = pd.read_csv(f"{DATA_PATH}/{dataset}/splits/{novel_type}/fold_{split_num}_valid.csv")['index']
    test_idx = pd.read_csv(f"{DATA_PATH}/{dataset}/splits/{novel_type}/fold_{split_num}_test.csv")['index']

    return list(train_idx), list(valid_idx), list(test_idx)


def get_train_val_test_df(dataset: str, novel_type: str, split_num: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Func:
        load dataframe for train val test
    Args:
        dataset: dataset name such as "davis"
        novel_type: warm / novel_pair / novel_drug / novel_prot
        split_num: 0 ~ 4
    Return:
        train_dataframe
        valid_dataframe
        tes_dataframe
    """
    prot_dict, drug_dict = load_prot_drug(dataset=dataset)
    train_idx, valid_idx, test_idx = load_split_idx(dataset=dataset, novel_type=novel_type, split_num=split_num)

    pair_df = pd.read_csv(f"{DATA_PATH}/{dataset}/pairs.csv", index_col='index')
    
    smiles = []
    prot_seqs = []

    for index, row in pair_df.iterrows():
        prot_seqs.append(prot_dict[row['prot_id']])
        smiles.append(drug_dict[row['drug_id']])

    pair_df['prot_seq'] = prot_seqs
    pair_df['iso_smiles'] = smiles

    return pair_df.loc[train_idx], pair_df.loc[valid_idx], pair_df.loc[test_idx]


def tackle_all():
    for dataset in ['davis', 'kiba', 'metz']:
        for setting in ['warm', 'novel_drug', 'novel_prot', 'novel_pair']:
            for fold in range(5):
                train, val, test = get_train_val_test_df(dataset=dataset, novel_type=setting, split_num=fold)
                train = pd.DataFrame(data={
                    'drug_key': train['drug_id'],
                    'compound_iso_smiles': train['iso_smiles'],
                    'target_key': train['prot_id'],
                    'target_sequence': train['prot_seq'],
                    'affinity': train['affinity'],
                })
                val = pd.DataFrame(data={
                    'drug_key': val['drug_id'],
                    'compound_iso_smiles': val['iso_smiles'],
                    'target_key': val['prot_id'],
                    'target_sequence': val['prot_seq'],
                    'affinity': val['affinity'],
                })
                test = pd.DataFrame(data={
                    'drug_key': test['drug_id'],
                    'compound_iso_smiles': test['iso_smiles'],
                    'target_key': test['prot_id'],
                    'target_sequence': test['prot_seq'],
                    'affinity': test['affinity'],
                })
                os.makedirs(f"{SAVE_PATH}/{dataset}/{setting}", exist_ok=True)
                train.to_csv(f"{SAVE_PATH}/{dataset}/{setting}/fold_{fold}_train.csv", index=False)
                val.to_csv(f"{SAVE_PATH}/{dataset}/{setting}/fold_{fold}_val.csv", index=False)
                test.to_csv(f"{SAVE_PATH}/{dataset}/{setting}/fold_{fold}_test.csv", index=False)


if __name__=="__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--dataset', required=True)
    # parser.add_argument('--type', required=True)
    # parser.add_argument('--split', required=True)
    # args = parser.parse_args()
    tackle_all()
