from pathlib import Path
from tqdm import tqdm
import re
from sklearn.metrics import pairwise_distances
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import pandas as pd
import numpy as np
import sys
import argparse

AALetter = [
    "A",
    "R",
    "N",
    "D",
    "C",
    "E",
    "Q",
    "G",
    "H",
    "I",
    "L",
    "K",
    "M",
    "F",
    "P",
    "S",
    "T",
    "W",
    "Y",
    "V",
]


def get_ecfp_encoding(smiles, radius=2, nBits=1024):
    ecfp_lst = []
    for smile in smiles:
        mol = Chem.MolFromSmiles(smile) #, sanitize=False
        if not mol:
            print(f"Unable to compile SMILES: {smile}")
            sys.exit()
        features_vec = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
        features = np.zeros((1,))
        DataStructs.ConvertToNumpyArray(features_vec, features)
        ecfp_lst.append(features)
    ecfp_lst = np.array(ecfp_lst)
    return ecfp_lst


def get_3mers():
    kmers = list()
    for i in AALetter:
        for j in AALetter:
            for k in AALetter:
                kmers.append(i + j + k)
    return kmers


def get_kmers_encoding(aa_sequences, kmers):
    kmers_encoding_lst = []
    for aa_sequence in tqdm(aa_sequences):
        kmers_encoding = []
        for i in kmers:
            kmers_encoding.append(len(re.findall(i, aa_sequence)))
        kmers_encoding_lst.append(kmers_encoding)
    return np.array(kmers_encoding_lst)


def drug_cluster_split(df: pd.DataFrame, seq_col: str, drug_threshold: float = 0.5) -> tuple[pd.DataFrame, np.ndarray]:
    smile_lst = df[seq_col].unique().tolist()
    drug_feature_lst = get_ecfp_encoding(smile_lst)

    # drug cluster
    smile_cluster_dict = {}
    distance_matrix = pairwise_distances(X=drug_feature_lst, metric="jaccard")
    cond_distance_matrix = squareform(distance_matrix)
    Z = linkage(cond_distance_matrix, method="single")
    cluster_labels = fcluster(Z, t=drug_threshold, criterion="distance")

    unique_values, counts = np.unique(cluster_labels, return_counts=True)
    print(len(unique_values), 'clusters')

    for smile, cluster_ in zip(smile_lst, cluster_labels):
        smile_cluster_dict[smile] = cluster_
    df["cluster"] = df[seq_col].map(smile_cluster_dict)
    return df, counts


def prot_cluster_split(df: pd.DataFrame, seq_col: str, target_threshold: float = 0.5) -> tuple[pd.DataFrame, np.ndarray]:
    aas_lst = df[seq_col].unique().tolist()
    kmers = get_3mers()
    target_feature_lst = get_kmers_encoding(aas_lst, kmers)

    # protein cluster
    target_cluster_dict = {}
    distance_matrix = pairwise_distances(X=target_feature_lst, metric="cosine")
    cond_distance_matrix = squareform(distance_matrix)
    Z = linkage(cond_distance_matrix, method="single")
    cluster_labels = fcluster(Z, t=target_threshold, criterion="distance")

    unique_values, counts = np.unique(cluster_labels, return_counts=True)
    print(len(unique_values), 'clusters')

    for aas, cluster_ in zip(aas_lst, cluster_labels):
        target_cluster_dict[aas] = cluster_
    df["cluster"] = df[seq_col].map(target_cluster_dict)
    return df, counts


def cluster_aware_kfold_split(df: pd.DataFrame, id_col: str, k: int = 5) -> pd.DataFrame:
    """
    按照聚类结果进行k折交叉验证分割
    
    参数:
        df: 包含'id'和'cluster'列的DataFrame
        k: 折数 (默认为5)
    
    返回:
        更新的DataFrame, 包含每个分子所在的split
    """
    # 计算每个cluster的大小
    cluster_sizes = df.groupby('cluster').size().reset_index(name='size')
    
    # 按cluster大小降序排序
    cluster_sizes = cluster_sizes.sort_values('size', ascending=False)
    
    # 初始化k个split，每个split记录分子数量和分子ID列表
    splits = [{'count': 0, 'molecules': set()} for _ in range(k)]
    
    # 遍历每个cluster，分配到当前分子数最少的split
    for _, row in cluster_sizes.iterrows():
        cluster_id = row['cluster']
        cluster_molecules = set(df[df['cluster'] == cluster_id][id_col].tolist())
        
        # 找到当前分子数最少的split
        min_split = splits.index(min(splits, key=lambda x: x['count']))
        
        # 将当前cluster的所有分子加入该split
        splits[min_split]['molecules'].update(cluster_molecules)
        splits[min_split]['count'] += len(cluster_molecules)

    for i, split in enumerate(splits):
        print(f"Split {i}: {split['count']} items")

    # 创建分子到split的映射字典
    mol_to_split = {}
    for split_idx, split in enumerate(splits):
        for mol in split['molecules']:
            mol_to_split[mol] = split_idx
    
    df['split'] = df[id_col].map(mol_to_split)
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path)
    args = parser.parse_args()
    root = args.root  # NOTE: Change to your dir

    drug_df = pd.read_csv(f"{root}/drugs.csv")
    drug_df, cluster_counts = drug_cluster_split(drug_df, 'iso_smiles', drug_threshold=0.5)
    np.save(f"{root}/drug_clu.npy", cluster_counts)
    drug_df = cluster_aware_kfold_split(drug_df, 'drug_id')
    drug_df.to_csv(f"{root}/drugs.csv", index=False)

    prot_df = pd.read_csv(f"{root}/prots.csv")
    prot_df, cluster_counts = prot_cluster_split(prot_df, 'prot_seq', target_threshold=0.5)
    np.save(f"{root}/prot_clu.npy", cluster_counts)
    prot_df = cluster_aware_kfold_split(prot_df, 'prot_id')
    prot_df.to_csv(f"{root}/prots.csv", index=False)
