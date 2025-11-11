import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data, DataLoader
from Bio.PDB import PDBParser
from tqdm import tqdm
import pickle

ROOT = '/data/qingyuyang/dta_ours/data'

aa_order = "ACDEFGHIKLMNPQRSTVWY"
aa_map = {
    'ALA':'A', 'ARG':'R', 'ASN':'N', 'ASP':'D', 'CYS':'C',
    'GLN':'Q', 'GLU':'E', 'GLY':'G', 'HIS':'H', 'ILE':'I',
    'LEU':'L', 'LYS':'K', 'MET':'M', 'PHE':'F', 'PRO':'P',
    'SER':'S', 'THR':'T', 'TRP':'W', 'TYR':'Y', 'VAL':'V'
}
aa_prop = {
    'A': [1.8, 0, 0, 89.1], 'R': [-4.5, +1, 1, 174.2], 'N': [-3.5, 0, 1, 132.1],
    'D': [-3.5, -1, 1, 133.1], 'C': [2.5, 0, 0, 121.2], 'Q': [-3.5, 0, 1, 146.2],
    'E': [-3.5, -1, 1, 147.1], 'G': [-0.4, 0, 0, 75.1], 'H': [-3.2, +1, 1, 155.2],
    'I': [4.5, 0, 0, 131.2], 'L': [3.8, 0, 0, 131.2], 'K': [-3.9, +1, 1, 146.2],
    'M': [1.9, 0, 0, 149.2], 'F': [2.8, 0, 0, 165.2], 'P': [-1.6, 0, 0, 115.1],
    'S': [-0.8, 0, 1, 105.1], 'T': [-0.7, 0, 1, 119.1], 'W': [-0.9, 0, 0, 204.2],
    'Y': [-1.3, 0, 1, 181.2], 'V': [4.2, 0, 0, 117.1], 'X': [0, 0, 0, 0]
}

# ----------------------------
# 1. PDB解析与数据预处理
# ----------------------------
def three_to_one(three_letter):
    """三字母氨基酸→单字母缩写"""
    return aa_map.get(three_letter, 'X')  # 未知残基用X标记

def parse_pdb(pdb_path):
    """解析PDB文件, 返回残基Ca坐标和氨基酸类型"""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_path)
    
    ca_coords = []  # Cα原子坐标 (N_residues, 3)
    aa_types = []   # 氨基酸类型（单字母）
    
    for model in structure:
        for chain in model:
            for res in chain:
                aa = res.get_resname()
                aa_1letter = three_to_one(aa)
                # 仅保留20种常见氨基酸
                if aa_1letter not in aa_order:
                    continue
                # 提取Cα坐标
                try:
                    ca = res["CA"]
                    ca_coords.append(ca.get_coord())
                    aa_types.append(aa_1letter)
                except KeyError:
                    continue  # 跳过缺失Cα的残基
            break  # 取第一条链
    return np.array(ca_coords), aa_types

# ----------------------------
# 2. 图结构构建（节点、边、特征）
# ----------------------------
def build_edges(ca_coords, distance_threshold=8.0):
    """基于Ca距离构建边(无向图)"""
    n_res = len(ca_coords)
    edges = []
    for i in range(n_res):
        for j in range(i+1, n_res):
            dist = np.linalg.norm(ca_coords[i] - ca_coords[j])
            if dist < distance_threshold:
                edges.append([i, j])
                edges.append([j, i])  # 双向边
    return np.array(edges).T  # 转为(2, N_edges)格式

def build_node_features(aa_types):
    """构建简化的节点特征(仅含one-hot和物理化学性质)"""
    # 1. 氨基酸one-hot编码（21维）
    aa_onehot = []
    for aa in aa_types:
        vec = [0]*21
        if aa in aa_order:
            vec[aa_order.index(aa)] = 1
        else:
            vec[-1] = 1  # 未知残基
        aa_onehot.append(vec)
    aa_onehot = np.array(aa_onehot)
    
    # 2. 物理化学性质（4维：疏水性、电荷、极性、分子量，已归一化）
    phys_chem = []
    
    for aa in aa_types:
        prop = aa_prop[aa]
        prop_norm = [
            (prop[0] + 4.5) / 9.0,  # 疏水性归一化到[0,1]
            (prop[1] + 1) / 2.0,    # 电荷归一化到[0,1]
            prop[2],                # 极性（0/1）
            prop[3] / 204.2         # 分子量归一化到[0,1]
        ]
        phys_chem.append(prop_norm)
    phys_chem = np.array(phys_chem)
    
    # 融合特征（21+4=25维）
    return np.hstack([aa_onehot, phys_chem]).astype(np.float32)

# ----------------------------
# 3. 构建PyG数据集
# ----------------------------
def build_pyg_data(ca_coords, aa_types, distance_threshold=8.0): #label, 
    """构建PyG的Data对象"""
    x = torch.tensor(build_node_features(aa_types), dtype=torch.float32)
    edge_index = torch.tensor(build_edges(ca_coords, distance_threshold), dtype=torch.long)
    # pos = torch.tensor(ca_coords, dtype=torch.float32)  # GCN可选项，但保留用于可视化
    # y = torch.tensor([label], dtype=torch.float32)
    return Data(x=x, edge_index=edge_index) #, pos=pos, y=y

def build_dataset(dataset):
    """构建数据集"""
    prot_df = pd.read_csv(f'{ROOT}/{dataset}/prots.csv')
    infos = list(zip(prot_df['prot_id'], prot_df['pdb']))

    graph_dict = {}
    for prot_id, pdb_path in tqdm(infos):  # , label
        try:
            ca_coords, aa_types = parse_pdb(pdb_path)
            # if len(ca_coords) < 10:
            #     continue
            data = build_pyg_data(ca_coords, aa_types) #, label
            graph_dict[prot_id] = data
        except Exception as e:
            print(f"处理失败 {pdb_path}: {e}")
    
    with open(f'{ROOT}/{dataset}/prot_graphs.pkl', 'wb') as f:
        pickle.dump(graph_dict, f)


if __name__ == '__main__':
    # build_dataset('davis')
    # build_dataset('kiba')
    build_dataset('metz')
