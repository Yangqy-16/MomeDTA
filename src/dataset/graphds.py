# from typing import Any
import os.path as osp
import pickle
# import numpy as np
import torch
import pandas as pd
# from torch_geometric.data import InMemoryDataset
# from torch_geometric import data as DATA
# from omegaconf import DictConfig
# from pytorch_lightning.trainer.states import RunningStage
# from coach_pl.configuration import configurable
# from coach_pl.dataset import DATASET_REGISTRY
from rdkit import Chem
# from rdkit.Chem import MolFromSmiles
import networkx as nx
from rdkit import Chem
from rdkit.Chem import ChemicalFeatures
from rdkit import RDConfig
from tqdm import tqdm
fdef_name = osp.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
chem_feature_factory = ChemicalFeatures.BuildFeatureFactory(fdef_name)


'''
Note that training and test datasets are the same as GraphDTA
Please see: https://github.com/thinng/GraphDTA
'''


VOCAB_PROTEIN = {"A": 1, "C": 2, "B": 3, "E": 4, "D": 5,
                 "G": 6, "F": 7, "I": 8, "H": 9, "K": 10,
                 "M": 11, "L": 12, "O": 13, "N": 14, "Q": 15,
                 "P": 16, "S": 17, "R": 18, "U": 19, "T": 20,
                 "W": 21, "V": 22, "Y": 23, "X": 24, "Z": 25}


def seqs2int(target):
    return [VOCAB_PROTEIN[s] for s in target]


# @DATASET_REGISTRY.register()
# class GNNDataset(InMemoryDataset):
#     @configurable
#     def __init__(self, cfg: DictConfig, stage: RunningStage, transform=None, pre_transform=None, pre_filter=None): #root, state='train'
#         cfg = cfg
#         root, ds, setting, fold = cfg.DATASET.ROOT, cfg.DATASET.DS, cfg.DATASET.SETTING, cfg.DATASET.FOLD
#         root = f'{root}/{ds}/{setting}'

#         super().__init__(root, transform, pre_transform, pre_filter)
#         if stage == 'train':
#             data, slices = torch.load(processed_paths[0])
#         elif stage == 'val':
#             data, slices = torch.load(processed_paths[1])
#         else:
#             data, slices = torch.load(processed_paths[2])

#     @classmethod
#     def from_config(cls, cfg: DictConfig, stage: RunningStage) -> dict[str, Any]:
#         return {
#             "cfg": cfg,
#             "stage": stage,
#         }

#     @property
#     def raw_file_names(self):
#         return [f'fold_{fold}_train.csv', f'fold_{fold}_val.csv', f'fold_{fold}_train.csv']

#     @property
#     def processed_file_names(self):
#         return ['processed_data_train.pt', 'processed_data_test.pt']

#     def download(self):
#         # Download to `raw_dir`.
#         pass

#     def _download(self):
#         pass

# def process_data(data_path, graph_dict):
#     df = pd.read_csv(data_path)

#     data_list = []
#     for i, row in df.iterrows():
#         smi = row['compound_iso_smiles']
#         sequence = row['target_sequence']
#         label = row['affinity']

#         x, edge_index, edge_attr = graph_dict[smi]

#         # caution
#         x = (x - x.min()) / (x.max() - x.min())

#         target = seqs2int(sequence)
#         target_len = 1200
#         if len(target) < target_len:
#             target = np.pad(target, (0, target_len- len(target)))
#         else:
#             target = target[:target_len]

#         # Get Labels
#         try:
#             data = DATA.Data(
#                 x=x,
#                 edge_index=edge_index,
#                 edge_attr=edge_attr,
#                 y=torch.FloatTensor([label]),
#                 target=torch.LongTensor([target])
#             )
#         except:
#             print("unable to process:", smi)

#         data_list.append(data)

#     return data_list

def process(root):
    # df_train = pd.read_csv(raw_paths[0])
    # df_test = pd.read_csv(raw_paths[1])
    df = pd.read_csv(f'{root}/drugs.csv') #concat([df_train, df_test])

    smiles = df['iso_smiles'].unique()
    graph_dict = dict()
    for smile in tqdm(smiles, total=len(smiles)):
        mol = Chem.MolFromSmiles(smile)
        g = mol2graph(mol)
        graph_dict[smile] = g
    
    with open(f'{root}/graph_dict.pkl', 'wb') as f:
        pickle.dump(graph_dict, f)

    # train_list = process_data(raw_paths[0], graph_dict)
    # test_list = process_data(raw_paths[1], graph_dict)

    # if pre_filter is not None:
    #     train_list = [train for train in train_list if pre_filter(train)]
    #     test_list = [test for test in test_list if pre_filter(test)]

    # if pre_transform is not None:
    #     train_list = [pre_transform(train) for train in train_list]
    #     test_list = [pre_transform(test) for test in test_list]

    # print('Graph construction done. Saving to file.')

    # data, slices = collate(train_list)
    # # save preprocessed train data:
    # torch.save((data, slices), processed_paths[0])

    # data, slices = collate(test_list)
    # # save preprocessed test data:
    # torch.save((data, slices), processed_paths[1])

def get_nodes(g):
    feat = []
    for n, d in g.nodes(data=True):
        h_t = []
        h_t += [int(d['a_type'] == x) for x in ['H', 'C', 'N', 'O', 'F', 'Cl', 'S', 'Br', 'I', ]]
        h_t.append(d['a_num'])
        h_t.append(d['acceptor'])
        h_t.append(d['donor'])
        h_t.append(int(d['aromatic']))
        h_t += [int(d['hybridization'] == x) \
                for x in (Chem.rdchem.HybridizationType.SP, \
                            Chem.rdchem.HybridizationType.SP2,
                            Chem.rdchem.HybridizationType.SP3)]
        h_t.append(d['num_h'])
        # 5 more
        h_t.append(d['ExplicitValence'])
        h_t.append(d['FormalCharge'])
        h_t.append(d['ImplicitValence'])
        h_t.append(d['NumExplicitHs'])
        h_t.append(d['NumRadicalElectrons'])
        feat.append((n, h_t))
    feat.sort(key=lambda item: item[0])
    node_attr = torch.FloatTensor([item[1] for item in feat])

    return node_attr

def get_edges(g):
    e = {}
    for n1, n2, d in g.edges(data=True):
        e_t = [int(d['b_type'] == x)
                for x in (Chem.rdchem.BondType.SINGLE, \
                            Chem.rdchem.BondType.DOUBLE, \
                            Chem.rdchem.BondType.TRIPLE, \
                            Chem.rdchem.BondType.AROMATIC)]

        e_t.append(int(d['IsConjugated'] == False))
        e_t.append(int(d['IsConjugated'] == True))
        e[(n1, n2)] = e_t

    if len(e) == 0:
        return torch.LongTensor([[0], [0]]), torch.FloatTensor([[0, 0, 0, 0, 0, 0]])

    edge_index = torch.LongTensor(list(e.keys())).transpose(0, 1)
    edge_attr = torch.FloatTensor(list(e.values()))
    return edge_index, edge_attr

def mol2graph(mol):
    if mol is None:
        return None
    feats = chem_feature_factory.GetFeaturesForMol(mol)
    g = nx.DiGraph()

    # Create nodes
    for i in range(mol.GetNumAtoms()):
        atom_i = mol.GetAtomWithIdx(i)
        g.add_node(i,
                    a_type=atom_i.GetSymbol(),
                    a_num=atom_i.GetAtomicNum(),
                    acceptor=0,
                    donor=0,
                    aromatic=atom_i.GetIsAromatic(),
                    hybridization=atom_i.GetHybridization(),
                    num_h=atom_i.GetTotalNumHs(),

                    # 5 more node features
                    ExplicitValence=atom_i.GetExplicitValence(),
                    FormalCharge=atom_i.GetFormalCharge(),
                    ImplicitValence=atom_i.GetImplicitValence(),
                    NumExplicitHs=atom_i.GetNumExplicitHs(),
                    NumRadicalElectrons=atom_i.GetNumRadicalElectrons(),
                    )

    for i in range(len(feats)):
        if feats[i].GetFamily() == 'Donor':
            node_list = feats[i].GetAtomIds()
            for n in node_list:
                g.nodes[n]['donor'] = 1
        elif feats[i].GetFamily() == 'Acceptor':
            node_list = feats[i].GetAtomIds()
            for n in node_list:
                g.nodes[n]['acceptor'] = 1

    # Read Edges
    for i in range(mol.GetNumAtoms()):
        for j in range(mol.GetNumAtoms()):
            e_ij = mol.GetBondBetweenAtoms(i, j)
            if e_ij is not None:
                g.add_edge(i, j,
                            b_type=e_ij.GetBondType(),
                            # 1 more edge features 2 dim
                            IsConjugated=int(e_ij.GetIsConjugated()),
                            )

    node_attr = get_nodes(g)
    edge_index, edge_attr = get_edges(g)

    return node_attr, edge_index, edge_attr


if __name__ == "__main__":
#     GNNDataset('data/davis')
#     GNNDataset('data/kiba')
    # process('/data/qingyuyang/dta_ours/data/davis')
    # process('/data/qingyuyang/dta_ours/data/kiba')
    process('/data/qingyuyang/dta_ours/data/metz')
