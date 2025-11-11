import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from rdkit import Chem

from molclr.ginet_finetune import GINet
from molclr.dataset_test import ATOM_LIST, CHIRALITY_LIST, BOND_LIST, BONDDIR_LIST
from utils import *


class EmbedNet(GINet):
    def forward(self, data: Data):
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        
        h = self.x_embedding1(x[:,0]) + self.x_embedding2(x[:,1])

        for layer in range(self.num_layer):
            h = self.gnns[layer](h, edge_index, edge_attr)
            h = self.batch_norms[layer](h)
            if layer == self.num_layer - 1:
                h = F.dropout(h, self.drop_ratio, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)

        # h = self.pool(h, data.batch)
        # h = self.feat_lin(h)
        
        return h#, self.pred_head(h)


def getitem(smiles):
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)

    N = mol.GetNumAtoms()
    M = mol.GetNumBonds()

    type_idx = []
    chirality_idx = []
    atomic_number = []
    for atom in mol.GetAtoms():
        type_idx.append(ATOM_LIST.index(atom.GetAtomicNum()))
        chirality_idx.append(CHIRALITY_LIST.index(atom.GetChiralTag()))
        atomic_number.append(atom.GetAtomicNum())

    x1 = torch.tensor(type_idx, dtype=torch.long).view(-1,1)
    x2 = torch.tensor(chirality_idx, dtype=torch.long).view(-1,1)
    x = torch.cat([x1, x2], dim=-1)

    row, col, edge_feat = [], [], []
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        row += [start, end]
        col += [end, start]
        edge_feat.append([
            BOND_LIST.index(bond.GetBondType()),
            BONDDIR_LIST.index(bond.GetBondDir())
        ])
        edge_feat.append([
            BOND_LIST.index(bond.GetBondType()),
            BONDDIR_LIST.index(bond.GetBondDir())
        ])

    edge_index = torch.tensor([row, col], dtype=torch.long)
    edge_attr = torch.tensor(np.array(edge_feat), dtype=torch.long)
    
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    return data


if __name__ == '__main__':
    args = parse_arguments()
    root = args.root

    output_dir = f'{root}/embed/molclr'
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(f'{root}/drugs.csv')
    info = list(set(zip( list(df['drug_id']), list(df['iso_smiles']) )))
    name_list = [pair[0] for pair in info]
    smiles_list = [pair[1] for pair in info]

    device = f'cuda:{str(args.gpu)}'
    model = EmbedNet()
    model_path = '/data/qingyuyang/dta_ours/weights/molclr/model.pth'
    state_dict = torch.load(model_path, map_location=device)
    model.load_my_state_dict(state_dict)
    print("Loaded trained model with success.")
    model.eval()
    model.to(device)

    with torch.no_grad():
        for name, smiles in tqdm(zip(name_list, smiles_list)):
            data = getitem(smiles)
            data = data.to(device)
            embed = model(data)
            torch.save(embed.clone().detach().cpu(), f'{output_dir}/{name}.pt')
