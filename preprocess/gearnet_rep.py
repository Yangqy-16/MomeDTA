"""
This script is based on the discussion in GearNet's repository (https://github.com/DeepGraphLearning/GearNet).
We support batch processing in this script.

NOTE: Go to the definition of data.Protein.from_pdb in line 53 and change line 328 to 
      `mol = Chem.MolFromPDBFile(pdb_file, sanitize=False, proximityBonding=True)` (set kwargs),
      otherwise reading some PDBs may cause error!

After running this script, you can load the generated protein instances by: 
    torch.load('###.pt')
"""

import os
from tqdm import tqdm
import torch
from torchdrug import data, models, layers
from torchdrug.layers import geometry
from utils import *
import warnings
warnings.filterwarnings('ignore')


if __name__ == "__main__":
    args = parse_arguments()
    batch_size = args.bs
    root = args.root
    output_dir = f'{root}/embed/gearnet'
    os.makedirs(output_dir, exist_ok=True)

    # graph
    graph_construction_model = layers.GraphConstruction(node_layers=[geometry.AlphaCarbonNode()], 
                                                        edge_layers=[geometry.SpatialEdge(radius=10.0, min_distance=5),
                                                                     geometry.KNNEdge(k=10, min_distance=5),
                                                                     geometry.SequentialEdge(max_distance=2)],
                                                        edge_feature="gearnet")

    # model
    gearnet_edge = models.GearNet(input_dim=21, hidden_dims=[512, 512, 512, 512, 512, 512],
                                  num_relation=7, edge_input_dim=59, num_angle_bin=8,
                                  batch_norm=True, concat_hidden=True, short_cut=True, readout="sum")
    pthfile = '/data/qingyuyang/dta_ours/weights/gearnet/mc_gearnet_edge.pth'  # NOTE: Change to your path
    net = torch.load(pthfile, map_location=torch.device(f"cuda:{args.gpu}"))
    gearnet_edge.load_state_dict(net)
    gearnet_edge.eval()
    print('Successfully load GearNet!')

    # --------------- generate representations ---------------
    df = pd.read_csv(f'{root}/prots.csv')
    all_list = list(set(zip(df['prot_id'], df['pdb'])))
    undone_pdbs = select_undone_items(all_list, output_dir)

    for idx in tqdm(range(0, len(undone_pdbs), batch_size)):  # reformulate to batches
        pdb_batch = undone_pdbs[idx : min(len(undone_pdbs), idx + batch_size)]

        # construct Protein instances
        proteins = []
        for prot_id, pdb_file in pdb_batch:
            protein = data.Protein.from_pdb(pdb_file, atom_feature="position", bond_feature="length", residue_feature="symbol") #
            protein.view = "residue"
            proteins.append(protein)
        assert len(proteins) == len(pdb_batch)

        # protein
        _protein = data.Protein.pack(proteins)
        _protein.view = "residue"
        final_protein = graph_construction_model(_protein)

        with torch.no_grad():
            output = gearnet_edge(final_protein, final_protein.node_feature.float(), all_loss=None, metric=None)

        # save
        counter = 0
        for idx, (prot_id, pdb_file) in enumerate(pdb_batch):  # idx: protein/graph id in this batch
            this_node_feature = output['node_feature'][counter : counter + final_protein.num_residues[idx], :]
            
            torch.save(this_node_feature, f"{output_dir}/{prot_id}.pt")
            counter += final_protein.num_residues[idx]
