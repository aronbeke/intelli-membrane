import torch
from torch.utils.data import DataLoader

from tqdm import tqdm

from HiMol.data_utils import *
from HiMol.gnn_model import GNN

import models.featurization
from rdkit import Chem

import pandas as pd
import numpy as np

import os

def group_node_rep(node_rep, batch_size, num_part):
    group = []
    super_group = []
    # print('num_part', num_part)
    count = 0
    for i in range(batch_size):
        num_atom = num_part[i][0]
        num_motif = num_part[i][1]
        num_all = num_atom + num_motif + 1
        group.append(node_rep[count:count + num_atom])
        super_group.append(node_rep[count + num_all -1])
        count += num_all
    return group, super_group

class HiMol_extraction(object):
    def __init__(self):
        self.device = self._get_device()

    def _get_device(self):
        if torch.cuda.is_available():
            device = 'cuda:0'
            torch.cuda.set_device(device)
        else:
            device = 'cpu'
        print("Running on:", device)
        return device

    def extract_representation(self, args):
            
        model = GNN(args['num_layer'], args['emb_dim'], JK=args['JK'], drop_ratio=args['dropout_ratio'], gnn_type=args['gnn_type'])
        model.from_pretrained(args['input_model_file'])
        model.to(self.device)
        model.eval()

        # Create MolTestDataset with the list of SMILES
        dataset = MoleculeDataset(args['dataset']) # directory of txt file

        # Initialize an empty list to store representations
        representations = []

        # Create a DataLoader for the dataset
        loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=lambda x:x)

        # with torch.no_grad():
        #     for data in loader:
        #         #data = data.to(self.device)
                
        #         # Pass through the model to get the representation
        #         h, _ = model(data)  # Get h for learned representation
        #         representations.append(h.cpu().numpy())  # Store representation
        #         print("Representation obtained.")
        
        with torch.no_grad():
            for step, batch in enumerate(tqdm(loader, desc="Iteration")):
                    batch_size = len(batch)

                    graph_batch = molgraph_to_graph_data(batch)
                    graph_batch = graph_batch.to(self.device)
                    node_rep = model(graph_batch.x, graph_batch.edge_index, graph_batch.edge_attr)
                    num_part = graph_batch.num_part
                    node_rep, super_node_rep = group_node_rep(node_rep, batch_size, num_part)

                    #representations.append(np.array(super_node_rep))  # Store representation
                    representations.append(super_node_rep[0].cpu().numpy()) # with CUDA
        
        return representations
    

def process_himol_dataset(input_file, aux_solute_smiles_file, aux_solvent_smiles_file, output_file, args):
    data_df = pd.read_csv(input_file, index_col='id')

    # Initialize HiMol
    himol = HiMol_extraction()

    # List of SMILES strings to extract representations from
    smiles_list = data_df['solute_smiles'].tolist()
    smiles_list_final = []

    for smi in smiles_list:
        mol_nondat = Chem.MolFromSmiles(smi, sanitize=False)
        mol_dat = models.featurization.set_dative_bonds(mol_nondat)
        smiles_dat = Chem.MolToSmiles(mol_dat)
        smiles_list_final.append(smiles_dat)

    with open(aux_solute_smiles_file, "w") as f:
        for smiles in smiles_list_final:
            f.write(smiles + "\n")

    args['dataset'] = aux_solute_smiles_file

    # Extract representations
    representations = himol.extract_representation(args=args)

    # Flatten each representation and create a DataFrame with sequentially labeled columns
    flattened_representations = [rep.flatten() for rep in representations]
    representations_df = pd.DataFrame(
        flattened_representations,
        columns=[f"solute_himol{str(i).zfill(3)}" for i in range(flattened_representations[0].shape[0])]
    )

    # Solvents
    solvent_smiles_list = data_df['solvent_smiles'].tolist()

    with open(aux_solvent_smiles_file, "w") as f:
        for smiles in solvent_smiles_list:
            f.write(smiles + "\n")

    args['dataset'] = aux_solvent_smiles_file

    solvent_representations = himol.extract_representation(args=args)
    flattened_solvent_representations = [rep.flatten() for rep in solvent_representations]
    solvent_representations_df = pd.DataFrame(
        flattened_solvent_representations,
        columns=[f"solvent_himol{str(i).zfill(3)}" for i in range(flattened_solvent_representations[0].shape[0])]
    )

    data_df.reset_index(drop=True, inplace=True)
    representations_df.reset_index(drop=True, inplace=True)
    solvent_representations_df.reset_index(drop=True, inplace=True)

    # Concatenate the representations with the original data_df
    data_df = pd.concat([data_df, representations_df, solvent_representations_df], axis=1)

    # Save or view the final dataframe with added columns
    data_df.to_csv(output_file, index_label='id')


def process_himol(input_file, aux_smiles_file, output_file, args, column):

    data_df = pd.read_csv(input_file, index_col='id')

    # Initialize HiMol
    himol = HiMol_extraction()

    # List of SMILES strings to extract representations from
    smiles_list = data_df[column].tolist()
    smiles_list_final = []

    for smi in smiles_list:
        mol_nondat = Chem.MolFromSmiles(smi, sanitize=False)
        mol_dat = models.featurization.set_dative_bonds(mol_nondat)
        smiles_dat = Chem.MolToSmiles(mol_dat)
        smiles_list_final.append(smiles_dat)

    with open(aux_smiles_file, "w") as f:
        for smiles in smiles_list_final:
            f.write(smiles + "\n")

    args['dataset'] = aux_smiles_file

    # Extract representations
    representations = himol.extract_representation(args=args)

    # Flatten each representation and create a DataFrame with sequentially labeled columns
    flattened_representations = [rep.flatten() for rep in representations]
    representations_df = pd.DataFrame(
        flattened_representations,
        columns=[f"{column}_himol{str(i).zfill(3)}" for i in range(flattened_representations[0].shape[0])]
    )

    data_df.reset_index(drop=True, inplace=True)
    representations_df.reset_index(drop=True, inplace=True)

    # Concatenate the representations with the original data_df
    data_df = pd.concat([data_df, representations_df], axis=1)

    # Save or view the final dataframe with added columns
    data_df.to_csv(output_file, index_label='id')