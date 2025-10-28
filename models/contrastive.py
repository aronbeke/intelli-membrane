import torch
from torch_geometric.data import DataLoader
import os

from MolCLR.dataset.dataset_test import MolTestDataset

import yaml
import pandas as pd
import models.featurization
from rdkit import Chem

class MolCLR_extraction(object):
    def __init__(self, config):
        self.config = config
        self.device = self._get_device()

    def _get_device(self):
        if torch.cuda.is_available() and self.config['gpu'] != 'cpu':
            device = self.config['gpu']
            torch.cuda.set_device(device)
        else:
            device = 'cpu'
        print("Running on:", device)
        return device

    def _load_pre_trained_weights(self, model):
        try:
            checkpoints_folder = os.path.join('./MolCLR/ckpt', self.config['load_model'], 'checkpoints')
            state_dict = torch.load(os.path.join(checkpoints_folder, 'model.pth'), map_location=self.device)
            model.load_state_dict(state_dict)
            print("Loaded pre-trained model successfully.")
        except FileNotFoundError:
            print("Pre-trained weights not found. Please ensure the checkpoint path is correct.")
        return model

    def extract_representation(self, smiles_list, model_type='gcn'):
        # Create MolTestDataset with the list of SMILES
        dataset = MolTestDataset(smiles_list=smiles_list, task="regression")
        
        # Load the GCN model and initialize it with pretrained weights
        if model_type == 'gcn':
            from MolCLR.models.gcn_molclr import GCN  # Adjust the import if necessary
            model = GCN(**self.config["model"]).to(self.device)
            model = self._load_pre_trained_weights(model)
            model.eval()  # Set to evaluation mode
        elif model_type == 'gin':
            from MolCLR.models.ginet_molclr import GINet  # Adjust the import if necessary
            model = GINet(**self.config["model"]).to(self.device)
            model = self._load_pre_trained_weights(model)
            model.eval()  # Set to evaluation mode

        # Initialize an empty list to store representations
        representations = []

        # Create a DataLoader for the dataset
        loader = DataLoader(dataset, batch_size=1, shuffle=False)

        with torch.no_grad():
            for data in loader:
                data = data.to(self.device)
                
                # Pass through the model to get the representation
                h, _ = model(data)  # Get h for learned representation
                representations.append(h.cpu().numpy())  # Store representation

        return representations
    

def process_molclr_dataset(input_file, output_file, config_file, model_type):
    # PROCESS CATALYST DATASET - GCN
    data_df = pd.read_csv(input_file, index_col='id')

    # Example usage
    config = yaml.load(open(config_file, "r"), Loader=yaml.FullLoader)

    # Initialize MolCLR
    molclr = MolCLR_extraction(config=config)

    # List of SMILES strings to extract representations from
    smiles_list = data_df['solute_smiles'].tolist()
    smiles_list_with_dative_bonds = []

    for smi in smiles_list:
        mol_nondat = Chem.MolFromSmiles(smi, sanitize=False)
        mol_dat = models.featurization.set_dative_bonds(mol_nondat)
        smiles_dat = Chem.MolToSmiles(mol_dat)
        smiles_list_with_dative_bonds.append(smiles_dat)

    # Extract representations
    representations = molclr.extract_representation(smiles_list_with_dative_bonds,model_type=model_type)

    # Flatten each representation and create a DataFrame with sequentially labeled columns
    flattened_representations = [rep.flatten() for rep in representations]
    representations_df = pd.DataFrame(
        flattened_representations,
        columns=[f"solute_molclr{str(i).zfill(3)}" for i in range(flattened_representations[0].shape[0])]
    )

    # Solvents
    solvent_smiles_list = data_df['solvent_smiles'].tolist()
    solvent_representations = molclr.extract_representation(solvent_smiles_list,model_type=model_type)
    flattened_solvent_representations = [rep.flatten() for rep in solvent_representations]
    solvent_representations_df = pd.DataFrame(
        flattened_solvent_representations,
        columns=[f"solvent_molclr{str(i).zfill(3)}" for i in range(flattened_solvent_representations[0].shape[0])]
    )

    data_df.reset_index(drop=True, inplace=True)
    representations_df.reset_index(drop=True, inplace=True)
    solvent_representations_df.reset_index(drop=True, inplace=True)

    # Concatenate the representations with the original data_df
    data_df = pd.concat([data_df, representations_df, solvent_representations_df], axis=1)

    # Save or view the final dataframe with added columns
    data_df.to_csv(output_file, index_label='id')


def process_molclr(input_file, output_file, config_file, model_type, column, drop=[]):
    # PROCESS CATALYST DATASET - GCN
    data_df = pd.read_csv(input_file, index_col='id')
    data_df = data_df[~data_df[column].isin(drop)]

    # Example usage
    config = yaml.load(open(config_file, "r"), Loader=yaml.FullLoader)

    # Initialize MolCLR
    molclr = MolCLR_extraction(config=config)

    # List of SMILES strings to extract representations from
    smiles_list = data_df[column].tolist()
    smiles_list_with_dative_bonds = []

    for smi in smiles_list:
        mol_nondat = Chem.MolFromSmiles(smi, sanitize=False)
        mol_dat = models.featurization.set_dative_bonds(mol_nondat)
        smiles_dat = Chem.MolToSmiles(mol_dat)
        smiles_list_with_dative_bonds.append(smiles_dat)

    # Extract representations
    representations = molclr.extract_representation(smiles_list_with_dative_bonds,model_type=model_type)

    # Flatten each representation and create a DataFrame with sequentially labeled columns
    flattened_representations = [rep.flatten() for rep in representations]
    representations_df = pd.DataFrame(
        flattened_representations,
        columns=[f"{column}_molclr{str(i).zfill(3)}" for i in range(flattened_representations[0].shape[0])]
    )
    
    data_df.reset_index(drop=True, inplace=True)
    representations_df.reset_index(drop=True, inplace=True)

    # Concatenate the representations with the original data_df
    data_df = pd.concat([data_df, representations_df], axis=1)

    # Save or view the final dataframe with added columns
    data_df.to_csv(output_file, index_label='id')