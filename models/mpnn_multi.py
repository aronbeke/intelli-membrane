import numpy as np
import pandas as pd

import torch
import torch.nn as torch_nn
from torch import Tensor

#from torch.utils.data import random_split, Subset

from chemprop import data, featurizers, nn
from chemprop.models import multi
from chemprop.nn import metrics
from chemprop.nn.transforms import ScaleTransform
from chemprop.data import BatchMolGraph

from rdkit import Chem

from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import KFold

from lightning import pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, Callback
from lightning.pytorch.loggers import TensorBoardLogger

import matplotlib.pyplot as plt

from models.processing import clip_rejection

import models.attention

import joblib

from typing import Iterable

'''
Message Passing NN rejection prediction models based on graph data (chemprop implementation)
'''

nworkers = 8
prog_bar = False

## -- DATASET -- ##

class RejectionDataset:
    def __init__(self, dataframe, smiles_columns, extra_continuous_columns, extra_categorical_columns, target_column, loss_weight_column='loss_weight'):
        self.data_df = dataframe

        # Careful with column order!
        self.extra_continuous_columns = extra_continuous_columns
        self.extra_categorical_columns = extra_categorical_columns
        self.extra_feature_columns = extra_continuous_columns + extra_categorical_columns
        self.smiles_columns = smiles_columns
        self.target_column = target_column
        self.loss_weight_column = loss_weight_column

        self.x_ds = self.data_df.loc[:, self.extra_feature_columns].values

        self.build_datapoints()
        
    def build_datapoints(self):
        # Create molecule datapoints for chemprop, including extra features
        
        smiss = self.data_df.loc[:, self.smiles_columns].values  # Extract SMILES

        try:
            if self.target_column[0] in self.data_df.columns:
                assert self.loss_weight_column in self.data_df.columns
                self.has_targets = True
                self.targets = self.data_df.loc[:, self.target_column].values  # Target values (e.g., rejection)
                self.loss_weights = self.data_df.loc[:, self.loss_weight_column].values
                self.all_data = [[data.MoleculeDatapoint.from_smi(smis[0], y, x_d=x_d, weight=w) for smis, y, x_d, w in zip(smiss, self.targets, self.x_ds, self.loss_weights)]]
                self.all_data += [[data.MoleculeDatapoint.from_smi(smis[i]) if smis[i].strip() != "" else data.MoleculeDatapoint.from_smi("[*]") for smis in smiss] for i in range(1, len(self.smiles_columns))] # dummy SMILES [*] for empty secondary SMILES
                
            else:
                self.has_targets = False
                self.all_data = [[data.MoleculeDatapoint.from_smi(smis[0], x_d=x_d) for smis, x_d in zip(smiss, self.x_ds)]]
                self.all_data += [[data.MoleculeDatapoint.from_smi(smis[i]) if smis[i].strip() != "" else data.MoleculeDatapoint.from_smi("[*]") for smis in smiss] for i in range(1, len(self.smiles_columns))] # dummy SMILES [*] for empty secondary SMILES
        except TypeError:
            print(smiss)

    def clip_rejection(self, rejection_column):
        self.data_df = clip_rejection(self.data_df, rejection_column=rejection_column, return_copy=False)

    def retransform_rejection(self, nlc_rejection_column):
        new_rejection_column = nlc_rejection_column.replace('nlc_', '')
        self.data_df[new_rejection_column] = self.data_df[nlc_rejection_column].apply(
            lambda x: 1 - 10**(-x) + 1e-3
        )
        return new_rejection_column

    def scale_extra_features(self, extra_scaler):
        full_cont_data = self.data_df[self.extra_continuous_columns].values
        scaled_cont_data = extra_scaler.transform(full_cont_data)
        full_cat_data = self.data_df[self.extra_categorical_columns].values
        self.x_ds = np.concatenate([scaled_cont_data, full_cat_data], axis=1)
        self.build_datapoints()

    def export_data(self, destination):
        self.data_df.to_csv(destination)      

    def create_mcdset(self, featurizer):
        self.mol_datasets = [data.MoleculeDataset(self.all_data[i], featurizer) for i in range(len(self.all_data))]
        self.mol_mcdset = data.MulticomponentDataset(self.mol_datasets)

    
## -- MODELS -- ##

class MultilayerMulticomponentMPNN(multi.MulticomponentMPNN):
    def __init__(self, 
                 n_smiles_components=2, 
                 n_mp_layers=[2,2], 
                 hidden_dim_mp=[128,128], 
                 hidden_dim_ffn=128, 
                 n_layers=3, dropout=0.1, 
                 scaler=None, 
                 extra_scaler=None, 
                 extra_scaler_params=None, 
                 extra_features_dim=0, 
                 activation='relu', 
                 learning_rate=1e-4,
                 message_passing='atom',
                 aggregation='mean'
                 ):

        # Define the multicomponent GNN
        MessagePassingClass = nn.AtomMessagePassing if message_passing == 'atom' else nn.BondMessagePassing

        mcmp = nn.MulticomponentMessagePassing(
            blocks=[
                MessagePassingClass(
                    d_h=hidden_dim_mp[i],
                    bias=True,
                    depth=n_mp_layers[i],
                    undirected=True,
                    dropout=dropout,
                    activation=activation
                )
                for i in range(n_smiles_components)
            ],
            n_components=n_smiles_components
        )

        if aggregation == 'mean':
            agg = nn.MeanAggregation()
        elif aggregation == 'sum':
            agg = nn.SumAggregation()
        else:
            raise ValueError(f"Unsupported aggregation type: {aggregation}. Use 'mean' or 'sum'.")


        # Define the final regression layer (FFN)
        ffn = nn.RegressionFFN(
            input_dim=mcmp.output_dim + extra_features_dim,
            hidden_dim=hidden_dim_ffn,
            n_layers=n_layers,
            dropout=dropout,
            output_transform=nn.UnscaleTransform.from_standard_scaler(scaler),
            activation=activation
        )

        # Define metrics
        metric_list = [metrics.RMSEMetric(), metrics.MAEMetric(), metrics.R2Metric()]

        # Initialize the parent class (MulticomponentMPNN)
        super().__init__(mcmp, 
                         agg, 
                         ffn, 
                         metrics=metric_list,
                         init_lr=learning_rate,
                         )

        if extra_scaler is not None:
            extra_scaler_params = {
                'mean': extra_scaler.mean_.tolist(),
                'scale': extra_scaler.scale_.tolist()
            }

        self.extra_scaler = extra_scaler
        self.extra_scaler_params = extra_scaler_params

        self.save_hyperparameters(ignore=['extra_scaler'])  # Avoid saving raw object
    
    @classmethod
    def load_from_checkpoint(cls, checkpoint_path, learning_rate_ft=None, **kwargs):
        # Load hyperparameters from the checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=kwargs.get('map_location', None))
        hparams = checkpoint['hyper_parameters']

        # Extract your custom arguments from the saved hyperparameters
        n_smiles_components = hparams.get('n_smiles_components', 2)
        n_mp_layers = hparams.get('n_mp_layers', [2,2])
        hidden_dim_mp = hparams.get('hidden_dim_mp', [128,128])
        hidden_dim_ffn = hparams.get('hidden_dim_ffn', 128)
        n_layers = hparams.get('n_layers', 3)
        dropout = hparams.get('dropout', 0.1)
        scaler = hparams.get('scaler', None)
        extra_scaler_params = hparams.get('extra_scaler_params', None)
        if extra_scaler_params is not None:
            extra_scaler = StandardScaler()
            extra_scaler.mean_ = np.array(extra_scaler_params['mean'])
            extra_scaler.scale_ = np.array(extra_scaler_params['scale'])
        else:
            print('Extra scaler parameters not found in mcmpnn checkpoint!')
            extra_scaler = None
        extra_features_dim = hparams.get('extra_features_dim', 0)
        activation = hparams.get('activation', 'relu')
        message_passing = hparams.get('message_passing', 'atom')
        aggregation = hparams.get('aggregation', 'mean')

        if learning_rate_ft is None:
            learning_rate = hparams.get('learning_rate', 1e-4)
            print(f"Using learning rate from checkpoint = {learning_rate}")
        else:
            learning_rate = learning_rate_ft
            print(f"Overriding ckpt learning rate: using learning_rate_ft = {learning_rate}")

        if scaler is None:
            print('Target scaler is None when loading!')

        # Check if defaults were used
        for param_name in [
            "n_smiles_components",
            "n_mp_layers",
            "hidden_dim_mp",
            "hidden_dim_ffn",
            "n_layers",
            "dropout",
            "scaler",
            "extra_scaler_params",
            "extra_features_dim",
            "activation",
            "message_passing",
            "aggregation"
        ]:
            if param_name not in hparams:
                print(f"Warning: '{param_name}' not found in checkpoint, using default value.")


        # Initialize the model with the extracted arguments
        model = cls(
            n_smiles_components=n_smiles_components,
            n_mp_layers=n_mp_layers,
            hidden_dim_mp=hidden_dim_mp,
            hidden_dim_ffn=hidden_dim_ffn,
            n_layers=n_layers,
            dropout=dropout,
            scaler=scaler,
            extra_scaler = extra_scaler,
            extra_features_dim=extra_features_dim,
            learning_rate = learning_rate,
            activation=activation,
            message_passing=message_passing,
            aggregation=aggregation
        )

        # Load state dict
        model.load_state_dict(checkpoint['state_dict'], strict=kwargs.get('strict', True))
        return model
    

class AttentionDistributedMulticomponentMPNN(multi.MulticomponentMPNN):
    def __init__(self,
                 n_solute_components=1, 
                 shared_mp_layer_config=(3, 128), 
                 solvent_mp_layer_config=(3, 128), 
                 hidden_dim_ffn=128, 
                 n_ffn_layers=3, 
                 dropout=0.1, 
                 scaler=None, 
                 extra_scaler=None, 
                 extra_scaler_params=None, 
                 extra_features_dim=0, 
                 activation='relu',
                 atom_attentive_aggregation = True,
                 learning_rate=1e-4,
                 message_passing='atom',
                 d_attn=8
                 ):

        self.n_solute_components = n_solute_components
        # For now this is hard-coded
        self.n_solvent_components = 1
        self.atom_attentive_aggregation = atom_attentive_aggregation

        MessagePassingClass = nn.AtomMessagePassing if message_passing == 'atom' else nn.BondMessagePassing

        shared_mp = MessagePassingClass(
            d_h=shared_mp_layer_config[1],
            depth=shared_mp_layer_config[0],
            bias=True,
            undirected=True,
            dropout=dropout,
            activation=activation
        )

        solvent_mp = MessagePassingClass(
            d_h=solvent_mp_layer_config[1],
            depth=solvent_mp_layer_config[0],
            bias=True,
            undirected=True,
            dropout=dropout,
            activation=activation
        )

        blocks = [shared_mp] * self.n_solute_components + [solvent_mp] * self.n_solvent_components
        mcmp = nn.MulticomponentMessagePassing(blocks=blocks, n_components= (self.n_solute_components + self.n_solvent_components))
        
        # Not used if atom attentive aggregation
        agg = nn.MeanAggregation()

        ffn = nn.RegressionFFN(
            input_dim=shared_mp_layer_config[1] + solvent_mp_layer_config[1] + extra_features_dim,
            hidden_dim=hidden_dim_ffn,
            n_layers=n_ffn_layers,
            dropout=dropout,
            output_transform=nn.UnscaleTransform.from_standard_scaler(scaler),
            activation=activation
        )

        metric_list = [metrics.RMSEMetric(), metrics.MAEMetric(), metrics.R2Metric()]

        super().__init__(mcmp, agg, ffn, metrics=metric_list, init_lr=learning_rate)

        self.attn = models.attention.AttentionAggregator(d_model=shared_mp_layer_config[1],d_attn=d_attn, temperature=3.0)

        if atom_attentive_aggregation:
            self.shared_atom_att_agg = models.attention.AtomAttentiveAggregation(output_size=shared_mp_layer_config[1], temperature=3.0)
            self.solvent_atom_att_agg = models.attention.AtomAttentiveAggregation(output_size=solvent_mp_layer_config[1], temperature=3.0)

        # Override BatchNorm
        self.bn = torch_nn.BatchNorm1d(shared_mp_layer_config[1]+solvent_mp_layer_config[1])

        if extra_scaler is not None:
            extra_scaler_params = {
                'mean': extra_scaler.mean_.tolist(),
                'scale': extra_scaler.scale_.tolist()
            }

        self.extra_scaler = extra_scaler
        self.extra_scaler_params = extra_scaler_params

        self.return_attn = False
        self.d_attn = d_attn

        self.save_hyperparameters(ignore=['extra_scaler'])

    def is_dummy(self, component: BatchMolGraph) -> torch.Tensor:
        """
        Check if a component (MolGraph inside BatchMolGraph) is a dummy datapoint.
        Assumes dummy datapoints are initialized with empty atom features (V) or have no atoms.
        """
        # Create a tensor of 1s and 0s based on whether the atom feature matrix is empty
        return torch.tensor(component.V.numel() == 0, dtype=torch.float, device=component.V.device).unsqueeze(0)

    def fingerprint(
        self,
        bmgs: list[BatchMolGraph],
        V_ds: list[Tensor | None],
        X_d: Tensor | None = None,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """
        Computes the molecular fingerprint by applying message passing, attention over solutes,
        and concatenating with solvent and optional extra features.

        Returns:
            - Fingerprint tensor of shape [B, D_total]
            - Optionally, attention weights of shape [B, n_solute_components] if self.return_attn is True
        """
        # Get hidden representations per component
        H_vs = self.message_passing(bmgs, V_ds)  # List of [N_atoms, D]

        if self.atom_attentive_aggregation:
            atom_attn_weights = []  # Will store per-component per-molecule weights
            Hs = []

            for i, (H_v, bmg) in enumerate(zip(H_vs, bmgs)):
                if i < self.n_solute_components:
                    H_i = self.shared_atom_att_agg(H_v, bmg.batch)
                    attn_module = self.shared_atom_att_agg
                else:
                    H_i = self.solvent_atom_att_agg(H_v, bmg.batch)
                    attn_module = self.solvent_atom_att_agg

                Hs.append(H_i)

                if isinstance(attn_module, models.attention.AtomAttentiveAggregation):
                    assert isinstance(attn_module.last_alphas, list), "Expected list of [n_atoms_i, 1] tensors"
                    atom_attn_weights.append(attn_module.last_alphas)  # list of [n_atoms_i, 1]
                else:
                    atom_attn_weights.append([None] * H_i.shape[0])  # Same batch size, but no weights

            # Transpose from [n_components][B] → [B][n_components]
            atom_attn_weights = list(map(list, zip(*atom_attn_weights)))
        else:
            #print([H_v.shape for H_v, bmg in zip(H_vs, bmgs)])
            Hs = [self.agg(H_v, bmg.batch) for H_v, bmg in zip(H_vs, bmgs)]  # List of [B, D]

        # Assume the first n_solute components are solutes; the rest are solvents
        solute_Hs = Hs[: self.n_solute_components]
        solvent_Hs = Hs[self.n_solute_components :]  # Could be multiple solvents if extended

        # Stack solute reps: [B, n_solutes, D]
        solute_reps = torch.stack(solute_Hs, dim=1)

        # # Build solute mask: 1 for valid, 0 for dummy
        # Mask of shape [B, n_solutes], initialized with 1s (assume all real)
        B = solute_Hs[0].size(0)
        device = solute_Hs[0].device
        solute_mask = torch.ones((B, self.n_solute_components), dtype=torch.bool, device=device)

        # Loop through each solute component's BatchMolGraph and identify dummies
        for j, bmg in enumerate(bmgs[:self.n_solute_components]):
            for i in range(B):
                atom_mask = (bmg.batch == i)
                if not atom_mask.any():  # No atoms => dummy
                    solute_mask[i, j] = 0

        # Attention over solute components
        solute_agg, attn_weights = self.attn(solute_reps, solute_mask)  # Both [B, D] and [B, n_solutes]

        # Process solvent(s) – assuming one solvent for now
        solvent_rep = solvent_Hs[0] if solvent_Hs else torch.zeros_like(solute_agg)

        # Combine solute and solvent representations: [B, D_combined]
        H = torch.cat([solute_agg, solvent_rep], dim=1)

        # Batch norm
        H = self.bn(H)

        # Optional extra features
        if X_d is not None:
            H = torch.cat([H, self.X_d_transform(X_d)], dim=1)
        
        # Each atom_attn_weights[i] is shaped [N_atoms_i, 1] and aligned with atom indices in bmgs[i]
        if self.return_attn and self.atom_attentive_aggregation:
            return H, attn_weights, atom_attn_weights
        elif self.return_attn and not self.atom_attentive_aggregation:
            return H, attn_weights
        else:
            return H

    def forward(
        self,
        bmgs: list[BatchMolGraph],
        V_ds: list[Tensor | None],
        X_d: Tensor | None = None,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """
        Forward pass: returns prediction or (prediction, attention_weights, atom_attention_weights).
        """
        fp = self.fingerprint(bmgs, V_ds, X_d)
        if self.return_attn and self.atom_attentive_aggregation:
            H, attn_weights, atom_attn_weights = fp
            return self.predictor(H), attn_weights, atom_attn_weights
        elif self.return_attn and not self.atom_attentive_aggregation:
            H, attn_weights = fp
            return self.predictor(H), attn_weights
        else:
            return self.predictor(fp)
    
    def set_return_attn(self, value):
        """Set whether attention should be returned in the forward pass."""
        self.return_attn = value

    @classmethod
    def load_from_checkpoint(cls, checkpoint_path, learning_rate_ft=None, **kwargs):
        # Load hyperparameters from the checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=kwargs.get('map_location', None))
        hparams = checkpoint['hyper_parameters']

        # Extract your custom arguments from the saved hyperparameters
        n_solute_components = hparams.get('n_solute_components', 1)
        shared_mp_layer_config = hparams.get('shared_mp_layer_config', (3, 128))
        solvent_mp_layer_config = hparams.get('solvent_mp_layer_config', (3, 128))
        hidden_dim_ffn = hparams.get('hidden_dim_ffn', 128)
        n_ffn_layers = hparams.get('n_ffn_layers', 3)
        dropout = hparams.get('dropout', 0.1)
        scaler = hparams.get('scaler', None)
        extra_scaler_params = hparams.get('extra_scaler_params', None)
        if extra_scaler_params is not None:
            extra_scaler = StandardScaler()
            extra_scaler.mean_ = np.array(extra_scaler_params['mean'])
            extra_scaler.scale_ = np.array(extra_scaler_params['scale'])
        else:
            print('Extra scaler parameters not found in mcmpnn checkpoint!')
            extra_scaler = None
        extra_features_dim = hparams.get('extra_features_dim', 0)
        activation = hparams.get('activation', 'relu')
        message_passing = hparams.get('message_passing', 'atom')
        d_attn = hparams.get('d_attn', 8)
        atom_attentive_aggregation = hparams.get('atom_attentive_aggregation', True)

        if learning_rate_ft is None:
            learning_rate = hparams.get('learning_rate', 1e-4)
            print(f"Using learning rate from checkpoint = {learning_rate}")
        else:
            learning_rate = learning_rate_ft
            print(f"Overriding ckpt learning rate: using learning_rate_ft = {learning_rate}")

        if scaler is None:
            print('Target scaler is None when loading!')

        # Check if defaults were used
        for param_name in [
            "shared_mp_layer_config",
            "solvent_mp_layer_config",
            "hidden_dim_ffn",
            "n_ffn_layers",
            "dropout",
            "scaler",
            "extra_scaler_params",
            "extra_features_dim",
            "activation",
            "d_attn",
            "message_passing",
            "atom_attentive_aggregation",
            "n_solute_components"
        ]:
            if param_name not in hparams:
                print(f"Warning: '{param_name}' not found in checkpoint, using default value.")


        # Initialize the model with the extracted arguments
        model = cls(
            n_solute_components=n_solute_components,
            shared_mp_layer_config=shared_mp_layer_config,
            solvent_mp_layer_config=solvent_mp_layer_config,
            hidden_dim_ffn=hidden_dim_ffn,
            n_ffn_layers=n_ffn_layers,
            dropout=dropout,
            scaler=scaler,
            extra_scaler = extra_scaler,
            extra_features_dim=extra_features_dim,
            learning_rate = learning_rate,
            activation=activation,
            d_attn=d_attn,
            message_passing=message_passing,
            atom_attentive_aggregation=atom_attentive_aggregation
        )

        # Load state dict
        model.load_state_dict(checkpoint['state_dict'], strict=kwargs.get('strict', True))
        return model


## -- PREDICTORS -- ##

class RejectionPredictor:
    def __init__(self, device_id=0):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #self.device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
        self.model_name = 'rejection_model'

    def assign_dataset(self, dataset: RejectionDataset):
        '''
        For general prediction purposes
        '''
        self.rejection_dataset = dataset

    def set_scaler(self, dataset: data.MulticomponentDataset):
        self.scaler = dataset.normalize_targets()

    def apply_scaler(self, dataset: data.MulticomponentDataset):
        dataset.normalize_targets(self.scaler)

    def dump_extra_scaler(self, extra_scaler_path):
        joblib.dump(self.extra_scaler, extra_scaler_path)

    def read_extra_scaler(self, extra_scaler_path):
        self.extra_scaler = joblib.load(extra_scaler_path)

    def data_preprocess_for_training(self, featurizer=featurizers.SimpleMoleculeMolGraphFeaturizer(), fine_tune=False, scaler = None, extra_scaler = None, scale_extras = False):
        self.featurizer = featurizer

        if self.featurizer is None:
            print('Falling back on simple featurizer.')
            self.featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()

        if fine_tune:
            self.scaler = scaler
            if self.scaler is None:
                raise ValueError("scaler is None for fine tuning! Make sure to load it from the pretrained model.")

        # --- Input (x_d) scaling ---
        # Get raw continuous and categorical data
        cont_cols = self.train_rejection_dataset.extra_continuous_columns

        # Fit scaler only on training data
        train_cont_data = self.train_rejection_dataset.data_df.loc[:, cont_cols].values

        if not fine_tune:
            self.extra_scaler = StandardScaler().fit(train_cont_data)
        else:
            self.extra_scaler = extra_scaler

        if self.extra_scaler is None:
            raise ValueError("extra_scaler is None!")

        # Apply scaler to all rows (train + val)
        if scale_extras:
            self.train_rejection_dataset.scale_extra_features(self.extra_scaler)
            self.val_rejection_dataset.scale_extra_features(self.extra_scaler)

        # --- Construct MC MoleculeDatapoints with updated x_d ---

        self.train_rejection_dataset.create_mcdset(featurizer=featurizer)
        self.val_rejection_dataset.create_mcdset(featurizer=featurizer)

        if fine_tune:
            self.apply_scaler(self.train_rejection_dataset.mol_mcdset)
            self.apply_scaler(self.val_rejection_dataset.mol_mcdset)
        else:
            # Normalize training targets and apply the same scaler to validation targets
            self.set_scaler(self.train_rejection_dataset.mol_mcdset)
            self.apply_scaler(self.val_rejection_dataset.mol_mcdset)  # Normalize validation targets using training scaler

        if self.scaler is None:
            raise ValueError("scaler is None! Make sure to load it from the pretrained model.")

    def data_preprocess_for_prediction(self, featurizer=featurizers.SimpleMoleculeMolGraphFeaturizer(), extra_scaler = None, scale_extras = False):
        self.featurizer = featurizer

        print('Loading extra_scaler for prediction')
        self.extra_scaler = extra_scaler
        if self.extra_scaler is None:
            raise ValueError("extra_scaler is None! Make sure to load it from the pretrained model.")
        
        if scale_extras:
            self.rejection_dataset.scale_extra_features(self.extra_scaler)
        self.rejection_dataset.create_mcdset(featurizer=featurizer)

    def train(self,
              train_rejection_dataset: RejectionDataset,
              val_rejection_dataset: RejectionDataset,
              model_type = 'mcmpnn',
              model_name='rejection_model',
              dir_path=None,
              n_smiles_components=2, 
              n_mp_layers=[2,2], # only for mcmpnn
              hidden_dim_mp=[128,128], # only for mcmpnn
              shared_mp_layer_config=(3, 128), # only for admpnn
              solvent_mp_layer_config=(3, 128), # only for admpnn
              hidden_dim_ffn=128, 
              n_ffn_layers=3, 
              dropout=0.1, 
              epochs=150, 
              learning_rate = 1e-4,
              d_attn=8, # only for admpnn
              atom_attentive_aggregation=True, # only for admpnn
              activation='relu',
              message_passing='atom',
              aggregation='mean', # only for mcmpnn
              scale_extras = False
              ):
        
        self.train_rejection_dataset = train_rejection_dataset
        self.val_rejection_dataset = val_rejection_dataset
        self.model_name = model_name

        if dir_path is None and model_type == 'mcmpnn':
            dir_path = "checkpoints_mcmpnn/"
        elif dir_path is None and model_type == 'admpnn':
            dir_path = "checkpoints_admpnn/"

        self.data_preprocess_for_training(scale_extras=scale_extras)

        # Create model
        if model_type == 'mcmpnn':
            self.model = MultilayerMulticomponentMPNN(
                n_smiles_components=n_smiles_components,
                n_mp_layers=n_mp_layers,
                hidden_dim_mp=hidden_dim_mp,
                hidden_dim_ffn=hidden_dim_ffn,
                n_layers=n_ffn_layers,
                dropout=dropout,
                scaler=self.scaler,
                extra_scaler=self.extra_scaler,
                learning_rate=learning_rate,
                extra_features_dim=len(self.train_rejection_dataset.extra_feature_columns),
                activation=activation,
                message_passing=message_passing,
                aggregation=aggregation
            )
        elif model_type == 'admpnn':
            self.model = AttentionDistributedMulticomponentMPNN(
                n_solute_components=n_smiles_components-1,
                shared_mp_layer_config=shared_mp_layer_config,
                solvent_mp_layer_config=solvent_mp_layer_config,
                hidden_dim_ffn=hidden_dim_ffn,
                n_ffn_layers=n_ffn_layers,
                dropout=dropout,
                scaler=self.scaler,
                extra_scaler=self.extra_scaler,
                learning_rate=learning_rate,
                extra_features_dim=len(self.train_rejection_dataset.extra_feature_columns),
                d_attn=d_attn,
                atom_attentive_aggregation=atom_attentive_aggregation,
                activation=activation,
                message_passing=message_passing
            )
        else:
            raise ValueError("Invalid model type. Choose 'mcmpnn' or 'admpnn'.")
        self.model = self.model.to(self.device)

        # Define checkpoints
        checkpoint_callback = ModelCheckpoint(
            dirpath=dir_path,  # Directory to save checkpoints
            filename=model_name+"-{epoch:02d}-{val_loss_weighted:.2f}",  # Filename format
            save_top_k=1,  # Save the top 1 model (best performance)
            monitor="val_loss_weighted",  # Metric to monitor (change to your validation metric)
            mode="min"  # Save model with the lowest validation loss
        )

        # # Define the early stopping callback
        # early_stopping_callback = EarlyStopping(
        #     monitor="val_loss_weighted",  # Metric to monitor, change to your validation metric
        #     patience=5,          # Number of epochs with no improvement after which training will be stopped
        #     mode="min"           # "min" for minimizing the monitored metric, "max" for maximizing
        # )
        # Trainer from PyTorch Lightning

        # Logger
        logger = TensorBoardLogger("logs", name=model_type+"_"+model_name)

        # Trainer
        trainer = pl.Trainer(
            callbacks = [checkpoint_callback],
            logger=logger,
            enable_checkpointing=True,
            enable_progress_bar=prog_bar,
            accelerator="gpu",
            devices=1,   #  [self.device.index],
            max_epochs=epochs
        )

        # Create dataloaders
        train_loader = data.build_dataloader(self.train_rejection_dataset.mol_mcdset, num_workers=nworkers)
        val_loader = data.build_dataloader(self.val_rejection_dataset.mol_mcdset, shuffle=False, num_workers=nworkers)

        # Train model
        trainer.fit(model=self.model, train_dataloaders=train_loader, val_dataloaders=val_loader)

        # Load the best model after training
        self.best_model_path = checkpoint_callback.best_model_path
        print(f"Best model saved at: {self.best_model_path}")
        if model_type == 'mcmpnn':
            self.model = MultilayerMulticomponentMPNN.load_from_checkpoint(self.best_model_path)
        elif model_type == 'admpnn':
            self.model = AttentionDistributedMulticomponentMPNN.load_from_checkpoint(self.best_model_path)

        # Test the model
        self.validation_results = trainer.validate(self.model, dataloaders=val_loader)
        assert len(self.validation_results) == 1

        # Logged metrics are returned in scaled space
        scaled_rmse = self.validation_results[0]['val/rmse']
        scaled_mae = self.validation_results[0]['val/mae']
        scaled_r2 = self.validation_results[0]['val/r2']

        self.validation_results[0]['val/rmse_original'] = (scaled_rmse * self.model.predictor.output_transform.scale.numpy()).item()
        self.validation_results[0]['val/mae_original'] = (scaled_mae * self.model.predictor.output_transform.scale.numpy()).item()

        rmse_original = self.validation_results[0]['val/rmse_original']
        mae_original = self.validation_results[0]['val/mae_original']

        print('Training validation results:')
        print(f'Val RMSE: {rmse_original:.4f}')
        print(f'Val MAE: {mae_original:.4f}')
        print(f'Val R2 (scaled): {scaled_r2:.4f}')

        return self.validation_results


    def predict(self, model_type = 'mcmpnn', checkpoint_path=None, featurizer=featurizers.SimpleMoleculeMolGraphFeaturizer(), model_name='', forget_predictions = False, return_attn = False, scale_extras=False):
        '''
        Prediction evaluation always returns metrics for rejection, not NLC rejection.
        '''

        if not hasattr(self, 'rejection_dataset'):
            raise ValueError("A RejectionDataset must be initiated.")

        # Load model from checkpoint if provided
        if model_type == 'mcmpnn':
            if checkpoint_path is not None:
                self.model = MultilayerMulticomponentMPNN.load_from_checkpoint(checkpoint_path)
            elif not hasattr(self, 'model'):
                raise ValueError("Model (self.model) must be initialized or a checkpoint must be provided.")
        elif model_type == 'admpnn':
            if checkpoint_path is not None:
                self.model = AttentionDistributedMulticomponentMPNN.load_from_checkpoint(checkpoint_path)
            elif not hasattr(self, 'model'):
                raise ValueError("Model (self.model) must be initialized or a checkpoint must be provided.")

        if self.model.extra_scaler is None:
            raise ValueError("No extra_scaler found in the mcmpnn model for predictions.")

        # Prepare the test dataset
        self.data_preprocess_for_prediction(extra_scaler=self.model.extra_scaler, featurizer=featurizer, scale_extras=scale_extras)
        # Create a test loader
        test_loader = data.build_dataloader(self.rejection_dataset.mol_mcdset, shuffle=False, num_workers=nworkers)

        # Predict with the model
        with torch.inference_mode():
            trainer = pl.Trainer(
                logger=None,
                enable_progress_bar=prog_bar,
                accelerator="gpu",
                devices=1, #[self.device.index]
            )

            if return_attn and model_type == 'admpnn':
                self.model.set_return_attn(True)
                outputs = trainer.predict(self.model, test_loader)
                if self.model.atom_attentive_aggregation:
                    preds, attn_weights_batchlist, atom_attn_weights_batchlist = zip(*outputs)
                else:
                    preds, attn_weights_batchlist = zip(*outputs)
                self.model.set_return_attn(False)
            else:
                preds = trainer.predict(self.model, test_loader)

        # Concatenate predictions
        preds = np.concatenate(preds, axis=0)

        # ATTENTION HANDLING
        if return_attn and model_type == 'admpnn':
            attn_weights = [aw.detach().cpu().numpy() for aw in attn_weights_batchlist]
            attn_weights = np.concatenate(attn_weights, axis=0)  # Shape: [N, n_solutes]

            # Add each attention weight column separately
            for i in range(attn_weights.shape[1]):
                self.rejection_dataset.data_df[f'attn_weight_{i}_{model_name}'] = attn_weights[:, i]


        if return_attn and model_type == 'admpnn' and self.model.atom_attentive_aggregation:
            # Flatten [B][n_components][n_atoms_i, 1] → [N_records][n_components][n_atoms_i, 1]
            flat_attn_weights = []
            for batch in atom_attn_weights_batchlist:
                for record_attns in batch:
                    flat_record = [aw.detach().cpu().numpy() for aw in record_attns]
                    flat_attn_weights.append(flat_record)

            # Sanity checks
            assert len(flat_attn_weights) == len(self.rejection_dataset.data_df), \
                f"Mismatch: {len(flat_attn_weights)} weights vs {len(self.rejection_dataset.data_df)} records"

            #print("Attention shape check:", flat_attn_weights[0][0].shape)

            n_solutes = self.model.n_solute_components
            atom_attns_dict = {j: [] for j in range(n_solutes)}
            raw_atom_attns_dict = {j: [] for j in range(n_solutes)}
            norm_raw_atom_attns_dict = {j: [] for j in range(n_solutes)}

            for i, row in self.rejection_dataset.data_df.iterrows():
                row_smiles = []
                if 'solute_smiles' in row and pd.notna(row['solute_smiles']):
                    row_smiles.append(row['solute_smiles'])
                if 'secondary_solute_smiles' in row and pd.notna(row['secondary_solute_smiles']):
                    row_smiles.append(row['secondary_solute_smiles'])

                for j in range(n_solutes):
                    smi = row_smiles[j]
                    molecule = Chem.MolFromSmiles(smi)

                    try:
                        frag_map = models.attention.get_ligand_fragment_map(molecule)
                    except ValueError:
                        print('Fragmentation failed for molecule:', smi)
                        atom_attns_dict[j].append({})
                        continue

                    if j >= len(flat_attn_weights[i]):
                        print(f"Warning: Missing attention weights for solute {j} in record {i}")
                        atom_attns_dict[j].append({})
                        continue

                    attn_array = flat_attn_weights[i][j]  # [n_atoms_i, 1]
                    if attn_array.ndim == 2 and attn_array.shape[1] == 1:
                        atom_attn = attn_array.squeeze(1)  # [n_atoms]
                    elif attn_array.ndim == 1:
                        atom_attn = attn_array  # already squeezed
                    else:
                        raise ValueError(f"Unexpected attention shape: {attn_array.shape}")

                    weights_dict = {}
                    if frag_map is not None:
                        for frag_name, atom_indices in frag_map.items():
                            weight_sum = float(sum(atom_attn[k] for k in atom_indices))
                            weights_dict[frag_name] = (weight_sum, len(atom_indices))

                    # --- Compute mean attention per heteroatom type ---
                    heteroatoms = ["O", "N", "P", "S", "F", "Cl"]

                    for symbol in heteroatoms:
                        atom_indices = [idx for idx, atom in enumerate(molecule.GetAtoms()) if atom.GetSymbol() == symbol]
                        if atom_indices:
                            mean_attn = float(np.mean([atom_attn[k] for k in atom_indices]))
                            weights_dict[symbol] = (mean_attn, len(atom_indices))
                        else:
                            weights_dict[symbol] = (None, 0)

                    atom_attns_dict[j].append(weights_dict)

                    # --- Compute mean attention for different carbon hybridizations ---
                    carbon_types = {
                        "C_primary": 1,    # carbon attached to 1 other heavy atom
                        "C_secondary": 2,  # carbon attached to 2 other heavy atoms
                        "C_tertiary": 3,   # carbon attached to 3 other heavy atoms
                        "C_quaternary": 4  # carbon attached to 4 other heavy atoms
                    }

                    for label, degree in carbon_types.items():
                        atom_indices = [
                            idx for idx, atom in enumerate(molecule.GetAtoms())
                            if atom.GetSymbol() == "C" and atom.GetDegree() == degree
                        ]
                        if atom_indices:
                            mean_attn = float(np.mean([atom_attn[k] for k in atom_indices]))
                            weights_dict[label] = (mean_attn, len(atom_indices))
                        else:
                            weights_dict[label] = (None, 0)

                    # --- Oxygen and Nitrogen hybridizations ---
                    # For simplicity: Primary = degree 1, Secondary = degree 2, Tertiary = degree 3
                    atom_types = {
                        "O_primary": ("O", 1),
                        "O_secondary": ("O", 2),
                        "N_primary": ("N", 1),
                        "N_secondary": ("N", 2),
                        "N_tertiary": ("N", 3)
                    }

                    for label, (symbol, degree) in atom_types.items():
                        atom_indices = [
                            idx for idx, atom in enumerate(molecule.GetAtoms())
                            if atom.GetSymbol() == symbol and atom.GetDegree() == degree
                        ]
                        if atom_indices:
                            mean_attn = float(np.mean([atom_attn[k] for k in atom_indices]))
                            weights_dict[label] = (mean_attn, len(atom_indices))
                        else:
                            weights_dict[label] = (None, 0)

                    # --- Atoms connecting to metals ---
                    metal_atomic_numbers = set(range(21, 31)) | set(range(39, 49)) | set(range(57, 81)) | set(range(89, 113))
                    for symbol in ["C", "N", "O"]:
                        atom_indices = []
                        for atom in molecule.GetAtoms():
                            if atom.GetSymbol() != symbol:
                                continue
                            # Check if any neighbor is a metal
                            if any(neigh.GetAtomicNum() in metal_atomic_numbers for neigh in atom.GetNeighbors()):
                                atom_indices.append(atom.GetIdx())
                        label = f"{symbol}_metal_bonded"
                        if atom_indices:
                            mean_attn = float(np.mean([atom_attn[k] for k in atom_indices]))
                            weights_dict[label] = (mean_attn, len(atom_indices))
                        else:
                            weights_dict[label] = (None, 0)

                    # === Raw atom-level attention dictionary ===
                    raw_weights_dict = {}
                    norm_raw_weights_dict = {}

                    num_atoms = molecule.GetNumAtoms()

                    for idx, atom in enumerate(molecule.GetAtoms()):
                        atom_label = f"{atom.GetSymbol()}{idx}"
                        raw_weights_dict[atom_label] = float(atom_attn[idx])
                        norm_raw_weights_dict[atom_label] = float(atom_attn[idx]) * num_atoms

                    raw_atom_attns_dict[j].append(raw_weights_dict)
                    norm_raw_atom_attns_dict[j].append(norm_raw_weights_dict)

            # Save into dataframe for available solutes
            for j in range(n_solutes):
                suffix = 'primary' if j == 0 else 'secondary'
                self.rejection_dataset.data_df[f'atom_attns_{suffix}_{model_name}'] = pd.Series(atom_attns_dict[j], dtype=object)
                self.rejection_dataset.data_df[f'raw_atom_attns_{suffix}_{model_name}'] = pd.Series(raw_atom_attns_dict[j], dtype=object)
                self.rejection_dataset.data_df[f'norm_raw_atom_attns_{suffix}_{model_name}'] = pd.Series(norm_raw_atom_attns_dict[j], dtype=object)

        # Save predictions to the dataset

        if not forget_predictions:
            self.rejection_dataset.data_df[self.rejection_dataset.target_column[0] + '_' + model_name] = preds

            if self.rejection_dataset.target_column[0] == 'rejection':
                self.rejection_dataset.clip_rejection(rejection_column=self.rejection_dataset.target_column[0] + '_' + model_name)
            elif self.rejection_dataset.target_column[0] == 'nlc_rejection':
                new_rejection_column = self.rejection_dataset.retransform_rejection(nlc_rejection_column=self.rejection_dataset.target_column[0] + '_' + model_name)
                self.rejection_dataset.clip_rejection(rejection_column=new_rejection_column)
            else:
                raise ValueError("Unsupported target column for rejection prediction. Use 'rejection' or 'nlc_rejection'.")

        if self.rejection_dataset.has_targets:
            # Always calculate on rejection

            true_values = self.rejection_dataset.data_df['rejection'].values  # True target values

            if self.rejection_dataset.target_column[0] == 'rejection':
                predicted_values = preds
            elif self.rejection_dataset.target_column[0] == 'nlc_rejection':
                predicted_values = self.rejection_dataset.data_df[new_rejection_column].values

            rmse = root_mean_squared_error(true_values, predicted_values)
            mae = mean_absolute_error(true_values, predicted_values)
            r2 = r2_score(true_values, predicted_values)  # Compute R²

            # Print metrics
            print(f"Test RMSE: {rmse:.4f}")
            print(f"Test MAE: {mae:.4f}")
            print(f"Test R2: {r2:.4f}")
        else:
            rmse = None
            mae = None
            r2 = None

        return rmse, mae, r2


    def fine_tune(self, 
                  train_rejection_dataset: RejectionDataset, 
                  val_rejection_dataset: RejectionDataset,
                  dir_path="checkpoints_mcmpnn/", 
                  mp_blocks_to_freeze = [0, 1], 
                  layers_to_freeze_per_block = {0: 1, 1: 1}, 
                  frzn_ffn_layers = 1, 
                  fine_tuned_model_name='rejection_model', 
                  pretrained_checkpoint_path=None, 
                  featurizer=featurizers.SimpleMoleculeMolGraphFeaturizer(), 
                  epochs=150, 
                  learning_rate_ft=None,
                  scale_extras=False):
        '''
        Fine-tune the model on a different dataset (transfer learning)
        '''
        
        self.train_rejection_dataset = train_rejection_dataset
        self.val_rejection_dataset = val_rejection_dataset

        if pretrained_checkpoint_path is not None:
            print(f"Loading model from checkpoint: {pretrained_checkpoint_path}")
            self.model = MultilayerMulticomponentMPNN.load_from_checkpoint(pretrained_checkpoint_path, learning_rate_ft=learning_rate_ft)
        elif not hasattr(self, 'mcmpnn'):
            raise ValueError("Model (self.model) must be initialized or a pretrained model checkpoint must be provided.")

        # Load scaler
        if hasattr(self.model.predictor, 'output_transform'):
            scaler = self.model.predictor.output_transform.to_standard_scaler()
        else:
            raise ValueError("Pretrained model does not have a valid output_transform.")

        if self.model.extra_scaler is None or not hasattr(self.model, 'extra_scaler'):
            raise ValueError("No extra_scaler found in the mcmpnn model for fine tuning.")

        # Process data
        self.data_preprocess_for_training(fine_tune=True, featurizer=featurizer, scaler=scaler, extra_scaler=self.model.extra_scaler, scale_extras=scale_extras)

        for i in mp_blocks_to_freeze:
            mp_block = self.model.message_passing.blocks[i]
            mp_layers = list(mp_block.children())

            num_layers_to_freeze = layers_to_freeze_per_block.get(i, 0)  # Default to 0 if not in dict

            for j, layer in enumerate(mp_layers):
                if j < num_layers_to_freeze:
                    layer.apply(lambda module: module.requires_grad_(False))
                    layer.eval()

                    if hasattr(layer, 'bn'):
                        layer.bn.eval()
                        layer.bn.apply(lambda module: module.requires_grad_(False))


        for idx in range(frzn_ffn_layers):
            if idx < len(self.model.predictor.ffn):
                self.model.predictor.ffn[idx].requires_grad_(False)
                self.model.predictor.ffn[idx].eval() # This was idx+1 in the documentation for some reason
            else:
                print(f"Warning: FFN layer index {idx} is out of range. Skipping.")

        # Trainer setup
        checkpoint_callback = ModelCheckpoint(
            dirpath=dir_path,  # Directory to save checkpoints
            filename=fine_tuned_model_name+"-{epoch:02d}-{val_loss_weighted:.2f}",  # Filename format
            save_top_k=1,  # Save the top 1 model (best performance)
            monitor="val_loss_weighted",  # Metric to monitor (change to your validation metric)
            mode="min"  # Save model with the lowest validation loss
        )

        logger = TensorBoardLogger("logs", name="mcmpnn_ft_"+fine_tuned_model_name)

        # Trainer from PyTorch Lightning
        trainer = pl.Trainer(
            callbacks = [checkpoint_callback],
            logger=logger,
            enable_checkpointing=True,
            enable_progress_bar=prog_bar,
            accelerator="gpu",
            devices=1, #[self.device.index],
            max_epochs=epochs
        )

        # Create dataloaders
        train_loader = data.build_dataloader(self.train_rejection_dataset.mol_mcdset, num_workers=nworkers)
        val_loader = data.build_dataloader(self.val_rejection_dataset.mol_mcdset, shuffle=False, num_workers=nworkers)

        # Train model
        trainer.fit(self.model, train_loader, val_loader)

        # Load the best model after training
        self.best_model_path = checkpoint_callback.best_model_path
        self.model = MultilayerMulticomponentMPNN.load_from_checkpoint(self.best_model_path)

        self.fine_tune_val_results = trainer.validate(self.model, val_loader)

        # Logged metrics are returned in scaled space
        scaled_ft_val_rmse = self.fine_tune_val_results[0]['val/rmse']
        scaled_ft_val_mae = self.fine_tune_val_results[0]['val/mae']
        scaled_r2 = self.fine_tune_val_results[0]['val/r2']

        self.fine_tune_val_results[0]['val/rmse_original'] = (scaled_ft_val_rmse * self.model.predictor.output_transform.scale.numpy()).item()
        self.fine_tune_val_results[0]['val/mae_original'] = (scaled_ft_val_mae * self.model.predictor.output_transform.scale.numpy()).item()

        rmse_ft_original = self.fine_tune_val_results[0]['val/rmse_original']
        mae_ft_original = self.fine_tune_val_results[0]['val/mae_original']

        print('Fine tuning validation results:')
        print(f'Val RMSE: {rmse_ft_original:.4f}')
        print(f'Val MAE: {mae_ft_original:.4f}')
        print(f'Val R2 (scaled): {scaled_r2:.4f}')

        return self.fine_tune_val_results
    
    def fine_tune_from_mpnn(self,
                            train_rejection_dataset: RejectionDataset, 
                            val_rejection_dataset: RejectionDataset,
                            dir_path="checkpoints_mcmpnn/",
                            n_smiles_components = 2,
                            n_mp_layers = [5, 3], # First is overriden from MPNN!
                            hidden_dim_mp = [512, 512],  # First is overriden from MPNN!
                            layers_to_freeze_in_mpnn = 1,
                            hidden_dim_ffn = 512,
                            n_ffn_layers = 1, 
                            fine_tuned_model_name='rejection_ft_model', 
                            pretrained_mpnn_path='checkpoints_mpnn/model.ckpt',
                            dropout = 0.2,
                            learning_rate_combi_ft=None,
                            epochs=150,
                            featurizer=featurizers.SimpleMoleculeMolGraphFeaturizer(),
                            activation='leakyrelu',
                            message_passing='atom',
                            aggregation='mean',
                            scale_extras=False):

        self.train_rejection_dataset = train_rejection_dataset
        self.val_rejection_dataset = val_rejection_dataset

        if pretrained_mpnn_path is not None:
            print(f"Loading MPNN model from checkpoint: {pretrained_mpnn_path}")
            from models.mpnn_single import MultilayerMPNN
            self.mpnn = MultilayerMPNN.load_from_checkpoint(pretrained_mpnn_path)
            pretrained_mp = self.mpnn.message_passing
        else:
            raise ValueError("Pretrained MPNN model checkpoint must be provided.")

        self.data_preprocess_for_training(featurizer=featurizer, scale_extras=scale_extras)

        # Create model
        self.model = MultilayerMulticomponentMPNN(
            n_smiles_components=n_smiles_components,
            n_mp_layers=n_mp_layers,
            hidden_dim_mp=hidden_dim_mp,
            hidden_dim_ffn=hidden_dim_ffn,
            n_layers=n_ffn_layers,
            dropout=dropout,
            scaler=self.scaler,
            extra_scaler=self.extra_scaler,
            learning_rate=learning_rate_combi_ft,
            extra_features_dim=len(self.train_rejection_dataset.extra_feature_columns),
            activation=activation,
            message_passing=message_passing,
            aggregation=aggregation
        )

        self.model.message_passing.blocks[0] = pretrained_mp  # Replace the first MP block
        print('First n_mp_layer and first hidden_dim_mp overriden from MPNN.')

        mp_block = self.model.message_passing.blocks[0]
        mp_layers = list(mp_block.children())

        for j, layer in enumerate(mp_layers):
            if j < layers_to_freeze_in_mpnn:
                layer.apply(lambda module: module.requires_grad_(False))
                layer.eval()

                if hasattr(layer, 'bn'):
                    layer.bn.eval()
                    layer.bn.apply(lambda module: module.requires_grad_(False))

        # Trainer setup
        checkpoint_callback = ModelCheckpoint(
            dirpath=dir_path,  # Directory to save checkpoints
            filename=fine_tuned_model_name+"-{epoch:02d}-{val_loss_weighted:.2f}",  # Filename format
            save_top_k=1,  # Save the top 1 model (best performance)
            monitor="val_loss_weighted",  # Metric to monitor (change to your validation metric)
            mode="min"  # Save model with the lowest validation loss
        )

        logger = TensorBoardLogger("logs", name="mcmpnn_combi_ft_"+fine_tuned_model_name)

        # Trainer from PyTorch Lightning
        trainer = pl.Trainer(
            callbacks = [checkpoint_callback],
            logger=logger,
            enable_checkpointing=True,
            enable_progress_bar=prog_bar,
            accelerator="gpu",
            devices=1, #[self.device.index],
            max_epochs=epochs
        )

        # Create dataloaders
        train_loader = data.build_dataloader(self.train_rejection_dataset.mol_mcdset, num_workers=nworkers)
        val_loader = data.build_dataloader(self.val_rejection_dataset.mol_mcdset, shuffle=False, num_workers=nworkers)

        # Train model
        trainer.fit(self.model, train_loader, val_loader)

        # Load the best model after training
        self.best_model_path = checkpoint_callback.best_model_path
        self.model = MultilayerMulticomponentMPNN.load_from_checkpoint(self.best_model_path)

        self.fine_tune_val_results = trainer.validate(self.model, val_loader)

        # Logged metrics are returned in scaled space
        scaled_ft_val_rmse = self.fine_tune_val_results[0]['val/rmse']
        scaled_ft_val_mae = self.fine_tune_val_results[0]['val/mae']
        scaled_r2 = self.fine_tune_val_results[0]['val/r2']

        self.fine_tune_val_results[0]['val/rmse_original'] = (scaled_ft_val_rmse * self.model.predictor.output_transform.scale.numpy()).item()
        self.fine_tune_val_results[0]['val/mae_original'] = (scaled_ft_val_mae * self.model.predictor.output_transform.scale.numpy()).item()

        rmse_ft_original = self.fine_tune_val_results[0]['val/rmse_original']
        mae_ft_original = self.fine_tune_val_results[0]['val/mae_original']

        print('Combined fine tuning validation results:')
        print(f'Val RMSE: {rmse_ft_original:.4f}')
        print(f'Val MAE: {mae_ft_original:.4f}')
        print(f'Val R2 (scaled): {scaled_r2:.4f}')

        return self.fine_tune_val_results