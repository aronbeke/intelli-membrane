import numpy as np
import pandas as pd
import torch

from chemprop import data, featurizers, models, nn
from chemprop.nn import metrics

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import KFold

from lightning import pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, Callback
from lightning.pytorch.loggers import TensorBoardLogger

import matplotlib.pyplot as plt

from models.processing import clip_rejection

import joblib

'''
Message Passing NN permeance prediction models based on graph data (chemprop implementation)
'''

nworkers = 4
accelerator = 'auto'

class MolDataset:
    def __init__(self, dataframe, smiles_column, extra_columns, target_column):
        self.data_df = dataframe

        self.extra_feature_columns = extra_columns
        self.smiles_column = smiles_column
        self.target_column = target_column

        if len(smiles_column) != 1:
            print('ERROR - Wrong number of molecular SMILES')

        smiss = self.data_df.loc[:, self.smiles_column].values  # Extract SMILES
        if target_column[0] in self.data_df.columns:
            self.has_targets = True
            self.targets = self.data_df.loc[:, self.target_column].values  # Target values (e.g., permeance)
        else:
            self.has_targets = False
        
        # Additional features like pressure, temperature, and one-hot encoded membrane
        self.x_ds = self.data_df.loc[:, self.extra_feature_columns].values
        
        # Create molecule datapoints for chemprop, including extra features
        if target_column[0] in self.data_df.columns:
            self.all_data = [data.MoleculeDatapoint.from_smi(smis[0], y, x_d=x_d) for smis, y, x_d in zip(smiss, self.targets, self.x_ds)]
        else:
            self.all_data = [data.MoleculeDatapoint.from_smi(smis[0], x_d=x_d) for smis, x_d in zip(smiss, self.x_ds)]

    def split_data(self, split_ratios=(0.9, 0.1), model_name = ''):
        """
        Splits data into training, validation, and test sets.
        """
        mols = [d.mol for d in self.all_data]

        if len(split_ratios) == 2:
            split_ratios = (split_ratios[0], 0.0, split_ratios[1])  # Add test set with 0%

        self.train_indices, __, self.val_indices = data.make_split_indices(mols, "random", split_ratios)

        self.train_data, __, self.val_data = data.split_data_by_indices(
            self.all_data, self.train_indices, None, self.val_indices
        )

        # Initialize columns with 0s
        self.data_df['train_'+model_name] = 0
        self.data_df['val_'+model_name] = 0
        self.data_df['test_'+model_name] = 0

        # Set corresponding rows to 1 based on the split indices
        self.data_df.loc[self.train_indices, 'train_'+model_name] = 1
        self.data_df.loc[self.val_indices, 'val_'+model_name] = 1
    
    def clip_rejection(self, rejection_column):
        self.data_df = clip_rejection(self.data_df, rejection_column=rejection_column, return_copy=False)

    def export_data(self, destination):
        self.data_df.to_csv(destination)
    

class MultilayerMPNN(models.MPNN):
    def __init__(self, n_mp_layers=1, hidden_dim_mp=128, hidden_dim_ffn=128, n_layers=3, dropout=0.1, scaler=None, extra_features_dim=0, activation='relu'):

        # Define the multicomponent GNN
        mp = nn.AtomMessagePassing(
                    d_h=hidden_dim_mp,  # Hidden dimension size
                    bias=True,
                    depth=n_mp_layers,  # Number of message passing iterations
                    undirected=True,
                    dropout=dropout,
                    activation=activation  # Activation function after each iteration
                )

        agg = nn.MeanAggregation()

        # Define the final regression layer (FFN)
        ffn = nn.RegressionFFN(
            input_dim=mp.output_dim + extra_features_dim,
            hidden_dim=hidden_dim_ffn,
            n_layers=n_layers,
            dropout=dropout,
            output_transform=nn.UnscaleTransform.from_standard_scaler(scaler)
        )

        # Define metrics
        metric_list = [metrics.RMSEMetric(), metrics.MAEMetric(), metrics.R2Metric()]

        # Batch norm?
        batch_norm = True

        # Initialize the parent class
        super().__init__(mp, agg, ffn, batch_norm, metrics=metric_list)
    
    @classmethod
    def load_from_checkpoint(cls, checkpoint_path, **kwargs):
        # Load hyperparameters from the checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=kwargs.get('map_location', None))
        hparams = checkpoint['hyper_parameters']

        # Extract your custom arguments from the saved hyperparameters
        n_mp_layers = hparams.get('n_mp_layers', 1)
        hidden_dim_mp = hparams.get('hidden_dim_mp', 128)
        hidden_dim_ffn = hparams.get('hidden_dim_ffn', 128)
        n_layers = hparams.get('n_layers', 3)
        dropout = hparams.get('dropout', 0.1)
        scaler = hparams.get('scaler', None)
        extra_features_dim = hparams.get('extra_features_dim', 0)
        activation = hparams.get('activation', 'relu')

    
        # Check if defaults were used
        for param_name in [
            "n_mp_layers",
            "hidden_dim_mp",
            "hidden_dim_ffn",
            "n_layers",
            "dropout",
            "scaler",
            "extra_features_dim",
            "activation"
        ]:
            if param_name not in hparams:
                print(f"Warning: '{param_name}' not found in checkpoint, using default value.")

        # Initialize the model with the extracted arguments
        model = cls(
            n_mp_layers=n_mp_layers,
            hidden_dim_mp=hidden_dim_mp,
            hidden_dim_ffn=hidden_dim_ffn,
            n_layers=n_layers,
            dropout=dropout,
            scaler=scaler,
            extra_features_dim=extra_features_dim,
            activation=activation
        )

        # Load state dict
        model.load_state_dict(checkpoint['state_dict'], strict=kwargs.get('strict', True))
        return model


class MolPredictor:
    def __init__(self, dataset: MolDataset):
        self.assign_dataset(dataset)

    def assign_dataset(self, dataset: MolDataset):
        self.dataset = dataset

    def set_scaler(self, dataset):
        self.scaler = dataset.normalize_targets()

    def apply_scaler(self, dataset):
        dataset.normalize_targets(self.scaler)
    
    def data_preprocess(self, model_name = '', split_ratios=(0.85, 0.15), featurizer=featurizers.SimpleMoleculeMolGraphFeaturizer()):
        # Define featurizers
        self.featurizer = featurizer

        self.dataset.split_data(split_ratios=split_ratios, model_name=model_name)

        self.train_dataset = data.MoleculeDataset(self.dataset.train_data, self.featurizer)
        self.val_dataset = data.MoleculeDataset(self.dataset.val_data, self.featurizer)
        
        # Normalize training targets and apply the same scaler to validation targets
        self.set_scaler(self.train_dataset)
        self.apply_scaler(self.val_dataset)  # Normalize validation targets using training scaler

    def train(self, model_name='mol_model', dir_path="checkpoints_mpnn/", split=(0.85,0.15), n_mp_layers=1, hidden_dim_mp=128, hidden_dim_ffn=128, n_ffn_layers=3, dropout=0.1, epochs=150, activation='relu'):

        self.data_preprocess(model_name=model_name, split_ratios= split)

        self.mpnn = MultilayerMPNN(
            n_mp_layers=n_mp_layers,
            hidden_dim_mp=hidden_dim_mp,
            hidden_dim_ffn=hidden_dim_ffn,
            n_layers=n_ffn_layers,
            dropout=dropout,
            scaler=self.scaler,
            extra_features_dim=len(self.dataset.extra_feature_columns),
            activation=activation
        )

        checkpoint_callback = ModelCheckpoint(
            dirpath=dir_path,  # Directory to save checkpoints
            filename=model_name+"-{epoch:02d}-{val_loss:.2f}",  # Filename format
            save_top_k=1,  # Save the top 1 model (best performance)
            monitor="val_loss",  # Metric to monitor (change to your validation metric)
            mode="min"  # Save model with the lowest validation loss
        )

        # # Define the early stopping callback
        # early_stopping_callback = EarlyStopping(
        #     monitor="val_loss",  # Metric to monitor, change to your validation metric
        #     patience=5,          # Number of epochs with no improvement after which training will be stopped
        #     mode="min"           # "min" for minimizing the monitored metric, "max" for maximizing
        # )
        # Trainer from PyTorch Lightning

        logger = TensorBoardLogger("logs", name="mpnn_"+model_name)

        trainer = pl.Trainer(
            callbacks = [checkpoint_callback],
            logger=logger,
            enable_checkpointing=True,
            enable_progress_bar=True,
            accelerator=accelerator,
            devices=1,
            max_epochs=epochs
        )

        # Create dataloaders
        train_loader = data.build_dataloader(self.train_dataset, num_workers=nworkers)
        val_loader = data.build_dataloader(self.val_dataset, shuffle=False, num_workers=nworkers)

        # Train model
        trainer.fit(self.mpnn, train_loader, val_loader)

        # Load the best model after training
        best_model_path = checkpoint_callback.best_model_path
        print(f"Best model saved at: {best_model_path}")
        self.mpnn = MultilayerMPNN.load_from_checkpoint(best_model_path)

        # Test the model
        self.validation_results = trainer.validate(self.mpnn, dataloaders=val_loader)

        # Logged metrics are returned in scaled space
        scaled_rmse = self.validation_results[0]['val/rmse']
        scaled_mae = self.validation_results[0]['val/mae']
        scaled_r2 = self.validation_results[0]['val/r2']

        self.validation_results[0]['val/rmse_original'] = (scaled_rmse * self.mpnn.predictor.output_transform.scale.numpy()).item()
        self.validation_results[0]['val/mae_original'] = (scaled_mae * self.mpnn.predictor.output_transform.scale.numpy()).item()

        rmse_original = self.validation_results[0]['val/rmse_original']
        mae_original = self.validation_results[0]['val/mae_original']

        print('Training validation results:')
        print(f'Val RMSE: {rmse_original:.4f}')
        print(f'Val MAE: {mae_original:.4f}')
        print(f'Val R2 (scaled): {scaled_r2:.4f}')

        return self.validation_results
    
    def predict(self, checkpoint_path=None, featurizer=None, model_name='', pad_tests = True):

        if not hasattr(self, 'dataset'):
            raise ValueError("A MolDataset must be initiated.")

        if checkpoint_path != None:
            self.mpnn = MultilayerMPNN.load_from_checkpoint(checkpoint_path)
        elif not hasattr(self, 'mpnn'):
            raise ValueError("Model (self.mpnn) must be initialized or a checkpoint must be provided.")

        if featurizer != None:
            self.featurizer = featurizer
        elif not hasattr(self, 'featurizer'):
            print("Featurizer not found. Using SimpleMoleculeMolGraphFeaturizer().")
            self.featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()

        dset = data.MoleculeDataset(self.dataset.all_data, self.featurizer)
        loader = data.build_dataloader(dset, shuffle=False, num_workers=nworkers)

        with torch.inference_mode():
            trainer = pl.Trainer(
                logger=None,
                enable_progress_bar=True,
                accelerator=accelerator,
                devices=1
            )
            preds = trainer.predict(self.mpnn, loader)
            
        preds = np.concatenate(preds, axis=0)
    
        if self.dataset.has_targets:
            # Inverse transform predictions and targets
            true_values = self.dataset.data_df[self.dataset.target_column[0]].values  # True target values

            rmse = root_mean_squared_error(true_values, preds)
            mae = mean_absolute_error(true_values, preds)
            r2 = r2_score(true_values, preds)  # Compute R²

            # Print metrics
            print(f"Test RMSE: {rmse:.4f}")
            print(f"Test MAE: {mae:.4f}")
            print(f"Test R2: {r2:.4f}")
        else:
            rmse = None
            mae = None
            r2 = None

        # Save predictions to the dataset

        if pad_tests:
            # Initialize columns with 0s
            self.dataset.data_df['train_'+model_name] = 0
            self.dataset.data_df['val_'+model_name] = 0
            self.dataset.data_df['test_'+model_name] = 1

        self.dataset.data_df[self.dataset.target_column[0] + '_' + model_name] = preds

        return rmse, mae, r2