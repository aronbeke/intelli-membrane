import torch
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import KFold

import torch.nn as nn
from torch.utils.data import Subset, DataLoader, Dataset, random_split

from lightning import pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, Callback
from lightning.pytorch.loggers import TensorBoardLogger

import matplotlib.pyplot as plt

from models.processing import clip_rejection

import joblib

'''
Standard rejection prediction models based on tabular data (featurized molecules)
'''

nworkers = 4
prog_bar = False

# Define the dataset class
class StandardRejectionDataset(Dataset):
    def __init__(self, dataframe, molecule_features_prefix, solvent_features_prefix, extra_continuous_columns, extra_categorical_columns, target_column):
        self.data_df = dataframe
        
        self.extra_continuous_columns = extra_continuous_columns
        self.extra_categorical_columns = extra_categorical_columns
        self.extra_feature_columns = extra_continuous_columns + extra_categorical_columns
        self.target_column = target_column
        
        self.molecule_rep_columns = [col for col in self.data_df.columns if col.startswith(molecule_features_prefix)]
        self.solvent_rep_columns = [col for col in self.data_df.columns if col.startswith(solvent_features_prefix)]

        molecule_reps = self.data_df[self.molecule_rep_columns]
        solvent_reps = self.data_df[self.solvent_rep_columns]
        extra_features = self.data_df[self.extra_feature_columns]

        self.molecule_reps = torch.tensor(molecule_reps.values, dtype=torch.float32)
        self.solvent_reps = torch.tensor(solvent_reps.values, dtype=torch.float32)
        self.extra_features = torch.tensor(extra_features.values, dtype=torch.float32)

        if target_column in self.data_df.columns:
            self.targets = torch.tensor(self.data_df[target_column].values, dtype=torch.float32)
            self.has_targets = True
        else:
            # Populate targets with zeros if target_column is not provided or doesn't exist
            self.targets = torch.zeros(len(self.data_df), dtype=torch.float32)
            self.has_targets = False

    def clip_rejection(self, rejection_column):
        self.data_df = clip_rejection(self.data_df, rejection_column=rejection_column, return_copy=False)
        
    def export_data(self, destination):
        self.data_df.to_csv(destination)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return (
            self.molecule_reps[idx],
            self.solvent_reps[idx],
            self.extra_features[idx],
            self.targets[idx],
        )


# Define the LightningModule
class RejectionModel(pl.LightningModule):
    def __init__(self, input_dim, number_of_hidden_layers=2, hidden_dim=512, dropout_rate=0.2, lr=0.001, extra_scaler=None):
        super(RejectionModel, self).__init__()

        self.save_hyperparameters()

        layers = [nn.Linear(input_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.LeakyReLU(0.01), nn.Dropout(dropout_rate)]

        # Dynamically add hidden layers
        for _ in range(number_of_hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.LeakyReLU(0.01))
            layers.append(nn.Dropout(dropout_rate))

        layers.append(nn.Linear(hidden_dim, 1))  # Output layer

        self.model = nn.Sequential(*layers)
        self.criterion = nn.MSELoss()
        self.lr = lr
        self.scaler = StandardScaler()
        self.scaler_fitted = False  # Flag to ensure scaler is fitted before use
        self.extra_scaler = extra_scaler

    def fit_scaler(self, dataset):
        """
        Fit the StandardScaler on the targets in the training dataset.
        Call this before training the model.
        """
        targets = [sample[-1].item() for sample in dataset]  # Extract targets from the dataset
        targets = torch.tensor(targets).unsqueeze(-1).numpy()  # Convert to numpy for StandardScaler
        self.scaler.fit(targets)
        self.scaler_fitted = True

    def on_save_checkpoint(self, checkpoint):
        checkpoint['scaler'] = self.scaler
        checkpoint['scaler_fitted'] = self.scaler_fitted
        checkpoint['extra_scaler'] = self.extra_scaler

    def on_load_checkpoint(self, checkpoint):
        self.scaler = checkpoint['scaler']
        self.scaler_fitted = checkpoint.get('scaler_fitted', False)
        self.extra_scaler = checkpoint['extra_scaler']

    def forward(self, molecule_rep, solvent_rep, extra_features):
        x = torch.cat((molecule_rep, solvent_rep, extra_features), dim=1)
        return self.model(x).squeeze()

    def _compute_metrics(self, predictions, scaled_targets):
        """Compute RMSE, MAE, and R2 in both scaled and original spaces."""

        # Ensure proper shapes
        predictions = predictions.view(-1)
        scaled_targets = scaled_targets.view(-1)

        loss = self.criterion(predictions, scaled_targets.squeeze())
        scaled_rmse = torch.sqrt(loss)
        scaled_mae = torch.mean(torch.abs(predictions - scaled_targets.squeeze()))

        # Convert back to original space
        std = self.scaler.scale_[0] if hasattr(self.scaler, "scale_") else 1.0
        original_rmse = scaled_rmse * std
        original_mae = scaled_mae * std

        # Convert to NumPy for sklearn metrics
        preds_np = predictions.cpu().detach().numpy()
        targets_np = scaled_targets.cpu().numpy()

        # Compute R2
        scaled_r2 = r2_score(targets_np, preds_np)

        original_predictions = self.scaler.inverse_transform(preds_np.reshape(-1, 1)).flatten()
        original_targets = self.scaler.inverse_transform(targets_np.reshape(-1, 1)).flatten()
        original_r2 = r2_score(original_targets, original_predictions)

        # original_predictions = self.scaler.inverse_transform(predictions.unsqueeze(-1).cpu().detach().numpy()).flatten()
        # original_r2 = r2_score(self.scaler.inverse_transform(scaled_targets.cpu().numpy()), original_predictions)

        return loss, scaled_rmse, scaled_mae, scaled_r2, original_rmse, original_mae, original_r2

    def training_step(self, batch, batch_idx):
        if not self.scaler_fitted:
            raise RuntimeError("Scaler is not fitted. Call `fit_scaler` before training.")

        molecule_rep, solvent_rep, extra_features, targets = batch
        scaled_targets = torch.tensor(self.scaler.transform(targets.unsqueeze(-1).cpu()), dtype=torch.float32).to(self.device)

        predictions = self(molecule_rep, solvent_rep, extra_features)
        loss, scaled_rmse, scaled_mae, scaled_r2, original_rmse, original_mae, original_r2 = self._compute_metrics(predictions, scaled_targets)

        self.log("train_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_rmse", scaled_rmse, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_mae", scaled_mae, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_r2", scaled_r2, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_rmse_original", original_rmse, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_mae_original", original_mae, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_r2_original", original_r2, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        molecule_rep, solvent_rep, extra_features, targets = batch
        scaled_targets = torch.tensor(self.scaler.transform(targets.unsqueeze(-1).cpu()), dtype=torch.float32).to(self.device)

        predictions = self(molecule_rep, solvent_rep, extra_features)
        loss, scaled_rmse, scaled_mae, scaled_r2, original_rmse, original_mae, original_r2 = self._compute_metrics(predictions, scaled_targets)

        self.log("val_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_rmse", scaled_rmse, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_mae", scaled_mae, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_r2", scaled_r2, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_rmse_original", original_rmse, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_mae_original", original_mae, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_r2_original", original_r2, on_epoch=True, prog_bar=True, logger=True)

    def test_step(self, batch, batch_idx):
        molecule_rep, solvent_rep, extra_features, targets = batch
        scaled_targets = torch.tensor(self.scaler.transform(targets.unsqueeze(-1).cpu()), dtype=torch.float32).to(self.device)

        predictions = self(molecule_rep, solvent_rep, extra_features)
        loss, scaled_rmse, scaled_mae, scaled_r2, original_rmse, original_mae, original_r2 = self._compute_metrics(predictions, scaled_targets)

        self.log("test_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        self.log("test_rmse", scaled_rmse, on_epoch=True, prog_bar=True, logger=True)
        self.log("test_mae", scaled_mae, on_epoch=True, prog_bar=True, logger=True)
        self.log("test_r2", scaled_r2, on_epoch=True, prog_bar=True, logger=True)
        self.log("test_rmse_original", original_rmse, on_epoch=True, prog_bar=True, logger=True)
        self.log("test_mae_original", original_mae, on_epoch=True, prog_bar=True, logger=True)
        self.log("test_r2_original", original_r2, on_epoch=True, prog_bar=True, logger=True)

    def predict_step(self, batch, batch_idx):
        molecule_rep, solvent_rep, extra_features, _ = batch
        predictions = self(molecule_rep, solvent_rep, extra_features)

        preds = predictions
        if isinstance(preds, torch.Tensor):
            preds = preds.detach().cpu().numpy()

        # Handle scalar, 1D, and 2D cases
        if np.isscalar(preds):  # scalar like 0.6486
            preds = np.array([[preds]])
        elif preds.ndim == 0:
            preds = preds.reshape(1, 1)
        elif preds.ndim == 1:
            preds = preds.reshape(-1, 1)

        # Inverse scale predictions to original space
        original_predictions = torch.tensor(
            self.scaler.inverse_transform(preds),
            dtype=torch.float32
        ).squeeze().to(self.device)
        #return original_predictions
        return original_predictions.view(-1)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


class StandardRejectionPredictor:
    def __init__(self, device_id=0):
        self.model_name = 'std_rejection_model'
        self.device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")

    def assign_dataset(self, new_rejection_dataset: StandardRejectionDataset):
        """Public method to assign a new dataset."""
        self._set_dataset(new_rejection_dataset, type='standard')

    def _set_dataset(self, rejection_dataset: StandardRejectionDataset, type='standard'):
        """Private method to set the dataset and calculate input dimensions."""
        if type == 'standard':
            self.dataset = rejection_dataset
            self.input_dim = (
                self.dataset.molecule_reps.shape[1]
                + self.dataset.solvent_reps.shape[1]
                + self.dataset.extra_features.shape[1]
            )
        elif type == 'training':
            self.training_dataset = rejection_dataset
            self.input_dim = (
                self.training_dataset.molecule_reps.shape[1]
                + self.training_dataset.solvent_reps.shape[1]
                + self.training_dataset.extra_features.shape[1]
            )
        elif type == 'validation':
            self.val_dataset = rejection_dataset
            self.input_dim = (
                self.val_dataset.molecule_reps.shape[1]
                + self.val_dataset.solvent_reps.shape[1]
                + self.val_dataset.extra_features.shape[1]
            )
        else:
            raise ValueError("Unsupported dataset type. Use 'standard', 'training', or 'validation'.")

    def dump_extra_scaler(self, extra_scaler_path):
        joblib.dump(self.extra_scaler, extra_scaler_path)

    def read_extra_scaler(self, extra_scaler_path):
        self.extra_scaler = joblib.load(extra_scaler_path)

    def data_preprocess_for_training(self):
        # --- Input (x_d) scaling ---
        # Get raw continuous and categorical data
        cont_cols = self.training_dataset.extra_continuous_columns
        cat_cols = self.training_dataset.extra_categorical_columns

        self.extra_scaler = StandardScaler().fit(self.training_dataset.data_df[cont_cols].values)
        if self.extra_scaler is None:
            raise ValueError("extra_scaler is None! Make sure to define it.")
        
        # Apply scaler to all training rows (train + val)
        full_cont_data = self.training_dataset.data_df[cont_cols].values
        scaled_cont_data = self.extra_scaler.transform(full_cont_data)
        full_cat_data = self.training_dataset.data_df[cat_cols].values
        self.training_dataset.extra_features = torch.tensor(np.concatenate([scaled_cont_data, full_cat_data], axis=1), dtype=torch.float32)
        
        # Apply scaler to all validation rows (train + val)
        full_cont_data = self.val_dataset.data_df[cont_cols].values
        scaled_cont_data = self.extra_scaler.transform(full_cont_data)
        full_cat_data = self.val_dataset.data_df[cat_cols].values
        self.val_dataset.extra_features = torch.tensor(np.concatenate([scaled_cont_data, full_cat_data], axis=1), dtype=torch.float32)
        

    def data_preprocess_for_prediction(self, extra_scaler = None):
        print('Loading extra_scaler for prediction')
        self.extra_scaler = extra_scaler
        if self.extra_scaler is None:
            raise ValueError("extra_scaler is None! Make sure to load it from the pretrained model.")
                
        # --- Input (x_d) scaling ---
        # Get raw continuous and categorical data
        cont_cols = self.dataset.extra_continuous_columns
        cat_cols = self.dataset.extra_categorical_columns
    
        full_cont_data = self.dataset.data_df[cont_cols].values
        scaled_cont_data = self.extra_scaler.transform(full_cont_data)
        full_cat_data = self.dataset.data_df[cat_cols].values

        # Concatenate normalized continuous + unscaled categorical
        self.dataset.extra_features = torch.tensor(np.concatenate([scaled_cont_data, full_cat_data], axis=1), dtype=torch.float32)

    
    def train(self,
              training_dataset: StandardRejectionDataset,
              val_dataset: StandardRejectionDataset,
              checkpoint_folder, 
              model_name = 'std_rejection_model', 
              number_of_hidden_layers=2, 
              max_epochs = 150, 
              batch_size = 32, 
              hidden_dim = 64, 
              dropout = 0.2):
        
        self._set_dataset(training_dataset, type='training')
        self._set_dataset(val_dataset, type='validation')
        self.data_preprocess_for_training()

        # Create model
        self.model = RejectionModel(input_dim=self.input_dim, number_of_hidden_layers=number_of_hidden_layers, hidden_dim=hidden_dim, dropout_rate=dropout, extra_scaler=self.extra_scaler)

        if self.model.extra_scaler is None:
            raise ValueError("No extra_scaler found in the mcmpnn model for training!")

        # Define checkpoints
        checkpoint_callback = ModelCheckpoint(
            dirpath=checkpoint_folder+"/",  # Directory to save checkpoints
            filename=model_name+"-{epoch:02d}-{val_loss:.2f}",  # Filename format
            save_top_k=1,  # Save the top 1 model (best performance)
            monitor="val_loss",  # Metric to monitor (change to your validation metric)
            mode="min"  # Save model with the lowest validation loss
        )

        # Logger
        logger = TensorBoardLogger("logs", name="standard_"+model_name)

        # Trainer
        trainer = pl.Trainer(
            callbacks = [checkpoint_callback],
            logger=logger,
            enable_checkpointing=True,
            enable_progress_bar=prog_bar,
            max_epochs=max_epochs,
            accelerator="gpu",
            devices=[self.device.index],
            #log_every_n_steps=1,
        )

        # Create dataloaders
        self.train_loader = DataLoader(self.training_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=batch_size, drop_last=True)

        # Fit scaler
        self.model.fit_scaler(self.training_dataset)

        # Train the model
        trainer.fit(self.model, self.train_loader, self.val_loader)

        # Load the best model after training
        self.best_model_path = checkpoint_callback.best_model_path
        print(f"Best model saved at: {self.best_model_path}")
        self.model = RejectionModel.load_from_checkpoint(self.best_model_path)

        # Test the model
        self.val_results = trainer.validate(self.model, self.val_loader)

        print('Model validation results:',self.val_results)

        return self.val_results
    
    
    def predict(self, checkpoint_path=None, model_name='', forget_predictions = False):

        if not hasattr(self, 'dataset'):
            raise ValueError("A StandardRejectionDataset must be initiated.")

        if checkpoint_path != None:
            self.model = RejectionModel.load_from_checkpoint(checkpoint_path)
        elif not hasattr(self, 'model'):
            raise ValueError("Model (self.model) must be initialized or a checkpoint must be provided.")
        
        if self.model.extra_scaler is None:
            raise ValueError("No extra_scaler found in the mcmpnn model for predictions!")

        self.data_preprocess_for_prediction(extra_scaler = self.model.extra_scaler)
        loader = DataLoader(self.dataset, batch_size=32, shuffle=False)

        with torch.inference_mode():
            trainer = pl.Trainer(
                #logger=None,
                enable_progress_bar=prog_bar,
                accelerator="gpu",
                devices=[self.device.index],
            )
            preds = trainer.predict(self.model, loader)
            
        # preds = np.concatenate(preds, axis=0)
        preds = np.concatenate([p.cpu().numpy() for p in preds], axis=0)

        if not forget_predictions:
            self.dataset.data_df[self.dataset.target_column+'_'+model_name] = preds
            self.dataset.clip_rejection(self.dataset.target_column+'_'+model_name)
    
        # Calculate RMSE and MAE using true values
        true_values = self.dataset.data_df[self.dataset.target_column].values  # True target values
        rmse = root_mean_squared_error(true_values, preds)  # RMSE
        mae = mean_absolute_error(true_values, preds)  # MAE
        r2 = r2_score(true_values, preds)  # R2

        # Print RMSE and MAE
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")
        print(f"R2: {r2:.4f}")

        return rmse, mae, r2