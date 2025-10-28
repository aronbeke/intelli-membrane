import os
import json
import math
import numpy as np
import time
import torch
import torch_geometric
import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import rejection_models.featurization as featurization
import torch_geometric.nn as geom_nn
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

class MoleculeDataset(Dataset):
    def __init__(self, smiles_list, atom_featurizer, bond_featurizer):
        self.atom_features_list = []
        self.bond_features_list = []
        self.pair_indices_list = []
        self.smiles_list = smiles_list
        self.atom_featurizer = atom_featurizer
        self.bond_featurizer = bond_featurizer

        for smiles in self.smiles_list:
            molecule = featurization.molecule_from_smiles(smiles)
            atom_features, bond_features, pair_indices = featurization.graph_from_molecule(molecule, self.atom_featurizer, self.bond_featurizer)

            self.atom_features_list.append(atom_features)
            self.bond_features_list.append(bond_features)
            self.pair_indices_list.append(pair_indices)

    def __len__(self):
        return len(self.smiles_list)

    def __getitem__(self, idx):
        return (self.atom_features_list[idx], self.bond_features_list[idx], self.pair_indices_list[idx])
    

def graphs_from_smiles(smiles_list, atom_featurizer, bond_featurizer):
    '''
    Usage:
    smiles_list = ["CCO", "CCN", "CCC"]
    atom_featurizer = YourAtomFeaturizer()  # Define your atom featurizer
    bond_featurizer = YourBondFeaturizer()  # Define your bond featurizer
    dataset = graphs_from_smiles(smiles_list, atom_featurizer, bond_featurizer)

    Create a DataLoader:
    data_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    '''

    dataset = MoleculeDataset(smiles_list, atom_featurizer, bond_featurizer)
    return dataset

########### BATCH DATA PROCESSING ##################

def prepare_batch(x_batch, y_batch):
    atom_features, bond_features, pair_indices = x_batch

    # Obtain number of atoms and bonds for each graph (molecule)
    num_atoms = atom_features.row_lengths()
    num_bonds = bond_features.row_lengths()

    # Obtain partition indices (molecule_indicator), which will be used to
    # gather (sub)graphs from global graph in model later on
    molecule_indices = torch.arange(len(num_atoms))
    molecule_indicator = molecule_indices.repeat_interleave(num_atoms)

    # Merge (sub)graphs into a global (disconnected) graph
    gather_indices = molecule_indices[:-1].repeat(num_bonds[1:])
    increment = torch.cumsum(num_atoms[:-1], dim=0)
    increment = torch.cat((torch.zeros(num_bonds[0]), increment))
    increment = increment[gather_indices].unsqueeze(1)
    pair_indices += increment
    atom_features = atom_features.merge_dims(outer_dim=0, inner_dim=1).to_tensor()
    bond_features = bond_features.merge_dims(outer_dim=0, inner_dim=1).to_tensor()

    return (atom_features, bond_features, pair_indices, molecule_indicator), y_batch


class GNNDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    

def collate_fn(batch):
    x_batch, y_batch = zip(*batch)
    return prepare_batch(x_batch, y_batch)


def get_dataloader(X, y, batch_size=32, shuffle=False):
    dataset = GNNDataset(X, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)

####################### STANDARD MODELS ############################

class GNNModel(nn.Module):
    '''
    General GNN model, layer type can be set parametrically to
    Graph Convolution Network (GCN)
    Graph Attention Network (GAT)
    Simple Graph Convolution (GraphConv)
    '''

    def __init__(self, c_in, c_hidden, c_out, num_layers=2, layer_name="GCN", dp_rate=0.1, **kwargs):
        """
        Inputs:
            c_in - Dimension of input features
            c_hidden - Dimension of hidden features
            c_out - Dimension of the output features. Usually number of classes in classification
            num_layers - Number of "hidden" graph layers
            layer_name - String of the graph layer to use
            dp_rate - Dropout rate to apply throughout the network
            kwargs - Additional arguments for the graph layer (e.g. number of heads for GAT)
        """
        super().__init__()

        gnn_layer_by_name = {
            "GCN": geom_nn.GCNConv,
            "GAT": geom_nn.GATConv,
            "GraphConv": geom_nn.GraphConv
        }

        gnn_layer = gnn_layer_by_name[layer_name]

        layers = []
        in_channels, out_channels = c_in, c_hidden
        for l_idx in range(num_layers-1):
            layers += [
                gnn_layer(in_channels=in_channels,
                          out_channels=out_channels,
                          **kwargs),
                nn.ReLU(inplace=True),
                nn.Dropout(dp_rate)
            ]
            in_channels = c_hidden
        layers += [gnn_layer(in_channels=in_channels,
                             out_channels=c_out,
                             **kwargs)]
        self.layers = nn.ModuleList(layers)

    def forward(self, x, edge_index):
        """
        Inputs:
            x - Input features per node
            edge_index - List of vertex index pairs representing the edges in the graph (PyTorch geometric notation)
        """
        for l in self.layers:
            # For graph layers, we need to add the "edge_index" tensor as additional input
            # All PyTorch Geometric graph layer inherit the class "MessagePassing", hence
            # we can simply check the class type.
            if isinstance(l, geom_nn.MessagePassing):
                x = l(x, edge_index)
            else:
                x = l(x)
        return x

class GraphLevelGNNModel(nn.Module):
    '''
    Model for graph-level classification
    '''

    def __init__(self, c_in, c_hidden, c_out, dp_rate_linear=0.5, **kwargs):
        """
        Inputs:
            c_in - Dimension of input features
            c_hidden - Dimension of hidden features
            c_out - Dimension of output features (usually number of classes, or 1 for single-value graph-level regression)
            dp_rate_linear - Dropout rate before the linear layer (usually much higher than inside the GNN)
            kwargs - Additional arguments for the GNNModel object
        """
        super().__init__()
        self.GNN = GNNModel(c_in=c_in,
                            c_hidden=c_hidden,
                            c_out=c_hidden, # Not our prediction output yet!
                            **kwargs)
        self.head = nn.Sequential(
            nn.Dropout(dp_rate_linear),
            nn.Linear(c_hidden, c_out)
        )

    def forward(self, x, edge_index, batch_idx):
        """
        Inputs:
            x - Input features per node
            edge_index - List of vertex index pairs representing the edges in the graph (PyTorch geometric notation)
            batch_idx - Index of batch element for each node
        """
        x = self.GNN(x, edge_index)
        x = geom_nn.global_mean_pool(x, batch_idx) # Average pooling
        x = self.head(x)
        return x


class MultiInputGNN(nn.Module):
    def __init__(self, mol_encoder, n_additional_features):
        super(MultiInputGNN, self).__init__()
        
        # Molecular encoders for both chemical and solvent SMILES
        self.chemical_encoder = mol_encoder
        self.solvent_encoder = mol_encoder
        
        # Fully connected layer for one-hot membrane and other numerical features
        self.fc_additional = nn.Linear(n_additional_features, 128)
        
        # Final regression layer
        self.fc_out = nn.Linear(2 * mol_encoder.output_size + 128, 1)  # Concatenating both SMILES outputs and additional features

    def forward(self, chemical_smiles, solvent_smiles, additional_features):
        # Encode chemical and solvent SMILES into graph representations
        chemical_repr = self.chemical_encoder(chemical_smiles)
        solvent_repr = self.solvent_encoder(solvent_smiles)
        
        # Process additional features (membrane one-hot + other numerical features)
        additional_repr = self.fc_additional(additional_features)
        
        # Concatenate all representations
        combined_repr = torch.cat([chemical_repr, solvent_repr, additional_repr], dim=1)
        
        # Final regression output (rejection prediction)
        output = self.fc_out(combined_repr)
        
        return output

################################################ COMPLEX MPNN ##################################################

class EdgeNetwork(nn.Module):
    '''
    Transforms and aggregates bond features in the graph.
    '''

    def __init__(self, atom_dim, bond_dim):
        super(EdgeNetwork, self).__init__()
        self.atom_dim = atom_dim
        self.bond_dim = bond_dim
        self.kernel = nn.Parameter(torch.Tensor(bond_dim, atom_dim * atom_dim))
        self.bias = nn.Parameter(torch.zeros(atom_dim * atom_dim))
        nn.init.xavier_uniform_(self.kernel)

    def forward(self, atom_features, bond_features, pair_indices):
        # Apply linear transformation to bond features
        bond_features = torch.matmul(bond_features, self.kernel) + self.bias

        # Reshape for neighborhood aggregation later
        bond_features = bond_features.view(-1, self.atom_dim, self.atom_dim)

        # Obtain atom features of neighbors
        atom_features_neighbors = atom_features[pair_indices[:, 1]]
        atom_features_neighbors = atom_features_neighbors.unsqueeze(-1)

        # Apply neighborhood aggregation
        transformed_features = torch.matmul(bond_features, atom_features_neighbors)
        transformed_features = transformed_features.squeeze(-1)
        aggregated_features = torch.zeros_like(atom_features)
        aggregated_features.index_add_(0, pair_indices[:, 0], transformed_features)
        return aggregated_features


class MessagePassing(nn.Module):
    '''
    Manages the iterative message passing and updates node states.
    1. Edge Network
    2. Gated Recurrent Unit Cell
    '''
    def __init__(self, units, steps=4):
        super(MessagePassing, self).__init__()
        self.units = units
        self.steps = steps
        self.pad_length = None
        self.message_step = None
        self.update_step = None

    def build(self, input_shape):
        self.atom_dim = input_shape[0][-1]
        self.message_step = EdgeNetwork(self.atom_dim, input_shape[1][-1])
        self.pad_length = max(0, self.units - self.atom_dim)
        # Gated Recurrent Unit Cell, is a type of recurrent neural network (RNN) cell used for modeling sequential data
        self.update_step = nn.GRUCell(self.atom_dim + self.pad_length, self.atom_dim + self.pad_length)

    def forward(self, atom_features, bond_features, pair_indices):
        # Pad atom features if number of desired units exceeds atom_features dim.
        # Alternatively, a dense layer could be used here.
        atom_features_updated = F.pad(atom_features, (0, self.pad_length))

        # Perform a number of steps of message passing
        for i in range(self.steps):
            # Aggregate information from neighbors
            atom_features_aggregated = self.message_step(atom_features_updated, bond_features, pair_indices)

            # Update node state via a step of GRU
            atom_features_updated = self.update_step(atom_features_aggregated, atom_features_updated)
        return atom_features_updated


class PartitionPadding(nn.Module):
    '''
    Handles padding and stacking of graph features for batch processing.
    '''

    def __init__(self, batch_size):
        super(PartitionPadding, self).__init__()
        self.batch_size = batch_size

    def forward(self, atom_features, molecule_indicator):
        # Obtain subgraphs
        atom_features_partitioned = [atom_features[molecule_indicator == i] for i in range(self.batch_size)]

        # Pad and stack subgraphs
        num_atoms = [f.shape[0] for f in atom_features_partitioned]
        max_num_atoms = max(num_atoms)
        atom_features_padded = [F.pad(f, (0, 0, 0, max_num_atoms - n)) for f, n in zip(atom_features_partitioned, num_atoms)]
        atom_features_stacked = torch.stack(atom_features_padded)

        # Remove empty subgraphs (usually for last batch in dataset)
        gather_indices = torch.nonzero(torch.sum(atom_features_stacked, dim=(1, 2)) != 0, as_tuple=True)[0]
        return atom_features_stacked[gather_indices]
    

class TransformerEncoderReadout(nn.Module):
    '''
    Applies transformer-based encoding and readout operations to graph features.
    When the message passing procedure ends, the k-step-aggregated node states are to be partitioned into subgraphs (corresponding to each molecule in the batch) and subsequently reduced to graph-level embeddings. 
    In the original paper, a set-to-set layer was used for this purpose. In this tutorial however, a transformer encoder + average pooling will be used. Specifically:
    1. the k-step-aggregated node states will be partitioned into the subgraphs (corresponding to each molecule in the batch);
    2. each subgraph will then be padded to match the subgraph with the greatest number of nodes, followed by a tf.stack(...);
    3. the (stacked padded) tensor, encoding subgraphs (each subgraph containing a set of node states), are masked to make sure the paddings don't interfere with training;
    4. finally, the tensor is passed to the transformer followed by average pooling.
    '''
    
    def __init__(self, num_heads=8, embed_dim=64, dense_dim=512, batch_size=32):
        super(TransformerEncoderReadout, self).__init__()

        self.partition_padding = PartitionPadding(batch_size)
        self.attention = nn.MultiheadAttention(embed_dim, num_heads)
        self.dense_proj = nn.Sequential(
            nn.Linear(embed_dim, dense_dim),
            nn.ReLU(),
            nn.Linear(dense_dim, embed_dim)
        )
        self.layernorm_1 = nn.LayerNorm(embed_dim)
        self.layernorm_2 = nn.LayerNorm(embed_dim)

    def forward(self, atom_features, molecule_indicator):
        x = self.partition_padding(atom_features, molecule_indicator)
        padding_mask = x.sum(dim=-1) != 0
        attention_output, _ = self.attention(x.transpose(0, 1), x.transpose(0, 1), x.transpose(0, 1), key_padding_mask=~padding_mask)
        proj_input = self.layernorm_1(x + attention_output.transpose(0, 1))
        proj_output = self.layernorm_2(proj_input + self.dense_proj(proj_input))
        return proj_output.mean(dim=1)


class MPNNModel(nn.Module):
    '''
    Based on https://keras.io/examples/graph/mpnn-molecular-graphs/
    Combines message passing, transformer encoding, and dense layers to produce final predictions.
    '''
    def __init__(self, atom_dim, bond_dim, batch_size=32, message_units=64, message_steps=4, num_attention_heads=8, dense_units=512):
        super(MPNNModel, self).__init__()
        
        # Define layers and parameters
        self.message_passing = MessagePassing(message_units, message_steps)
        self.transformer_encoder_readout = TransformerEncoderReadout(num_attention_heads, message_units, dense_units, batch_size)
        self.dense1 = nn.Linear(dense_units, dense_units)
        self.dense2 = nn.Linear(dense_units, 1)

    def forward(self, atom_features, bond_features, pair_indices, molecule_indicator):
        # Message Passing
        x = self.message_passing(atom_features, bond_features, pair_indices)
        
        # Transformer Encoder Readout
        x = self.transformer_encoder_readout(x, molecule_indicator)
        
        # Dense layers
        x = F.relu(self.dense1(x))
        x = torch.sigmoid(self.dense2(x))
        return x


################################### TRAINING (LIGHTNING) #################################################

class GraphLevelGNNClassifier(pl.LightningModule):

    def __init__(self, **model_kwargs):
        super().__init__()
        # Saving hyperparameters
        self.save_hyperparameters()

        self.model = GraphLevelGNNModel(**model_kwargs)
        # BCE = binary_cross_entropy_loss
        self.loss_module = nn.BCEWithLogitsLoss() if self.hparams.c_out == 1 else nn.CrossEntropyLoss()

    def forward(self, data, mode="train"):
        x, edge_index, batch_idx = data.x, data.edge_index, data.batch
        x = self.model(x, edge_index, batch_idx)
        x = x.squeeze(dim=-1)

        if self.hparams.c_out == 1:
            preds = (x > 0).float()
            data.y = data.y.float()
        else:
            preds = x.argmax(dim=-1)
        loss = self.loss_module(x, data.y)
        acc = (preds == data.y).sum().float() / preds.shape[0]
        return loss, acc

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=1e-2, weight_decay=0.0) # High lr because of small dataset and small model
        return optimizer

    def training_step(self, batch, batch_idx):
        loss, acc = self.forward(batch, mode="train")
        self.log('train_loss', loss)
        self.log('train_acc', acc)
        return loss

    def validation_step(self, batch, batch_idx):
        _, acc = self.forward(batch, mode="val")
        self.log('val_acc', acc)

    def test_step(self, batch, batch_idx):
        _, acc = self.forward(batch, mode="test")
        self.log('test_acc', acc)


class GraphLevelGNNRegressor(pl.LightningModule):

    def __init__(self, **model_kwargs):
        super().__init__()
        # Saving hyperparameters
        self.save_hyperparameters()

        # Initialize the GNN model
        self.model = GraphLevelGNNModel(**model_kwargs)
        # Use Mean Squared Error Loss for regression
        self.loss_module = nn.MSELoss()

    def forward(self, data):
        """
        Forward pass of the model.
        Args:
            data: Data object containing x (features), edge_index (edges), and batch (batch indices).
        Returns:
            tuple: (loss, predictions, accuracy) for regression, accuracy is not used.
        """
        x, edge_index, batch_idx = data.x, data.edge_index, data.batch
        x = self.model(x, edge_index, batch_idx)
        x = x.squeeze(dim=-1)  # Flatten the output if necessary

        # Compute loss
        loss = self.loss_module(x, data.y)
        # Predictions for regression do not involve a classification threshold
        # Accuracy is not a useful metric for regression
        return loss, x

    def configure_optimizers(self):
        """
        Configure the optimizer for training.
        Returns:
            optimizer: Optimizer used for training.
        """
        optimizer = optim.AdamW(self.parameters(), lr=1e-2, weight_decay=0.0)
        return optimizer

    def training_step(self, batch, batch_idx):
        """
        Training step for each batch.
        Args:
            batch: Data batch from the DataLoader.
            batch_idx: Index of the batch.
        Returns:
            loss: Loss value for optimization.
        """
        loss, _ = self.forward(batch)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Validation step for each batch.
        Args:
            batch: Data batch from the DataLoader.
            batch_idx: Index of the batch.
        Returns:
            None: Logs validation loss.
        """
        loss, _ = self.forward(batch)
        self.log('val_loss', loss)

    def test_step(self, batch, batch_idx):
        """
        Test step for each batch.
        Args:
            batch: Data batch from the DataLoader.
            batch_idx: Index of the batch.
        Returns:
            None: Logs test loss.
        """
        loss, _ = self.forward(batch)
        self.log('test_loss', loss)


def train_graph_classifier(model_name, dataset, CHECKPOINT_PATH, device, graph_train_loader, graph_val_loader, graph_test_loader,  **model_kwargs):
    pl.seed_everything(42)

    # Create a PyTorch Lightning trainer with the generation callback
    root_dir = os.path.join(CHECKPOINT_PATH, "GraphLevel" + model_name)
    os.makedirs(root_dir, exist_ok=True)
    trainer = pl.Trainer(default_root_dir=root_dir,
                         callbacks=[ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_acc")],
                         accelerator="gpu" if str(device).startswith("cuda") else "cpu",
                         devices=1,
                         max_epochs=500,
                         enable_progress_bar=False)
    trainer.logger._default_hp_metric = None # Optional logging argument that we don't need

    # Check whether pretrained model exists. If yes, load it and skip training

    pretrained_filename = os.path.join(CHECKPOINT_PATH, f"GraphLevel{model_name}.ckpt")
    if os.path.isfile(pretrained_filename):
        print("Found pretrained model, loading...")
        model = GraphLevelGNNRegressor.load_from_checkpoint(pretrained_filename)
    else:
        pl.seed_everything(42)
        model = GraphLevelGNNRegressor(c_in=dataset.num_node_features,
                              c_out=1 if dataset.num_classes==2 else dataset.num_classes,
                              **model_kwargs)
        trainer.fit(model, graph_train_loader, graph_val_loader)
        model = GraphLevelGNNRegressor.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
        
    # Test best model on validation and test set
    train_result = trainer.test(model, graph_train_loader, verbose=False)
    test_result = trainer.test(model, graph_test_loader, verbose=False)
    result = {"test": test_result[0]['test_acc'], "train": train_result[0]['test_acc']}
    return model, result


def train_graph_regressor(model_name, dataset, CHECKPOINT_PATH, device, graph_train_loader, graph_val_loader, graph_test_loader,  **model_kwargs):
    pl.seed_everything(42)

    # Create a PyTorch Lightning trainer with the generation callback
    root_dir = os.path.join(CHECKPOINT_PATH, "GraphLevel" + model_name)
    os.makedirs(root_dir, exist_ok=True)
    trainer = pl.Trainer(default_root_dir=root_dir,
                         callbacks=[ModelCheckpoint(save_weights_only=True, mode="min", monitor="val_loss")],
                         accelerator="gpu" if str(device).startswith("cuda") else "cpu",
                         devices=1,
                         max_epochs=500,
                         enable_progress_bar=False)
    trainer.logger._default_hp_metric = None # Optional logging argument that we don't need

    # Check whether pretrained model exists. If yes, load it and skip training

    pretrained_filename = os.path.join(CHECKPOINT_PATH, f"GraphLevel{model_name}.ckpt")
    if os.path.isfile(pretrained_filename):
        print("Found pretrained model, loading...")
        model = GraphLevelGNNClassifier.load_from_checkpoint(pretrained_filename)
    else:
        pl.seed_everything(42)
        model = GraphLevelGNNClassifier(c_in=dataset.num_node_features,
                              c_out=1 if dataset.num_classes==2 else dataset.num_classes,
                              **model_kwargs)
        trainer.fit(model, graph_train_loader, graph_val_loader)
        model = GraphLevelGNNClassifier.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
        
    # Test best model on validation and test set
    train_result = trainer.test(model, graph_train_loader, verbose=False)
    test_result = trainer.test(model, graph_test_loader, verbose=False)
    result = {"test": test_result[0]['test_loss'], "train": train_result[0]['test_loss']}
    return model, result