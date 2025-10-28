import numpy as np
import pandas as pd

import torch
import torch.nn as torch_nn
from torch import Tensor

#from torch.utils.data import random_split, Subset

from chemprop import nn
from chemprop.nn import Aggregation
from rdkit import Chem

from typing import List, Optional, Dict
import ast


class AttentionAggregator(torch_nn.Module):
    def __init__(self, d_model, d_attn=None, temperature: float = 1.0):
        super().__init__()
        if d_attn is None:
            d_attn = min(16, max(8, d_model // 8))
            print('Using default attention embedding size:', d_attn)
        else:
            print('Using attention embedding size:', d_attn)

        self.query = torch_nn.Linear(d_model, d_attn)
        self.key = torch_nn.Linear(d_model, d_attn)
        self.value = torch_nn.Linear(d_model, d_model)
        self.temperature = temperature

    def forward(self, reps, mask):
        """
        reps: [B, n, d_model]
        mask: [B, n] with 1s for valid entries, 0s for padding
        """
        B, n, _ = reps.shape

        if n == 1:
            # No need for attention — just return the input
            return reps.squeeze(1), torch.ones(B, n, 1, device=reps.device)

        Q = self.query(reps)  # [B, n, d_attn]
        K = self.key(reps)    # [B, n, d_attn]
        V = self.value(reps)  # [B, n, d_model]

        logits = torch.sum(Q * K, dim=-1)  # [B, n]

        # Apply temperature scaling
        logits = logits / self.temperature

        # Mask invalid entries
        logits = logits.masked_fill(mask == 0, -1e9)

        weights = torch.softmax(logits, dim=-1).unsqueeze(-1)  # [B, n, 1]

        # Handle possible NaNs from all-zero masks
        weights = torch.nan_to_num(weights, nan=0.0)

        agg = torch.sum(weights * V, dim=1)  # [B, d_model]
        return agg, weights  # [B, d_model], [B, n, 1]


# class AttentionAggregatorMLP(torch_nn.Module):
#     def __init__(self, hidden_size: int):
#         super().__init__()
#         self.attn_mlp = torch_nn.Sequential(
#             torch_nn.Linear(hidden_size, hidden_size),
#             torch_nn.ReLU(),
#             torch_nn.Linear(hidden_size, 1),
#         )

#     def forward(self, solute_reps: torch.Tensor, solute_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
#         """
#         Args:
#             solute_reps: Tensor of shape [B, n_solutes, D]
#             solute_mask: Tensor of shape [B, n_solutes], 1 for real, 0 for dummy

#         Returns:
#             - attn_output: [B, D] attention-weighted solute representation
#             - attn_weights: [B, n_solutes] normalized attention weights (zeros for dummies)
#         """
#         # Compute unnormalized attention scores: [B, n_solutes, 1]
#         attn_logits = self.attn_mlp(solute_reps)  # [B, n_solutes, 1]
#         attn_logits = attn_logits.squeeze(-1)     # [B, n_solutes]

#         # Mask out dummy components by setting logits to large negative value
#         attn_logits[solute_mask == 0] = -1e9

#         # Softmax to get attention weights
#         attn_weights = torch.softmax(attn_logits, dim=1)  # [B, n_solutes]

#         # Apply attention
#         attn_output = torch.sum(attn_weights.unsqueeze(-1) * solute_reps, dim=1)  # [B, D]

#         return attn_output, attn_weights
    

def extract_attention_entry(row, k, suffix, key='metal'):
    """
    Function to extract the first value of the <key> attention tuple.
    k: number of folds
    row: pandas row
    """
    attns = []
    for i in range(k):
        col_name = f'atom_attns_primary_admpnn_combi{suffix}_fold{i}'
        if col_name in row and pd.notnull(row[col_name]):
            try:
                attn_dict = ast.literal_eval(row[col_name]) if isinstance(row[col_name], str) else row[col_name]
                if (
                    isinstance(attn_dict, dict)
                    and key in attn_dict
                    and isinstance(attn_dict[key], (list, tuple))
                    and len(attn_dict[key]) > 0
                    and attn_dict[key][0] is not None
                ):
                    attns.append(attn_dict[key][0])
            except Exception:
                print(f'Unexpected entry in {col_name}')
                continue

    valid_attns = [a for a in attns if a is not None]
    return np.mean(valid_attns) if valid_attns else np.nan


def get_ligand_fragment_map(mol: Chem.Mol) -> Dict[str, List[int]]:
    metal_atomic_numbers = set(range(21, 31)) | set(range(39, 49)) | set(range(57, 81)) | set(range(89, 113))
    atom_to_fragment = {}

    # Step 1: Identify metal atoms
    metal_atoms = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum() in metal_atomic_numbers]
    if not metal_atoms:
        # Treat entire molecule as one ligand if no metal is present
        all_atoms = list(range(mol.GetNumAtoms()))
        return {'ligand_0': all_atoms}

    # Step 2: Identify metal-ligand bonds
    metal_ligand_bonds = []
    for bond in mol.GetBonds():
        a1, a2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if (a1 in metal_atoms and a2 not in metal_atoms) or (a2 in metal_atoms and a1 not in metal_atoms):
            metal_ligand_bonds.append(bond.GetIdx())

    # Step 3: Fragment the molecule by breaking metal-ligand bonds
    fragmented_mol = Chem.FragmentOnBonds(mol, metal_ligand_bonds, addDummies=False)
    frags = Chem.GetMolFrags(fragmented_mol, asMols=False, sanitizeFrags=False)

    frag_map: Dict[str, List[int]] = {'metal': metal_atoms}
    ligand_id = 0
    for frag in frags:
        frag_set = set(frag)
        if not frag_set & set(metal_atoms):  # skip fragment if it is just the metal
            frag_map[f'ligand_{ligand_id}'] = list(frag_set)
            ligand_id += 1

    return frag_map


def get_ligand_fragment_map_complex(mol: Chem.Mol) -> Dict[str, List[int]]:
    '''
    For multiple disconnected substructures in a single Mol
    '''
    metal_atomic_numbers = set(range(21, 31)) | set(range(39, 49)) | set(range(57, 81)) | set(range(89, 113))

    # Get disconnected components with atom index mapping
    frags_info = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False, returnAtomIndices=True)

    frag_map: Dict[str, List[int]] = {}
    global_ligand_id = 0

    for frag_idx, (frag_mol, atom_indices) in enumerate(frags_info):
        metal_atoms_local = [
            atom.GetIdx() for atom in frag_mol.GetAtoms()
            if atom.GetAtomicNum() in metal_atomic_numbers
        ]
        metal_atoms_global = [atom_indices[i] for i in metal_atoms_local]

        if not metal_atoms_local:
            # No metal: treat entire fragment as one ligand
            frag_map[f'ligand_{global_ligand_id}'] = atom_indices
            global_ligand_id += 1
            continue

        # Identify metal-ligand bonds
        metal_ligand_bonds = []
        for bond in frag_mol.GetBonds():
            a1, a2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            if (a1 in metal_atoms_local and a2 not in metal_atoms_local) or \
               (a2 in metal_atoms_local and a1 not in metal_atoms_local):
                metal_ligand_bonds.append(bond.GetIdx())

        # Fragment by breaking metal-ligand bonds
        fragmented = Chem.FragmentOnBonds(frag_mol, metal_ligand_bonds, addDummies=False)
        frags = Chem.GetMolFrags(fragmented, asMols=False, sanitizeFrags=False)

        # Add metal fragment
        frag_map[f'metal_{frag_idx}'] = metal_atoms_global

        # Add ligand fragments
        for subfrag in frags:
            subfrag_set = set(subfrag)
            if not subfrag_set & set(metal_atoms_local):
                global_atom_indices = [atom_indices[i] for i in subfrag_set]
                frag_map[f'ligand_{global_ligand_id}'] = global_atom_indices
                global_ligand_id += 1

    return frag_map


# class FragmentAttentiveAggregation(Aggregation):
#     def forward(self, H: Tensor, batch: Tensor, mol_list: List[Chem.Mol], atom_to_graph: Optional[List[List[int]]] = None) -> Tensor:
#         device = H.device
#         hidden_size = H.size(1)
#         self.output_net = torch_nn.Linear(hidden_size, 1)  # for attention
        
#         # Step 1: Atom -> Fragment mapping
#         fragment_embeddings = []
#         frag_batch = []
#         frag_to_mol_map = []
        
#         index_offset = 0  # offset due to batch concatenation

#         for i, mol in enumerate(mol_list):
#             atom_indices = (batch == i).nonzero(as_tuple=True)[0] + index_offset
#             atom_h = H[atom_indices]

#             # Compute fragment map: dict {frag_id: [atom_indices]}
#             frag_map = get_ligand_fragment_map(mol)  # user-defined

#             for frag_id, atom_ids in frag_map.items():
#                 atom_tensor_ids = [index_offset + atom_id for atom_id in atom_ids]
#                 frag_h = H[atom_tensor_ids]
#                 frag_embedding = frag_h.mean(dim=0)
#                 fragment_embeddings.append(frag_embedding)
#                 frag_batch.append(i)
#                 frag_to_mol_map.append(i)

#             index_offset += (batch == i).sum().item()

#         # Now have fragment_embeddings: [n_frags, hidden_size]
#         frag_H = torch.stack(fragment_embeddings, dim=0)
#         frag_batch_tensor = torch.tensor(frag_batch, dtype=torch.long, device=device)

#         # Step 2: Fragment-level attention
#         attention_logits = self.output_net(frag_H).exp()
#         dim_size = batch.max().item() + 1

#         Z = torch.zeros(dim_size, 1, dtype=H.dtype, device=H.device).scatter_reduce_(
#             0, frag_batch_tensor.unsqueeze(1), attention_logits, reduce="sum", include_self=False
#         )
#         alphas = attention_logits / Z[frag_batch_tensor]

#         index_torch = frag_batch_tensor.unsqueeze(1).repeat(1, hidden_size)
#         mol_repr = torch.zeros(dim_size, hidden_size, dtype=H.dtype, device=H.device).scatter_reduce_(
#             0, index_torch, alphas * frag_H, reduce="sum", include_self=False
#         )

#         return mol_repr
    

class AtomAttentiveAggregation(Aggregation):
    def __init__(self, dim: int = 0, *args, output_size: int, temperature: float = 1.0, **kwargs):
        super().__init__(dim, *args, **kwargs)

        self.W = torch_nn.Linear(output_size, 1)
        self.temperature = temperature
        self.last_alphas = None  # Store last attention weights

    def forward(self, H: Tensor, batch: Tensor) -> Tensor:
        dim_size = int(batch.max().item()) + 1  # number of molecules in batch

        # Step 1: Compute raw scores
        scores = self.W(H).squeeze(-1)         # [N_atoms]

        # Step 2: Initialize attention weights per molecule
        alphas = torch.zeros_like(scores)

        # Step 3: Compute softmax separately for each molecule
        for i in range(dim_size):
            mask = (batch == i)
            if mask.any():
                scaled_scores = scores[mask] / self.temperature
                alphas[mask] = torch.softmax(scaled_scores, dim=0)

        self.last_alphas = [alphas[batch == i].unsqueeze(-1) for i in range(dim_size)]  # keep same shape

        # Step 4: Weighted sum aggregation
        output = torch.zeros(dim_size, H.shape[1], dtype=H.dtype, device=H.device).scatter_add_(
            self.dim, batch.unsqueeze(1).expand(-1, H.shape[1]), alphas.unsqueeze(1) * H
        )

        # === NaN Debugging ===
        if torch.isnan(scores).any():
            print("NaNs in attention scores")
        if torch.isnan(alphas).any():
            print("NaNs in attention weights")
        if torch.isnan(output).any():
            print("NaNs in final aggregated output")
            print("alphas stats:", alphas.min().item(), alphas.max().item())
            print("H stats:", H.min().item(), H.max().item())
            print("output stats:", output.min().item(), output.max().item())

        return output


        # attention_logits = self.W(H).exp()  # [N_atoms, 1]

        # # Compute sum of weights per molecule
        # Z = torch.zeros(dim_size, 1, dtype=H.dtype, device=H.device).scatter_reduce_(
        #     self.dim, batch.unsqueeze(1), attention_logits, reduce="sum", include_self=False
        # )

        # if torch.any(Z == 0):
        #     print("Zero denominator in AtomAttentiveAggregation!")
        # if torch.any(torch.isnan(Z)):
        #     print("NaN detected in denominator Z!")

        # alphas = attention_logits / Z[batch]  # [N_atoms, 1]

        # # Save attention weights per molecule
        # self.last_alphas = []
        # for i in range(dim_size):
        #     mask = (batch == i)
        #     self.last_alphas.append(alphas[mask])  # [n_atoms_i, 1]

        # # Weighted sum aggregation
        # index_torch = batch.unsqueeze(1).repeat(1, H.shape[1])
        # return torch.zeros(dim_size, H.shape[1], dtype=H.dtype, device=H.device).scatter_reduce_(
        #     self.dim, index_torch, alphas * H, reduce="sum", include_self=False
        # )