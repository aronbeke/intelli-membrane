from copy import deepcopy
from lightning import pytorch as pl
from pathlib import Path

import pandas as pd
import numpy as np
import torch

from dataclasses import InitVar, dataclass
from typing import List, Sequence, Tuple, Union, Optional
from rdkit import Chem
from rdkit.Chem.rdchem import Atom, Bond, BondType

from chemprop import data, featurizers;

import shap


class CustomMultiHotAtomFeaturizer(featurizers.atom.MultiHotAtomFeaturizer):
    """A custom MultiHotAtomFeaturizer that allows for selective feature ablation.
        
    Parameters
    ----------
    keep_features : List[bool], optional
    a list of booleans to indicate which atom features to keep. If None, all features are kept. If False, corresponding feature is set to zeros. Useful for ablation and SHAP analysis.
    """
    
    def __init__(self,
                 atomic_nums: Sequence[int],
                 degrees: Sequence[int],
                 formal_charges: Sequence[int],
                 chiral_tags: Sequence[int],
                 num_Hs: Sequence[int],
                 hybridizations: Sequence[int],
                 keep_features: List[bool] = None):
        super().__init__(atomic_nums, degrees, formal_charges, chiral_tags, num_Hs, hybridizations)
        
        if keep_features is None:
            keep_features = [True] * (len(self._subfeats) + 2)
        self.keep_features = keep_features

    def __call__(self, a: Atom | None) -> np.ndarray:
        x = np.zeros(self._MultiHotAtomFeaturizer__size)
        if a is None:
            return x
        
        feats = [
            a.GetAtomicNum(),
            a.GetTotalDegree(),
            a.GetFormalCharge(),
            int(a.GetChiralTag()),
            int(a.GetTotalNumHs()),
            a.GetHybridization(),
        ]
        
        i = 0
        for feat, choices, keep in zip(feats, self._subfeats, self.keep_features[:len(feats)]):
            j = choices.get(feat, len(choices))
            if keep:
                x[i + j] = 1
            i += len(choices) + 1
        
        if self.keep_features[len(feats)]:
            x[i] = int(a.GetIsAromatic())
        if self.keep_features[len(feats) + 1]:
            x[i + 1] = 0.01 * a.GetMass()

        return x

    def zero_mask(self) -> np.ndarray:
        """Featurize the atom by setting all bits to zero."""
        return np.zeros(len(self))
    

class CustomMultiHotBondFeaturizer(featurizers.bond.MultiHotBondFeaturizer):
    """A custom MultiHotBondFeaturizer that allows for selective feature ablation.
    
    Parameters
    ----------
    keep_features : List[bool], optional
    a list of booleans to indicate which bond features to keep except for nullity. If None, all features are kept. If False, corresponding feature is set to zeros. Useful for ablation and SHAP analysis.
    """
    
    def __init__(self,
                 bond_types: Sequence[BondType] | None = None,
                 stereos: Sequence[int] | None = None,
                 keep_features: List[bool] = None):
        super().__init__(bond_types, stereos)
        
        self._MultiHotBondFeaturizer__size = 1 + len(self.bond_types) + 2 + (len(self.stereo) + 1)

        if keep_features is None:
            keep_features = [True] * 4 
        self.keep_features = keep_features        

    def __len__(self) -> int:
        return self._MultiHotBondFeaturizer__size

    def __call__(self, b: Bond) -> np.ndarray:
        x = np.zeros(len(self), int)

        if b is None:
            x[0] = 1
            return x
        i = 1
        bond_type = b.GetBondType()
        bt_bit, size = self.one_hot_index(bond_type, self.bond_types)
        if self.keep_features[0] and bt_bit != size:
            x[i + bt_bit] = 1
        i += size - 1

        if self.keep_features[1]:
            x[i] = int(b.GetIsConjugated())
        if self.keep_features[2]:
            x[i + 1] = int(b.IsInRing())
        i += 2

        if self.keep_features[3]:
            stereo_bit, _ = self.one_hot_index(int(b.GetStereo()), self.stereo)
            x[i + stereo_bit] = 1

        return x

    def zero_mask(self) -> np.ndarray:
        """Featurize the bond by setting all bits to zero."""
        return np.zeros(len(self), int)

    @classmethod
    def one_hot_index(cls, x, xs: Sequence) -> tuple[int, int]:
        """Returns a tuple of the index of ``x`` in ``xs`` and ``len(xs) + 1`` if ``x`` is in ``xs``.
        Otherwise, returns a tuple with ``len(xs)`` and ``len(xs) + 1``."""
        n = len(xs)
        return xs.index(x) if x in xs else n, n + 1
    

@dataclass
class CustomSimpleMoleculeMolGraphFeaturizer(featurizers.molgraph.molecule.SimpleMoleculeMolGraphFeaturizer):
    """A custom SimpleMoleculeMolGraphFeaturizer with additional feature control."""
    
    keep_atom_features: Optional[List[bool]] = None
    keep_bond_features: Optional[List[bool]] = None
    keep_atoms: Optional[List[bool]] = None
    keep_bonds: Optional[List[bool]] = None

    def __post_init__(self, extra_atom_fdim: int = 0, extra_bond_fdim: int = 0):
        super().__post_init__(extra_atom_fdim, extra_bond_fdim)

        if isinstance(self.atom_featurizer, CustomMultiHotAtomFeaturizer) and self.keep_atom_features is not None:
            self.atom_featurizer.keep_features = self.keep_atom_features
        if isinstance(self.bond_featurizer, CustomMultiHotBondFeaturizer) and self.keep_bond_features is not None:
            self.bond_featurizer.keep_features = self.keep_bond_features

    def __call__(
        self,
        mol: Chem.Mol,
        atom_features_extra: np.ndarray | None = None,
        bond_features_extra: np.ndarray | None = None,
    ) -> data.molgraph.MolGraph:
        n_atoms = mol.GetNumAtoms()
        n_bonds = mol.GetNumBonds()

        if self.keep_atoms is None:
            self.keep_atoms = [True] * n_atoms
        if self.keep_bonds is None:
            self.keep_bonds = [True] * n_bonds

        if atom_features_extra is not None and len(atom_features_extra) != n_atoms:
            raise ValueError(
                "Input molecule must have same number of atoms as `len(atom_features_extra)`!"
                f"got: {n_atoms} and {len(atom_features_extra)}, respectively"
            )
        if bond_features_extra is not None and len(bond_features_extra) != n_bonds:
            raise ValueError(
                "Input molecule must have same number of bonds as `len(bond_features_extra)`!"
                f"got: {n_bonds} and {len(bond_features_extra)}, respectively"
            )
        if n_atoms == 0:
            V = np.zeros((1, self.atom_fdim), dtype=np.single)
        else:
            V = np.array([self.atom_featurizer(a) if self.keep_atoms[a.GetIdx()] else self.atom_featurizer.zero_mask()
                          for a in mol.GetAtoms()], dtype=np.single)

        if atom_features_extra is not None:
            V = np.hstack((V, atom_features_extra))

        E = np.empty((2 * n_bonds, self.bond_fdim))
        edge_index = [[], []]

        i = 0
        for u in range(n_atoms):
            for v in range(u + 1, n_atoms):
                bond = mol.GetBondBetweenAtoms(u, v)
                if bond is None:
                    continue

                x_e = self.bond_featurizer(bond) if self.keep_bonds[bond.GetIdx()] else self.bond_featurizer.zero_mask()

                if bond_features_extra is not None:
                    x_e = np.concatenate((x_e, bond_features_extra[bond.GetIdx()]), dtype=np.single)

                E[i: i + 2] = x_e
                edge_index[0].extend([u, v])
                edge_index[1].extend([v, u])
                i += 2

        rev_edge_index = np.arange(len(E)).reshape(-1, 2)[:, ::-1].ravel()
        edge_index = np.array(edge_index, int)
        return data.molgraph.MolGraph(V, E, edge_index, rev_edge_index)

################################################################################

def get_predictions_with_mol_ablation(model,
                    keep_atoms: Optional[List[bool]], 
                    keep_bonds: Optional[List[bool]], 
                    mol: str,
                    solvent_smiles: str,
                    x_d,
                    atom_featurizer = CustomMultiHotAtomFeaturizer.v2(), 
                    bond_featurizer = CustomMultiHotBondFeaturizer()) -> float:
    '''
    A helper function to get predictions from a molecule with ability to keep or remove specific atom and bond features
    or atoms and bonds
    '''

    solute_featurizer = CustomSimpleMoleculeMolGraphFeaturizer(
        atom_featurizer=atom_featurizer,
        bond_featurizer=bond_featurizer,
        keep_atoms=keep_atoms,
        keep_bonds=keep_bonds
    )

    solvent_featurizer = featurizers.molgraph.molecule.SimpleMoleculeMolGraphFeaturizer()

    all_data = [[data.MoleculeDatapoint.from_smi(mol, x_d=x_d)]]
    all_data += [[data.MoleculeDatapoint.from_smi(solvent_smiles)]] 
    
    dsets = [data.MoleculeDataset(all_data[0], solute_featurizer), data.MoleculeDataset(all_data[1], solvent_featurizer)]
    mcdset = data.MulticomponentDataset(dsets)
    test_loader = data.build_dataloader(mcdset, batch_size=1, shuffle=False)

    with torch.inference_mode():
        trainer = pl.Trainer(
            logger=None,
            enable_progress_bar=False,
            accelerator="auto",
            devices=1
        )
        test_preds = trainer.predict(model, test_loader)
    
    return test_preds[0][0]


######################################

class MoleculeModelWrapper:
    '''
    An example wrapper class for use as the model input in SHAP explainer
    '''
    def __init__(self, mol: str, model, n_atoms: int, n_bonds: int, x_d, solvent_smiles):
        self.mol = mol
        self.solvent_smiles = solvent_smiles
        self.x_d = x_d
        self.model = model
        self.n_atoms = n_atoms
        self.n_bonds = n_bonds
    
    def __call__(self, X):
        preds = []
        for keep_features in X:
            try:
                # unpacking X, indices corresponds to atom.GetIdx() and bond.GetIdx() from rdkit mol, adapt as needed
                keep_atoms = keep_features[:self.n_atoms]
                keep_bonds = keep_features[self.n_atoms:self.n_atoms + self.n_bonds]
            except Exception as e:
                print(f"Invalid input: {keep_features}")
                raise e
            pred = get_predictions_with_mol_ablation(self.model, keep_atoms, keep_bonds, self.mol, self.solvent_smiles, self.x_d)
            preds.append([pred.item()])
        return np.array(preds)
    

def binary_masker(binary_mask, x):
    masked_x = deepcopy(x)
    masked_x[binary_mask == 0] = 0
    return np.array([masked_x])

################################################

def shap_explainer(model_wrapper, feature_choice, max_evals = 1000, masker = binary_masker):
    '''
    feature_choice: 2D numpy array
    '''
    import logging
    logging.getLogger("lightning.pytorch.utilities.rank_zero").setLevel(logging.WARNING)

    np.random.seed(42)
    explainer = shap.PermutationExplainer(model_wrapper, masker=masker)
    explanation = explainer(feature_choice, max_evals=max_evals)
    
    return explanation