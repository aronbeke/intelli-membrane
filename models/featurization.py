import numpy as np
from rdkit import Chem
from rdkit import RDLogger

import torch

# Temporary suppression of RDKit logs
RDLogger.DisableLog("rdApp.*")

class Featurizer:
    '''
    General scaffold for featurizers
    '''
    def __init__(self, allowable_sets):
        self.dim = 0
        self.features_mapping = {}
        for k, s in allowable_sets.items():
            s = sorted(list(s))
            self.features_mapping[k] = dict(zip(s, range(self.dim, len(s) + self.dim)))
            self.dim += len(s)

    def encode(self, inputs):
        output = np.zeros((self.dim,))
        for name_feature, feature_mapping in self.features_mapping.items():
            feature = getattr(self, name_feature)(inputs)
            if feature not in feature_mapping:
                continue
            output[feature_mapping[feature]] = 1.0
        return output


class AtomFeaturizer(Featurizer):
    '''
    For featurizing atoms
    '''
    def __init__(self, allowable_sets):
        super().__init__(allowable_sets)

    def symbol(self, atom):
        return atom.GetSymbol()

    def n_valence(self, atom):
        return atom.GetTotalValence()

    def n_hydrogens(self, atom):
        return atom.GetTotalNumHs()

    def hybridization(self, atom):
        return atom.GetHybridization().name.lower()


class BondFeaturizer(Featurizer):
    '''
    For featurizing bonds
    '''
    def __init__(self, allowable_sets):
        super().__init__(allowable_sets)
        self.dim += 1

    def encode(self, bond):
        output = np.zeros((self.dim,))
        if bond is None:
            output[-1] = 1.0
            return output
        output = super().encode(bond)
        return output

    def bond_type(self, bond):
        return bond.GetBondType().name.lower()

    def conjugated(self, bond):
        return bond.GetIsConjugated()

################# MOLECULE DATA PROCESSING ##################

def molecule_from_smiles(smiles):
    # MolFromSmiles(m, sanitize=True) should be equivalent to
    # MolFromSmiles(m, sanitize=False) -> SanitizeMol(m) -> AssignStereochemistry(m, ...)
    
    """sanitisation = cleaning the molecule"""
    molecule = Chem.MolFromSmiles(smiles, sanitize=False)

    # If sanitization is unsuccessful, catch the error, and try again without
    # the sanitization step that caused the error
    flag = Chem.SanitizeMol(molecule, catchErrors=True)
    if flag != Chem.SanitizeFlags.SANITIZE_NONE:
        Chem.SanitizeMol(molecule, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ flag)

    Chem.AssignStereochemistry(molecule, cleanIt=True, force=True)
    return molecule

def graph_from_molecule(molecule,atom_featurizer,bond_featurizer):
    # Initialize graph
    atom_features = []
    bond_features = []
    pair_indices = []

    for atom in molecule.GetAtoms():
        atom_features.append(atom_featurizer.encode(atom))

        # Add self-loops
        pair_indices.append([atom.GetIdx(), atom.GetIdx()])
        bond_features.append(bond_featurizer.encode(None))

        for neighbor in atom.GetNeighbors():
            bond = molecule.GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx())
            pair_indices.append([atom.GetIdx(), neighbor.GetIdx()])
            bond_features.append(bond_featurizer.encode(bond))

    return torch.tensor(atom_features, dtype=torch.float32), \
           torch.tensor(bond_features, dtype=torch.float32), \
           torch.tensor(pair_indices, dtype=torch.int64)
           
    # return np.array(atom_features), np.array(bond_features), np.array(pair_indices)


################# MOLECULE INFORMATION PROCESSING ##################

def is_transition_metal(at):
    n = at.GetAtomicNum()
    return (n>=22 and n<=29) or (n>=40 and n<=47) or (n>=72 and n<=79)

def set_dative_bonds(mol, fromAtoms=(7,8,15)):
    """ convert some bonds to dative

    Replaces some single bonds between metals and atoms with atomic numbers in fomAtoms
    with dative bonds. The replacement is only done if the atom has "too many" bonds.

    Returns the modified molecule.

    """
    pt = Chem.GetPeriodicTable()
    rwmol = Chem.RWMol(mol)
    rwmol.UpdatePropertyCache(strict=False)
    metals = [at for at in rwmol.GetAtoms() if is_transition_metal(at)]
    for metal in metals:
        for nbr in metal.GetNeighbors():
            if nbr.GetAtomicNum() in fromAtoms and \
               nbr.GetExplicitValence()>pt.GetDefaultValence(nbr.GetAtomicNum()) and \
               rwmol.GetBondBetweenAtoms(nbr.GetIdx(),metal.GetIdx()).GetBondType() == Chem.BondType.SINGLE:
                rwmol.RemoveBond(nbr.GetIdx(),metal.GetIdx())
                rwmol.AddBond(nbr.GetIdx(),metal.GetIdx(),Chem.BondType.DATIVE)
    return rwmol


def canonical_smiles(smi):
    """Convert a SMILES string to its canonical form using RDKit."""
    mol = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(mol, canonical=True) if mol else None

def complex_canonical_smiles(smiles: str, kekule: bool = False) -> str:
    """
    Canonicalizes a SMILES string using RDKit.

    Parameters:
        smiles (str): Input SMILES string.
        kekule (bool): Whether to return a kekulized (non-aromatic) SMILES.

    Returns:
        str: Canonical (optionally kekulized) SMILES string, or None if invalid.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        # Remove stereochemistry or other info if needed
        Chem.SanitizeMol(mol)
        
        # Optional: Kekulize the molecule before generating SMILES
        if kekule:
            Chem.Kekulize(mol, clearAromaticFlags=True)
        
        return Chem.MolToSmiles(mol, canonical=True, kekuleSmiles=kekule)
    except Exception as e:
        print(f"[Error] Failed to canonicalize SMILES '{smiles}': {e}")
        return None

def standardize_smiles(origin_df, solute_data_df, nf10k_smiles_col='solute_smiles', solute_data_smiles_col='solute_smiles'):
    # Create a mapping from canonical SMILES in nf10k to their indices
    print("Canonicalizing SMILES in nf10k...")
    origin_df = origin_df.copy()
    origin_df['canonical_nf10k'] = origin_df[nf10k_smiles_col].apply(canonical_smiles)

    print("Canonicalizing SMILES in solute_data...")
    solute_data_df = solute_data_df.copy()
    solute_data_df['canonical_solute'] = solute_data_df[solute_data_smiles_col].apply(canonical_smiles)

    # Create a mapping from canonical SMILES to the version from solute_data
    replacement_dict = dict(zip(solute_data_df['canonical_solute'], solute_data_df[solute_data_smiles_col]))

    # Replace matching canonical SMILES
    print("Replacing matching SMILES in nf10k with solute_data version...")
    def replace_smiles(row):
        canon = row['canonical_nf10k']
        return replacement_dict.get(canon, row[nf10k_smiles_col])  # Replace if found, else keep original

    origin_df[nf10k_smiles_col] = origin_df.apply(replace_smiles, axis=1)

    # Drop the helper column
    origin_df.drop(columns=['canonical_nf10k'], inplace=True)
    return origin_df


def get_metal_from_smiles(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    metals = [atom.GetSymbol() for atom in mol.GetAtoms() if atom.GetAtomicNum() > 20 and atom.GetAtomicNum() not in range(34, 53)]  # skip non-transition metals
    return metals[0] if metals else None