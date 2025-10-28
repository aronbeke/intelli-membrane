import numpy as np
import models.mpnn_shap
from rdkit import Chem
from rdkit.Chem import AllChem

def calculate_rejection_contributions(shapley_values, mol_rejection, shap_baseline):
    '''
    shapley contributions in np.array
    Handling baseline in a heuristic and conservative way.
    num_elem is the number of elements in shapley_values, designed for number of atoms or bonds.
    '''

    num_elem = len(shapley_values)
    rejection_contributions = shapley_values + (shap_baseline/num_elem)
    normalized_rejection_contributions = rejection_contributions / (mol_rejection/num_elem)

    return rejection_contributions, normalized_rejection_contributions


def calculate_shap_and_rejection_contributions(mcmpnn, smi, solvent_smi, x_d, rejection, max_eval=1000, method='separate'):
    '''
    Calculate SHAP values and rejection contributions for the molecules in a single record of multicomponent MPNN model.
    Args:
        mcmpnn: Multicomponent MPNN model.
        smis: List of SMILES strings representing molecules.
        x_d: Input features for the molecules.
        rejection: Rejection value for the molecules.
        max_eval: Maximum number of evaluations for SHAP.
        method: Method to use for SHAP calculation ('atom', 'separate', or 'unified').
    '''

    # SHAP EXPLANATION

    n_atoms = Chem.MolFromSmiles(smi).GetNumAtoms()
    n_bonds = Chem.MolFromSmiles(smi).GetNumBonds()

    # FEATURE MASK

    keep_features = []

    if method == 'atom':
        keep_features += [1] * n_atoms + [0] * n_bonds
    elif method == 'unified':
        keep_features += [1] * n_atoms + [1] * n_bonds
    else:
        raise ValueError(f"Unknown method: {method}")

    feature_choice = np.array([keep_features])

    # WRAPPER AND EXPLANATION
    pred = models.mpnn_shap.get_predictions_with_mol_ablation(mcmpnn, [1]*n_atoms, [1]*n_bonds, smi, solvent_smiles=solvent_smi, x_d=x_d.astype(np.float64))
    bv = models.mpnn_shap.get_predictions_with_mol_ablation(mcmpnn, [0]*n_atoms, [0]*n_bonds, smi, solvent_smiles=solvent_smi, x_d=x_d.astype(np.float64))
    print('Inputted prediction:', rejection)
    print('Calculated prediction: ',pred)
    print('BV: ',bv)
    model_wrapper = models.mpnn_shap.MoleculeModelWrapper(smi, mcmpnn, n_atoms, n_bonds, x_d.astype(np.float64), solvent_smi)
    explanation = models.mpnn_shap.shap_explainer(model_wrapper, feature_choice, max_evals=max_eval)
    print('Sum of values', np.sum(explanation.values[0]))
    print('Base value',  float(explanation.base_values))

    # REJECTION CONTRIBUTIONS

    shap_values = explanation.values[0]
    atom_shap_values = shap_values[:n_atoms]
    bond_shap_values = shap_values[n_atoms:]
    print('Sum of atom values', np.sum(atom_shap_values))
    print('Sum of bond values', np.sum(bond_shap_values))

    atom_rejcon_dict = {} # Rejection contribution dictionary
    bond_rejcon_dict = {}
    atom_norm_rejcon_dict = {} # Normalized rejection contribution dictionary
    bond_norm_rejcon_dict = {}
    atom_shap_dict = {}  # SHAP values dictionary
    bond_shap_dict = {}

    base_values = explanation.base_values

    if method == 'atom':
        atom_values, atom_norm_values = calculate_rejection_contributions(atom_shap_values, rejection, float(explanation.base_values))
        bond_values, bond_norm_values = np.zeros(n_bonds), np.ones(n_bonds)

    elif method == 'unified':
        rejection_contributions, norm_rejection_contributions = calculate_rejection_contributions(shap_values, rejection, float(explanation.base_values))
        atom_values = rejection_contributions[:n_atoms]
        bond_values = rejection_contributions[n_atoms:]
        atom_norm_values = norm_rejection_contributions[:n_atoms]
        bond_norm_values = norm_rejection_contributions[n_atoms:]

    atom_rejcon_dict[0] = atom_values
    bond_rejcon_dict[0] = bond_values
    atom_norm_rejcon_dict[0] = atom_norm_values
    bond_norm_rejcon_dict[0] = bond_norm_values
    atom_shap_dict[0] = atom_shap_values
    bond_shap_dict[0] = bond_shap_values

    return explanation, atom_rejcon_dict, bond_rejcon_dict, atom_norm_rejcon_dict, bond_norm_rejcon_dict, atom_shap_dict, bond_shap_dict, base_values


def replace_dative_bonds_with_single(mol):
    """
    Converts all dative bonds (coordinate bonds) in the molecule to single bonds.
    """
    rw_mol = Chem.RWMol(mol)  # Editable version of the molecule

    for bond in mol.GetBonds():
        if bond.GetBondType() == Chem.BondType.DATIVE:
            idx1 = bond.GetBeginAtomIdx()
            idx2 = bond.GetEndAtomIdx()
            rw_mol.RemoveBond(idx1, idx2)
            rw_mol.AddBond(idx1, idx2, Chem.BondType.SINGLE)

    return rw_mol.GetMol()

def combine_fragments(scaffold_smiles, fg_smiles):
    '''
    Example usage:

    mol = combine_fragments(
    Chem.MolFromSmiles("c1cc([*])ccc1"),  # scaffold
    Chem.MolFromSmiles("[*]C(=O)O")       # functional group
    )
    print(Chem.MolToSmiles(mol))
    '''
    # Combine scaffold and functional group
    scaffold = Chem.MolFromSmiles(scaffold_smiles) # scaffold
    fg = Chem.MolFromSmiles(fg_smiles) 
    combo = Chem.CombineMols(scaffold, fg)
    em = Chem.EditableMol(combo)

    # Find dummy atoms (usually atomic number 0)
    dummies = [atom.GetIdx() for atom in combo.GetAtoms() if atom.GetAtomicNum() == 0]
    if len(dummies) != 2:
        raise ValueError("Expected exactly two dummy atoms (one in scaffold, one in fg).")

    # Add a bond between the dummy atoms' neighbors
    neighbors = []
    for d in dummies:
        atom = combo.GetAtomWithIdx(d)
        nbr = [n.GetIdx() for n in atom.GetNeighbors()][0]
        neighbors.append(nbr)

    em.AddBond(neighbors[0], neighbors[1], Chem.BondType.SINGLE)

    # Remove dummy atoms
    em.RemoveAtom(dummies[1])  # higher index first
    em.RemoveAtom(dummies[0])

    em_mol = em.GetMol()

    return Chem.MolToSmiles(em_mol)


def combine_fragments_multiple(scaffold_smiles, fg_smiles):
    '''
    Attach a copy of the fragment to each dummy site in the scaffold.

    Assumes:
    - Scaffold has one or more dummy atoms ([*]) to attach to
    - Fragment has exactly one dummy atom ([*]) for connection
    '''

    scaffold = Chem.MolFromSmiles(scaffold_smiles)
    scaffold_dummies = [a.GetIdx() for a in scaffold.GetAtoms() if a.GetAtomicNum() == 0]

    if len(scaffold_dummies) == 0:
        raise ValueError("No dummy atoms found in scaffold.")

    # Convert fragment once and store the position of its dummy atom and its neighbor
    frag = Chem.MolFromSmiles(fg_smiles)
    frag_dummy_idx = [a.GetIdx() for a in frag.GetAtoms() if a.GetAtomicNum() == 0]
    if len(frag_dummy_idx) != 1:
        raise ValueError("Fragment must contain exactly one dummy atom.")

    frag_dummy = frag.GetAtomWithIdx(frag_dummy_idx[0])
    frag_neighbor_idx = [n.GetIdx() for n in frag_dummy.GetNeighbors()][0]

    # Prepare editable mol starting from the scaffold
    base = Chem.RWMol(scaffold)
    offset = base.GetNumAtoms()

    for dummy_idx in scaffold_dummies:
        # Copy fragment
        frag_copy = Chem.Mol(frag)

        # Map frag atoms into base
        frag_atom_map = {}
        for atom in frag_copy.GetAtoms():
            if atom.GetIdx() != frag_dummy_idx[0]:
                new_idx = base.AddAtom(atom)
                frag_atom_map[atom.GetIdx()] = new_idx

        # Add bonds from fragment into base
        for bond in frag_copy.GetBonds():
            a1 = bond.GetBeginAtomIdx()
            a2 = bond.GetEndAtomIdx()
            if a1 in frag_atom_map and a2 in frag_atom_map:
                base.AddBond(frag_atom_map[a1], frag_atom_map[a2], bond.GetBondType())

        # Connect scaffold dummy neighbor to fragment dummy neighbor
        scaffold_nbr = [n.GetIdx() for n in base.GetAtomWithIdx(dummy_idx).GetNeighbors()][0]
        frag_nbr_in_base = frag_atom_map[frag_neighbor_idx]
        base.AddBond(scaffold_nbr, frag_nbr_in_base, Chem.BondType.SINGLE)

    # Remove all dummy atoms (starting from highest index)
    dummy_indices = [a.GetIdx() for a in base.GetAtoms() if a.GetAtomicNum() == 0]
    for idx in sorted(dummy_indices, reverse=True):
        base.RemoveAtom(idx)

    # Replace dative bonds
    #mol = replace_dative_bonds_with_single(base.GetMol())
    mol = base.GetMol()

    # Canonical SMILES for ChemDraw
    clean_smiles = Chem.MolToSmiles(mol, canonical=True, kekuleSmiles=True)

    return clean_smiles


def find_matching_smiles(df, smarts_pattern, smiles_column="solute_smiles", return_indices = False):
    # Convert SMARTS to a query molecule
    pattern = Chem.MolFromSmarts(smarts_pattern)
    if pattern is None:
        raise ValueError("Invalid SMARTS pattern.")

    match_smiles = []

    # Iterate through SMILES
    print(f"Matches for SMARTS pattern: {smarts_pattern}")
    for i, row in df.iterrows():
        smiles = row[smiles_column]
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print(f"Skipping invalid SMILES: {smiles}")
            continue

        if return_indices:
            substruct_matches = mol.GetSubstructMatches(pattern)
            if substruct_matches:
                # Append a tuple: (SMILES, list of atom index tuples)
                match_smiles.append((smiles, substruct_matches))
        else:
            if mol.HasSubstructMatch(pattern):
                match_smiles.append(smiles)

    return match_smiles


def substructure_contribution(solute_smiles, substructure_smarts, atom_contributions, bond_contributions=None):
    pattern = Chem.MolFromSmarts(substructure_smarts)
    mol = Chem.MolFromSmiles(solute_smiles)
    
    if mol is None or pattern is None:
        return float('nan')  # Handle invalid inputs gracefully

    substruct_matches = mol.GetSubstructMatches(pattern)  # list of atom index tuples
    
    if not substruct_matches:
        return 0.0  # No match found

    match_contributions = []
    for match in substruct_matches:
        # --- atom contributions ---
        atom_sum = sum(atom_contributions[i] for i in match)

        # --- bond contributions ---
        bond_sum = 0.0
        if bond_contributions is not None:
            # loop over bonds in the substructure match
            atom_set = set(match)
            for bond in mol.GetBonds():
                a1, a2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                if a1 in atom_set and a2 in atom_set:  # bond fully inside match
                    bond_sum += bond_contributions[bond.GetIdx()]

        match_contributions.append(atom_sum + bond_sum)

    # Return average across matches
    return np.mean(match_contributions)