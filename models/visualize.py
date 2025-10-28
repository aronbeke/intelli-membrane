import pandas as pd
import numpy as np
import models.processing

import matplotlib.pyplot as plt

from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem import AllChem, rdDepictor

import matplotlib.cm as cm
import matplotlib.colors as mcolors

# def get_color(value, min_val, max_val, vcenter=1, ref_max=None, ref_min=None, return_mapper=False):
#     'Blue-Red'
#     if ref_max is None:
#         ref_max = max_val
#     if ref_min is None:
#         ref_min = min_val

#     if ref_min < vcenter and ref_max > vcenter:
#         # Mixed positive/negative → blue-white-red
#         cmap = cm.get_cmap("seismic")
#         norm = mcolors.TwoSlopeNorm(vmin=ref_min-(vcenter-ref_min)*0.2,
#                                     vcenter=vcenter,
#                                     vmax=1.3*ref_max)
#     elif ref_max > vcenter:  
#         # Only positive → white-red
#         cmap = cm.get_cmap("Reds")
#         norm = mcolors.Normalize(vmin=vcenter, vmax=1.3*ref_max)
#     else:
#         # Only negative → blue-white
#         cmap = cm.get_cmap("Blues_r")
#         norm = mcolors.Normalize(vmin=ref_min-(vcenter-ref_min)*0.2,
#                                  vmax=vcenter)

#     if return_mapper:
#         return cmap, norm  # for colorbar
#     return cmap(norm(value))[:3]


def get_color(value, min_val, max_val,  ref_max=None, ref_min=None, return_mapper=False):
    """
    Continuous colormap:
        min_val → blue
        0       → white
        1       → grey
        max_val → red
    """
    c0 = 'blue'
    c1 = 'white'
    c2 = 'silver'
    c3 = 'red'

    # Default references
    max_val = ref_max if ref_max is not None else max_val
    min_val = ref_min if ref_min is not None else min_val

    # Expand range if needed
    max_val = max(max_val, 1)
    min_val = min(min_val, 0)

    # Determine anchors and colors
    if np.isclose(min_val, 0) and np.isclose(max_val, 1):
        anchors_raw = np.array([0, 1])
        colors = [c1,c2]

    elif np.isclose(max_val, 1):
        anchors_raw = np.array([min_val, 0, 1])
        colors = [c0,c1,c2]

    elif np.isclose(min_val, 0):
        anchors_raw = np.array([0, 1, max_val])
        colors = [c1,c2,c3]

    else:
        anchors_raw = np.array([min_val, 0, 1, max_val])
        colors = [c0,c1,c2,c3]

    # Normalize to [0, 1]
    anchors_norm = (anchors_raw - min_val) / (max_val - min_val)
    anchors_norm = np.clip(anchors_norm, 0, 1)

    cmap = mcolors.LinearSegmentedColormap.from_list(
        "blue_white_gray_red", list(zip(anchors_norm, colors))
    )
    norm = mcolors.Normalize(vmin=min_val, vmax=max_val)

    if return_mapper:
        return cmap, norm

    return cmap(norm(value))[:3]



def visualize_molecule(atom_label_array, 
                       bond_label_array, 
                       mol, 
                       identifier, 
                       target_path, 
                       ref_max = None, 
                       ref_min = None, 
                       color_atoms=True, 
                       color_bonds=True, 
                       label_atoms=True, 
                       label_bonds=True, 
                       color_bar=True, 
                       display_mols=False, 
                       black_and_white=True):
    '''
    atom_label_array: 1D np.array
    bond_label_array: 1D np.array or None
    mol: Chem.Mol
    identifier: str
    target_path: str
    color_bar: Boolean - whether to display color scale
    display_mols: Boolean - works in Jupyter Notebook
    black_and_white: Boolean - renders molecules in pure black
    '''

    # --- Get min/max values ---
    min_shap_atom, max_shap_atom = min(atom_label_array), max(atom_label_array)

    if bond_label_array is not None:
        min_shap_bond, max_shap_bond = min(bond_label_array), max(bond_label_array)
    else:
        min_shap_bond, max_shap_bond = min_shap_atom, max_shap_atom  # fallback

    min_shap = min(min_shap_atom, min_shap_bond)
    max_shap = max(max_shap_atom, max_shap_bond)

    # --- Atom labels/colors ---
    atom_labels = {atom.GetIdx(): f'{atom_label_array[atom.GetIdx()]:.3f}' 
                   for atom in mol.GetAtoms() if label_atoms}
    
    atom_colors = {
        atom.GetIdx(): get_color(atom_label_array[atom.GetIdx()],
                                 min_shap, max_shap,
                                 ref_min=ref_min, ref_max=ref_max)
        if atom_label_array[atom.GetIdx()] != 0 else (0.8, 0.8, 0.8)
        for atom in mol.GetAtoms()
    }

    # --- Bond labels/colors ---
    bond_labels, bond_colors = {}, {}
    if bond_label_array is not None:  
        # normal case: use provided bond values
        bond_labels = {
            bond.GetIdx(): f'{bond_label_array[bond.GetIdx()]:.3f}'
            for bond in mol.GetBonds()
            if label_bonds and bond_label_array[bond.GetIdx()] != 0
        }
        bond_colors = {
            bond.GetIdx(): get_color(bond_label_array[bond.GetIdx()],
                                     min_shap, max_shap,
                                     ref_min=ref_min, ref_max=ref_max)
            for bond in mol.GetBonds()
        }
    elif color_bonds:  
        # fallback: color bonds by average of two connected atoms
        for bond in mol.GetBonds():
            a1, a2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            avg_val = (atom_label_array[a1] + atom_label_array[a2]) / 2.0
            bond_colors[bond.GetIdx()] = get_color(avg_val,
                                                   min_shap, max_shap,
                                                   ref_min=ref_min, ref_max=ref_max)
        # note: no bond_labels here

    # --- Build mol copy for drawing ---
    mol_with_shap = Chem.Mol(mol)

    # --- Generate 2D coordinates for proper layout ---
    rdDepictor.Compute2DCoords(mol_with_shap)
    rdMolDraw2D.PrepareMolForDrawing(mol_with_shap, kekulize=True)
    #AllChem.Compute2DCoords(mol_with_shap)

    if label_atoms:
        for atom in mol_with_shap.GetAtoms():
            atom_idx = atom.GetIdx()
            if atom_idx in atom_labels:
                atom.SetProp('atomNote', atom_labels[atom_idx])

    if label_bonds and bond_labels:
        for bond in mol_with_shap.GetBonds():
            bond_idx = bond.GetIdx()
            if bond_idx in bond_labels:
                bond.SetProp('bondNote', bond_labels[bond_idx])

    # --- Drawing setup ---
    drawer_png = rdMolDraw2D.MolDraw2DCairo(300, 300)
    drawer_svg = rdMolDraw2D.MolDraw2DSVG(300, 300)

    drawer_png.drawOptions().fontName = "Arial"
    drawer_svg.drawOptions().fontName = "Arial"
    drawer_png.drawOptions().atomHighlightsAreCircles = True
    drawer_svg.drawOptions().atomHighlightsAreCircles = True

    if black_and_white:
        drawer_png.drawOptions().useBWAtomPalette()
        drawer_svg.drawOptions().useBWAtomPalette()

    drawer_png.drawOptions().highlightAtoms = list(atom_colors.keys())
    drawer_svg.drawOptions().highlightAtoms = list(atom_colors.keys())

    # --- Apply highlights ---
    if color_atoms and color_bonds:
        rdMolDraw2D.PrepareAndDrawMolecule(
            drawer_png, mol_with_shap,
            highlightAtoms=list(atom_colors.keys()),
            highlightAtomColors=atom_colors,
            highlightBonds=list(bond_colors.keys()),
            highlightBondColors=bond_colors
        )
        rdMolDraw2D.PrepareAndDrawMolecule(
            drawer_svg, mol_with_shap,
            highlightAtoms=list(atom_colors.keys()),
            highlightAtomColors=atom_colors,
            highlightBonds=list(bond_colors.keys()),
            highlightBondColors=bond_colors
        )
    elif color_atoms:
        rdMolDraw2D.PrepareAndDrawMolecule(
            drawer_png, mol_with_shap,
            highlightAtoms=list(atom_colors.keys()),
            highlightAtomColors=atom_colors
        )
        rdMolDraw2D.PrepareAndDrawMolecule(
            drawer_svg, mol_with_shap,
            highlightAtoms=list(atom_colors.keys()),
            highlightAtomColors=atom_colors
        )
    elif color_bonds:
        rdMolDraw2D.PrepareAndDrawMolecule(
            drawer_png, mol_with_shap,
            highlightBonds=list(bond_colors.keys()),
            highlightBondColors=bond_colors
        )
        rdMolDraw2D.PrepareAndDrawMolecule(
            drawer_svg, mol_with_shap,
            highlightBonds=list(bond_colors.keys()),
            highlightBondColors=bond_colors
        )
    else:
        rdMolDraw2D.PrepareAndDrawMolecule(drawer_png, mol_with_shap)
        rdMolDraw2D.PrepareAndDrawMolecule(drawer_svg, mol_with_shap)

    # --- Save images ---
    drawer_png.FinishDrawing()
    shap_path_png = f"{target_path}/{identifier}.png"
    drawer_png.WriteDrawingText(shap_path_png)

    drawer_svg.FinishDrawing()
    shap_path_svg = f"{target_path}/{identifier}.svg"
    with open(shap_path_svg, "w") as f:
        f.write(drawer_svg.GetDrawingText())

    # --- Save colorbar ---
    if color_bar:
        fig, ax = plt.subplots(figsize=(2, 4))

        # Get the exact cmap + norm used for coloring
        cmap, norm = get_color(0, min_shap, max_shap,
                            ref_min=ref_min, ref_max=ref_max,
                            return_mapper=True)

        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        cb = plt.colorbar(sm, ax=ax)
        cb.set_label("SHAP Contribution")

        plt.savefig(f"{target_path}/{identifier}_colorbar.svg")
        plt.close()

    if display_mols:
        try:
            from IPython.display import Image, display
            display(Image(filename=shap_path_png))
        except ImportError:
            print(f"Image for molecule {identifier} saved at {target_path}.")

    print(f"Saved molecule visualization to {shap_path_png} and {shap_path_svg}")