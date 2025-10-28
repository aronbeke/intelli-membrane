import random
import numpy as np
import pandas as pd
import torch
import os
import re
import shutil


def read_txt_to_list(txt_file, as_int=False):
    with open(txt_file, "r") as file:
        if as_int:
            my_list_from_file = [int(line.strip()) for line in file]
        else:
            my_list_from_file = [line.strip() for line in file]
    
    return my_list_from_file

def copy_and_rename_file(source_path, destination_path, replace=False):
    """
    Copies a file from source_path to destination_path.
    
    :param source_path:    The full path of the source file.
    :param destination_path:  The full path to destination.
    """
    # Ensure the destination directory exists
    os.makedirs(os.path.dirname(destination_path), exist_ok=True)

    if not replace:
        # Only copy if the file does not already exist
        if not os.path.exists(destination_path):
            shutil.copy2(source_path, destination_path)
            print(f"File copied from {source_path} to {destination_path}.")
        else:
            print(f"File '{destination_path}' already exists. Skipping copy.")
    elif replace:
        shutil.copy2(source_path, destination_path)
        print(f"File copied from {source_path} to {destination_path}.")
            

def load_feature_columns(type = 'mcmpnn', nlc = False, dset='', n_solutes=1):
    '''
    Universal function to load features - ensures the same feature order
    '''
    assert n_solutes in [1,2]

    extra_continuous_columns = ['mwco', 'contact_angle', 'zeta_potential', 'permeance', 'pressure', 'temperature']
    #extra_continuous_columns = ['permeance', 'pressure', 'temperature']
    extra_categorical_columns = ['contact_angle_missing', 'zeta_potential_missing']
    #extra_categorical_columns = []

    solvent_properties = ['solvent_mw','solvent_logp','solvent_viscosity','solvent_density','solvent_dielectric_constant','solvent_dd','solvent_dp','solvent_dh','solvent_molar_volume']
    
    if type == 'admpnn' and n_solutes==2:
        solute_properties = ['solute_mw', 'solute_logp', 'secondary_solute_mw', 'secondary_solute_logp','secondary_solute_ratio']
    else:
        solute_properties = ['solute_mw', 'solute_logp']

    membrane_columns = read_txt_to_list('data/aux/membrane_columns'+dset+'.txt')
    process_columns = read_txt_to_list('data/aux/process_configuration_columns'+dset+'.txt')

    extra_continuous_columns.extend(solvent_properties)
    extra_continuous_columns.extend(solute_properties)

    extra_categorical_columns.extend(membrane_columns)
    extra_categorical_columns.extend(process_columns)

    if nlc:
        target_column = ['nlc_rejection']
    else:
        target_column = ['rejection']

    if type == 'mcmpnn':
        smiles_columns = ['solute_smiles', 'solvent_smiles']
        return smiles_columns, target_column, extra_continuous_columns, extra_categorical_columns
    
    elif type == 'admpnn':
        '''
        Attention distributed GNN
        '''
        smiles_columns = ['solute_smiles', 'secondary_solute_smiles', 'solvent_smiles']
        return smiles_columns, target_column, extra_continuous_columns, extra_categorical_columns

    elif type == 'molclr_gcn':
        molecule_features_prefix = 'solute_smiles_molclr'
        solvent_features_prefix = 'solvent_smiles_molclr'
        target_column = target_column[0]
        return molecule_features_prefix, solvent_features_prefix, target_column, extra_continuous_columns, extra_categorical_columns

    elif type == 'himol_gcn':
        molecule_features_prefix = 'solute_smiles_himol'
        solvent_features_prefix = 'solvent_smiles_himol'
        target_column = target_column[0]
        return molecule_features_prefix, solvent_features_prefix, target_column, extra_continuous_columns, extra_categorical_columns

    else:
        print('Unknown type for feature column loading')
        return


def get_models_dict(folder_path,model_label):
    models_dict = {}
    #pattern = re.compile(model_label+'_fold(\d+)-epoch=\d+-val_loss=[\d\.]+\.ckpt')
    pattern = re.compile(rf"{model_label}_fold(\d+)-epoch=\d+-val_loss_weighted=[\d\.]+\.ckpt")
    
    for filename in os.listdir(folder_path):
        match = pattern.search(filename)
        if match:
            fold = f"fold{match.group(1)}"
            models_dict[fold] = os.path.join(folder_path, filename)
    
    return models_dict


def apply_filter(df, filter_tuple):
    '''
    Example filter_tuple: ("solute_category", "metal_catalyst", "==")
    Usage: df[apply_filter(df, filter_tuple)]
    '''
    assert len(filter_tuple) == 3
    assert filter_tuple[2] in ["==", "!="]

    if filter_tuple[2] == "==":
        return df[filter_tuple[0]] == filter_tuple[1]
    
    elif filter_tuple[2] == "!=":
        return df[filter_tuple[0]] != filter_tuple[1]