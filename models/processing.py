import numpy as np
import pandas as pd
import models.auxiliary
import os
from sklearn.preprocessing import PowerTransformer
from sklearn.model_selection import StratifiedKFold, train_test_split
from scipy import stats
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, DataStructs


def clip_rejection(df, rejection_column, return_copy = False):
    if return_copy:
        working_df = df.copy()
    else:
        working_df = df

    working_df[rejection_column] = working_df[rejection_column].clip(upper=1.0)

    return working_df


def clip_permeance(df, permeance_column, return_copy = False):
    if return_copy:
        working_df = df.copy()
    else:
        working_df = df

    working_df[permeance_column] = working_df[permeance_column].clip(lower=0.0)

    return working_df


def filter_csv_records(csv_file, txt_file, target_path):
    """
    Processes a CSV and a TXT file:
    - Loads the CSV into a DataFrame with 'id' as the index.
    - For each row in the TXT file:
      - Splits identifier and SMILES.
      - Filters the DataFrame for matching and non-matching SMILES.
      - Saves the filtered results and the rest into separate CSV files.

    Args:
        csv_file (str): Path to the input CSV file.
        txt_file (str): Path to the input TXT file.
        target_path (str): Path to the target location.
    """
    # Load the CSV into a DataFrame
    df = pd.read_csv(csv_file, index_col='id')

    # Open and read the TXT file line by line
    with open(txt_file, 'r') as f:
        lines = f.readlines()

    # Process each line in the TXT file
    for line in lines:
        identifier, smiles = line.strip().split(',')

        # Create a copy of the DataFrame
        df_copy = df.copy()

        # Filter the DataFrame for matching and non-matching SMILES
        matching_records = df_copy[df_copy['solute_smiles'] == smiles]
        rest_records = df_copy[df_copy['solute_smiles'] != smiles]

        # Generate filenames
        base_name = os.path.splitext(os.path.basename(csv_file))[0]
        matching_file = f"{target_path}/{base_name}_{identifier}_filtered.csv"
        rest_file = f"{target_path}/{base_name}_{identifier}_rest.csv"

        # Save the filtered results to CSV
        matching_records.to_csv(matching_file)
        rest_records.to_csv(rest_file)

        print(f"Processed {identifier}: {matching_file}, {rest_file} saved.")


def get_uniques_from_two(df1, df2, feature):
    uniques_df1 = df1[feature].unique()
    uniques_df2 = df2[feature].unique()

    # Create a complete list of unique membrane families from both datasets
    all_uniques = sorted(set(uniques_df1).union(set(uniques_df2)))

    return uniques_df1, uniques_df2, all_uniques

def get_uniques(df, feature):
    all_uniques = df[feature].unique()

    return all_uniques


def one_hot_encode(dataframe, feature, all_unique, exceptions=[]):
    original_feature = dataframe[feature]  # Save the original feature column
    
    dataframe = pd.get_dummies(dataframe, columns=[feature])
    one_hot_columns = [col for col in dataframe.columns if col.startswith(f"{feature}_")]
    
    for elem in exceptions:
        if elem in one_hot_columns:
            one_hot_columns.remove(elem)
    
    dataframe[one_hot_columns] = dataframe[one_hot_columns].astype(int)
    
    # Ensure all categories in `all_unique` exist as one-hot columns
    dataframe = dataframe.reindex(columns=dataframe.columns.tolist() + 
                                            [feature+f'_{feat}' for feat in all_unique 
                                             if feature+f'_{feat}' not in dataframe.columns], 
                                            fill_value=0)
    
    dataframe[feature] = original_feature  # Restore the original feature column
    
    return dataframe, one_hot_columns


def transform_to_one_hot_encode(dataframe, feature, one_hot_columns):
    """
    Transforms a categorical column into a one-hot encoded format using pre-defined one-hot column names.
    
    Parameters:
        dataframe (pd.DataFrame): The dataframe containing the categorical column.
        feature (str): The name of the categorical column to be transformed.
        one_hot_columns (list): List of expected one-hot encoded column names.
    
    Returns:
        pd.DataFrame: The transformed dataframe with one-hot encoded columns.
    """
    

    missing_cols = [col for col in one_hot_columns if col not in dataframe.columns]

    if missing_cols:
        filler_df = pd.DataFrame(0, index=dataframe.index, columns=missing_cols)
        dataframe = pd.concat([dataframe, filler_df], axis=1)
    
    # Populate the one-hot columns based on the feature values
    for index, row in dataframe.iterrows():
        one_hot_col_name = f"{feature}_{row[feature]}"
        if one_hot_col_name in one_hot_columns:
            dataframe.at[index, one_hot_col_name] = 1
    
    # Drop the original categorical column
    # dataframe = dataframe.drop(columns=[feature])
    
    return dataframe
    

def remove_incomplete_records(df):
    initial_count = len(df)
    df = df.dropna()
    final_count = len(df)
    dropped_records = initial_count - final_count
    print(f'Number of records dropped from dataframe: {dropped_records}')
    return df


def remove_duplicates(df):
    len_before = len(df)
    df = df.drop_duplicates()
    len_after = len(df)
    print("No. of duplicates dropped from dataframe:",len_before-len_after)
    return df


def rejection_processing(df):
    '''
    Optional Gaussian noise and log modification
    NLC Rejection: negative logarithmic complement rejection
    '''
    df['nlc_rejection'] = df['rejection']

    # noise = np.random.normal(loc=0.9950, scale=0.002, size=df[df['rejection'] == 1.00].shape[0])
    # df.loc[df['rejection'] == 1.00, 'corr_rejection'] = noise.clip(0.9900, 0.9999)

    df['nlc_rejection'] = - np.log10(1 - df['nlc_rejection'] + 1e-3)  # Adding a small constant to handle R == 1.0 case


def complex_rejection_processing(df1,df2):
    rejection_processing(df1)
    rejection_processing(df2)

    concat_df = pd.concat([df1['corr_rejection'], df2['corr_rejection']])

    # Yeo-Johnson transformation
    pt = PowerTransformer(method='yeo-johnson')  # Yeo-Johnson is bijective
    combined_transformed_yj = pt.fit_transform(concat_df.values.reshape(-1, 1))

    df1['yj_corr_rejection'] = combined_transformed_yj[:len(df1)]
    df2['yj_corr_rejection'] = combined_transformed_yj[len(df1):]

    # Box-Cox transf
    #shifted_data = concat_df + abs(concat_df.min()) + 0.001
    shifted_data = concat_df + 10
    combined_transformed_bc, _ = stats.boxcox(shifted_data)
    df1['bc_corr_rejection'] = combined_transformed_bc[:len(df1)]
    df2['bc_corr_rejection'] = combined_transformed_bc[len(df1):]


def unify_datasets(df1, df2):
    '''
    Keeps one instance of duplicates, the one in df2. Dfs should have a 'source' feature
    '''
    # Read the two CSV files into DataFrames
    len_1 = len(df1)
    len_2 = len(df2)

    # Concatenate the DataFrames
    combined_df = pd.concat([df2,df1], ignore_index=True)
    combined_df = combined_df.drop_duplicates(subset=[col for col in combined_df.columns if col != 'source'], keep='first')

    len_combined = len(combined_df)

    print("Number of shared records dropped (first instances kept): ")
    print(len_1+len_2-len_combined)

    combined_df.reset_index(drop=True, inplace=True)
    combined_df.index.name = 'id'  # Set the index name to 'id'
    
    return combined_df

def process_smiles(csv_file, column_name, output_file, type='txt'):
    """
    Extracts a specified column from a CSV file, removes duplicates, and writes 
    the unique entries to a TXT or CSV file, one per line.

    Args:
        csv_file (str): Path to the input CSV file.
        column_name (str): The name of the column to process.
        output_txt_file (str): Path to the output TXT file.
    """
    try:
        # Load the CSV file
        df = pd.read_csv(csv_file)

        # Check if the specified column exists
        if column_name not in df.columns:
            print(f"Column '{column_name}' not found in the CSV file.")
            return

        # Extract the column and drop duplicates
        unique_smiles = df[column_name].drop_duplicates()

        if type == 'txt':
            # Write unique SMILES entries to a text file
            with open(output_file, 'w') as f:
                for smiles in unique_smiles:
                    f.write(f"{smiles}\n")
            print(f"Processed SMILES have been written to {output_file}.")
        elif type == 'csv':
            unique_smiles.reset_index(drop=True, inplace=True)
            unique_smiles.to_csv(output_file, index_label='id')
            print(f"Processed SMILES have been written to {output_file}.")
        else:
            print('Unknown file type.')

    except Exception as e:
        print(f"An error occurred: {e}")


def process_membrane_and_solvent_features(df):
    '''
    Unviversal zeta and contact angle for membranes - this drops more duplicates
    '''

    # Solute properties
    def calc_mw(smiles):
        mol = Chem.MolFromSmiles(smiles)
        return Descriptors.MolWt(mol) if mol else None

    def calc_logp(smiles):
        mol = Chem.MolFromSmiles(smiles)
        return Descriptors.MolLogP(mol) if mol else None

    # Membrane properties
    membrane_df = pd.read_csv('data/membrane_data.csv')
    membrane_families = membrane_df.set_index('membrane')['family'].to_dict()
    membrane_mwcos = membrane_df.set_index('membrane')['mwco'].to_dict()
    membrane_contact_angles = membrane_df.set_index('membrane')['contact_angle'].to_dict()
    membrane_zeta_potential = membrane_df.set_index('membrane')['zeta_potential'].to_dict()
    membrane_contact_angle_missings = membrane_df.set_index('membrane')['contact_angle_missing'].to_dict()
    membrane_zeta_potential_missings = membrane_df.set_index('membrane')['zeta_potential_missing'].to_dict()

    # Solvent properties
    solvent_df = pd.read_csv('data/solvents/solvent_data.csv')
    solvent_smiles = solvent_df.set_index('solvent')['solvent_smiles'].to_dict()
    solvent_mw = solvent_df.set_index('solvent')['solvent_mw'].to_dict()
    solvent_logp = solvent_df.set_index('solvent')['solvent_logp'].to_dict()
    solvent_viscosity = solvent_df.set_index('solvent')['solvent_viscosity'].to_dict()
    solvent_density = solvent_df.set_index('solvent')['solvent_density'].to_dict()
    solvent_dielectric_constant = solvent_df.set_index('solvent')['solvent_dielectric_constant'].to_dict()
    solvent_dd = solvent_df.set_index('solvent')['solvent_dd'].to_dict()
    solvent_dp = solvent_df.set_index('solvent')['solvent_dp'].to_dict()
    solvent_dh = solvent_df.set_index('solvent')['solvent_dh'].to_dict()
    solvent_molar_volume = solvent_df.set_index('solvent')['solvent_molar_volume'].to_dict()

    df['membrane_family'] = df['membrane'].map(membrane_families)
    df['mwco'] = df['membrane'].map(membrane_mwcos)
    df['contact_angle'] = df['membrane'].map(membrane_contact_angles)
    df['zeta_potential'] = df['membrane'].map(membrane_zeta_potential)
    df['contact_angle_missing'] = df['membrane'].map(membrane_contact_angle_missings)
    df['zeta_potential_missing'] = df['membrane'].map(membrane_zeta_potential_missings)
    df['solvent_smiles'] = df['solvent'].map(solvent_smiles)
    df['solvent_mw'] = df['solvent'].map(solvent_mw)
    df['solvent_logp'] = df['solvent'].map(solvent_logp)
    df['solvent_viscosity'] = df['solvent'].map(solvent_viscosity)
    df['solvent_density'] = df['solvent'].map(solvent_density)
    df['solvent_dielectric_constant'] = df['solvent'].map(solvent_dielectric_constant)
    df['solvent_dd'] = df['solvent'].map(solvent_dd)
    df['solvent_dp'] = df['solvent'].map(solvent_dp)
    df['solvent_dh'] = df['solvent'].map(solvent_dh)
    df['solvent_molar_volume'] = df['solvent'].map(solvent_molar_volume)

    df['solute_mw'] = df['solute_smiles'].apply(calc_mw)
    df['solute_logp'] = df['solute_smiles'].apply(calc_logp)
    try:
        df['secondary_solute_mw'] = df['secondary_solute_smiles'].apply(calc_mw)
        df['secondary_solute_logp'] = df['secondary_solute_smiles'].apply(calc_logp)
    except KeyError:
        print('No secondary solutes found.')

    return df


def complete_input_processing(file_path, dset):
    res_df = pd.read_csv(file_path)
    res_df = process_membrane_and_solvent_features(res_df)
    mem_cols = models.auxiliary.read_txt_to_list('data/aux/membrane_columns'+dset+'.txt')
    pro_con_cols = models.auxiliary.read_txt_to_list('data/aux/process_configuration_columns'+dset+'.txt')
    res_df = transform_to_one_hot_encode(res_df, 'membrane', mem_cols)
    res_df = transform_to_one_hot_encode(res_df, 'process_configuration', pro_con_cols)
    res_df.to_csv(file_path)


def get_representations(input: str, output: str, source_solutes: str, source_solvents: str, no_features: int, model_type: str, drop = []):
    """
    Merges solute and solvent representations from source files into the main dataset.

    Parameters:
    - input (str): Path to the input CSV file containing solute and solvent information.
    - output (str): Path to save the output CSV file with merged representations.
    - source_solutes (str): Path to the CSV file containing solute representations.
    - source_solvents (str): Path to the CSV file containing solvent representations.
    - no_features (int): Number of feature columns to copy.
    - model_type (str): The model type prefix for feature column names.

    Returns:
    - None: Saves the merged DataFrame to the specified output path.
    """

    # Read input and source dataframes
    raw_df = pd.read_csv(input)

    data_df = raw_df.copy()
    data_df = data_df[~data_df['solute_smiles'].isin(drop)]

    source_solutes_df = pd.read_csv(source_solutes)
    source_solvents_df = pd.read_csv(source_solvents)

    # Ensure solute merge is based on 'solute_smiles'
    solute_feature_columns = [f"solute_smiles_{model_type}{str(i).zfill(3)}" for i in range(no_features)]
    
    if 'solute_smiles' not in data_df.columns or 'solute_smiles' not in source_solutes_df.columns:
        raise KeyError("Column 'solute_smiles' must exist in both input and solute source files.")
    
    merged_df = data_df.merge(
        source_solutes_df[['solute_smiles'] + solute_feature_columns], 
        on='solute_smiles', 
        how='left'
    )

    # Ensure solvent merge is based on 'solvent' instead of 'solvent_smiles'
    solvent_feature_columns = [f"solvent_smiles_{model_type}{str(i).zfill(3)}" for i in range(no_features)]
    
    if 'solvent' not in data_df.columns or 'solvent' not in source_solvents_df.columns:
        raise KeyError("Column 'solvent' must exist in both input and solvent source files.")
    
    merged_df = merged_df.merge(
        source_solvents_df[['solvent'] + solvent_feature_columns], 
        on='solvent', 
        how='left'
    )
    
    # Save the final dataframe
    merged_df.to_csv(output, index=False)


def set_zeroshot_test(og_df, zero_shot_solute_smiles_list, secondary_active = False):
    if secondary_active:
        zero_shot_mask = (
            og_df["solute_smiles"].isin(zero_shot_solute_smiles_list) |
            og_df["secondary_solute_smiles"].isin(zero_shot_solute_smiles_list)
        )
    else:
        zero_shot_mask = og_df["solute_smiles"].isin(zero_shot_solute_smiles_list)

    og_df.loc[zero_shot_mask, 'test'] = 1
    og_df.loc[~zero_shot_mask, 'test'] = 0
    return og_df


def rejection_weight(R, base=0.5, max_w=2.0, R_cutoff=0.5):
    """
    Continuous weight function based on rejection R ∈ [0,1].
    - Constant weight = base until R_cutoff.
    - Then linearly increases to max_w at R = 1.0.
    """
    R = np.clip(R, 0, 1)  # ensure R is in [0,1]
    
    ramp = np.where(
        R > R_cutoff,
        (R - R_cutoff) / (1 - R_cutoff),  # linear increase from 0 to 1
        0.0
    )

    return base + (max_w - base) * ramp


def data_weight(
        df: pd.DataFrame,
        weight_losses = True,
        filter=("source", "cat", "=="), 
        target_bias_dict=None,
        class_bias_factor = 0,
        base=1.0, 
        max_w=1.0, 
        R_cutoff=0.5,
        verbose=False
    ) -> pd.DataFrame:

    """
    Applies a three-stage loss weighting protocol.
    
    1. Rejection-based continuous weighting.
    2. Source-specific normalization to control bias.
    3. Membrane-specific weighting to control class imbalance.

    Concepts:
    Target bias factor: 1 means equal weight on target and general datasets. If tbf == n_t/n_g -> w_g == w_t == 1
    Class bias factor: 0 means no class imbalance correction, 1 means full correction (equal contribution among classes)

    Parameters:
    - df: Input dataframe with 'rejection' and 'membrane' columns.
    - filter: Tuple indicating which column and value to treat as 'target' (e.g., ('source', 'cat', '==')).
    - target_bias_dict: Dictionary mapping membranes to target bias factors (e.g., {'DM300': 2.0}).
    - base, max_w, R_cutoff, steepness: Parameters for the rejection-based weight curve.

    Returns:
    - Modified dataframe with new column 'loss_weight'.
    """

    if weight_losses == False:
        df['loss_weight'] = 1.0
        return df

    # Normalized weight based on rejection
    df['loss_weight'] = df['rejection'].apply(lambda r: rejection_weight(r, base=base, max_w=max_w, R_cutoff=R_cutoff))
    alpha = len(df) / df['loss_weight'].sum()
    df['loss_weight'] *= alpha

    # Normalized target equalization
    if target_bias_dict is not None:
        target_bias_dict = target_bias_dict.copy()
        target_bias_dict['__remainder__'] = 1.0
        for membrane, target_bias in target_bias_dict.items():
            if membrane == '__remainder__':
                target_mask = (models.auxiliary.apply_filter(df,filter)) & (~df['membrane'].isin(target_bias_dict.keys()))
                og_mask = (~models.auxiliary.apply_filter(df,filter)) & (~df['membrane'].isin(target_bias_dict.keys()))
            else:
                target_mask = (models.auxiliary.apply_filter(df,filter)) & (df['membrane'] == membrane)
                og_mask = (~models.auxiliary.apply_filter(df,filter)) & (df['membrane'] == membrane)

            n_t = target_mask.sum()
            n_g = og_mask.sum()

            if n_t > 0 and n_g > 0 and n_g > n_t:
                w_g = (n_g + n_t) / ((1 + target_bias) * n_g)
                w_t = target_bias * (n_g / n_t) * w_g
            else:
                w_g = 1.0
                w_t = 1.0
            
            df.loc[target_mask, 'loss_weight'] *= w_t
            df.loc[og_mask, 'loss_weight'] *= w_g
            
            if verbose:
                print(f"Membrane: {membrane}, n_t: {n_t}, n_g: {n_g}, w_t: {w_t:.3f}, w_g: {w_g:.3f}")

    mean_records_per_membrane = df.groupby('membrane').size().mean()

    # Normalized class equalization
    for membrane in df['membrane'].unique().tolist():
        membrane_mask = df['membrane'] == membrane
        records_per_membrane = len(df[membrane_mask])

        w_m = (records_per_membrane + class_bias_factor*(mean_records_per_membrane - records_per_membrane))/records_per_membrane

        df.loc[membrane_mask, 'loss_weight'] *= w_m

    return df


def train_val_split(focus_df, protocol, target_column, stratification, bins, seed, k):
    if protocol == 'standard':
        focus_df["binned_target"] = pd.qcut(focus_df[target_column], q=bins, labels=False, duplicates="drop")

        if stratification == "rejection":
            focus_df["stratify_label"] = focus_df["binned_target"]
        elif stratification == "membrane":
            focus_df["stratify_label"] = focus_df["membrane"]

        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)    
        set_folds = list(skf.split(focus_df, focus_df["stratify_label"]))

    elif protocol == 'solute_disjoint':
        solute_to_records = focus_df.groupby("solute_smiles").indices

        # Create a list of (solute_smiles, num_records) tuples
        solute_sizes = [(solute, len(indices)) for solute, indices in solute_to_records.items()]
        
        # Sort solutes by number of records (descending) to assign large ones first
        solute_sizes.sort(key=lambda x: -x[1])

        # Initialize empty folds and fold sizes
        solute_folds = [[] for _ in range(k)]
        fold_sizes = [0 for _ in range(k)]

        # Greedy assignment to keep fold sizes balanced
        for solute, size in solute_sizes:
            # Find the fold with the smallest current size
            set_fold = fold_sizes.index(min(fold_sizes))
            solute_folds[set_fold].append(solute)
            fold_sizes[set_fold] += size

        # Convert solute assignments to record indices
        fold_indices = []
        for fold_solutes in solute_folds:
            indices = []
            for solute in fold_solutes:
                indices.extend(solute_to_records[solute])
            fold_indices.append(np.array(indices))

        # Create (train_idx, val_idx) tuples
        set_folds = []
        for i in range(k):
            val_idx = fold_indices[i]
            train_idx = np.concatenate([fold_indices[j] for j in range(k) if j != i])
            set_folds.append((train_idx, val_idx))

    elif protocol == 'none':
        # All data in training, empty validation
        all_indices = np.arange(len(focus_df))
        set_folds = [(all_indices, np.array([], dtype=int)) for _ in range(k)]
        
    return set_folds


def stratified_kfold_split(
    og_df, 
    target_column,
    k=5, 
    seed=42, 
    bins=10,
    focus='all',
    general_validation_type = 'none',
    target_validation_type = 'standard',
    filter_condition = ("source","cat","=="),
    stratification = 'rejection',
    test_type='standard',
    weight_parameters = None
):
    """
    Creates K stratified splits (train-validation) on the main data.
    Handles generalization (target) bias, class imbalance, and rejection bias via weighting.
    For validation only rejection weighting is applied.

    Args:
        og_df (pd.DataFrame): The original dataframe to split. Must contain a 'source' column.
        target_column (str): The continuous target column.
        k (int): Number of folds.
        seed (int): Random seed for reproducibility.
        bins (int): Number of bins to discretize the target variable for stratification.
        focus (str): what to include: all, target, general
        general_validation_type (str): 'standard', 'none', 'solute_disjoint'. If 'solute_disjoint': complete solute record set are chosen for validation.
        target_validation_type (str): 'standard', 'none', 'solute_disjoint'.
        filter_condition (tuple): A tuple of (column_name, value) to filter records for target dataset. Default is ("source", "cat").
        stratification (str): Stratify either by 'rejection' or 'membrane'
        test_type (str): which test set to consider (solute_disjoint or standard). If 'solute_disjoint_outer_val', the generated test folds are the outer validation set, the true test set is dropped.
        weight_parameters (dict): Parameters for data weighting. If None, no weighting is applied.
            Should contain:
                - 'weight_losses': Boolean to trigger weighting.
                - 'filter_condition': Tuple for filtering records.
                - 'target_bias_dict': Dictionary for target bias factors.
                - 'class_bias_factor': Float for class bias factor
    Returns:
        list: A list of (train_df, val_df, test_df) tuples, each with 'id_original' to track original indices.
    """

    assert stratification in ['rejection','membrane']
    assert target_validation_type in ['standard', 'solute_disjoint', 'none']
    assert general_validation_type in ['standard', 'solute_disjoint', 'none']
    assert focus in ['all', 'target', 'general']
    assert test_type in ['standard', 'solute_disjoint', 'solute_disjoint_outer_val']

    if weight_parameters and ('target_bias_dict_val' not in weight_parameters):
        weight_parameters['target_bias_dict_val'] = weight_parameters['target_bias_dict']
    
    if weight_parameters and ('class_bias_factor_val' not in weight_parameters):
        weight_parameters['class_bias_factor_val'] = weight_parameters['class_bias_factor']

    ### --- SET UP DF AND ZERO SHOT DF --- ###

    full_df = og_df.copy()
    full_df["is_augmented"] = 0
    full_df["id_original"] = full_df.index

    if test_type == 'solute_disjoint_outer_val':
        # test_solute_disjoint == 1 records are dropped, test_solute_disjoint_outer_val == 1 records are used in test folds
        keep_mask = ~ (full_df['test_solute_disjoint'] == 1)
        full_df = full_df[keep_mask].reset_index(drop=True)

    test_col = 'test_' + test_type

    target_full_df = full_df[(models.auxiliary.apply_filter(full_df,filter_condition))].reset_index(drop=True)
    general_full_df = full_df[(~models.auxiliary.apply_filter(full_df,filter_condition))].reset_index(drop=True)

    target_test_mask = target_full_df[test_col] == 1
    target_test_df = target_full_df[target_test_mask].reset_index(drop=True)
    target_df = target_full_df[~target_test_mask].reset_index(drop=True)

    general_test_mask = general_full_df[test_col] == 1
    general_test_df = general_full_df[general_test_mask].reset_index(drop=True)
    general_df = general_full_df[~general_test_mask].reset_index(drop=True)

    general_folds = train_val_split(general_df, general_validation_type, target_column, stratification, bins, seed, k)
    target_folds = train_val_split(target_df, target_validation_type, target_column, stratification, bins, seed, k)

    # Test sets
    test_sets = []
    if focus == 'all':
        test_sets.append(general_test_df)
        test_sets.append(target_test_df)
    elif focus == 'target':
        test_sets.append(general_full_df)
        test_sets.append(target_test_df)
    elif focus == 'general':
        test_sets.append(general_test_df)
        test_sets.append(target_full_df)

    test_df = pd.concat(test_sets, ignore_index=True)
    test_df = test_df.sample(frac=1, random_state=seed).reset_index(drop=True)

    # Training and validation
    folds = []

    for fold_idx in range(k):
        train_sets = []
        val_sets = []

        if focus in ['all', 'general']:
            train_general = general_df.iloc[general_folds[fold_idx][0]].reset_index(drop=True)
            val_general = general_df.iloc[general_folds[fold_idx][1]].reset_index(drop=True)

            train_sets.append(train_general)
            if general_validation_type != 'none':
                val_sets.append(val_general)
        
        if focus in ['all', 'target']:
            train_target = target_df.iloc[target_folds[fold_idx][0]].reset_index(drop=True)
            val_target = target_df.iloc[target_folds[fold_idx][1]].reset_index(drop=True)

            train_sets.append(train_target)
            if target_validation_type != 'none':
                val_sets.append(val_target)
            
        train_df = pd.concat(train_sets, ignore_index=True)
        val_df = pd.concat(val_sets, ignore_index=True)

        train_df = train_df.sample(frac=1, random_state=fold_idx).reset_index(drop=True)
        val_df = val_df.sample(frac=1, random_state=fold_idx).reset_index(drop=True)

        # Weighting
        if weight_parameters:
            train_df = data_weight(
                train_df,
                weight_losses = weight_parameters['weight_losses'],
                filter=weight_parameters['filter_condition'], 
                target_bias_dict=weight_parameters['target_bias_dict'],
                class_bias_factor=weight_parameters['class_bias_factor'],
                base=weight_parameters['base'],
                R_cutoff=weight_parameters['R_cutoff'],
                max_w=weight_parameters['max_w']
            )

            ## VALIDATION WEIGHTING - CAREFUL!
            val_df = data_weight(
                val_df,
                weight_losses = weight_parameters['weight_losses'],
                filter = weight_parameters['filter_condition'],
                target_bias_dict=weight_parameters['target_bias_dict_val'],
                class_bias_factor=weight_parameters['class_bias_factor_val'],
                base=weight_parameters['base'],
                R_cutoff=weight_parameters['R_cutoff'],
                max_w=weight_parameters['max_w']
            )

            test_df = data_weight(test_df, weight_losses = False)

        for _df in [train_df, val_df]:
            _df.drop(columns=["binned_target", "stratify_label"], inplace=True, errors='ignore')

        folds.append((train_df, val_df, test_df.copy()))

        # Sanity check
        used_ids = set(train_df["id_original"]) | set(val_df["id_original"]) | set(test_df["id_original"])
        assert used_ids.issubset(set(full_df["id_original"]))
        assert len(used_ids) == len(full_df)

    return folds


def rebuild_original_df(k_folds, original_df):
    """Rebuilds the original dataframe from K stratified folds using 'id_original' for proper reconstruction.
    Doesn't work with zero-shot splits.

    Args:
        k_folds (list): A list of (train_df, val_df, test_df) tuples with 'id_original' column.
        original_df (pd.DataFrame): The original dataframe before splitting.

    Returns:
        pd.DataFrame: The reconstructed dataframe with additional columns.
    """
    original_df = original_df.copy()  # Ensure we don't modify the original

    # Collect all new columns that appear in any fold but are NOT in the original dataframe
    original_columns = set(original_df.columns)
    new_columns = set()

    for train_df, val_df, test_df in k_folds:
        new_columns.update(set(train_df.columns) - original_columns)
        new_columns.update(set(val_df.columns) - original_columns)
        new_columns.update(set(test_df.columns) - original_columns)

    new_columns = list(new_columns)  # Convert to list for iteration

    # Initialize new columns with NaN
    for col in new_columns:
        original_df[col] = float("nan")

    # Helper function to filter out augmented rows
    def filter_non_augmented(df):
        if "is_augmented" in df.columns:
            return df[df["is_augmented"] == 0]
        else:
            return df

    # Populate new columns using 'id_original' for correct alignment
    for train_df, val_df, test_df in k_folds:
        for df in [train_df, val_df, test_df]:
            df_filtered = filter_non_augmented(df)
            for col in new_columns:
                if col in df_filtered.columns:
                    original_df.loc[df_filtered["id_original"], col] = df_filtered[col].values

    return original_df
    

def select_solutes_with_membrane_limit(combined_df, low_limit, high_limit, membrane_limit=None, secondary_active=False):
    # Filter to only include 'cat' sources
    filtered_df = combined_df[combined_df['source'] == 'cat']

    # Prepare solute counts (for primary or both solutes)
    if not secondary_active:
        solute_counts = filtered_df['solute_smiles'].value_counts()
    else:
        all_solutes = pd.concat([
            filtered_df['solute_smiles'],
            filtered_df['secondary_solute_smiles']
        ]).dropna()
        solute_counts = all_solutes.value_counts()

    # Filter based on frequency
    eligible_solutes = solute_counts[(solute_counts >= low_limit) & (solute_counts <= high_limit)].index

    # Apply the frequency filtering to the dataframe
    if not secondary_active:
        filtered_df = filtered_df[filtered_df['solute_smiles'].isin(eligible_solutes)]
    else:
        filtered_df = filtered_df[
            filtered_df['solute_smiles'].isin(eligible_solutes) |
            filtered_df['secondary_solute_smiles'].isin(eligible_solutes)
        ]

    if filtered_df.empty:
        raise ValueError("No solutes meet frequency criteria after filtering.")

    # Category-wise solute collection
    categories = ['metal_catalyst', 'organocatalyst', 'ligand']
    selected_smiles_by_category = {cat: [] for cat in categories}

    # Check each SMILES
    for smi in eligible_solutes:
        if not secondary_active:
            cat_rows = filtered_df[filtered_df['solute_smiles'] == smi]
        else:
            cat_rows = filtered_df[
                (filtered_df['solute_smiles'] == smi) |
                (filtered_df['secondary_solute_smiles'] == smi)
            ]

        if cat_rows.empty:
            continue

        category = cat_rows['solute_category'].mode().iloc[0]  # assume most frequent category
        if category not in selected_smiles_by_category:
            continue

        # Temporarily test adding this SMILES
        temp_selected = []
        for cat in categories:
            temp_selected.extend(selected_smiles_by_category[cat])
        temp_selected.append(smi)

        # Create mask for selected
        if not secondary_active:
            selected_df = combined_df[combined_df['solute_smiles'].isin(temp_selected)]
            remaining_df = combined_df[~combined_df['solute_smiles'].isin(temp_selected)]
        else:
            mask_selected = (
                combined_df['solute_smiles'].isin(temp_selected) |
                combined_df['secondary_solute_smiles'].isin(temp_selected)
            )
            selected_df = combined_df[mask_selected]
            remaining_df = combined_df[~mask_selected]

        # Check membrane limit condition
        if membrane_limit is not None:
            membrane_counts = remaining_df['membrane'].value_counts()
            low_count_membranes = membrane_counts[membrane_counts < membrane_limit].index
            selected_df = selected_df[~selected_df['membrane'].isin(low_count_membranes)]

        # Check if SMILES has enough data left in selected_df
        if not secondary_active:
            smi_count = selected_df[selected_df['solute_smiles'] == smi].shape[0]
        else:
            smi_count = selected_df[
                (selected_df['solute_smiles'] == smi) |
                (selected_df['secondary_solute_smiles'] == smi)
            ].shape[0]

        if smi_count < low_limit:
            continue

        selected_smiles_by_category[category].append(smi)

    # Flatten final list and add any required SMILES manually

    return selected_smiles_by_category


def select_samples_by_tanimoto_diversity(smiles_list, n_samples = 5, start_idx = 0):
    """
    Selects a diverse set of SMILES strings based on Tanimoto similarity.

    Args:
        smiles_list (list): List of SMILES strings to select from.
        n_samples (int): Number of diverse samples to select.

    Returns:
        list: A list of selected diverse SMILES strings.
    """

    # Step 1: Convert to RDKit mol objects and compute fingerprints
    mols = [Chem.MolFromSmiles(s) for s in smiles_list]
    fps = [AllChem.GetMorganFingerprintAsBitVect(m, radius=2, nBits=1024) for m in mols]

    # Step 2: Calculate pairwise Tanimoto similarity matrix
    n = len(fps)
    similarity_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            sim = DataStructs.TanimotoSimilarity(fps[i], fps[j])
            similarity_matrix[i, j] = sim
            similarity_matrix[j, i] = sim

    # Step 3: Diversity selection
    # Greedy algorithm: pick one molecule, then iteratively pick the one farthest from chosen set

    def select_diverse(fps, k):
        selected = [start_idx]  # Start molecule index
        candidates = set(range(1, len(fps)))
        while len(selected) < k:
            max_min_dist = -1
            next_mol = None
            for c in candidates:
                # compute minimum similarity to selected molecules
                min_sim = min(DataStructs.TanimotoSimilarity(fps[c], fps[s]) for s in selected)
                dist = 1 - min_sim  # distance
                if dist > max_min_dist:
                    max_min_dist = dist
                    next_mol = c
            selected.append(next_mol)
            candidates.remove(next_mol)
        return selected

    selected_indices = select_diverse(fps, n_samples)
    selected_smiles = [smiles_list[i] for i in selected_indices]

    return selected_smiles


# def stratified_kfold_split_complex(
#     og_df, 
#     target_column, 
#     continuous_extra_features = None,
#     model_name = 'rejection_model',
#     augmentation="none",
#     zero_shot_solute_smiles_list=None,
#     k=5, 
#     seed=42, 
#     bins=10,
#     epsilon = 0.01,
#     secondary_active=False,
#     train_val_split = (0.9,0.1),
#     noncat_val_type = 'complete',
#     filter_condition = ("source","cat"),
#     sample_training = -1,
#     shuffle_training = False,
#     stratification = 'rejection'
# ):
#     """
#     Creates K stratified splits ([train-validation]-test) on the main data, ensuring equal catalyst distribution per fold,
#     and with the option to augment the training set with noisy catalyst replicates. Option for zero-shot smiles exclusion.

#     Args:
#         og_df (pd.DataFrame): The original dataframe to split. Must contain a 'source' column.
#         target_column (str): The continuous target column.
#         continuous_extra_features (list): List of column names to apply Gaussian noise for augmentation.
#         augmentation (str): "none", "membrane", or "triple".
#         zero_shot_solute_smiles_list (list): List of SMILES string for zero-shot solute. If None, no zero-shot split is performed.
#         k (int): Number of folds.
#         seed (int): Random seed for reproducibility.
#         bins (int): Number of bins to discretize the target variable for stratification.
#         epsilon (float): Gaussian noise.
#         secondary_active (bool): If True, considers secondary solute SMILES for zero-shot split.
#         train_val_split (tuple): Train-validation split ratios.
#         noncat_val_type (str): 'complete', 'sampled', 'none'
#         filter_condition (tuple): A tuple of (column_name, value) to filter records for stratified augmentation. Default is ("source", "cat").
#         sample_training (int): If not -1: sampling the training set to match the number of (augmented) catalyst records times 'sample_training'. If -1: complete
#         stratification (str): Stratify either by 'rejection' or 'membrane'
#     Returns:
#         list: A list of (train_df, val_df, test_df) tuples, each with 'id_original' to track original indices.
#         pd.DataFrame (optional): Zero-shot test set if `zero_shot_solute_smiles` is provided.
#     """

#     assert noncat_val_type in ['complete', 'sampled', 'none'], "noncat_val_type must be one of: 'complete', 'sampled', 'none'"
#     assert augmentation in ['none', 'triple', 'membrane']
#     assert stratification in ['rejection','membrane']
#     assert continuous_extra_features is not None or augmentation == "none", "continuous_extra_features must be provided for augmentation"

#     ### --- SET UP DF AND ZERO SHOT DF --- ###

#     full_df = og_df.copy()
#     full_df["is_augmented"] = 0
#     full_df["id_original"] = full_df.index
#     fold_column = 'test_fold_' + model_name
#     full_df[fold_column] = float("nan")
    
#     if zero_shot_solute_smiles_list is None:
#         df = full_df.copy()
#     else:
#         if secondary_active:
#             zero_shot_mask = (
#                 full_df["solute_smiles"].isin(zero_shot_solute_smiles_list) |
#                 full_df["secondary_solute_smiles"].isin(zero_shot_solute_smiles_list)
#             )
#         else:
#             zero_shot_mask = full_df["solute_smiles"].isin(zero_shot_solute_smiles_list)

#         full_df.loc[zero_shot_mask, fold_column] = -1
#         zero_shot_test_df = full_df[zero_shot_mask].reset_index(drop=True)
#         df = full_df[~zero_shot_mask].reset_index(drop=True)

#     catalyst_df = df[df[filter_condition[0]] == filter_condition[1]].reset_index(drop=True)
#     non_cat_df = df[df[filter_condition[0]] != filter_condition[1]].reset_index(drop=True)

#     ### --- STRATIFICATION --- ###

#     non_cat_df["binned_target"] = pd.qcut(non_cat_df[target_column], q=bins, labels=False, duplicates="drop")
#     print(f"Number of non_cat rejection bins used: {non_cat_df['binned_target'].nunique()}")

#     catalyst_df["binned_target"] = pd.qcut(catalyst_df[target_column], q=bins, labels=False, duplicates="drop")
#     print(f"Number of catalyst rejection bins used: {catalyst_df['binned_target'].nunique()}")

#     if stratification == "rejection":
#         non_cat_df["stratify_label"] = non_cat_df["binned_target"]
#         catalyst_df["stratify_label"] = catalyst_df["binned_target"]
#     elif stratification == "membrane":
#         non_cat_df["stratify_label"] = non_cat_df["membrane"]
#         catalyst_df["stratify_label"] = catalyst_df["membrane"]

#     skf_non_cat = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
#     skf_cat = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)

#     non_cat_folds = list(skf_non_cat.split(non_cat_df, non_cat_df["stratify_label"]))
#     catalyst_folds = list(skf_cat.split(catalyst_df, catalyst_df["stratify_label"]))

#     rng = np.random.default_rng(seed)
#     folds = []

#     ### --- ASSIGN TEST FOLDS --- ###

#     for fold_idx, (trainval_idx, test_idx) in enumerate(non_cat_folds):
#         non_cat_df.loc[non_cat_df.index[test_idx], fold_column] = fold_idx

#     for fold_idx, (trainval_idx, test_idx) in enumerate(catalyst_folds):
#         catalyst_df.loc[catalyst_df.index[test_idx], fold_column] = fold_idx

#     ### --- CREATE FOLDS --- ###

#     for fold_idx in range(k):
#         # Test sets
#         test_non_cat = non_cat_df.iloc[non_cat_folds[fold_idx][1]].reset_index(drop=True)
#         test_cat = catalyst_df.iloc[catalyst_folds[fold_idx][1]].reset_index(drop=True)
#         test_df = pd.concat([test_non_cat, test_cat], ignore_index=True)

#         # Train+Val sets
#         train_val_non_cat = non_cat_df.iloc[non_cat_folds[fold_idx][0]].reset_index(drop=True)
#         train_val_cat = catalyst_df.iloc[catalyst_folds[fold_idx][0]].reset_index(drop=True)

#         # Train-val split for cat
#         train_idx_cat, val_idx_cat = train_test_split(
#             train_val_cat.index,
#             stratify=train_val_cat["stratify_label"],
#             test_size=train_val_split[1],
#             random_state=seed
#         )
#         train_cat = train_val_cat.loc[train_idx_cat].reset_index(drop=True)
#         val_cat = train_val_cat.loc[val_idx_cat].reset_index(drop=True)

#         # Non-cat split
#         if noncat_val_type == 'complete':
#             train_idx_nc, val_idx_nc = train_test_split(
#                 train_val_non_cat.index,
#                 stratify=train_val_non_cat["stratify_label"],
#                 test_size=train_val_split[1],
#                 random_state=seed
#             )
#             train_non_cat = train_val_non_cat.loc[train_idx_nc].reset_index(drop=True)
#             val_non_cat = train_val_non_cat.loc[val_idx_nc].reset_index(drop=True)
#         elif noncat_val_type == 'sampled':
#             val_non_cat = train_val_non_cat.sample(n=len(val_cat), random_state=seed)
#             train_non_cat = train_val_non_cat.drop(val_non_cat.index).reset_index(drop=True)
#             val_non_cat = val_non_cat.reset_index(drop=True)
#         else:  # 'none'
#             train_non_cat = train_val_non_cat.reset_index(drop=True)
#             val_non_cat = None

#         # AUGMENTATION on catalyst training set
#         augmented_catalysts = []
#         if augmentation == "triple":
#             std_devs = train_cat[continuous_extra_features].std() * epsilon
#             for _ in range(2):  # triple = 2 augmentations added to original
#                 noisy = train_cat.copy()
#                 noisy["id_original"] = train_cat["id_original"].values
#                 noisy["is_augmented"] = 1
#                 for col in continuous_extra_features:
#                     noise = rng.normal(loc=0, scale=std_devs[col], size=len(noisy))
#                     noisy[col] += noise
#                 augmented_catalysts.append(noisy)

#         elif augmentation == "membrane":
#             membrane_counts = train_cat['membrane'].value_counts()
#             max_count = membrane_counts.max()
#             std_devs = train_cat[continuous_extra_features].std() * epsilon

#             for membrane, count in membrane_counts.items():
#                 if count < max_count:
#                     # Determine the number of augmentations (integer between 1 and 3)
#                     possible_augmentations = min((max_count // count) - 1, 3)
#                     if possible_augmentations >= 1:
#                         subset = train_cat[train_cat['membrane'] == membrane]
#                         for _ in range(possible_augmentations):
#                             noisy = subset.copy()
#                             noisy["id_original"] = subset["id_original"].values
#                             noisy["is_augmented"] = 1
#                             for col in continuous_extra_features:
#                                 noise = rng.normal(loc=0, scale=std_devs[col], size=len(noisy))
#                                 noisy[col] += noise
#                             augmented_catalysts.append(noisy)
#         train_cat_augmented = pd.concat([train_cat] + augmented_catalysts, ignore_index=True)

#         # Sampling training
#         if sample_training != -1:
#             # Determine how many samples to take per membrane
#             aug_counts = train_cat_augmented['membrane'].value_counts()
#             sample_counts = (sample_training * aug_counts).astype(int)

#             # For each membrane, sample the corresponding number from train_non_cat
#             sampled_dfs = []
#             for membrane, count in sample_counts.items():
#                 subset = train_non_cat[train_non_cat['membrane'] == membrane]
#                 n_sample = min(count, len(subset))
#                 sampled = subset.sample(n=n_sample, random_state=42)
#                 sampled_dfs.append(sampled)

#             # Concatenate all sampled subsets
#             train_non_cat_sampled = pd.concat(sampled_dfs, ignore_index=True)
#         else:
#             train_non_cat_sampled = train_non_cat

#         # Final train and val
#         train_df = pd.concat([train_non_cat_sampled, train_cat_augmented], ignore_index=True)
#         if shuffle_training:
#             train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)

#         if noncat_val_type == 'none':
#             val_df = val_cat.reset_index(drop=True)
#         else:
#             val_df = pd.concat([val_non_cat, val_cat], ignore_index=True)

#         for _df in [train_df, val_df, test_df]:
#             _df.drop(columns=["binned_target", "stratify_label"], inplace=True, errors='ignore')

#         folds.append((train_df, val_df, test_df))

#     if zero_shot_solute_smiles_list is None:
#         return folds
#     else:
#         return folds, zero_shot_test_df
