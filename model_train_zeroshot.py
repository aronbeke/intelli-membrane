import models.flow
import sys
import json
import os

if __name__ == '__main__':
    if len(sys.argv) != 6:
        print("Usage: python model_train.py <architecture> <type> <version> <zeroshot_list_path> <zeroshot_type>")
        sys.exit(1)

    model_archi = sys.argv[1]
    model_type = sys.argv[2]
    model_version = sys.argv[3]
    zsh_path = sys.argv[4]
    zsh_type = sys.argv[5]

    multiprocess = True
    assert model_archi in ['mcmpnn', 'admpnn'], "Invalid model architecture"
    assert model_type in ['combi'], "Invalid model type"
    assert zsh_type in ('metal_catalysts', 'non_metal_catalysts', 'organocatalysts', 'ligands', 'mixed')

    dset = 'DSN'
    dir_path = 'checkpoints_'+model_archi+'/zeroshot'
    results_folder = 'results/zeroshot/'+zsh_type

    if zsh_type == 'metal_catalysts':
        filter = ('solute_category', 'metal_catalyst','==')
    elif zsh_type == 'non_metal_catalysts':
        filter = ('solute_category', 'metal_catalyst','!=')
    elif zsh_type == 'organocatalysts':
        filter = ('solute_category', 'organocatalyst','==')
    elif zsh_type == 'ligands':
        filter = ('solute_category', 'ligand','==')
    elif zsh_type == 'mixed':
        filter = ('source', 'cat','==')
    else:
        raise ValueError(f"Unknown target: {zsh_type}")

    train_type = 'standard'
    
    ### --- ###

    target_bias_dict_val = {
        'GMT-oNF-2': 1,
        'SS336': 1,
        'DM300': 1,
        'DM150': 1,
        'DM500': 1,
        'SM122': 1
    }
    target_bias_dict = {
        'GMT-oNF-2': 1,
        'SS336': 1,
        'DM300': 1,
        'DM150': 1,
        'DM500': 1,
        'SM122': 1

    }

    weight_parameters = {'weight_losses': True,
                        'filter_condition': filter,
                        'class_bias_factor': 0.3,
                        'class_bias_factor_val': 0.3,
                        'base': 0.5,
                        'R_cutoff': 0.2,
                        'max_w': 1.5,
                        'target_bias_dict_val': target_bias_dict_val,
                        'target_bias_dict': target_bias_dict
                        }

    split_parameters = {
            'focus': 'all',
            'general_validation_type': 'solute_disjoint',
            'target_validation_type': 'solute_disjoint',
            'filter_condition': filter,
            'stratification': 'rejection',
            'test_type': 'solute_disjoint'
        }

    if model_version == 'v41a' and model_archi == 'mcmpnn' and model_type == 'combi':
        model_parameters = {
            'dropout': 0.0474,
            'n_mp_layers': [6,2],
            'n_ffn_layers': 3,
            'hidden_dim_mp': [128, 128],
            'hidden_dim_ffn': 1024,
            'epochs': 1500,
            'activation': 'leakyrelu',
            'message_passing': 'atom',
            'aggregation': 'mean',
            'n_solutes': 1,
            'scale_extras': True
        }

    elif model_version == 'v41d' and model_archi == 'mcmpnn' and model_type == 'combi':
        model_parameters = {
            'dropout': 0.08691,
            'n_mp_layers': [6,4],
            'n_ffn_layers': 2,
            'hidden_dim_mp': [512, 512],
            'hidden_dim_ffn': 1024,
            'epochs': 1500,
            'activation': 'leakyrelu',
            'message_passing': 'atom',
            'aggregation': 'mean',
            'n_solutes': 1,
            'scale_extras': True
        }

    elif model_version == 'v41c' and model_archi == 'mcmpnn' and model_type == 'combi':
        model_parameters = {
            'dropout': 0.0494,
            'n_mp_layers': [2,4],
            'n_ffn_layers': 3,
            'hidden_dim_mp': [256, 128],
            'hidden_dim_ffn': 512,
            'epochs': 1500,
            'activation': 'leakyrelu',
            'message_passing': 'atom',
            'aggregation': 'mean',
            'n_solutes': 1,
            'scale_extras': True
        }

    else:
        raise ValueError
    
    ### --- Multiprocessing --- ###
    if multiprocess:
        gpu_ids = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if gpu_ids.strip() == "":
            raise RuntimeError("No GPUs visible: CUDA_VISIBLE_DEVICES is empty.")
        print(f"Available GPUs for this SLURM job: {gpu_ids}")
        gpu_list = list(map(int, gpu_ids.split(",")))
    else:
        gpu_list = None
    ### --- ###

    input_folder = 'data/nf10k'
    input_file_name = 'nf10kcat_processed'+dset+'.csv'

    with open(zsh_path, 'r') as f:
        zero_shot_smiles_list = [line.strip() for line in f if line.strip()]

    for smi_idx, smile in enumerate(zero_shot_smiles_list):
        model_version_smi = model_version + '_smi' + str(smi_idx)
        trainer = models.flow.Workflow(
                    model_archi, 
                    model_type,
                    model_version_smi,
                    dset,
                    input_folder,
                    input_file_name,
                    results_folder,
                    dir_path, 
                    split_parameters,
                    weight_parameters,
                    model_parameters,
                    gpu_list=gpu_list,
                    nlc = False,
                    train_type = train_type,
                    n_folds = 5,
                    zeroshot=smile
                    )
        trainer.train()