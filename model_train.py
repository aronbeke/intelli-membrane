import models.flow
import sys
import json
import os

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print("Usage: python model_train.py <architecture> <type> <version>")
        sys.exit(1)

    model_archi = sys.argv[1]
    model_type = sys.argv[2]
    model_version = sys.argv[3]

    multiprocess = False

    ###
    if model_version.startswith('v42'):
        dset = 'DSN'
    else:
        dset = 'DSO'
    ###

    input_folder = 'data/nf10k'
    if model_archi == 'mcmpnn' or model_archi == 'admpnn':
        input_file_name = 'nf10kcat_processed'+dset+'.csv'
    else:
        input_file_name = 'nf10kcat_'+model_archi+'_'+model_type+'_'+dset+'.csv'

    dir_path = 'checkpoints_'+model_archi+'/'+model_type
    results_folder = 'results/'+model_archi
    pt_models_json = 'results/mcmpnn/mcmpnn_pt_v40_best_models.json'
    
    filter = ('source', 'cat', '==')
    weight_parameters = {'weight_losses': True,
                        'filter_condition':  filter,
                        'class_bias_factor': 0.0,
                        'base': 0.5,
                        'R_cutoff': 0.2,
                        'max_w': 1.5,
                        'target_bias_dict': None
                        }

    split_parameters = {
            'focus': 'all',
            'general_validation_type': 'standard',
            'target_validation_type': 'standard',
            'filter_condition': filter,
            'stratification': 'rejection',
            'test_type': 'standard'
        }

    ## --- PRETRAINING --- ##
    if model_archi == 'mcmpnn' and model_type == 'pt':
        train_type = 'standard'
        split_parameters['focus'] = 'general'

        model_parameters = {
            'dropout': 0.045,
            'n_mp_layers': [6,4],
            'n_ffn_layers': 2,
            'hidden_dim_mp': [512, 512],
            'hidden_dim_ffn': 512,
            'epochs': 1000,
            'activation': 'leakyrelu',
            'message_passing': 'atom',
            'aggregation': 'mean',
            'n_solutes': 1,
            'scale_extras': True
        }


    ## --- FINE TUNING --- ##
    elif model_archi == 'mcmpnn' and model_type == 'ft':
        train_type = 'fine_tune'

        with open(pt_models_json, 'r') as f:
            pretrained_models = json.load(f)  # data is now a dictionary

        split_parameters['focus'] = 'target'
        model_parameters = {
            'mp_blocks_to_freeze': [0,1],
            'layers_to_freeze_per_block': {0: 3, 1: 0},
            'frzn_ffn_layers': 0,
            'learning_rate_ft': 1.20e-4,
            'epochs': 1000,
            'pretrained_models': pretrained_models,
            'n_solutes': 1,
            'scale_extras': True
        }

    ## --- COMBINED --- ##
    elif model_archi == 'mcmpnn' and model_type == 'combi':
        train_type = 'standard'

        if model_version == 'v42a':
            filter = ('solute_category', 'metal_catalyst', '==')
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
            weight_parameters = {'weight_losses': True,
                                'filter_condition':  filter,
                                'class_bias_factor': 0.3,
                                'class_bias_factor_val': 0.3,
                                'target_bias_dict_val': {
                                    'GMT-oNF-2': 1,
                                    'SS336': 1,
                                    'DM300': 1,
                                    'DM150': 1,
                                    'DM500': 1,
                                    'SM122': 1
                                },
                                'base': 0.5,
                                'R_cutoff': 0.2,
                                'max_w': 1.5,
                                'target_bias_dict': {
                                    'GMT-oNF-2': 1,
                                    'SS336': 1,
                                    'DM300': 1,
                                    'DM150': 1,
                                    'DM500': 1,
                                    'SM122': 1
                                }
                                }

            split_parameters = {
                    'focus': 'all',
                    'general_validation_type': 'solute_disjoint',
                    'target_validation_type': 'solute_disjoint',
                    'filter_condition': filter,
                    'stratification': 'rejection',
                    'test_type': 'solute_disjoint'
                }

        elif model_version == 'v42d':
            filter = ('solute_category', 'organocatalyst', '==')
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
            weight_parameters = {'weight_losses': True,
                                'filter_condition':  filter,
                                'class_bias_factor': 0.3,
                                'class_bias_factor_val': 0.3,
                                'target_bias_dict_val': {
                                    'GMT-oNF-2': 1,
                                    'SS336': 1,
                                    'DM300': 1,
                                    'DM150': 1,
                                    'DM500': 1,
                                    'SM122': 1
                                },
                                'base': 0.5,
                                'R_cutoff': 0.2,
                                'max_w': 1.5,
                                'target_bias_dict': {
                                    'GMT-oNF-2': 1,
                                    'SS336': 1,
                                    'DM300': 1,
                                    'DM150': 1,
                                    'DM500': 1,
                                    'SM122': 1
                                }
                                }

            split_parameters = {
                    'focus': 'all',
                    'general_validation_type': 'solute_disjoint',
                    'target_validation_type': 'solute_disjoint',
                    'filter_condition': filter,
                    'stratification': 'rejection',
                    'test_type': 'solute_disjoint'
                }
            
        elif model_version == 'v42c':
            filter = ('solute_category', 'ligand', '==')
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
            weight_parameters = {'weight_losses': True,
                                'filter_condition':  filter,
                                'class_bias_factor': 0.3,
                                'class_bias_factor_val': 0.3,
                                'target_bias_dict_val': {
                                    'GMT-oNF-2': 1,
                                    'SS336': 1,
                                    'DM300': 1,
                                    'DM150': 1,
                                    'DM500': 1,
                                    'SM122': 1
                                },
                                'base': 0.5,
                                'R_cutoff': 0.2,
                                'max_w': 1.5,
                                'target_bias_dict': {
                                    'GMT-oNF-2': 1,
                                    'SS336': 1,
                                    'DM300': 1,
                                    'DM150': 1,
                                    'DM500': 1,
                                    'SM122': 1
                                }
                                }

            split_parameters = {
                    'focus': 'all',
                    'general_validation_type': 'solute_disjoint',
                    'target_validation_type': 'solute_disjoint',
                    'filter_condition': filter,
                    'stratification': 'rejection',
                    'test_type': 'solute_disjoint'
                }
            
        else:
            model_parameters = {
                'dropout': 0.08691,
                'n_mp_layers': [6,4],
                'n_ffn_layers': 2,
                'hidden_dim_mp': [512, 512],
                'hidden_dim_ffn': 1024,
                'epochs': 1000,
                'activation': 'leakyrelu',
                'message_passing': 'atom',
                'aggregation': 'mean',
                'n_solutes': 1,
                'scale_extras': True
            }

    ## --- COMBINED FINE TUNED --- ##
    elif model_archi == 'mcmpnn' and model_type == 'combi_ft':
        
        # Only atom MP and relu so far

        train_type = 'fine_tune_from_mpnn'
        model_parameters = {
            'dropout': 0.08691,
            'n_mp_layers': [6,4],
            'n_ffn_layers': 2,
            'hidden_dim_mp': [512, 512],
            'hidden_dim_ffn': 1024,
            'epochs': 1000,
            'layers_to_freeze_in_mpnn': 1,
            'learning_rate_combi_ft': 4.90e-5,
            'pretrained_mpnn_path': 'checkpoints_mpnn/mpnn_appcat_logp_pretrain_v40-epoch=998-val_loss=0.04.ckpt',
            'activation': 'leakyrelu',
            'message_passing': 'atom',
            'aggregation': 'mean',
            'n_solutes': 1,
            'scale_extras': True
        }


    ## --- ADMPNN COMBINED --- ##
    elif model_archi == 'admpnn' and model_type == 'combi':
        train_type = 'standard'
        multiprocess = True

        if model_version == 'v42a':
            filter = ('solute_category', 'metal_catalyst', '==')

            class_bias_factor_val = 0.3 
            target_bias_dict_val = {
                'GMT-oNF-2': 1,
                'SS336': 1,
                'DM300': 1,
                'DM150': 1,
                'DM500': 1,
                'SM122': 1
            }

            weight_parameters = {'weight_losses': True,
                                'filter_condition':  filter,
                                'class_bias_factor': 0.3,
                                'class_bias_factor_val': class_bias_factor_val,
                                'target_bias_dict_val': target_bias_dict_val,
                                'base': 0.5,
                                'R_cutoff': 0.2,
                                'max_w': 1.5,
                                'target_bias_dict': {
                                    'GMT-oNF-2': 1,
                                    'SS336': 1,
                                    'DM300': 1,
                                    'DM150': 1,
                                    'DM500': 1,
                                    'SM122': 1
                                }
                                }

            split_parameters = {
                    'focus': 'all',
                    'general_validation_type': 'solute_disjoint',
                    'target_validation_type': 'solute_disjoint',
                    'filter_condition': filter,
                    'stratification': 'rejection',
                    'test_type': 'solute_disjoint'
                }

            model_parameters = {
                'dropout': 0.2,
                'shared_mp_layer_config': (3, 128),
                'n_ffn_layers': 3,
                'solvent_mp_layer_config': (2, 128),
                'hidden_dim_ffn': 512,
                'epochs': 1500,
                'd_attn': 4,
                'activation': 'leakyrelu',
                'message_passing': 'atom',
                'n_solutes': 1,
                'atom_attentive_aggregation': True,
                'scale_extras': True
            }
        else:
            model_parameters = {
                'dropout': 0.2,
                'shared_mp_layer_config': (3, 256),
                'n_ffn_layers': 2,
                'solvent_mp_layer_config': (2, 256),
                'hidden_dim_ffn': 512,
                'epochs': 1500,
                'd_attn': 4,
                'activation': 'leakyrelu',
                'message_passing': 'atom',
                'n_solutes': 2,
                'atom_attentive_aggregation': True,
                'scale_extras': True
            }

    ## --- STANDARD MODELS --- ##
    elif model_archi == 'molclr_gcn':
        train_type = 'standard'
        model_parameters = {
            'number_of_hidden_layers': 3,
            'hidden_dim': 256,
            'max_epochs': 1000,
            'dropout': 0.228,
            'n_solutes': 1,
            'scale_extras': True
        }

    elif model_archi == 'himol_gcn':
        train_type = 'standard'
        model_parameters = {
            'number_of_hidden_layers': 4,
            'hidden_dim': 256,
            'max_epochs': 1000,
            'dropout': 0.0245,
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

    trainer = models.flow.Workflow(
                model_archi, 
                model_type,
                model_version,
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
                n_folds = 5
                )

    trainer.train()