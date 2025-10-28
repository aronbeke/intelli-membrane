import pandas as pd
import models.auxiliary
import models.processing
import numpy as np
import json
import shutil
import os
import subprocess

class Workflow:
    def __init__(
            self, 
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
            gpu_list = None,
            nlc = False,
            train_type = 'standard',
            n_folds = 5,
            zeroshot = None,
            hypopt_mode = False
        ):
        '''
        model_archi: 'mcmpnn', 'admpnn', 'himol_gcn', 'molclr_gcn'
        model_type: 'pt', 'ft', 'combi', 'combi_ft', 'pubchem', 'appcat', 'zinc'
        model_version: like 'vXXnlc' or 'v17a_smi0'
        model_parameters (dict): contains model parameters
        train_type: 'standard', 'fine_tune', or 'fine_tune_from_mpnn'
        '''

        assert train_type in ['standard', 'fine_tune', 'fine_tune_from_mpnn']
        assert model_archi in ['mcmpnn', 'admpnn', 'himol_gcn', 'molclr_gcn']
        if gpu_list is not None:
            assert len(gpu_list) == n_folds, "Number of GPUs must match number of folds."

        if zeroshot is not None:
            self.mode = 'zeroshot'
            assert split_parameters['test_type'] == 'solute_disjoint'
        elif hypopt_mode:
            self.mode = 'hypopt'
        else:
            self.mode = 'standard'
        
        self.gpu_list = gpu_list
        self.input_folder = input_folder
        self.results_folder = results_folder
        self.dir_path = dir_path
        self.zeroshot = zeroshot
        
        self.split_pm = split_parameters
        self.marchi = model_archi
        self.mtype = model_type
        self.mversion = model_version
        self.pm = model_parameters
        self.wpm = weight_parameters
        self.k = n_folds
        self.dset = dset
        self.nlc = nlc
        self.train_type = train_type

        self.model_label = self.marchi + '_' + self.mtype + '_' + self.mversion
        self.input_file_path = os.path.join(self.input_folder, input_file_name)
        base_name, ext = os.path.splitext(input_file_name)
        results_file_name = f"{base_name}_{self.marchi}_{self.mtype}_{self.mversion}{ext}"
        self.results_file_path = os.path.join(self.results_folder, results_file_name)
        self.replace = True

    def data_process(self):
        models.auxiliary.copy_and_rename_file(self.input_file_path, self.results_file_path, replace=self.replace)
        self.data_df = pd.read_csv(self.results_file_path)

        if self.mode == 'zeroshot' and self.marchi == 'admpnn':
            self.data_df['test_zeroshot'] = (
                (self.data_df['solute_smiles'] == self.zeroshot) | 
                (self.data_df['secondary_solute_smiles'] == self.zeroshot)
            ).astype(int)
        elif self.mode == 'zeroshot':
            self.data_df['test_zeroshot'] = (self.data_df['solute_smiles'] == self.zeroshot).astype(int)

        self.folds = models.processing.stratified_kfold_split(og_df=self.data_df,
                                                                target_column= self.target_column[0] if self.marchi in {'mcmpnn', 'admpnn'} else self.target_column, 
                                                                k=self.k,
                                                                seed=42,
                                                                bins=10,
                                                                focus=self.split_pm['focus'],                                                                                   
                                                                general_validation_type=self.split_pm['general_validation_type'],
                                                                target_validation_type=self.split_pm['target_validation_type'],
                                                                filter_condition=self.split_pm['filter_condition'],
                                                                stratification=self.split_pm['stratification'],
                                                                test_type=self.split_pm['test_type'],
                                                                weight_parameters=self.wpm)

    def train(self):
        self.report = {}
        self.best_ckpts = {}

        # Save config
        config_dict = {
            "model_archi": self.marchi,
            "model_type": self.mtype,
            "model_version": self.mversion,
            "model_label": self.model_label,
            "dset": self.dset,
            "input_folder": self.input_folder,
            "input_file_name": os.path.basename(self.input_file_path),
            "results_folder": self.results_folder,
            "dir_path": self.dir_path,
            "split_parameters": self.split_pm,
            "weight_parameters": self.wpm,
            "model_parameters": self.pm,
            "nlc": self.nlc,
            "train_type": self.train_type,
            "n_folds": self.k
        }

        subresults_folder_name = f"res_{self.marchi}_{self.mtype}_{self.mversion}"
        self.subresults_folder_path = os.path.join(self.results_folder, subresults_folder_name)
        self.folds_folder_path = os.path.join(self.subresults_folder_path, 'folds')
        os.makedirs(self.subresults_folder_path, exist_ok=True)
        os.makedirs(self.folds_folder_path, exist_ok=True)
        config_file_name = f"{self.marchi}_{self.mtype}_{self.mversion}_config.json"
        config_file_path = os.path.join(self.subresults_folder_path, config_file_name)

        config_dict["folds_folder_path"] = self.folds_folder_path
        config_dict["res_folder_path"] = self.subresults_folder_path

        with open(config_file_path, "w") as f:
            json.dump(config_dict, f)

        # Preprocess and create folds
        if self.marchi in ['mcmpnn', 'admpnn']:
            self.smiles_columns, self.target_column, self.extra_continuous_columns, self.extra_categorical_columns = models.auxiliary.load_feature_columns(type=self.marchi, dset=self.dset, n_solutes=self.pm['n_solutes'])
            self.data_process()
        else:
            self.molecule_features_prefix, self.solvent_features_prefix, self.target_column, self.extra_continuous_columns, self.extra_categorical_columns = models.auxiliary.load_feature_columns(type=self.marchi, dset=self.dset)
            self.data_process()

        # Save folds to disk
        for fold_idx in range(self.k):
            for segment_idx, segment in {0: 'train', 1: 'val', 2: 'test'}.items():
                self.folds[fold_idx][segment_idx].to_csv(os.path.join(self.folds_folder_path, f"fold_{fold_idx}_{segment}.csv"), index=False)

        log_dir = os.path.join(self.subresults_folder_path, "logs")
        os.makedirs(log_dir, exist_ok=True)
        if self.gpu_list is None:
            for fold_idx in range(self.k):
                log_file = os.path.join(log_dir, f"fold_{fold_idx}.log")
                cmd = [
                    "python", "-m", "models.fold",
                    "--fold", str(fold_idx),
                    "--config_path", config_file_path,
                    "--gpu_id", "0"
                ]
                print(f"Launching process for fold {fold_idx}: {' '.join(cmd)}")
                with open(log_file, "w") as f:
                    subprocess.run(cmd, stdout=f, stderr=f, cwd=".")
        else:
            # Spawn subprocesses for each fold
            processes = []
            for fold_idx in range(self.k):
                cmd = [
                    "python", "-m", "models.fold",
                    "--fold", str(fold_idx),
                    "--config_path", config_file_path,
                    "--gpu_id", str(self.gpu_list[fold_idx]) if self.gpu_list else "0",
                ]
                log_file = os.path.join(log_dir, f"fold_{fold_idx}.log")
                f = open(log_file, "w")  # open without 'with' block
                print(f"Launching subprocess for fold {fold_idx}: {' '.join(cmd)}")
                p = subprocess.Popen(cmd, stdout=f, stderr=f, cwd=".")
                processes.append((p, f))

            # Wait for all to finish
            print("Waiting for subprocesses to complete...")
            for p, f in processes:
                p.wait()
                f.close()
            print("All subprocesses finished.")

        # Collect results from per-fold JSONs
        val_rmse_values = []
        test_rmse_values = []

        for fold_idx in range(self.k):
            report_path = os.path.join(self.subresults_folder_path, f"{self.model_label}_fold{fold_idx}_report.json")
            with open(report_path) as f:
                self.report[f"fold{fold_idx}"] = json.load(f)
            val_rmse_values.append(self.report[f"fold{fold_idx}"]['val_rmse'])
            test_rmse_values.append(self.report[f"fold{fold_idx}"]['test_rmse'])
            self.best_ckpts[f"fold{fold_idx}"] = self.report[f"fold{fold_idx}"]['best_model_path']

        self.mean_val_rmse = np.mean(val_rmse_values)
        self.mean_test_rmse = np.mean(test_rmse_values)

        new_folds = []
        for fold_idx in range(self.k):
            fold_data = (
                pd.read_csv(os.path.join(self.folds_folder_path, f"fold_{fold_idx}_train.csv")),
                pd.read_csv(os.path.join(self.folds_folder_path, f"fold_{fold_idx}_val.csv")),
                pd.read_csv(os.path.join(self.folds_folder_path, f"fold_{fold_idx}_test.csv"))
            )
            new_folds.append(fold_data)

        ### ---- Reconstruct and save ---- ###
        if self.mode == 'standard':
            self.res_df = models.processing.rebuild_original_df(k_folds=new_folds, original_df=self.data_df)
            self.res_df.to_csv(self.results_file_path)

            # Save combined report
            report_file_name = self.model_label + ".json"
            with open(os.path.join(self.results_folder, report_file_name), "w") as f:
                json.dump(self.report, f)

            best_models_file_name = self.model_label + "_best_models.json"
            with open(os.path.join(self.results_folder, best_models_file_name), "w") as f:
                json.dump(self.best_ckpts, f)

        elif self.mode == 'zeroshot':
            self.res_df = models.processing.rebuild_original_df(k_folds=new_folds, original_df=self.data_df)
            test_df = self.res_df[self.res_df['test_zeroshot'] == 1].copy()

            test_results_file_name = self.model_label + "_zeroshot_test.csv"
            test_results_file_path = os.path.join(self.results_folder, test_results_file_name)
            test_df.to_csv(test_results_file_path)

            report_file_name = self.model_label + ".json"
            with open(os.path.join(self.results_folder, report_file_name), "w") as f:
                json.dump(self.report, f)

            if os.path.exists(self.subresults_folder_path):
                shutil.rmtree(self.subresults_folder_path)

            if os.path.exists(self.results_file_path):
                os.remove(self.results_file_path)

        elif self.mode == 'hypopt':
            if os.path.exists(self.subresults_folder_path):
                shutil.rmtree(self.subresults_folder_path)
            
            if os.path.exists(self.results_file_path):
                os.remove(self.results_file_path)