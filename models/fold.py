import argparse
import json
import pandas as pd
import models.auxiliary
from datetime import datetime
import os

class Fold:
    def __init__(self,fold_idx, config, gpu_id):
        self.config = config
        self.idx = fold_idx
        self.gpu_id = gpu_id

        self.marchi = config['model_archi']
        self.mtype = config['model_type']
        self.mversion = config['model_version']
        self.model_label = config['model_label']
        self.dset = config['dset']
        self.dir_path = config['dir_path']
        self.split_pm = config['split_parameters']
        self.pm = config['model_parameters']
        self.nlc = config['nlc']
        self.train_type = config['train_type']
        self.folds_folder_path = config['folds_folder_path']
        self.res_folder_path = config['res_folder_path']

        self.subreport = {}
        self.model_name = f"{self.model_label}_fold{self.idx}"
        self.train_fold_path = os.path.join(self.folds_folder_path, f"fold_{fold_idx}_train.csv")
        self.val_fold_path = os.path.join(self.folds_folder_path, f"fold_{fold_idx}_val.csv")
        self.test_fold_path = os.path.join(self.folds_folder_path, f"fold_{fold_idx}_test.csv")

        self.train_fold_df = pd.read_csv(self.train_fold_path)
        self.val_fold_df = pd.read_csv(self.val_fold_path)
        self.test_fold_df = pd.read_csv(self.test_fold_path)

    def predict_fold(self, trainer, dataset, model_type=None, prefix ='', return_attn=False, forget_predictions=False, scale_extras = False):
        if len(dataset.data_df) == 0:
            print(f"Dataset is empty for {prefix}fold {self.idx}, skipping prediction.")
            self.subreport[prefix+'test_rmse'] = 0.0
            self.subreport[prefix+'test_mae'] = 0.0
            self.subreport[prefix+'test_r2'] = 0.0
            return
        trainer.assign_dataset(dataset)

        if self.marchi in ['molclr_gcn', 'himol_gcn']:
            rmse, mae, r2 = trainer.predict(model_name=self.model_name, forget_predictions = forget_predictions)
        elif self.marchi in ['mcmpnn', 'admpnn']:
            rmse, mae, r2 = trainer.predict(model_name=self.model_name, model_type=model_type, forget_predictions = forget_predictions, return_attn=return_attn, scale_extras=scale_extras)
        else:
            raise ValueError

        self.subreport[prefix+'test_rmse'] = rmse
        self.subreport[prefix+'test_mae'] = mae
        self.subreport[prefix+'test_r2'] = r2

    def train_fold(self):
        if self.marchi in ['mcmpnn', 'admpnn']:
            self.smiles_columns, self.target_column, self.extra_continuous_columns, self.extra_categorical_columns = models.auxiliary.load_feature_columns(type=self.marchi, dset=self.dset)
            self.train_gnn_fold()
        else:
            self.molecule_features_prefix, self.solvent_features_prefix, self.target_column, self.extra_continuous_columns, self.extra_categorical_columns = models.auxiliary.load_feature_columns(type=self.marchi, dset=self.dset, n_solutes=self.pm['n_solutes'])
            self.train_std_fold()

        subreport_path = os.path.join(self.res_folder_path, f"{self.model_label}_fold{self.idx}_report.json")
        with open(subreport_path, "w") as f:
            json.dump(self.subreport, f)

    def train_gnn_fold(self):

        rejection_training_dataset = models.mpnn_multi.RejectionDataset(self.train_fold_df, self.smiles_columns, self.extra_continuous_columns, self.extra_categorical_columns, self.target_column)
        rejection_val_dataset = models.mpnn_multi.RejectionDataset(self.val_fold_df, self.smiles_columns, self.extra_continuous_columns, self.extra_categorical_columns, self.target_column)
        #trainer = models.mpnn_multi.RejectionPredictor(device_id=self.gpu_id)
        trainer = models.mpnn_multi.RejectionPredictor()

        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting training {self.model_label}, fold {self.idx}, type: {self.train_type}, on GPU {self.gpu_id}", flush=True)

        if self.train_type == 'standard' and self.marchi == 'mcmpnn':
            results = trainer.train(model_name=self.model_name,
                                train_rejection_dataset=rejection_training_dataset,
                                val_rejection_dataset=rejection_val_dataset,
                                model_type=self.marchi,
                                dir_path=self.dir_path,
                                n_smiles_components=2,
                                n_mp_layers=self.pm['n_mp_layers'], 
                                hidden_dim_mp=self.pm['hidden_dim_mp'], 
                                hidden_dim_ffn=self.pm['hidden_dim_ffn'], 
                                n_ffn_layers=self.pm['n_ffn_layers'], 
                                dropout=self.pm['dropout'], 
                                epochs=self.pm['epochs'],
                                activation=self.pm['activation'],
                                message_passing=self.pm['message_passing'],
                                aggregation=self.pm['aggregation'],
                                scale_extras=self.pm['scale_extras']
                                )
            
        elif self.train_type == 'fine_tune' and self.marchi == 'mcmpnn':
            results = trainer.fine_tune(dir_path=self.dir_path,
                                train_rejection_dataset=rejection_training_dataset,
                                val_rejection_dataset=rejection_val_dataset,
                                pretrained_checkpoint_path=self.pm['pretrained_models']['fold'+str(self.idx)],  
                                mp_blocks_to_freeze = self.pm['mp_blocks_to_freeze'], 
                                layers_to_freeze_per_block = self.pm['layers_to_freeze_per_block'], 
                                frzn_ffn_layers = self.pm['frzn_ffn_layers'], 
                                fine_tuned_model_name=self.model_name, 
                                learning_rate_ft=self.pm['learning_rate_ft'], 
                                epochs=self.pm['epochs'],
                                scale_extras=self.pm['scale_extras'])
            
        elif self.train_type == 'fine_tune_from_mpnn' and self.marchi == 'mcmpnn':
            results = trainer.fine_tune_from_mpnn(dir_path=self.dir_path,
                                train_rejection_dataset=rejection_training_dataset,
                                val_rejection_dataset=rejection_val_dataset,
                                pretrained_mpnn_path=self.pm['pretrained_mpnn_path'],
                                n_smiles_components=2,
                                n_mp_layers=self.pm['n_mp_layers'],
                                hidden_dim_mp=self.pm['hidden_dim_mp'],
                                layers_to_freeze_in_mpnn=self.pm['layers_to_freeze_in_mpnn'],
                                hidden_dim_ffn=self.pm['hidden_dim_ffn'],
                                n_ffn_layers=self.pm['n_ffn_layers'],
                                fine_tuned_model_name=self.model_name,
                                dropout=self.pm['dropout'],
                                learning_rate_combi_ft=self.pm['learning_rate_combi_ft'],
                                epochs=self.pm['epochs'],
                                activation=self.pm['activation'],
                                message_passing=self.pm['message_passing'],
                                aggregation=self.pm['aggregation'],
                                scale_extras=self.pm['scale_extras'])

        elif self.train_type == 'standard' and self.marchi == 'admpnn':
            results = trainer.train(model_name=self.model_name,
                                train_rejection_dataset=rejection_training_dataset,
                                val_rejection_dataset=rejection_val_dataset,
                                model_type=self.marchi,
                                dir_path=self.dir_path,
                                n_smiles_components = self.pm['n_solutes'] + 1,
                                shared_mp_layer_config=self.pm['shared_mp_layer_config'],
                                solvent_mp_layer_config=self.pm['solvent_mp_layer_config'],
                                hidden_dim_ffn=self.pm['hidden_dim_ffn'], 
                                n_ffn_layers=self.pm['n_ffn_layers'], 
                                dropout=self.pm['dropout'], 
                                epochs=self.pm['epochs'],
                                d_attn=self.pm['d_attn'],
                                atom_attentive_aggregation=self.pm['atom_attentive_aggregation'],
                                activation=self.pm['activation'],
                                message_passing=self.pm['message_passing'],
                                scale_extras=self.pm['scale_extras']
                                )      
        else:
            raise ValueError

        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Finished training {self.model_label}, fold {self.idx}, type: {self.train_type}, on GPU {self.gpu_id}", flush=True)

        self.subreport['best_model_path'] = trainer.best_model_path
        self.subreport['val_rmse_scaled'] = results[0]['val/rmse']
        self.subreport['val_mae_scaled'] = results[0]['val/mae']
        self.subreport['val_r2'] = results[0]['val/r2']
        self.subreport['val_rmse'] = results[0]['val/rmse_original']
        self.subreport['val_mae'] = results[0]['val/mae_original']

        return_attn = True if self.marchi == 'admpnn' else False

        trainer.assign_dataset(rejection_training_dataset)
        _rmse, _mae, _r2 = trainer.predict(model_name=self.model_name, model_type=self.marchi, return_attn=return_attn, scale_extras = self.pm['scale_extras'])
        trainer.assign_dataset(rejection_val_dataset)
        _rmse, _mae, _r2 = trainer.predict(model_name=self.model_name, model_type=self.marchi, return_attn=return_attn, scale_extras = self.pm['scale_extras'])

        rejection_training_dataset.export_data(self.train_fold_path)
        rejection_val_dataset.export_data(self.val_fold_path)

        # Test

        rejection_test_dataset = models.mpnn_multi.RejectionDataset(self.test_fold_df, self.smiles_columns, self.extra_continuous_columns, self.extra_categorical_columns, self.target_column)
        self.predict_fold(trainer=trainer, dataset=rejection_test_dataset, model_type=self.marchi, prefix ='', return_attn=return_attn, forget_predictions=False, scale_extras = self.pm['scale_extras'])
        rejection_test_dataset.export_data(self.test_fold_path)

        # Just General

        general_test_dataset = models.mpnn_multi.RejectionDataset(self.test_fold_df[~(models.auxiliary.apply_filter(self.test_fold_df,self.split_pm['filter_condition']))], self.smiles_columns, self.extra_continuous_columns, self.extra_categorical_columns, self.target_column)
        self.predict_fold(trainer=trainer, dataset=general_test_dataset, model_type=self.marchi, prefix ='general_', return_attn=False, forget_predictions=True, scale_extras = self.pm['scale_extras'])

        # Just Target
        
        target_rejection_dataset = models.mpnn_multi.RejectionDataset(self.test_fold_df[(models.auxiliary.apply_filter(self.test_fold_df,self.split_pm['filter_condition']))], self.smiles_columns, self.extra_continuous_columns, self.extra_categorical_columns, self.target_column)
        self.predict_fold(trainer=trainer, dataset=target_rejection_dataset, model_type=self.marchi, prefix ='target_', return_attn=False, forget_predictions=True, scale_extras = self.pm['scale_extras'])


    def train_std_fold(self):
        print(f"Training {self.model_label}, fold {self.idx}, type: {self.train_type}")

        train_dataset = models.standard.StandardRejectionDataset(dataframe= self.train_fold_df, 
                                                                    molecule_features_prefix = self.molecule_features_prefix,
                                                                    solvent_features_prefix = self.solvent_features_prefix,
                                                                    extra_continuous_columns= self.extra_continuous_columns,
                                                                    extra_categorical_columns=self.extra_categorical_columns,
                                                                    target_column = self.target_column)
        val_dataset = models.standard.StandardRejectionDataset(dataframe= self.val_fold_df, 
                                                                    molecule_features_prefix = self.molecule_features_prefix,
                                                                    solvent_features_prefix = self.solvent_features_prefix,
                                                                    extra_continuous_columns=self.extra_continuous_columns,
                                                                    extra_categorical_columns=self.extra_categorical_columns,
                                                                    target_column = self.target_column)
        #trainer = models.standard.StandardRejectionPredictor(device_id=self.gpu_id)
        trainer = models.standard.StandardRejectionPredictor()

        results = trainer.train(checkpoint_folder=self.dir_path,
                                    training_dataset= train_dataset,
                                    val_dataset= val_dataset,
                                    model_name=self.model_name, 
                                    number_of_hidden_layers=self.pm['number_of_hidden_layers'], 
                                    hidden_dim=self.pm['hidden_dim'], 
                                    max_epochs=self.pm['max_epochs'], 
                                    dropout=self.pm['dropout'])
        
        self.subreport['best_model_path'] = trainer.best_model_path
        self.subreport['val_rmse_scaled'] = results[0]['val_rmse']
        self.subreport['val_mae_scaled'] = results[0]['val_mae']
        self.subreport['val_r2'] = results[0]['val_r2_original']
        self.subreport['val_rmse'] = results[0]['val_rmse_original']
        self.subreport['val_mae'] = results[0]['val_mae_original']

        trainer.assign_dataset(train_dataset)
        _rmse, _mae, _r2 = trainer.predict(model_name=self.model_name)
        trainer.assign_dataset(val_dataset)
        _rmse, _mae, _r2 = trainer.predict(model_name=self.model_name)     

        train_dataset.export_data(self.train_fold_path)
        val_dataset.export_data(self.val_fold_path)

        # Test

        rejection_test_dataset = models.standard.StandardRejectionDataset(dataframe= self.test_fold_df, 
                                                                    molecule_features_prefix = self.molecule_features_prefix,
                                                                    solvent_features_prefix = self.solvent_features_prefix,
                                                                    extra_continuous_columns=self.extra_continuous_columns,
                                                                    extra_categorical_columns=self.extra_categorical_columns,
                                                                    target_column = self.target_column)
        self.predict_fold(trainer=trainer, dataset=rejection_test_dataset, prefix ='', forget_predictions=False)
        rejection_test_dataset.export_data(self.val_fold_path)

        # Just General

        general_test_dataset = models.standard.StandardRejectionDataset(dataframe= self.test_fold_df[~(models.auxiliary.apply_filter(self.test_fold_df,self.split_pm['filter_condition']))],
                                                                    molecule_features_prefix = self.molecule_features_prefix,
                                                                    solvent_features_prefix = self.solvent_features_prefix,
                                                                    extra_continuous_columns=self.extra_continuous_columns,
                                                                    extra_categorical_columns=self.extra_categorical_columns,
                                                                    target_column = self.target_column)
        self.predict_fold(trainer=trainer, dataset=general_test_dataset, prefix ='general_', forget_predictions=True)

        # Just Target
        
        target_rejection_dataset = models.standard.StandardRejectionDataset(dataframe= self.test_fold_df[(models.auxiliary.apply_filter(self.test_fold_df,self.split_pm['filter_condition']))],
                                                                    molecule_features_prefix = self.molecule_features_prefix,
                                                                    solvent_features_prefix = self.solvent_features_prefix,
                                                                    extra_continuous_columns=self.extra_continuous_columns,
                                                                    extra_categorical_columns=self.extra_categorical_columns,
                                                                    target_column = self.target_column)
        self.predict_fold(trainer=trainer, dataset=target_rejection_dataset, prefix ='target_', forget_predictions=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--config_path", type=str, required=True)  # path to a saved JSON config
    parser.add_argument("--gpu_id", type=str, required=True)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    print(f"Running fold {args.fold} on GPU {args.gpu_id}")

    import models.mpnn_multi
    import models.standard

    # Load config
    with open(args.config_path) as f:
        cfg = json.load(f)

    gpu_id = int(args.gpu_id)

    fold_obj = Fold(fold_idx=args.fold, config=cfg, gpu_id=gpu_id)
    fold_obj.train_fold()

if __name__ == "__main__":
    main()