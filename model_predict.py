import pandas as pd
import numpy as np
import models.mpnn_multi
import models.auxiliary
import models.processing
import scipy.stats as stats
import os

def rejection_prediction(k, model_architecture, input_params, output_params, models_folder_path, model_label, nlc, dset, replace=True, needs_processing=True, scale_extras=False):
    assert model_architecture in ('mcmpnn', 'admpnn')

    input_path = os.path.join(input_params['folder'], input_params['file'])
    output_path = os.path.join(output_params['folder'], output_params['file'])

    models.auxiliary.copy_and_rename_file(input_path, output_path, replace=replace)

    models_dict = models.auxiliary.get_models_dict(models_folder_path, model_label)
    assert len(models_dict) == k
    to_print = "\n".join([f"models_dict['{key}'] = '{value}'" for key, value in models_dict.items()])
    print(to_print)

    smiles_columns, target_column, extra_continuous_columns, extra_categorical_columns = models.auxiliary.load_feature_columns(type=model_architecture, nlc=nlc, dset=dset)

    if needs_processing:
        models.processing.complete_input_processing(output_path, dset=dset)
    
    for label, model_path in models_dict.items():

        model_name = model_label +'_' + label
        df = pd.read_csv(output_path)

        rejection_dataset = models.mpnn_multi.RejectionDataset(df,  smiles_columns, extra_continuous_columns, extra_categorical_columns, target_column)
        trainer = models.mpnn_multi.RejectionPredictor()
        trainer.assign_dataset(rejection_dataset)

        if model_architecture == 'mcmpnn':
            _rmse, _mae, _r2 = trainer.predict(model_name=model_name, checkpoint_path=model_path, scale_extras= scale_extras)
        else:
            _rmse, _mae, _r2 = trainer.predict(model_name=model_name, model_type='admpnn', checkpoint_path=model_path, return_attn=True, scale_extras= scale_extras)
        trainer.rejection_dataset.export_data(output_path)
    
    # Compute confidence intervals for each row in predicted_df

    res_df = pd.read_csv(output_path)
    if model_architecture == 'mcmpnn':
        startswith_type = 'rejection_mcmpnn'
    else:
        startswith_type = 'rejection_admpnn'

    predicted_columns = [col for col in res_df.columns if col.startswith(startswith_type)]
    assert len(predicted_columns) == len(models_dict.items()), "Mismatch in models and predicted columns."

    ci_lower_list, ci_upper_list, mean_list = [], [], []

    for index, row in res_df.iterrows():
        L = row[predicted_columns].values
        n = len(L)
        mean_L = np.mean(L)
        std_L = np.std(L, ddof=1) if n > 1 else 0  # Handle single-value case
        t_crit = stats.t.ppf(0.975, df=n-1) if n > 1 else 0
        margin_of_error = t_crit * (std_L / np.sqrt(n)) if n > 1 else 0
        ci_lower_list.append(mean_L - margin_of_error)
        ci_upper_list.append(mean_L + margin_of_error)
        mean_list.append(mean_L)

    # Attach results to results_df
    res_df[model_architecture+'_'+target_column[0]+'_mean_predicted'] = mean_list
    res_df[model_architecture+'_'+target_column[0]+'_ci_lower'] = ci_lower_list
    res_df[model_architecture+'_'+target_column[0]+'_ci_upper'] = ci_upper_list

    res_df.to_csv(output_path)

if __name__ == "__main__":
    model_architecture = 'mcmpnn'
    nlc = False
    k = 5
    ckpt_path = 'checkpoints_mcmpnn/'
    needs_processing = True

    ### --- TASKS AND RUNS --- ###

    dset = 'DSN'
    scale_extras=True
    tasks = []
    model_type = 'combi'
    suffix = '_v42a'
    label_model = 'mcmpnn_' + model_type + suffix
    models_folder_path = ckpt_path + model_type 
    tasks.append(({'folder':'example', 'file': 'prediction_input.csv'}, {'folder': 'example', 'file':  'prediction_output.csv'}))
    for (input_params, output_params) in tasks:
        rejection_prediction(k, model_architecture, input_params, output_params, models_folder_path, label_model, nlc, dset, scale_extras=scale_extras)