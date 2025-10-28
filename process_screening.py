import pandas as pd
import model_predict
import os


### OSNCat SOLUTES ####

smi_col = 'stripped_solute_smiles'

appcat_solutes_df = pd.read_csv('data/solutes/appcat/appcat_solutes_sample.csv')
solutes_df = pd.read_csv('data/solutes/solute_data.csv')

# Split solute_df by category
df_metal = solutes_df[solutes_df['solute_category'] == 'metal_catalyst']
df_organo = solutes_df[solutes_df['solute_category'] == 'organocatalyst']
df_ligand = solutes_df[solutes_df['solute_category'] == 'ligand']

mem_sol_combos = [
    ('GMT-oNF-2', 'Acetonitrile', 0.153),
    ('GMT-oNF-2', 'Ethanol', 0.196),
    ('GMT-oNF-2', 'Toluene', 0.341),
    ('GMT-oNF-2', 'Ethyl acetate', 3.730),
    ('PMPerformance', 'Acetonitrile', 2.531),
    ('PMPerformance', 'Ethanol', 0.539),
    ('PMPerformance', 'Toluene', 0.717),
    ('PMPerformance', 'Ethyl acetate', 2.036),
    ('SS336', 'Acetonitrile', 0.456),
    ('SS336', 'Ethanol', 0.321),
    ('SS336', 'Toluene', 0.798),
    ('SS336', 'Ethyl acetate', 1.000),
    ('DM300', 'Acetonitrile', 0.519),
    ('DM300', 'Ethanol', 0.078),
    ('DM300', 'Toluene', 0.127),
    ('DM300', 'Ethyl acetate', 0.073),
    ('SM122', 'Toluene', 0.180)
]

pressure = 30
temperature = 25
process_configuration = 'CF'

for mem, sol, perm in mem_sol_combos:
    for cat_df, cat_name in zip(
        [df_metal, df_organo, df_ligand],
        ['metal_catalyst', 'organocatalyst', 'ligand']
    ):
        records = []
        for idx, row in cat_df.iterrows():
            records.append({
                "membrane": mem,
                "solvent": sol,
                "solute_smiles": row[smi_col],
                "secondary_solute_smiles": row[smi_col],
                "secondary_solute_ratio": 0.5,
                "pressure": pressure,
                "temperature": temperature,
                "permeance": perm,
                "process_configuration": process_configuration,
                "solute_category": row['solute_category']
            })

        result_df = pd.DataFrame(records)
        result_df.to_csv(f"data/test_sets/tox_pmi_maps/xtest_{cat_name}_{mem}_{sol}.csv", index=False)

    # Also write appcat solutes
    records = []
    for idx, row in appcat_solutes_df.iterrows():
        records.append({
            "membrane": mem,
            "solvent": sol,
            "solute_smiles": row['solute_smiles'],
            "secondary_solute_smiles": row['solute_smiles'],
            "secondary_solute_ratio": 0.5,
            "pressure": pressure,
            "temperature": temperature,
            "permeance": perm,
            "process_configuration": process_configuration,
            "solute_category": 'appcat'
        })

    result_df = pd.DataFrame(records)
    result_df.to_csv(f"data/test_sets/tox_pmi_maps/xtest_appcat_{mem}_{sol}.csv", index=False)


### FDA Solutes ###

fda_df = pd.read_csv('data/fda/fda_solute_solvent_combinations.csv')
fda_solutes = list(set(fda_df['input_solute'].to_list()))

mem_sol_combos2 = [
    ('GMT-oNF-2', 'Toluene', 0.341),
    ('PMPerformance', 'Toluene', 0.717),
    ('SS336', 'Toluene', 0.798),
    ('DM300', 'Toluene', 0.127),
    ('SM122', 'Toluene', 0.180)
]

pressure = 30
temperature = 25
process_configuration = 'CF'

for mem, sol, perm in mem_sol_combos2:
    records = []
    for smi in fda_solutes:
        records.append({
            "membrane": mem,
            "solvent": sol,
            "solute_smiles": smi,
            "secondary_solute_smiles": smi,
            "secondary_solute_ratio": 0.5,
            "pressure": pressure,
            "temperature": temperature,
            "permeance": perm,
            "process_configuration": process_configuration,
            "solute_category": 'fda'
        })

    result_df = pd.DataFrame(records)
    result_df.to_csv(f"data/test_sets/tox_pmi_maps/xtest_fda_{mem}_{sol}.csv", index=False)

### HUB Solutes ###

cat_df = pd.read_csv('data/test_sets/hubs/only_catalysts.csv')
hub_df = pd.read_csv('data/test_sets/hubs/catalyst_hubs.csv')
mem_sol_combos2 = [
    ('GMT-oNF-2', 'Toluene', 0.341),
    ('PMPerformance', 'Toluene', 0.717),
    ('SS336', 'Toluene', 0.798),
    ('DM300', 'Toluene', 0.127),
    ('SM122', 'Toluene', 0.180)
]

pressure = 30
temperature = 25
process_configuration = 'CF'

for cat_type in ['metal_catalyst', 'organocatalyst', 'ligand']:
    records = []
    for mem, sol, perm in mem_sol_combos2:
        for idx, row in cat_df[cat_df['solute_category'] == cat_type].iterrows():
            records.append({
                "membrane": mem,
                "solvent": sol,
                "solute_smiles": row['catalyst_smiles'],
                "secondary_solute_smiles": row['catalyst_smiles'],
                "secondary_solute_ratio": 0.5,
                "pressure": pressure,
                "temperature": temperature,
                "permeance": perm,
                "process_configuration": process_configuration,
                "solute_category": cat_type
            })

    result_df = pd.DataFrame(records)
    result_df.to_csv(f"data/test_sets/tox_pmi_maps/xtest_onlycat_{cat_type}.csv", index=False)

    records = []
    for mem, sol, perm in mem_sol_combos2:
        for idx, row in hub_df[hub_df['solute_category'] == cat_type].iterrows():
            records.append({
                "membrane": mem,
                "solvent": sol,
                "solute_smiles": row['catalyst_hub_smiles'],
                "secondary_solute_smiles": row['catalyst_hub_smiles'],
                "secondary_solute_ratio": 0.5,
                "pressure": pressure,
                "temperature": temperature,
                "permeance": perm,
                "process_configuration": process_configuration,
                "solute_category": cat_type,
                "catalyst_smiles": row['catalyst_smiles'],
                "hub": row['hub']
            })

    result_df = pd.DataFrame(records)
    result_df.to_csv(f"data/test_sets/tox_pmi_maps/xtest_hubs_{cat_type}.csv", index=False)



### PREDICTION FOR SCREENING ###

model_architecture = 'mcmpnn'
nlc = False
k = 5
ckpt_path = 'checkpoints_mcmpnn/'
needs_processing = True


dset = 'DSN'
suffix = '_v42a'
model_type = 'combi'
scale_extras = True
label_model = 'mcmpnn_' + model_type + suffix
models_folder_path = ckpt_path + model_type 
tasks = []
for combo in mem_sol_combos:
    input_folder = 'data/test_sets/tox_pmi_maps'
    output_folder = 'results/general'
    tasks.append(({'folder':input_folder, 'file': 'xtest_metal_catalyst_'+combo[0]+'_'+combo[1]+'.csv'}, {'folder': output_folder, 'file':  'xtest_metal_catalyst_'+combo[0]+'_'+combo[1]+'_predicted.csv'}))
for (input_params, output_params) in tasks:
    model_predict.rejection_prediction(k, model_architecture, input_params, output_params, models_folder_path, label_model, nlc, dset, scale_extras=scale_extras)

suffix = '_v42d'
model_type = 'combi'
scale_extras = True
label_model = 'mcmpnn_' + model_type + suffix
models_folder_path = ckpt_path + model_type
tasks = []
for combo in mem_sol_combos:
    input_folder = 'data/test_sets/tox_pmi_maps'
    output_folder = 'results/general'
    tasks.append(({'folder':input_folder, 'file': 'xtest_organocatalyst_'+combo[0]+'_'+combo[1]+'.csv'}, {'folder': output_folder, 'file':  'xtest_organocatalyst_'+combo[0]+'_'+combo[1]+'_predicted.csv'}))
    tasks.append(({'folder':input_folder, 'file': 'xtest_appcat_'+combo[0]+'_'+combo[1]+'.csv'}, {'folder': output_folder, 'file':  'xtest_appcat_'+combo[0]+'_'+combo[1]+'_predicted.csv'}))
for (input_params, output_params) in tasks:
    model_predict.rejection_prediction(k, model_architecture, input_params, output_params, models_folder_path, label_model, nlc, dset, scale_extras=scale_extras)

suffix = '_v42c'
model_type = 'combi'
scale_extras = True
label_model = 'mcmpnn_' + model_type + suffix
models_folder_path = ckpt_path + model_type
tasks = []
for combo in mem_sol_combos:
    input_folder = 'data/test_sets/tox_pmi_maps'
    output_folder = 'results/general'
    tasks.append(({'folder':input_folder, 'file': 'xtest_ligand_'+combo[0]+'_'+combo[1]+'.csv'}, {'folder': output_folder, 'file':  'xtest_ligand_'+combo[0]+'_'+combo[1]+'_predicted.csv'}))
for (input_params, output_params) in tasks:
    model_predict.rejection_prediction(k, model_architecture, input_params, output_params, models_folder_path, label_model, nlc, dset, scale_extras=scale_extras)

dset = 'DSO'
suffix = '_v40'
model_type = 'combi'
label_model = 'mcmpnn_' + model_type + suffix
models_folder_path = ckpt_path + model_type
scale_extras = True
tasks = []
for combo in mem_sol_combos2:
    input_folder = 'data/test_sets/tox_pmi_maps'
    output_folder = 'results/general'
    tasks.append(({'folder':input_folder, 'file': 'xtest_fda_'+combo[0]+'_'+combo[1]+'.csv'}, {'folder': output_folder, 'file':  'xtest_fda_'+combo[0]+'_'+combo[1]+'_predicted.csv'}))
for (input_params, output_params) in tasks:
    model_predict.rejection_prediction(k, model_architecture, input_params, output_params, models_folder_path, label_model, nlc, dset, scale_extras=scale_extras)

### PREDICTION FOR HUBS ###


model_architecture = 'mcmpnn'
nlc = False
k = 5
ckpt_path = 'checkpoints_mcmpnn/'
needs_processing = True


dset = 'DSN'
suffix = '_v42a'
model_type = 'combi'
scale_extras = True
label_model = 'mcmpnn_' + model_type + suffix
models_folder_path = ckpt_path + model_type 
tasks = []
input_folder = 'data/test_sets/tox_pmi_maps'
output_folder = 'results/general'
tasks.append(({'folder':input_folder, 'file': 'xtest_onlycat_metal_catalyst.csv'}, {'folder': output_folder, 'file':  'xtest_onlycat_metal_catalyst_predicted.csv'}))
tasks.append(({'folder':input_folder, 'file': 'xtest_hubs_metal_catalyst.csv'}, {'folder': output_folder, 'file':  'xtest_hubs_metal_catalyst_predicted.csv'}))
for (input_params, output_params) in tasks:
    model_predict.rejection_prediction(k, model_architecture, input_params, output_params, models_folder_path, label_model, nlc, dset, scale_extras=scale_extras)

suffix = '_v42d'
model_type = 'combi'
scale_extras = True
label_model = 'mcmpnn_' + model_type + suffix
models_folder_path = ckpt_path + model_type
tasks = []
input_folder = 'data/test_sets/tox_pmi_maps'
output_folder = 'results/general'
tasks.append(({'folder':input_folder, 'file': 'xtest_onlycat_organocatalyst.csv'}, {'folder': output_folder, 'file':  'xtest_onlycat_organocatalyst_predicted.csv'}))
tasks.append(({'folder':input_folder, 'file': 'xtest_hubs_organocatalyst.csv'}, {'folder': output_folder, 'file':  'xtest_hubs_organocatalyst_predicted.csv'}))
for (input_params, output_params) in tasks:
    model_predict.rejection_prediction(k, model_architecture, input_params, output_params, models_folder_path, label_model, nlc, dset, scale_extras=scale_extras)

suffix = '_v42c'
model_type = 'combi'
scale_extras = True
label_model = 'mcmpnn_' + model_type + suffix
models_folder_path = ckpt_path + model_type
tasks = []
input_folder = 'data/test_sets/tox_pmi_maps'
output_folder = 'results/general'
tasks.append(({'folder':input_folder, 'file': 'xtest_onlycat_ligand.csv'}, {'folder': output_folder, 'file':  'xtest_onlycat_ligand_predicted.csv'}))
tasks.append(({'folder':input_folder, 'file': 'xtest_hubs_ligand.csv'}, {'folder': output_folder, 'file':  'xtest_hubs_ligand_predicted.csv'}))
for (input_params, output_params) in tasks:
    model_predict.rejection_prediction(k, model_architecture, input_params, output_params, models_folder_path, label_model, nlc, dset, scale_extras=scale_extras)


out_folder = 'results/general/'
out_files = ['xtest_onlycat_metal_catalyst_predicted.csv','xtest_onlycat_organocatalyst_predicted.csv','xtest_onlycat_ligand_predicted.csv']
dfs = [pd.read_csv(os.path.join(out_folder, f)) for f in out_files]
df_concat = pd.concat(dfs, ignore_index=True)
out_path = os.path.join(out_folder, 'xtest_onlycat_all_predicted.csv')
df_concat.to_csv(out_path, index=False)

out_folder = 'results/general/'
out_files = ['xtest_hubs_metal_catalyst_predicted.csv','xtest_hubs_organocatalyst_predicted.csv','xtest_hubs_ligand_predicted.csv']
dfs = [pd.read_csv(os.path.join(out_folder, f)) for f in out_files]
df_concat = pd.concat(dfs, ignore_index=True)
out_path = os.path.join(out_folder, 'xtest_hubs_all_predicted.csv')
df_concat.to_csv(out_path, index=False)




### PREDICTION FOR CASE STUDIES ###

dset = 'DSN'
suffix = '_v42a'
model_type = 'combi'
label_model = 'mcmpnn_' + model_type + suffix
models_folder_path = ckpt_path + model_type
scale_extras = True
tasks = []
tasks.append(({'folder':'data/test_sets/case_studies', 'file': 'xtest_opti_metal.csv'}, {'folder': 'results/optimization/predictions', 'file':  'xtest_opti_metal_predicted.csv'}))
for (input_params, output_params) in tasks:
    model_predict.rejection_prediction(k, model_architecture, input_params, output_params, models_folder_path, label_model, nlc, dset, scale_extras=scale_extras)

dset = 'DSN'
suffix = '_v42d'
model_type = 'combi'
label_model = 'mcmpnn_' + model_type + suffix
models_folder_path = ckpt_path + model_type
scale_extras = True
tasks = []
tasks.append(({'folder':'data/test_sets/case_studies', 'file': 'xtest_opti_organo.csv'}, {'folder': 'results/optimization/predictions', 'file':  'xtest_opti_organo_predicted.csv'}))
for (input_params, output_params) in tasks:
    model_predict.rejection_prediction(k, model_architecture, input_params, output_params, models_folder_path, label_model, nlc, dset, scale_extras=scale_extras)

dset = 'DSO'
suffix = '_v40'
model_type = 'combi'
label_model = 'mcmpnn_' + model_type + suffix
models_folder_path = ckpt_path + model_type
scale_extras = True
tasks = []
tasks.append(({'folder':'data/test_sets/case_studies', 'file': 'xtest_opti_general.csv'}, {'folder': 'results/optimization/predictions', 'file':  'xtest_opti_general_predicted.csv'}))
for (input_params, output_params) in tasks:
    model_predict.rejection_prediction(k, model_architecture, input_params, output_params, models_folder_path, label_model, nlc, dset, scale_extras=scale_extras)

dset = 'DSO'
suffix = '_v40'
model_type = 'combi'
label_model = 'mcmpnn_' + model_type + suffix
models_folder_path = ckpt_path + model_type
scale_extras = True
tasks = []
tasks.append(({'folder':'data/test_sets/shap_att', 'file': 'xtest_substructures_for_pred.csv'}, {'folder': 'results/xai/shap_att', 'file':  'xtest_substructures_for_pred_predicted.csv'}))
for (input_params, output_params) in tasks:
    model_predict.rejection_prediction(k, model_architecture, input_params, output_params, models_folder_path, label_model, nlc, dset, scale_extras=scale_extras)