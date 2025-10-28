import sys
import models.dataset_calculations
import pandas as pd
import numpy as np

cratio_0 = 0.1
cratio_target = 1
c0 = 1 # mol /m3
ctarget = 10 # mol /m3
heat_of_evap = 33200

input_path = 'data/tea_samples/tea_samples_osncat_fda_best_membrane_Toluene.csv'
results_path = 'results/tea/osncat_fda_only_tnf_best_membrane_Toluene_results.csv'

models.dataset_calculations.ternary_separation_set(input_path, results_path, cratio_0, cratio_target, heat_of_evap=heat_of_evap, max_reference_concentration = 10) # mol /m3

input_path = 'data/tea_samples/tea_samples_osncat_GMT-oNF-2_Toluene.csv'
results_path = 'results/tea/osncat_only_bnf_GMT-oNF-2_Toluene_results.csv'

models.dataset_calculations.binary_separation_set(input_path, results_path, c0, ctarget, heat_of_evap=heat_of_evap)

input_path = 'data/tea_samples/tea_samples_osncat_best_membrane_Toluene.csv'
results_path = 'results/tea/osncat_only_bnf_best_membrane_Toluene_results.csv'

models.dataset_calculations.binary_separation_set(input_path, results_path, c0, ctarget, heat_of_evap=heat_of_evap)

input_path = 'data/tea_samples/tea_samples_hubs_GMT-oNF-2_Toluene.csv'
results_path = 'results/tea/hubs_only_bnf_GMT-oNF-2_Toluene_results.csv'

models.dataset_calculations.binary_separation_set(input_path, results_path, c0, ctarget, heat_of_evap=heat_of_evap)

input_path = 'data/tea_samples/tea_samples_cats_GMT-oNF-2_Toluene.csv'
results_path = 'results/tea/cats_only_bnf_GMT-oNF-2_Toluene_results.csv'

models.dataset_calculations.binary_separation_set(input_path, results_path, c0, ctarget, heat_of_evap=heat_of_evap)

input_path = 'data/tea_samples/tea_samples_hubs_best_membrane_Toluene.csv'
results_path = 'results/tea/hubs_only_bnf_best_membrane_Toluene_results.csv'

models.dataset_calculations.binary_separation_set(input_path, results_path, c0, ctarget, heat_of_evap=heat_of_evap)

input_path = 'data/tea_samples/tea_samples_cats_best_membrane_Toluene.csv'
results_path = 'results/tea/cats_only_bnf_best_membrane_Toluene_results.csv'

models.dataset_calculations.binary_separation_set(input_path, results_path, c0, ctarget, heat_of_evap=heat_of_evap)