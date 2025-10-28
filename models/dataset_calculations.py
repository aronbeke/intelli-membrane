import pandas as pd
import numpy as np
from math import floor
import models.energetics

# ALL ENERGIES ARE MOLAR

def reduction_calculation(e_ref,e_novel):
    if e_ref == 0:
        return 0
    elif e_ref == float('inf') and e_novel != e_ref:
        return 1
    elif e_ref == float('inf') and e_novel == float('inf'):
        return 0
    else:
        return max(0,(e_ref-e_novel) / e_ref)


def is_equal(A,B):
    if A == float('inf') or B == float('inf'):
        return False
    elif B == 0 and A == 0:
        return True
    elif B == 0 and A != 0:
        return False
    
    if abs(A-B)/B < 0.001:
        return True
    else:
        return False


def ternary_separation_set(csv_path, results_path, cratio_0, cratio_target, heat_of_evap, max_reference_concentration = 10):
    input_df = pd.read_csv(csv_path)
    heat_integration_efficiency = 0.5

    property_df = pd.read_csv('data/solvents/solvent_data.csv')

    records = []

    for idx, row in input_df.iterrows():
        solvent = row['solvent']
        solute_smiles_1 = row['solute_smiles_1']
        solute_smiles_2 = row['solute_smiles_2']
        membrane = row['membrane']
        solvent_smiles = row['solvent_smiles']
        solvent_viscosity = row['solvent_viscosity']
        solvent_density = row['solvent_density']
        permeance = row['permeance']
        solute_diffusivity_1 = row['solute_diffusivity_1']
        solute_diffusivity_2 = row['solute_diffusivity_2']
        solute_density_1 = row['solute_density_1']
        solute_density_2 = row['solute_density_2']
        solute_mw_1 = row['solute_mw_1']
        solute_mw_2 = row['solute_mw_2']
        solute_solubility_1 = row['solute_solubility_1']
        solute_solubility_2 = row['solute_solubility_2']
        solute_rejection_1 = row['solute_rejection_1']
        solute_rejection_2 = row['solute_rejection_2']
        solute_molar_volume_1 = row['solute_molar_volume_1']
        solute_molar_volume_2 = row['solute_molar_volume_2']
        solute_permeance_1 = row['solute_permeance_1']
        solute_permeance_2 = row['solute_permeance_2']
        solute_cost_1 = row['solute_cost_1']

        record = {
            'solute_smiles_1': solute_smiles_1,
            'solute_smiles_2': solute_smiles_2,
            'solvent': solvent,
            'membrane': membrane,
            'solvent_smiles': solvent_smiles,
            'solvent_viscosity': solvent_viscosity,
            'solvent_density': solvent_density,
            'permeance': permeance,
            'solute_diffusivity_1': solute_diffusivity_1,
            'solute_diffusivity_2': solute_diffusivity_2,
            'solute_density_1': solute_density_1,
            'solute_density_2': solute_density_2,
            'solute_mw_1': solute_mw_1,
            'solute_mw_2': solute_mw_2,
            'solute_solubility_1': solute_solubility_1,
            'solute_solubility_2': solute_solubility_2,
            'solute_rejection_1': solute_rejection_1,
            'solute_rejection_2': solute_rejection_2,
            'solute_molar_volume_1': solute_molar_volume_1,
            'solute_molar_volume_2': solute_molar_volume_2,
            'solute_permeance_1': solute_permeance_1,
            'solute_permeance_2': solute_permeance_2,
            'solute_cost_1': solute_cost_1,
        }

        pm = models.energetics.initiate_separation_parameters(solvent,permeance,heat_integration_efficiency, sep_type='impurity_removal')

        pm['L'].append(solute_permeance_1)
        pm['L'].append(solute_permeance_2)
        pm['solubility'].append(solute_solubility_1)
        pm['solubility'].append(solute_solubility_2)
        pm['M'].append(solute_mw_1)
        pm['M'].append(solute_mw_2)
        pm['D'].append(solute_diffusivity_1)
        pm['D'].append(solute_diffusivity_2)
        pm['nu'].append(solute_molar_volume_1)
        pm['nu'].append(solute_molar_volume_2)
        pm['R'].append(solute_rejection_1)
        pm['R'].append(solute_rejection_2)
        pm['viscosity'] = solvent_viscosity
        pm['density'] = solvent_density
        pm['solvent_molar_mass'] = property_df[property_df['solvent_smiles'] == solvent_smiles]['solvent_mw'].iloc[0] / 1000
        pm['solvent_heat_of_evaporation'] = heat_of_evap
        pm['catalyst_cost'] = solute_cost_1

        reference_solubility = min(0.25*pm['solubility'][0],0.25*pm['solubility'][1], max_reference_concentration)
            
        if cratio_0 >= 1:
            c0 = reference_solubility
            c1 = c0 / cratio_0
            c0_actual = [c0, c1]
        else:
            c1 = reference_solubility
            c0 = cratio_0 * c1
            c0_actual = [c0, c1]               

        tnf_molar_energy, tnf_recovery, tnf_n_stages, tnf_area, tnf_c_final, productivity, costs, membrane_cost, waste_cost = models.energetics.impurity_removal_energy_only_tnf(c0_actual, cratio_target, pm)

        record['tnf_ternary_energy'] = tnf_molar_energy
        record['tnf_ternary_recovery'] = tnf_recovery
        record['tnf_ternary_no_stages'] = tnf_n_stages
        record['tnf_ternary_total_area'] = tnf_area
        record['tnf_ternary_final_concentration'] = tnf_c_final
        record['tnf_ternary_productivity'] = productivity
        for key in costs.keys():
            record['tnf_ternary_cost_'+key] = costs[key]
        record['tnf_ternary_cost_membrane'] = membrane_cost
        record['tnf_ternary_cost_waste'] = waste_cost

        records.append(record)
        print('Sample',idx,'rejections:',[round(pm['R'][0],2),round(pm['R'][1],2)])

    results_df = pd.DataFrame(records)
    results_df.to_csv(results_path)


def binary_separation_set(csv_path, results_path, c0, ctarget, heat_of_evap):
    input_df = pd.read_csv(csv_path)
    heat_integration_efficiency = 0.5

    property_df = pd.read_csv('data/solvents/solvent_data.csv')

    records = []

    for idx, row in input_df.iterrows():
        solvent = row['solvent']
        solute_smiles_1 = row['solute_smiles_1']
        membrane = row['membrane']
        solvent_smiles = row['solvent_smiles']
        solvent_viscosity = row['solvent_viscosity']
        solvent_density = row['solvent_density']
        permeance = row['permeance']
        solute_diffusivity_1 = row['solute_diffusivity_1']
        solute_density_1 = row['solute_density_1']
        solute_mw_1 = row['solute_mw_1']
        solute_solubility_1 = row['solute_solubility_1']
        solute_rejection_1 = row['solute_rejection_1']
        solute_molar_volume_1 = row['solute_molar_volume_1']
        solute_permeance_1 = row['solute_permeance_1']
        solute_cost_1 = row['solute_cost_1']

        record = {
            'solute_smiles_1': solute_smiles_1,
            'solvent': solvent,
            'membrane': membrane,
            'solvent_smiles': solvent_smiles,
            'solvent_viscosity': solvent_viscosity,
            'solvent_density': solvent_density,
            'permeance': permeance,
            'solute_diffusivity_1': solute_diffusivity_1,
            'solute_density_1': solute_density_1,
            'solute_mw_1': solute_mw_1,
            'solute_solubility_1': solute_solubility_1,
            'solute_rejection_1': solute_rejection_1,
            'solute_molar_volume_1': solute_molar_volume_1,
            'solute_permeance_1': solute_permeance_1,
            'solute_cost_1': solute_cost_1,
        }

        pm = models.energetics.initiate_separation_parameters(solvent,permeance,heat_integration_efficiency,sep_type='solute_concentration')

        pm['L'].append(solute_permeance_1)
        pm['solubility'].append(solute_solubility_1)
        pm['M'].append(solute_mw_1)
        pm['D'].append(solute_diffusivity_1)
        pm['nu'].append(solute_molar_volume_1)
        pm['R'].append(solute_rejection_1)
        pm['viscosity'] = solvent_viscosity
        pm['density'] = solvent_density
        pm['solvent_molar_mass'] = property_df[property_df['solvent_smiles'] == solvent_smiles]['solvent_mw'].iloc[0] / 1000
        pm['solvent_heat_of_evaporation'] = heat_of_evap
        pm['catalyst_cost'] = solute_cost_1       

        molar_energy_demand, recovery, n_stages, total_area, productivity, costs, membrane_cost, waste_cost = models.energetics.concentration_only_bnf(c0,ctarget,pm)

        record['bnf_binary_energy'] = molar_energy_demand
        record['bnf_binary_recovery'] = recovery
        record['bnf_binary_no_stages'] = n_stages
        record['bnf_binary_total_area'] = total_area
        record['bnf_binary_productivity'] = productivity
        for key in costs.keys():
            record['bnf_binary_cost_'+key] = costs[key]
        record['bnf_binary_cost_membrane'] = membrane_cost
        record['bnf_binary_cost_waste'] = waste_cost

        records.append(record)
        print('Sample',idx,'rejection:',[round(pm['R'][0],2)])

    results_df = pd.DataFrame(records)
    results_df.to_csv(results_path)