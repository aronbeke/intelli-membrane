import pandas as pd
import numpy as np
import math
import models.opti_initialize
import models.opti_model
import models.energetics

def multiobjective_optimization_multistart(opti_type,
                                           target_folder,
                                           no_of_models,
                                           pm,
                                           pm_sim,
                                           constraint_type,
                                           objective_type,
                                           constraint_levels,
                                           config='3s',
                                           transport_model='sdc',
                                           model_x=False,
                                           model_d=False,
                                           model_r=False,
                                           additional_rec_constraint=0):

    prefix = 'opti_'+transport_model+'_'
    if model_x:
        prefix += 'x_'
    if model_d:
        prefix += 'd_'
    if model_r:
        prefix += 'r_'

    file_name = prefix+str(no_of_models)+'models_'+opti_type
    
    dp_max = pm['dp_max']

    length = len(constraint_levels)*no_of_models
    WATER_REC = np.zeros(length)
    BIVALENT_REC = np.zeros(length)
    SEP_FACTOR = np.zeros(length)
    SP_POWER = np.zeros(length)
    FINAL_CONC1 = np.zeros(length)
    FINAL_CONC2 = np.zeros(length)
    FEED_PRESSURE = np.zeros(length)
    PERMEATE_PRESSURES_ARR = [np.zeros(length),np.zeros(length),np.zeros(length)]
    DILUTIONS_ARR = [np.zeros(length),np.zeros(length),np.zeros(length)]
    OMEGA_ARR = [np.zeros(length),np.zeros(length)]
    OPTIMAL = np.zeros(length)


    i = 0
    for cl in constraint_levels:
        predictors = []

        init_pm = {
            'p_feed_min' : 10,
            'p_feed_max' : dp_max,
            'dilution_max' : 5,
        }

        for _ in range(no_of_models): 
            print(i)
            model_dict = {}
            if additional_rec_constraint == 0:
                constraints={constraint_type:cl}
            else:
                constraints={constraint_type:cl, 'recovery':additional_rec_constraint}
            objective = objective_type

            # BUILD MODEL
            
            if model_x and model_r and model_d:
                model_5 = models.opti_model.model_xrd(constraints, objective, pm, membrane_model=transport_model, model_x =model_x, config=config)
            else:
                model_5 = models.opti_model.model(constraints,objective,pm, membrane_model=transport_model, model_x =model_x, config=config)

            # INITIALIZE
            model_5, init_parameters = models.opti_initialize.random_initialization(model_5,pm_sim,init_pm,model_x=model_x,model_d=model_d,model_r = model_r,transport_model=transport_model)

            # OPTIMIZE AND TRANSFER
            model_5, solv_results_5, optimal_5 = models.opti_model.opti(model_5,solver='ipopt')

            model_dict['init'] = {}
            model_dict['init']['p_feed'] = init_parameters[0]
            model_dict['init']['pp_list'] = init_parameters[1]
            model_dict['init']['F_dil_list'] = init_parameters[2]
            model_dict['optimal'] = optimal_5
            model_dict['molar_power'] = model_5.molar_power.value
            model_dict['recovery'] = model_5.recovery.value
            model_dict['final_concentration1'] = model_5.final_concentration[0].value
            model_dict['final_concentration2'] = model_5.final_concentration[1].value
            model_dict['solvent_recovery'] = model_5.solvent_recovery.value
            model_dict['separation_factor'] = model_5.separation_factor.value
            model_dict['dilutions'] = {}
            model_dict['split_ratios'] = {}
            model_dict['feed_pressure'] = model_5.p_feed.value
            model_dict['permeate_pressures'] = {}
            for j in range(pm['nst']):
                if model_d:
                    model_dict['dilutions'][j] = model_5.F_dilution[j].value
                else:
                    model_dict['dilutions'][j] = 0
                if model_r:
                    model_dict['split_ratios'][j] = model_5.omega[j].value
                else:
                    model_dict['split_ratios'][j] = 0
                model_dict['permeate_pressures'][j] = model_5.stages[j].pp.value

            OPTIMAL[i] = optimal_5
            BIVALENT_REC[i] = model_5.recovery.value
            WATER_REC[i] = model_5.solvent_recovery.value
            FINAL_CONC1[i] = model_5.final_concentration[0].value
            FINAL_CONC2[i] = model_5.final_concentration[1].value
            SEP_FACTOR[i] = model_5.separation_factor.value
            SP_POWER[i] = model_5.molar_power.value
            FEED_PRESSURE[i] = model_dict['feed_pressure']
            for j in range(1,pm['nst']):
                OMEGA_ARR[j-1][i] = model_dict['split_ratios'][j]
            for j in range(pm['nst']):
                DILUTIONS_ARR[j][i] = model_dict['dilutions'][j]
                PERMEATE_PRESSURES_ARR[j][i] = model_dict['permeate_pressures'][j]
            
            predictors.append(model_dict)
            i += 1


    res_data_all = {
        'OPTIMAL': OPTIMAL,
        'SOLVENT_REC': WATER_REC,
        'SOLUTE_REC': BIVALENT_REC,
        'SEP_FACTOR': SEP_FACTOR,
        'MOLAR_POWER': SP_POWER,
        'FINAL_CONC1': FINAL_CONC1,
        'FINAL_CONC2': FINAL_CONC2,
        'FEED_PRESSURE': FEED_PRESSURE
    }

    for i_stage in range(pm['nst']):
        res_data_all['DILUTION_'+str(i_stage)] = DILUTIONS_ARR[i_stage]
        res_data_all['PERMEATE_PRESSURE_'+str(i_stage)] = PERMEATE_PRESSURES_ARR[i_stage]
    
    for i_stage in range(1,pm['nst']):
        res_data_all['OMEGA_'+str(i_stage)] = OMEGA_ARR[i_stage-1]

    results_all_df = pd.DataFrame(res_data_all)

    results_all_df.to_csv(target_folder+file_name+".csv")


def pareto_selector(file_name_without_csv,objective1,objective2,competing='true'):
    '''
    competing: 
    true -> classical pareto trade-off
    1min -> minimizing objective 1, maximizing objective 2
    2min -> minimizing objective 2, maximizing objective 1
    '''

    # Step 1: Read the CSV file
    input_file_path = file_name_without_csv+'.csv'
    output_file_path = file_name_without_csv+'_pareto.csv'

    df = pd.read_csv(input_file_path)

    # Step 2: Identify Pareto-optimal records based on objectives A and B
    pareto_optimal = []
    pareto_optimal_points = []

    for index, row in df.iterrows():
        current_point = (round(row[objective1],4), round(row[objective2],4))
        is_pareto_optimal = True
        is_feasible = True
        # if np.abs(row['SIM_SEP_FACTOR'] - row['SEP_FACTOR'])/row['SIM_SEP_FACTOR'] > 0.1:
        #     is_feasible = False
        for idx, rw in df.iterrows():
            current_point2 = (round(rw[objective1],4), round(rw[objective2],4))
            if competing == 'true':
                if (current_point2[0] >= current_point[0] and current_point2[1] >= current_point[1] and (current_point2[0] != current_point[0] or current_point2[1] != current_point[1])) or current_point in pareto_optimal_points:
                    is_pareto_optimal = False
                    break
            elif competing == '1min':
                if (current_point2[0] <= current_point[0] and current_point2[1] >= current_point[1] and (current_point2[0] != current_point[0] or current_point2[1] != current_point[1])) or current_point in pareto_optimal_points:
                    is_pareto_optimal = False
                    break
            elif competing == '2min':
                if (current_point2[0] >= current_point[0] and current_point2[1] <= current_point[1] and (current_point2[0] != current_point[0] or current_point2[1] != current_point[1])) or current_point in pareto_optimal_points:
                    is_pareto_optimal = False
                    break
        if is_pareto_optimal and row['OPTIMAL'] == 1 and is_feasible:
            pareto_optimal.append(row)
            pareto_optimal_points.append(current_point)

    pareto_df = pd.DataFrame(pareto_optimal)

    # Step 3: Save the Pareto-optimal records as another CSV file
    pareto_df.to_csv(output_file_path, index=False)


##########


def evaporation_energy_and_gwp_calculation(input_file, pm, target_range = None, name='efuel'):
    # dH: J/mol
    # rho: kg/m3
    # M_solvent: kg/mol

    input_file_path = input_file + '.csv'
    df = pd.read_csv(input_file_path)

    electricity_kg_co2_eq = {
        'EUR': 0.430,
        'USA': 0.438,
        'IND': 1.222,
        'CHN': 0.899
    }

    heat_kg_co2_eq = {
        'EUR': 0.457,
        'USA': 0.457,
        'IND': 0.457,
        'CHN': 0.457
    }

    electricity_usd = {
        'EUR': 0.449,
        'USA': 0.146,
        'IND': 0.117,
        'CHN': 0.088
    }

    heat_usd = {
        'EUR': 0.194,
        'USA': 0.053,
        'IND': 0.060,
        'CHN': 0.045
    }

    co2eq_membrane = 5.357  # kg CO2eq/m2
    cost_membrane = 100 # USD/m2
    membrane_lifetime = 8766  # h

    # Ensure necessary columns exist before assignment
    df['EVAP_ENERGY'] = np.nan
    df['NF_ENERGY'] = np.nan
    df['EVAP_ENERGY_THERMAL'] = np.nan
    df['EVAP_ENERGY_ELECTRIC'] = np.nan
    df['ENERGY_REDUCTION'] = np.nan
    for key in electricity_kg_co2_eq.keys():
        df[key + '_GWP_REDUCTION'] = np.nan
        df[key + '_COST_REDUCTION'] = np.nan
        df[key + '_BREAKEVEN_COST'] = np.nan

    for i in range(len(df)):  # Fixed the iteration range
        df.at[i, 'MOLAR_POWER_KW'] = df.at[i, 'MOLAR_POWER'] * 100000 / (1000 * 3600) # to KW
        # Inputs:
        ctarget = df.at[i, 'FINAL_CONC1']  # mol/m3
        c0 = pm['c_feed'][0]  # mol/m3
        nf_molar_energy_demand = 3600 * 1000 * (df.at[i, 'MOLAR_POWER_KW']) / (pm['F_feed'] * c0)  # J/mol

        if pm['vacuum']:
            evap_molar_energy_demand, evap_molar_energy_demand_thermal, evap_molar_energy_demand_electric = models.energetics.evaporation_energy_vacuum(c0, ctarget, pm) # J/mol
        else:
            evap_molar_energy_demand = models.energetics.evaporation_energy(c0, ctarget, pm) # J/mol
            evap_molar_energy_demand_thermal = evap_molar_energy_demand
            evap_molar_energy_demand_electric = 0

        area_mem = pm['nst'] * pm['ne'] * pm['A']  # m2

        prod_rate = (1 - df.at[i, 'SOLVENT_REC']) * pm['F_feed'] * ctarget
        if prod_rate == 0:
            continue  # Avoid division by zero

        membrane_co2eq_per_mol_product = (co2eq_membrane * area_mem) / (membrane_lifetime * prod_rate)
        membrane_cost_per_mol_product = (cost_membrane * area_mem) / (membrane_lifetime * prod_rate)

        df.at[i, 'NF_ENERGY'] = nf_molar_energy_demand
        df.at[i, 'EVAP_ENERGY'] = evap_molar_energy_demand
        df.at[i, 'EVAP_ENERGY_THERMAL'] = evap_molar_energy_demand_thermal
        df.at[i, 'EVAP_ENERGY_ELECTRIC'] = evap_molar_energy_demand_electric
        df.at[i, 'ENERGY_REDUCTION'] = (evap_molar_energy_demand - nf_molar_energy_demand) / evap_molar_energy_demand

        for key in electricity_kg_co2_eq.keys():
            reference_co2 = (heat_kg_co2_eq[key] / 3.6e6) * evap_molar_energy_demand_thermal + (electricity_kg_co2_eq[key] / 3.6e6) * evap_molar_energy_demand_electric
            novel_co2 = (electricity_kg_co2_eq[key] / 3.6e6) * nf_molar_energy_demand + membrane_co2eq_per_mol_product
            co2eq_reduction = (reference_co2 - novel_co2) / reference_co2

            df.at[i, key + '_GWP_REDUCTION'] = co2eq_reduction  # kgCO2eq / mol product
            
        for key in electricity_usd.keys():
            reference_cost = (heat_usd[key] / 3.6e6) * evap_molar_energy_demand_thermal + (electricity_usd[key] / 3.6e6) * evap_molar_energy_demand_electric
            novel_cost = (electricity_usd[key] / 3.6e6) * nf_molar_energy_demand + membrane_cost_per_mol_product
            cost_reduction = (reference_cost - novel_cost) / reference_cost

            df.at[i, key + '_COST_REDUCTION'] = cost_reduction  # USD / mol product

            df.at[i, key + '_BREAKEVEN_COST'] = ((reference_cost - (electricity_usd[key] / 3.6e6) * nf_molar_energy_demand) * membrane_lifetime * prod_rate) / area_mem  # USD / m2

    df.to_csv(input_file_path, index=False)

    # NATIONAL MAPPING

    if target_range is not None:
        tea_by_country_path = "results/optimization/technoeconomic_results_by_country.csv"
        countries_df = pd.read_csv(tea_by_country_path, index_col=0)
        countries_length = len(countries_df['country'])

        co2eq_reduction_per_country = np.zeros(countries_length)
        cost_reduction_per_country = np.zeros(countries_length)

        for j in range(countries_length):
            data_df = pd.read_csv(input_file_path)
            mask = (data_df[target_range[0]] >= target_range[1]) & (data_df[target_range[0]] <= target_range[2])
            match_row = data_df[mask].iloc[0]

            area_mem = pm['nst'] * pm['ne'] * pm['A']  # m2
            ctarget = match_row[target_range[0]]  # mol/m3
            prod_rate = (1 - match_row['SOLVENT_REC']) * pm['F_feed'] * ctarget
            if prod_rate == 0:
                continue  # Avoid division by zero

            membrane_co2eq_per_mol_product = (co2eq_membrane * area_mem) / (membrane_lifetime * prod_rate)
            membrane_cost_per_mol_product = (cost_membrane * area_mem) / (membrane_lifetime * prod_rate)

            cost_heat = countries_df['s_h_usd_kwh'].iloc[j]
            cost_elec = countries_df['s_e_usd_kwh'].iloc[j]
            co2eq_heat = countries_df['h_kg_co2_eq'].iloc[j]
            co2eq_elec = countries_df['e_kg_co2_eq'].iloc[j]

            reference_co2 = (co2eq_heat / 3.6e6) * match_row['EVAP_ENERGY_THERMAL'] + (co2eq_elec / 3.6e6) * match_row['EVAP_ENERGY_ELECTRIC']
            novel_co2 = (co2eq_elec / 3.6e6) * match_row['NF_ENERGY'] + membrane_co2eq_per_mol_product
            co2eq_reduction = (reference_co2 - novel_co2) / reference_co2

            reference_cost = (cost_heat / 3.6e6) * match_row['EVAP_ENERGY_THERMAL'] + (cost_elec / 3.6e6) * match_row['EVAP_ENERGY_ELECTRIC']
            novel_cost = (cost_elec / 3.6e6) * nf_molar_energy_demand + membrane_cost_per_mol_product
            cost_reduction = (reference_cost - novel_cost) / reference_cost
            
            co2eq_reduction_per_country[j] = co2eq_reduction
            cost_reduction_per_country[j] = cost_reduction

        countries_df['co2_reduction_' + name] = co2eq_reduction_per_country
        countries_df['cost_reduction_' + name] = cost_reduction_per_country
        countries_df.to_csv(tea_by_country_path)



def extraction_energy_and_gwp_calculation(input_file, pm):

    print('Starting energy assessment')

    input_file_path = input_file + '.csv'
    df = pd.read_csv(input_file_path)

    electricity_kg_co2_eq = {
        'EUR': 0.430,
        'USA': 0.438,
        'IND': 1.222,
        'CHN': 0.899
    }

    heat_kg_co2_eq = {
        'EUR': 0.457,
        'USA': 0.457,
        'IND': 0.457,
        'CHN': 0.457
    }

    electricity_usd = {
        'EUR': 0.449,
        'USA': 0.146,
        'IND': 0.117,
        'CHN': 0.088
    }

    heat_usd = {
        'EUR': 0.194,
        'USA': 0.053,
        'IND': 0.060,
        'CHN': 0.045
    }


    # Ensure necessary columns exist before assignment
    df['EXTRACTION_ENERGY'] = np.nan
    df['ENERGY_REDUCTION'] = np.nan
    for key in electricity_kg_co2_eq.keys():
        df[key + '_GWP_REDUCTION'] = np.nan
        df[key + '_COST_REDUCTION'] = np.nan
        df[key + '_BREAKEVEN_COST'] = np.nan

    for i in range(len(df)):  # Fixed the iteration range
        df.at[i, 'MOLAR_POWER_KW'] = df.at[i, 'MOLAR_POWER'] * 100000 / (1000 * 3600) # to KW
        # Inputs:
        c_target1 = df.at[i, 'FINAL_CONC1']  # mol/m3
        c_target2 = df.at[i, 'FINAL_CONC2']  # mol/m3
        c_target_ratio = c_target1 / c_target2

        c0 = pm['c_feed']  # mol/m3
        nf_molar_energy_demand = 3600 * 1000 * (df.at[i, 'MOLAR_POWER_KW']) / (pm['F_feed'] * c0[0])  # J/mol

        case, cA_1_final, cB_1_final, c_ratio_1_final, cA_2_final, cB_2_final, c_ratio_2_final, recovery_A, recovery_B = models.energetics.extraction_wrapper(c0,c_target_ratio,pm['logphi'][0],pm['logphi'][1],just_impurity=True)

        if case == 'A-standard':
            cA_from_ext = cA_1_final
        elif case == 'B-standard':
            cA_from_ext = cA_2_final

        evap_molar_energy_demand = models.energetics.evaporation_energy(cA_from_ext, c0[0], pm) # J/mol

        co2eq_membrane = 5.357  # kg CO2eq/m2
        cost_membrane = 100 # USD/m2
        membrane_lifetime = 8766  # h

        area_mem = pm['nst'] * pm['ne'] * pm['A']  # m2

        prod_rate = (1 - df.at[i, 'SOLVENT_REC']) * pm['F_feed'] * c_target1
        if prod_rate == 0:
            continue  # Avoid division by zero

        membrane_co2eq_per_mol_product = (co2eq_membrane * area_mem) / (membrane_lifetime * prod_rate)
        membrane_cost_per_mol_product = (cost_membrane * area_mem) / (membrane_lifetime * prod_rate)

        df.at[i, 'EXTRACTION_ENERGY'] = evap_molar_energy_demand
        df.at[i, 'NF_ENERGY'] = nf_molar_energy_demand
        df.at[i, 'EXTRACTION_ENERGY'] = evap_molar_energy_demand
        df.at[i, 'EXTRACTION_ENERGY_THERMAL'] = evap_molar_energy_demand
        df.at[i, 'EXTRACTION_ENERGY_ELECTRIC'] = 0.0

        if evap_molar_energy_demand == 0:
            df.at[i, 'ENERGY_REDUCTION'] = 0
            for key in electricity_kg_co2_eq.keys():
                df.at[i, key + '_GWP_REDUCTION'] = 0
            for key in electricity_usd.keys():
                df.at[i, key + '_COST_REDUCTION'] = 0
                df.at[i, key + '_BREAKEVEN_COST'] = 1000000
        else:
            df.at[i, 'ENERGY_REDUCTION'] = max(0,(evap_molar_energy_demand - nf_molar_energy_demand) / evap_molar_energy_demand)
            for key in electricity_kg_co2_eq.keys():
                reference_co2 = (heat_kg_co2_eq[key] / 3.6e6) * evap_molar_energy_demand
                novel_co2 = (electricity_kg_co2_eq[key] / 3.6e6) * nf_molar_energy_demand + membrane_co2eq_per_mol_product
                co2eq_reduction = (reference_co2 - novel_co2) / reference_co2

                df.at[i, key + '_GWP_REDUCTION'] = co2eq_reduction  # kgCO2eq / mol product
            for key in electricity_usd.keys():
                reference_cost = (heat_usd[key] / 3.6e6) * evap_molar_energy_demand
                novel_cost = (electricity_usd[key] / 3.6e6) * nf_molar_energy_demand + membrane_cost_per_mol_product
                cost_reduction = (reference_cost - novel_cost) / reference_cost

                df.at[i, key + '_COST_REDUCTION'] = cost_reduction  # USD / mol product

                print(reference_cost, nf_molar_energy_demand, prod_rate)
                df.at[i, key + '_BREAKEVEN_COST'] = ((reference_cost - (electricity_usd[key] / 3.6e6) * nf_molar_energy_demand) * membrane_lifetime * prod_rate) / area_mem  # USD / m2

    df.to_csv(input_file_path, index=False)