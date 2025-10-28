import sys
import numpy as np
import models.opti_multiobjective
import models.opti_initialize

################ OPTIMIZATION #################

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python process_optimization.py <run_no>")
        sys.exit(1)

    run = int(sys.argv[1])

    config = '3s'
    no_of_models = 5
    stages_per_config = 3
    elements_per_stage = 3
    nodes_per_element = 10
    no_collocation = 6
    soft_init = True
    model_x = True
    model_d = True
    model_r = True
    fixed_k_mass = False # for single stage flat sheet

    if run == 0:
        transport_model = 'pf' # sd or pf
        problem = 'etandem' # 'suzuki' or 'etandem'
        constraint_type = 'recovery'
        objective_type = 'separation_factor' #'molar_power' or 'final_concentration' or 'separation_factor'
        alternative = 'evaporation' # extraction or evaporation
        add_rec_const = 0
        constraint_levels = [0.1,0.2,0.3,0.4,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9] # for recovery-constrained sep_factor optimization
        target_range = None
    elif run == 1:
        transport_model = 'sd' # sd or pf
        problem = 'etandem' # 'suzuki' or 'etandem'
        constraint_type = 'recovery'
        objective_type = 'separation_factor' #'molar_power' or 'final_concentration' or 'separation_factor'
        alternative = 'evaporation' # extraction or evaporation
        add_rec_const = 0
        constraint_levels = [0.1,0.2,0.3,0.4,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9] # for recovery-constrained sep_factor optimization
        target_range = None
    elif run == 2:
        transport_model = 'sd' # sd or pf
        problem = 'etandem' # 'suzuki' or 'etandem'
        constraint_type = 'final_concentration'
        objective_type = 'molar_power' #'molar_power' or 'final_concentration' or 'separation_factor'
        alternative = 'evaporation' # extraction or evaporation
        add_rec_const = 0
        constraint_levels = [1.5,2,2.5,3,3.5,4,4.5,5]
        target_range = ('FINAL_CONC1',4.95,5.05)
    elif run == 3:
        transport_model = 'pf' # sd or pf
        problem = 'api' # 'suzuki' or 'etandem'
        constraint_type = 'recovery'
        objective_type = 'separation_factor' #'molar_power' or 'final_concentration' or 'separation_factor'
        alternative = 'extraction' # extraction or evaporation
        add_rec_const = 0
        constraint_levels = [0.1,0.2,0.3,0.4,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9] # for recovery-constrained sep_factor optimization
    elif run == 4:
        transport_model = 'sd' # sd or pf
        problem = 'api' # 'suzuki' or 'etandem'
        constraint_type = 'recovery'
        objective_type = 'separation_factor' #'molar_power' or 'final_concentration' or 'separation_factor'
        alternative = 'extraction' # extraction or evaporation
        add_rec_const = 0
        constraint_levels = [0.1,0.2,0.3,0.4,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9] # for recovery-constrained sep_factor optimization
    elif run == 5:
        transport_model = 'sd' # sd or pf
        problem = 'api' # 'suzuki' or 'etandem'
        constraint_type = 'separation_factor'
        objective_type = 'molar_power' #'molar_power' or 'final_concentration' or 'separation_factor'
        alternative = 'extraction' # extraction or evaporation
        add_rec_const = 0
        constraint_levels = [1.5,2,2.5,3,3.5,4,4.5,5]


    opti_name = 'obj_'+objective_type+'_con_'+constraint_type+'_3stage_sw'
    target_folder = 'results/optimization/'+problem+'/'

    samples = ['sample_0', 'sample_1', 'sample_2', 'sample_3', 'sample_4', 'sample_5', 'sample_6', 'sample_7', 'sample_8', 'sample_9',0,1,2,3,4,'mean']

    # for fold in folds:
    for fold in samples:

        if problem == 'etandem':
            pm, pm_sim = models.opti_initialize.load_etandem_separation_problem(fold,stages_per_config,elements_per_stage,nodes_per_element,no_collocation,fixed_k_mass,soft_init)
        elif problem == 'api':
            pm, pm_sim = models.opti_initialize.load_api_separation_problem(fold,stages_per_config,elements_per_stage,nodes_per_element,no_collocation,fixed_k_mass,soft_init)
        else:
            print('Unkown problem')
            continue

        opti_type = opti_name + 'fold' + str(fold)

        # models.opti_multiobjective.multiobjective_optimization_multistart(opti_type=opti_type,
        #                                                                         target_folder=target_folder,
        #                                                                         no_of_models=no_of_models,
        #                                                                         pm=pm,
        #                                                                         pm_sim=pm_sim,
        #                                                                         constraint_type=constraint_type,
        #                                                                         objective_type=objective_type,
        #                                                                         constraint_levels=constraint_levels,
        #                                                                         config=config,
        #                                                                         transport_model=transport_model,
        #                                                                         model_x=model_x,
        #                                                                         model_d=model_d,
        #                                                                         model_r=model_r,
        #                                                                         additional_rec_constraint=add_rec_const)

        input_file = 'results/optimization/'+problem+'/opti_'+transport_model+'_'
            
        if model_x:
            input_file += 'x_'
        if model_d:
            input_file += 'd_'
        if model_r:
            input_file += 'r_'
        
        input_file = input_file + str(no_of_models)+'models_'+opti_type

        # ### VALIDATION

        # models.opti_initialize.sim_validation(input_file, pm_sim, model_x=model_x, model_d=model_d, model_r = model_r, transport_model=transport_model)
        # input_file = input_file + '_validation'

        ### PARETO SELECTION
    
        # if objective_type != 'molar_power' and objective_type == 'separation_factor':
        #     models.opti_multiobjective.pareto_selector(input_file,'SOLUTE_REC','SEP_FACTOR',competing='true')
        # elif objective_type != 'molar_power' and objective_type == 'final_concentration':
        #     models.opti_multiobjective.pareto_selector(input_file,'SOLUTE_REC','FINAL_CONC1',competing='true')
        # elif objective_type == 'molar_power' and constraint_type == 'separation_factor':
        #     models.opti_multiobjective.pareto_selector(input_file,'SEP_FACTOR','MOLAR_POWER',competing='2min')
        # elif objective_type == 'molar_power' and constraint_type == 'final_concentration':
        #     models.opti_multiobjective.pareto_selector(input_file,'FINAL_CONC1','MOLAR_POWER',competing='2min')
        # else:
        #     print('Unknown pareto')

        input_file = input_file + '_pareto'

        ### ENERGY CALCULATION

        if alternative == 'extraction':
            models.opti_multiobjective.extraction_energy_and_gwp_calculation(input_file, pm)
        elif alternative == 'evaporation':
            models.opti_multiobjective.evaporation_energy_and_gwp_calculation(input_file, pm, target_range=target_range)
        