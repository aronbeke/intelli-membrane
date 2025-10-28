import models.nf_simulation
import models.membrane_transport
import numpy as np
import pandas as pd
import warnings


def load_etandem_separation_problem(fold,stages_per_config,elements_per_stage,nodes_per_element,no_collocation,fixed_k_mass,soft_init):
    '''
    eFuel tandem separation problem
    Technology comparison: concentration with NF and vacuum distillation/evaporation
    Octene-mix (assuming nonanol is evaporated with the solvent) -- PMPerformance
    Octene-mix: 50% 1-octene, 25% 1-nonanol, 25% 1-nonanal
    '''

    Rej1_ensemble = [0.97767234, 0.991549, 0.9506162, 0.9041277, 0.985317] # monomer
    Rej2_ensemble = [0.44454163, 0.3605274, 0.3236177, 0.3210490, 0.39032608] # nonanol

    np.random.seed(42)  # replace 42 with any integer you like

    # 1. Mean and std
    mean1, std1 = np.mean(Rej1_ensemble), np.std(Rej1_ensemble, ddof=1)
    mean2, std2 = np.mean(Rej2_ensemble), np.std(Rej2_ensemble, ddof=1)

    # print(f"Monomer: mean = {mean1:.4f}, std = {std1:.4f}")
    # print(f"Nonanol: mean = {mean2:.4f}, std = {std2:.4f}")

    # 2. Define Normal distributions
    dist1 = np.random.normal(mean1, std1, 10)
    dist2 = np.random.normal(mean2, std2, 10)

    # 3. Clip values above 1.0
    dist1 = np.clip(dist1, None, 1.0)
    dist2 = np.clip(dist2, None, 1.0)

    # print("\nSamples (Monomer):", dist1)
    # print("Samples (Nonanol):", dist2)

    # Samples are taken from assuming normal distribution for epistemic uncertainty in both rejections separately (clipped to 1.0)
    Rej1_samples = dist1.tolist()
    Rej2_samples = dist2.tolist()

    if fold == 'mean':
        Rej1 = float(np.mean(Rej1_ensemble))
        Rej2 = float(np.mean(Rej2_ensemble))
    elif isinstance(fold, str) and fold.startswith('sample'):
        sample_index = int(fold.split('_')[1])
        Rej1 = Rej1_samples[sample_index]
        Rej2 = Rej2_samples[sample_index]
    else:
        Rej1 = Rej1_ensemble[fold]
        Rej2 = Rej2_ensemble[fold]
    
    ### SYSTEM PARAMETERS
    pm = {
        'type': 'solute_concentration',
        'F_feed': 1.5,  # Volumetric flow rate of the feed stream m3/h
        'F_dil_feed': 1.5, # m3/h
        'A': 40.0,  # m2
        'l_module': 0.965,  # m
        'eta': 13.1, # viscosity in kg/mh (1 mPas = 3.6 kg/mh)
        'l_mesh': 0.00327,  # m
        'df': 0.00033,  # m
        'theta': 1.83259571,  # rad
        'n_env': 28,
        'b_env': 0.85,  # m
        'h': 0.0007,  # m
        'rho': 769.5, #kg/m3,
        'M_solvent': 0.12773, #kg/mol
        'T': 303.0,  # K
        'dp_max': 40,  # bar
        'pump_eff': 0.85,  # pump efficiency
        'pex_eff': 0.85,  # pressure exchanger efficiency
        'P0': 0.13e-3, # m3/m2hbar
        'P1': 0, # m/hbar2
        'b0': 0.20,  # concentration polarization coeff.
        'sf': False,  # solution-friction
        'k_mass_fix': 1e-4 * 3600, #m/h
        'dH_solvent': 40440 * 0.5 + 77000 * 0.25 + 51000 * 0.25, # J/mol # weighted avg of octane, nonanol, decanol
        'cp_solvent': 0.5* 235 + 0.5 * 356 # J/molK approx (octene and nonanol)

    }

    pm['F_lim'] = pm['F_feed'] + pm['F_dil_feed']
    pm['Di_list'] = [3.6e-6, 3.6e-6] #m2/h
    pm['c_feed'] = [1.0,10.0] # mol/m3
    pm['c_feed_dict'] = {}
    for index, value in enumerate(pm['c_feed']):
        pm['c_feed_dict'][index] = value

    # https://webbook.nist.gov/cgi/cbook.cgi?ID=C111660&Mask=4&Plot=on&Type=ANTOINE
    pm['vacuum'] = True
    pm['antoine_A'] = 4.05752	 # use octene values as approximation, should return K
    pm['antoine_B'] = 1353.486
    pm['antoine_C'] = -60.386
    pm['antoine_P_units'] = 'bar'
    pm['solvent_recovery'] = False
    pm['density'] = pm['rho']
    pm['solvent_molar_mass'] = pm['M_solvent']
    pm['solvent_heat_of_evaporation'] = pm['dH_solvent']
    pm['evap_eff'] = 2.6
    pm['heat_integration_eff'] = 0.5
    pm['cp'] = pm['cp_solvent']
    pm['operating_pressure'] = 5000 #Pa
    pm['vap_heat_capacity_ratio'] = 1.3
    pm['feed_temp'] = 298.15 #K

    solute_molar_volume = 1e-4
    p_inference = 5e6 # measured at 50 bar

    Li1 = models.membrane_transport.solute_permeance_from_rejection(Rej1, solute_molar_volume, pm['P0']/1e5, p_inference)
    Li2 = models.membrane_transport.solute_permeance_from_rejection(Rej2, solute_molar_volume, pm['P0']/1e5, p_inference)

    pm['Li_list'] = [Li1, Li2]
    pm['Rej_list'] = [Rej1, Rej2]
    pm['nn'] = nodes_per_element
    pm['nst'] = stages_per_config
    pm['fixed_k_mass'] = fixed_k_mass
    pm['soft_init'] = soft_init
    pm['nk'] = no_collocation
    pm['ne'] = elements_per_stage
    pm['ns'] = len(pm['c_feed']) # no. of solutes
    
    pm_sim = pm.copy()
    pm_sim['P0'] = pm_sim['P0'] / 1e5
    pm_sim['P1'] = pm_sim['P1'] / 1e10
    pm_sim['dp_max'] = pm_sim['dp_max'] * 1e5

    return pm, pm_sim

def load_api_separation_problem(fold,stages_per_config,elements_per_stage,nodes_per_element,no_collocation,fixed_k_mass,soft_init):
    '''
    Technology comparison: impurity removal with NF and extraction-evaporation
    Toluene -- PMPerformance
    _ext: extractor solvent (water)
    _sol: original solvent (toluene)
    '''

    Rej1_ensemble = [0.52882767, 0.6034495, 0.5519976, 0.66399205, 0.7087406]
    Rej2_ensemble = [0.06497818, 0.108692676, 0.10580775, 0.0905714, 0.16907242]


    np.random.seed(42)  # replace 42 with any integer you like

    # Data

    # 1. Mean and std
    mean1, std1 = np.mean(Rej1_ensemble), np.std(Rej1_ensemble, ddof=1)
    mean2, std2 = np.mean(Rej2_ensemble), np.std(Rej2_ensemble, ddof=1)

    # print(f"Monomer: mean = {mean1:.4f}, std = {std1:.4f}")
    # print(f"Nonanol: mean = {mean2:.4f}, std = {std2:.4f}")

    # 2. Define Normal distributions
    dist1 = np.random.normal(mean1, std1, 10)
    dist2 = np.random.normal(mean2, std2, 10)

    # 3. Clip values above 1.0
    dist1 = np.clip(dist1, None, 1.0)
    dist2 = np.clip(dist2, None, 1.0)

    # print("\nSamples (Monomer):", dist1)
    # print("Samples (Nonanol):", dist2)

    # Samples are taken from assuming normal distribution for epistemic uncertainty in both rejections separately (clipped to 1.0)
    Rej1_samples = dist1.tolist()
    Rej2_samples = dist2.tolist()

    if fold == 'mean':
        Rej1 = float(np.mean(Rej1_ensemble))
        Rej2 = float(np.mean(Rej2_ensemble))
    elif isinstance(fold, str) and fold.startswith('sample'):
        sample_index = int(fold.split('_')[1])
        Rej1 = Rej1_samples[sample_index]
        Rej2 = Rej2_samples[sample_index]
    else:
        Rej1 = Rej1_ensemble[fold]
        Rej2 = Rej2_ensemble[fold]
    
    ### SYSTEM PARAMETERS
    pm = {
        'type': 'impurity_removal',
        'F_feed': 9.0,  # Volumetric flow rate of the feed stream m3/h
        'F_dil_feed': 9.0,
        'A': 40.0,  # m2
        'l_module': 0.965,  # m
        'eta': 0.59 * 3.6, # in kg/mh (1 mPas = 3.6 kg/mh)
        'l_mesh': 0.00327,  # m
        'df': 0.00033,  # m
        'theta': 1.83259571,  # rad
        'n_env': 28,
        'b_env': 0.85,  # m
        'h': 0.0007,  # m (fitted from pressure drops)
        'rho': 867.00, #kg/m3,
        'M_solvent': 0.092141, #kg/mol
        'T': 303.0,  # K
        'dp_max': 40,  # bar
        'pump_eff': 0.85,  # pump efficiency
        'pex_eff': 0.85,  # pressure exchanger efficiency
        'P0': 0.717e-3, # m3/m2hbar
        'P1': 0, # m/hbar2
        'b0': 0.20,  # concentration polarization coeff.
        'sf': False,  # solution-friction
        'k_mass_fix': 1e-4 * 3600, #m/h
        'dH_solvent': 38060, # J/mol
        'logS_ext': [-2.93635992078529,-2.185854509], #log10(mol/L) # WATER
        'logS_sol': [-1.02521434156488,0.4266031323395092], #log10(mol/L) # TOLUENE
    }

    pm['logphi'] = []
    for idx in range(len(pm['logS_ext'])):
        pm['logphi'].append(pm['logS_ext'][idx] - pm['logS_sol'][idx])

    pm['evap_eff'] = 2.6
    pm['heat_integration_eff'] = 0.5
    pm['solvent_recovery'] = False
    pm['density'] = pm['rho']
    pm['solvent_molar_mass'] = pm['M_solvent']
    pm['solvent_heat_of_evaporation'] = pm['dH_solvent']

    pm['F_lim'] = pm['F_feed'] + pm['F_dil_feed']
    pm['Di_list'] = [3.6e-6, 3.6e-6] #m2/h
    pm['c_feed'] = [1.0,10.0]
    pm['c_feed_dict'] = {}
    for index, value in enumerate(pm['c_feed']):
        pm['c_feed_dict'][index] = value

    solute_molar_volume = 1e-4
    p_inference = 3e6

    Li1 = models.membrane_transport.solute_permeance_from_rejection(Rej1, solute_molar_volume, pm['P0']/1e5, p_inference)
    Li2 = models.membrane_transport.solute_permeance_from_rejection(Rej2, solute_molar_volume, pm['P0']/1e5, p_inference)

    pm['Li_list'] = [Li1, Li2]
    pm['Rej_list'] = [Rej1, Rej2]
    pm['nn'] = nodes_per_element
    pm['nst'] = stages_per_config
    pm['fixed_k_mass'] = fixed_k_mass
    pm['soft_init'] = soft_init
    pm['nk'] = no_collocation
    pm['ne'] = elements_per_stage
    pm['ns'] = len(pm['c_feed']) # no. of solutes
    
    pm_sim = pm.copy()
    pm_sim['P0'] = pm_sim['P0'] / 1e5
    pm_sim['P1'] = pm_sim['P1'] / 1e10
    pm_sim['dp_max'] = pm_sim['dp_max'] * 1e5

    return pm, pm_sim


def generate_random_combinations(total_sum, num_max, sample_size):
    '''
    Generate random process parameters
    '''
    num_set = np.random.uniform(0,num_max,size=(sample_size))
    while np.sum(num_set) >= total_sum:
        num_set = np.random.uniform(0,num_max,size=(sample_size))
    return list(num_set)

def random_initialization(model,pm_sim,init_pm,model_x=False,model_d=False,model_r = False,transport_model='sdc'):
    '''
    Main function to be called. Loops with random process parameters until finds a feasible one.
    '''
    feasible = 0
    warnings.simplefilter('error', RuntimeWarning)
    while feasible == 0:
        try:
            p_feed = np.random.uniform(init_pm['p_feed_min'],init_pm['p_feed_max'])*1e5 # Pa
            pp_list = list(np.random.uniform(0,p_feed/1e5,size=(pm_sim['nst']))*1e5) # Pa
            if model_d and not pm_sim['soft_init']:
                F_dil_list = generate_random_combinations(pm_sim['F_dil_feed'],init_pm['dilution_max'],pm_sim['nst']) #m3/h
            else:
                F_dil_list = [0] * pm_sim['nst']
            if model_r and not pm_sim['soft_init']:
                split_list = list(np.random.uniform(0,1,size=(pm_sim['nst'])))
            else:
                split_list = [0] * pm_sim['nst']
            init_parameters = [p_feed, pp_list, F_dil_list, split_list]
            print('Initializing with ',init_parameters)
            model, status = nf_linear_initialization(model,init_parameters,pm_sim,model_x=model_x,model_d=model_d,model_r = model_r,transport_model=transport_model)
        except RuntimeWarning:
             status = True
        
        if status:
            feasible = 0
        else:
            feasible = 1
    
    return model, init_parameters


def reject_initialization(desc,sol_dict,n_stages,n_solutes,n_elements):
    '''
    Checks whether simulation based on random process parameters is feasible
    '''
    init_error = False
    final_conc1,final_conc2,recovery,water_recovery,sep_factor,p_pump,power,molar_power,dil_power,rec_power = desc
    for i in range(n_stages):
        if dil_power[i] < 0:
            init_error = True
            print('Negative dil power')
            break
        if rec_power[i] < 0:
            init_error = True
            print('Negative rec power')
            break

    for el in [final_conc1,final_conc2,recovery,water_recovery,sep_factor,p_pump,power,molar_power]:
        if el < 0:
            init_error = True
            print('Negative characteristic')
            print(el)
            break

    for i in range(n_stages):
        for d in ['Fr','Fp','F0','p0','pr','pp']:
            if sol_dict[i][d] < 0:
                init_error = True
                print('Negative flow or pressure')
                break
        for d in ['c0','cr','cp']:
            for j in range(n_solutes):
                if sol_dict[i][d][j] < 0:
                    init_error = True
                    print('Negative concentration')
                    break
        
        for el in range(n_elements):
            if sol_dict[i]['elements'][el]['p0'] - sol_dict[i]['elements'][el]['pressure_drop'] - sol_dict[i]['elements'][el]['pp'] < 0:
                init_error = True
                break
            for d in ['Fr','Fp','F0','p0','pr','pp']:
                if sol_dict[i]['elements'][el][d] < 0:
                    # print('Infeasible element: ',el,sol_dict[i]['elements'][el][d])
                    init_error = True
                    print('Negative flow or pressure')
                    break
            for d in ['c0','cr','cp']:
                for j in range(n_solutes):
                    if sol_dict[i]['elements'][el][d][j] < 0:
                        # print('Infeasible element: ',el,sol_dict[i]['elements'][el][d])
                        init_error = True
                        print('Negative concentration')
                        break

    return init_error


def nf_linear_initialization(model,init_parameters,pm_sim,model_x=False,model_d=False,model_r = False,transport_model='sdc'):
    '''
    Initializes Pyomo model. Rejects initialization if infeasible.
    Status: False = ok, True = infeasible initialization
    model_type: 'xrd', 'xd', 'x', '0'
    x: pressure exchange
    r: recycle (not considered in initialization)
    d: dilution
    0: conventional linear cascade
    transport_model: 'sdc' or 'sd' or 'pf'
    '''

    status = False
    n_stages = pm_sim['nst']
    n_solutes = pm_sim['ns']
    n_elements = pm_sim['ne']

    p_feed, pp_list, F_dil_list, split_list = init_parameters
    act_init_pm = init_parameters
    desc, sol_dict = models.nf_simulation.nf_linear_simulation(act_init_pm, pm_sim,model_x=model_x,model_d=model_d,model=transport_model)
    print('Simulation done')
    
    final_conc1, final_conc2,recovery,solvent_recovery,sep_factor,p_pump,power,molar_power,dil_power,rec_power = desc

    status = reject_initialization(desc,sol_dict,n_stages,n_solutes,n_elements)
    if status:
        return model, status
    print("Feasible model found")

    model.recovery.value = recovery
    model.separation_factor.value = sep_factor
    model.solvent_recovery.value = solvent_recovery
    if model_x == False:
        model.p_feed = p_feed /1e5
    else:
        model.p_pump.value = p_pump /1e5
        model.p_feed.value = p_feed /1e5
    model.power.value = power /1e5
    model.molar_power.value = molar_power /1e5

    if model_d:
        for i in range(n_stages):
            model.F_dilution[i].value = F_dil_list[i]
            model.power_dilution[i].value = dil_power[i] /1e5
    if model_r:
        for i in range(n_stages):
            model.omega[i].value = split_list[i]
            model.recirc_power[i].value = rec_power[i] /1e5

    model.final_flow.value = sol_dict[n_stages-1]['Fr']
    model.final_pressure.value = sol_dict[n_stages-1]['pr'] /1e5
    for i in range(n_solutes):
        model.final_concentration[i].value = sol_dict[n_stages-1]['cr'][i]

    for st in range(n_stages):
        model.stages[st].p0.value = sol_dict[st]['p0'] /1e5
        model.stages[st].pp.value = sol_dict[st]['pp'] /1e5
        model.stages[st].pr.value = sol_dict[st]['pr'] /1e5
        model.stages[st].F0.value = sol_dict[st]['F0']
        model.stages[st].Fr.value = sol_dict[st]['Fr']
        model.stages[st].Fp.value = sol_dict[st]['Fp']
        for i in range(n_solutes):
            model.stages[st].C0[i].value = sol_dict[st]['c0'][i]
            model.stages[st].Cr[i].value = sol_dict[st]['cr'][i]
            model.stages[st].Cp[i].value = sol_dict[st]['cp'][i]

        for el in range(pm_sim['ne']):
            model.stages[st].elems[el].p0.value = sol_dict[st]['elements'][el]['p0'] /1e5
            model.stages[st].elems[el].pp.value = sol_dict[st]['elements'][el]['pp'] /1e5
            model.stages[st].elems[el].pdrop.value = (sol_dict[st]['elements'][el]['p0'] - sol_dict[st]['elements'][el]['pr']) /1e5
            if transport_model == 'sdc':
                model.stages[st].elems[el].beta.value = sol_dict[st]['elements'][el]['beta']
            elif transport_model == 'sd' or transport_model == 'pf':
                for sol in range(pm_sim['ns']):
                    model.stages[st].elems[el].k_mass[sol].value = sol_dict[st]['elements'][el]['k_mass'][sol]
            model.stages[st].elems[el].F0.value = sol_dict[st]['elements'][el]['F0']
            model.stages[st].elems[el].Fr.value = sol_dict[st]['elements'][el]['Fr']
            model.stages[st].elems[el].Fp.value = sol_dict[st]['elements'][el]['Fp']
            dp = (sol_dict[st]['elements'][el]['pr'] - sol_dict[st]['elements'][el]['pp']) /1e5
            for i in range(n_solutes):
                model.stages[st].elems[el].C0[i].value = sol_dict[st]['elements'][el]['c0'][i]
                model.stages[st].elems[el].Cr[i].value = sol_dict[st]['elements'][el]['cr'][i]
                model.stages[st].elems[el].Cp[i].value = sol_dict[st]['elements'][el]['cp'][i]

            for no in range(pm_sim['nn']):
                if transport_model == 'sdc':
                    j_list = sol_dict[st]['elements'][el]['nodes'][no]['sim'][3:(3+n_solutes)]
                    cr_list = sol_dict[st]['elements'][el]['nodes'][no]['sim'][(3+n_solutes):(3+2*n_solutes)]
                    cm_list = sol_dict[st]['elements'][el]['nodes'][no]['sim'][(3+2*n_solutes):(3+3*n_solutes)]
                    c_reduced_list = sol_dict[st]['elements'][el]['nodes'][no]['sim'][(3+3*n_solutes+(pm_sim['nk']-1)):(3+3*n_solutes+(pm_sim['nk']-1)+(pm_sim['nk']-1)*n_solutes)]

                    j, cr, cm = np.array([j_list]), np.transpose(np.array([cr_list])), np.transpose(np.array([cm_list]))
                    c_reduced_vector = np.array([c_reduced_list])

                    C_reduced_shape = (pm_sim['nk']-1,n_solutes)

                    J = np.tile(j,(pm_sim['nk']-1,1))
                    V = np.vander(np.linspace(0,1,pm_sim['nk']), pm_sim['nk'], increasing=True)
                    V_inv = np.linalg.inv(V)
                    D_coeff = np.tile(np.array([range(pm_sim['nk'])]),(pm_sim['nk']-1,1))
                    D_vander = np.hstack((np.zeros((pm_sim['nk']-1,1)),np.vander(np.linspace(0,1,pm_sim['nk']-1), pm_sim['nk']-1, increasing=True)))
                    D = np.multiply(D_coeff,D_vander)
                    C_reduced = c_reduced_vector.reshape(C_reduced_shape)
                    C = np.vstack((np.transpose(cm),C_reduced))

                    DCDx = D @ (V_inv @ C)

                    Fr_node = sol_dict[st]['elements'][el]['nodes'][no]['sim'][1]
                    Fp_node = sol_dict[st]['elements'][el]['nodes'][no]['sim'][2]
                    F0_node = sol_dict[st]['elements'][el]['nodes'][no]['F0']
                    model.stages[st].elems[el].nodes[no].beta.value = np.exp(pm_sim['b0']*(Fp_node/F0_node))
                    model.stages[st].elems[el].nodes[no].dp.value = dp
                    model.stages[st].elems[el].nodes[no].Fr.value = Fr_node
                    model.stages[st].elems[el].nodes[no].Fp.value = Fp_node
                    model.stages[st].elems[el].nodes[no].F0.value = F0_node
                    model.stages[st].elems[el].nodes[no].Jv.value = sol_dict[st]['elements'][el]['nodes'][no]['sim'][0]
                    for i in range(n_solutes):
                        model.stages[st].elems[el].nodes[no].J[i].value = j_list[i]
                        model.stages[st].elems[el].nodes[no].C0[i].value = sol_dict[st]['elements'][el]['nodes'][no]['c0'][i]
                        model.stages[st].elems[el].nodes[no].Cr[i].value = cr_list[i]
                        for j in range(pm_sim['nk']):
                            model.stages[st].elems[el].nodes[no].C[j,i].value = C[j,i]
                        for j in range(1,pm_sim['nk']):
                            model.stages[st].elems[el].nodes[no].dCdx[j,i].value = DCDx[j-1,i]

                elif transport_model == 'sd' or transport_model == 'pf':
                    j_list = sol_dict[st]['elements'][el]['nodes'][no]['sim'][3:(3+n_solutes)]
                    cr_list = sol_dict[st]['elements'][el]['nodes'][no]['sim'][(3+n_solutes):(3+2*n_solutes)]
                    cm_list = sol_dict[st]['elements'][el]['nodes'][no]['sim'][(3+2*n_solutes):(3+3*n_solutes)]
                    cp_list = sol_dict[st]['elements'][el]['nodes'][no]['sim'][(3+3*n_solutes):(3+4*n_solutes)]

                    Fr_node = sol_dict[st]['elements'][el]['nodes'][no]['sim'][1]
                    Fp_node = sol_dict[st]['elements'][el]['nodes'][no]['sim'][2]
                    F0_node = sol_dict[st]['elements'][el]['nodes'][no]['F0']
                    model.stages[st].elems[el].nodes[no].dp.value = dp
                    model.stages[st].elems[el].nodes[no].Fr.value = Fr_node
                    model.stages[st].elems[el].nodes[no].Fp.value = Fp_node
                    model.stages[st].elems[el].nodes[no].F0.value = F0_node
                    model.stages[st].elems[el].nodes[no].Jv.value = sol_dict[st]['elements'][el]['nodes'][no]['sim'][0]
                    for i in range(n_solutes):
                        model.stages[st].elems[el].nodes[no].k_mass[i].value = sol_dict[st]['elements'][el]['k_mass'][i]
                        model.stages[st].elems[el].nodes[no].J[i].value = j_list[i]
                        model.stages[st].elems[el].nodes[no].C0[i].value = sol_dict[st]['elements'][el]['nodes'][no]['c0'][i]
                        model.stages[st].elems[el].nodes[no].Cr[i].value = cr_list[i]
                        model.stages[st].elems[el].nodes[no].Cp[i].value = cp_list[i]
                        model.stages[st].elems[el].nodes[no].Cm[i].value = cm_list[i]

    return model, status


def sim_validation(input_file,pm_sim,model_x=False,model_d=False,model_r=False,transport_model = 'sdc'):
    '''
    pm_sim: simulation parameters dictionary
    '''

    
    df = pd.read_csv(input_file+'.csv')

    length = len(df.index)
    SIM_WATER_REC = np.zeros(length)
    SIM_BIVALENT_REC = np.zeros(length)
    SIM_SEP_FACTOR = np.zeros(length)
    SIM_SP_POWER = np.zeros(length)
    SIM_FIN_CONC1 = np.zeros(length)
    SIM_FIN_CONC2 = np.zeros(length)

    for i in range(length):
        p_feed = df['FEED_PRESSURE'].iloc[i]*1e5
        pp_list = [(df[f'PERMEATE_PRESSURE_{i_stage}'].iloc[i] * 1e5) for i_stage in range(pm_sim['nst'])]
        F_dil_list = [df[f'DILUTION_{i_stage}'].iloc[i] for i_stage in range(pm_sim['nst'])]
        split_list = [0] + [df[f'OMEGA_{i_stage}'].iloc[i] for i_stage in range(1, pm_sim['nst'])]

        init_parameters = [p_feed, pp_list, F_dil_list, split_list]
        desc, sol_dict = models.nf_simulation.nf_linear_simulation(init_parameters,pm_sim,model_x=model_x,model_d=model_d,model_r=model_r,model=transport_model)

        final_conc1,final_conc2,recovery,water_recovery,sep_factor,p_pump,power,molar_power,dil_power, rec_power = desc

        SIM_WATER_REC[i] = water_recovery
        SIM_BIVALENT_REC[i] = recovery
        SIM_SEP_FACTOR[i] = sep_factor
        SIM_SP_POWER[i] = molar_power
        SIM_FIN_CONC1[i] = final_conc1
        SIM_FIN_CONC2[i] = final_conc2

    df['SIM_SOLVENT_REC'] = SIM_WATER_REC
    df['SIM_SOLUTE_REC'] = SIM_BIVALENT_REC
    df['SIM_SEP_FACTOR'] = SIM_SEP_FACTOR
    df['SIM_MOLAR_POWER'] = SIM_SP_POWER
    df['SIM_FINAL_CONC1'] = SIM_FIN_CONC1
    df['SIM_FINAL_CONC2'] = SIM_FIN_CONC2

    df.to_csv(input_file+'_validation.csv')