import models.membrane_transport
import numpy as np
import pandas as pd

'''
Nanofiltration cascade simulations.
Main focus here: SDC, SD models
'''

def nf_stage_sim(stage_parameters, pm, model='sdc'):
    F_feed, c_feed, A, T, nk, ne, p0, pp, F_dil, F_rec, c_rec = stage_parameters

    ns = len(c_feed)
    F0 = F_feed + F_dil + F_rec
    c0 = []
    for i in range(len(c_feed)):
        c0.append((F_feed*c_feed[i] + F_rec*c_rec[i])/F0)

    solutions = []
    stage_solutions = {}
    
    if model == 'sdc':
        parameters = [F0, c0, A, T, nk, ns, p0, pp]
        els = models.membrane_transport.sdc_spiral_wound_mesh_module(parameters,pm)
        solutions.append(els)
    elif model == 'sd':
        parameters = [F0, c0, A, T, ns, p0, pp]
        els = models.membrane_transport.sd_spiral_wound_mesh_module(parameters,pm)
        solutions.append(els)
    elif model == 'pf':
        parameters = [F0, c0, A, T, ns, p0, pp]
        els = models.membrane_transport.pf_spiral_wound_mesh_module(parameters,pm)
        solutions.append(els)        
    else:
        print('Unrecognized model')
        return       

    for i in range(ne-1):
        if model == 'sdc':
            parameters = [solutions[-1]['Fr'], solutions[-1]['cr'], A, T, nk, ns, solutions[-1]['pr'], pp]
            els = models.membrane_transport.sdc_spiral_wound_mesh_module(parameters,pm)
            solutions.append(els)
        elif model == 'sd':
            parameters = [solutions[-1]['Fr'], solutions[-1]['cr'], A, T, ns, solutions[-1]['pr'], pp]
            els = models.membrane_transport.sd_spiral_wound_mesh_module(parameters,pm)
            solutions.append(els)
        elif model == 'pf':
            parameters = [solutions[-1]['Fr'], solutions[-1]['cr'], A, T, ns, solutions[-1]['pr'], pp]
            els = models.membrane_transport.pf_spiral_wound_mesh_module(parameters,pm)
            solutions.append(els)

    Fp_final = np.sum(i['Fp'] for i in solutions)
    cp_final = []
    for i in range(len(c_feed)):
        cp_final.append(np.sum(j['Fp']*j['cp'][i] for j in solutions)/Fp_final)

    stage_solutions['Fr'] = solutions[-1]['Fr']
    stage_solutions['cr'] = solutions[-1]['cr']
    stage_solutions['pr'] = solutions[-1]['pr']
    stage_solutions['F0'] = F0
    stage_solutions['c0'] = c0
    stage_solutions['p0'] = p0
    stage_solutions['Fp'] = Fp_final
    stage_solutions['cp'] = cp_final
    stage_solutions['pp'] = pp
    stage_solutions['elements'] = solutions
    return stage_solutions

##########################

def nf_linear_simulation(process_pm,pm,model_x=False,model_d=False, model_r = False, model ='sdc'):
    n_stages = pm['nst']

    sol_dict = {}
    p_feed, pp_list, F_dil_list, split_list = process_pm

    F_rec_list = [0.0] * n_stages
    c_rec_list = [[0.0]*pm['ns']]*n_stages

    if model_d == False:
        for i in range(len(F_dil_list)):
            F_dil_list[i] = 0.0

    if model_r == False:
        for i in range(len(split_list)):
            split_list[i] = 0.0

    def out_of_tolerance(a,a_new):
        if a == 0:
            if np.abs(a_new-a) <= 0.02:
                return False
            else:
                return True
        else:
            if np.abs(a_new-a)/a <= 0.05:
                return False
            else:
                return True
    
    def list_out_of_tolerance(a_list,a_list_new):
        ret = False
        for i in range(len(a_list)):
            if out_of_tolerance(a_list[i],a_list_new[i]):
                ret = True
                break
        return ret

    
    has_error = True

    counter = 0
    while has_error and counter < 20:
        if model_r:
            print('New iter', F_rec_list)

        # 1st stage
        stage_parameters_0 = [pm['F_feed'], pm['c_feed'], pm['A'], pm['T'], pm['nk'], pm['ne'], p_feed, pp_list[0], F_dil_list[0], F_rec_list[0], c_rec_list[0]]
        sol_dict[0] = nf_stage_sim(stage_parameters_0, pm, model)

        for i_stage in range(1,n_stages):
            stage_parameters = [sol_dict[i_stage-1]['Fr'], sol_dict[i_stage-1]['cr'], pm['A'], pm['T'], pm['nk'], pm['ne'], sol_dict[i_stage-1]['pr'], pp_list[i_stage], F_dil_list[i_stage], F_rec_list[i_stage], c_rec_list[i_stage]]
            sol_dict[i_stage] = nf_stage_sim(stage_parameters, pm, model)

        if model_r:
            F_rec_list_new = []
            for j_stage in range(1,n_stages):
                F_rec_list_new.append(sol_dict[j_stage]['Fp']*split_list[j_stage])
            F_rec_list_new.append(0.0)

            c_rec_list_new = []
            for j_stage in range(1,n_stages):
                c_rec_list_new.append(sol_dict[j_stage]['cp'])
            c_rec_list_new.append([0.0]*pm['ns'])

            has_error = list_out_of_tolerance(F_rec_list,F_rec_list_new)
            F_rec_list = F_rec_list_new
            c_rec_list = c_rec_list_new
            counter += 1
        else:
            has_error = False

    if model_r and counter == 20:
        print('Recycle iteration stopped')

    # Performance descriptors
    recovery = (sol_dict[n_stages-1]['Fr'] * (sol_dict[n_stages-1]['cr'][0])) / (pm['F_feed'] * (pm['c_feed'][0]))
    solvent_recovery = 1 - (sol_dict[n_stages-1]['Fr']/(pm['F_feed'] + np.sum(F_dil_list)))
    sep_factor = ((sol_dict[n_stages-1]['cr'][0]) / (sol_dict[n_stages-1]['cr'][1])) / ((pm['c_feed'][0]) / (pm['c_feed'][1]))
    if model_x:
        p_pump = (pm['F_feed']*p_feed - pm['pex_eff']*sol_dict[n_stages-1]['Fr']*sol_dict[n_stages-1]['pr'])/pm['F_feed']
    else: 
        p_pump = p_feed

    dil_power = {}
    for i in range(len(F_dil_list)):
        dil_power[i] = (1/pm['pump_eff']) * F_dil_list[i] * sol_dict[i]['p0']
    
    rec_power = {}
    rec_power[0] = 0.0
    for i_stage in range(1,n_stages):
        rec_power[i_stage] = (1/pm['pump_eff']) * sol_dict[i_stage]['Fp'] * split_list[i_stage] * (sol_dict[i_stage-1]['p0'] - sol_dict[i_stage]['pp'])

    power = (1/pm['pump_eff']) * pm['F_feed'] * p_pump + np.sum(dil_power[i] for i in range(len(F_dil_list))) + np.sum(rec_power[i] for i in range(pm['nst']))

    molar_power = power / recovery

    final_conc1 = sol_dict[n_stages-1]['cr'][0]
    final_conc2 = sol_dict[n_stages-1]['cr'][1]
    
    desc = [final_conc1, final_conc2,recovery,solvent_recovery,sep_factor,p_pump,power,molar_power,dil_power,rec_power]

    return desc, sol_dict