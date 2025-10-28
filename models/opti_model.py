import numpy as np
import pyomo
from pyomo.environ import *
from pyomo.opt import SolverFactory, SolverStatus, TerminationCondition
from pyomo.util.infeasible import log_infeasible_constraints

# CONFIGURATIONS ######################################################################

def model(constraints, objective, pm, membrane_model='sd', model_x =False, config='3s'):
    # Create a Pyomo model
    model = ConcreteModel()

    if membrane_model == 'sd':
        import models.optimization_sd as opt
    elif membrane_model == 'sdc':
        import models.optimization_sdc as opt
    elif membrane_model == 'pf':
        import models.optimization_pf as opt
    else:
        print('Unknown model')
        return

    opt.initialize_model(pm)
    # Problem-level variables

    model.recovery = Var(within=NonNegativeReals, bounds=(0,1), initialize = 1.0)
    model.separation_factor = Var(within=NonNegativeReals, bounds=(1,100), initialize = 1.0)
    model.solvent_recovery = Var(within=NonNegativeReals, bounds=(0,1), initialize = 0.0)
    model.p_feed = Var(within=NonNegativeReals, bounds = (0,pm['dp_max']), initialize = 0.0)
    model.p_pump = Var(within=NonNegativeReals, bounds = (0,pm['dp_max']), initialize = 0.0)
    model.power = Var(within=NonNegativeReals, bounds = (0,2000), initialize = 1500.0) # power necessary
    model.molar_power = Var(within=NonNegativeReals, bounds = (0,3000), initialize = 1500.0)

    model.final_flow = Var(within=NonNegativeReals, bounds=(0,pm['F_lim']), initialize = pm['F_feed'])
    model.final_pressure = Var(within=NonNegativeReals, bounds = (0,pm['dp_max']), initialize = 0.0)
    model.final_concentration = Var(pm['solutes'],within=NonNegativeReals, bounds = lambda b, i: pm['bounds']['Cr'][i], initialize = pm['c_feed_dict'])


    # Create pm['stages'] as Pyomo Blocks
    model.stages = Block(pm['stages'], rule=lambda b: opt.stage_rule(b,pm))

    # 2D2S
    if config == '3s':
        def omb_0_rule(model):
            return model.stages[0].F0 == pm['F_feed']
        def pre_0_rule(model):
            return model.stages[0].p0 == model.p_feed
        def cmb_0_rule(model,s):
            return model.stages[0].F0 * model.stages[0].C0[s] == pm['F_feed'] * pm['c_feed_dict'][s]

        def omb_1_rule(model):
            return model.stages[1].F0 == model.stages[0].Fr
        def pre_1_rule(model):
            return model.stages[1].p0 == model.stages[0].pr
        def cmb_1_rule(model,s):
            return model.stages[1].F0 * model.stages[1].C0[s] == model.stages[0].Fr * model.stages[0].Cr[s]

        def omb_2_rule(model):
            return model.stages[2].F0 == model.stages[1].Fr
        def pre_2_rule(model):
            return model.stages[2].p0 == model.stages[1].pr
        def cmb_2_rule(model,s):
            return model.stages[2].F0 * model.stages[2].C0[s] == model.stages[1].Fr * model.stages[1].Cr[s]

        def final_mix_rule(model):
            return model.final_flow == model.stages[2].Fr
        def final_pressure_rule(model):
            return model.final_pressure == model.stages[2].pr
        def final_component_rule(model,s):
            return model.final_flow * model.final_concentration[s] == model.stages[2].Fr * model.stages[2].Cr[s]
        
        model.omb_0 = Constraint(rule=omb_0_rule)
        model.cmb_0 = Constraint(pm['solutes'],rule=cmb_0_rule)
        model.omb_1 = Constraint(rule=omb_1_rule)
        model.cmb_1 = Constraint(pm['solutes'],rule=cmb_1_rule)
        model.omb_2 = Constraint(rule=omb_2_rule)
        model.cmb_2 = Constraint(pm['solutes'],rule=cmb_2_rule)
        model.pre_0 = Constraint(rule=pre_0_rule)
        model.pre_1 = Constraint(rule=pre_1_rule)
        model.pre_2 = Constraint(rule=pre_2_rule)

    elif config == '1s':
        def omb_0_rule(model):
            return model.stages[0].F0 == pm['F_feed']
        def pre_0_rule(model):
            return model.stages[0].p0 == model.p_feed
        def cmb_0_rule(model,s):
            return model.stages[0].F0 * model.stages[0].C0[s] == pm['F_feed'] * pm['c_feed_dict'][s]
        
        def final_mix_rule(model):
            return model.final_flow == model.stages[0].Fr
        def final_pressure_rule(model):
            return model.final_pressure == model.stages[0].pr
        def final_component_rule(model,s):
            return model.final_flow * model.final_concentration[s] == model.stages[0].Fr * model.stages[0].Cr[s]
        
        model.omb_0 = Constraint(rule=omb_0_rule)
        model.cmb_0 = Constraint(pm['solutes'],rule=cmb_0_rule)
        model.pre_0 = Constraint(rule=pre_0_rule)
    else:
        print('Unknown config')
        return

    model.final_mix_constr = Constraint(rule=final_mix_rule)
    model.final_pressure_constr = Constraint(rule=final_pressure_rule)
    model.final_component_constr = Constraint(pm['solutes'],rule=final_component_rule)


    # Performance metrics
    def recovery_rule(model):
        return (model.final_flow * (model.final_concentration[0])) / (pm['F_feed'] * (pm['c_feed_dict'][0])) == model.recovery

    def solvent_recovery_rule(model):
        return 1 - (model.final_flow/(pm['F_feed'])) == model.solvent_recovery

    def separation_factor_rule(model):
        return model.separation_factor == ((model.final_concentration[0]) / (model.final_concentration[1])) / ((pm['c_feed_dict'][0]) / (pm['c_feed_dict'][1]))

    model.recovery_constr = Constraint(rule=recovery_rule)
    model.water_recovery_constr = Constraint(rule=solvent_recovery_rule)
    model.separation_factor_constr = Constraint(rule=separation_factor_rule)


    # Constraints on pressure exchanger and power
    if model_x == False:
        def pex_rule(model):
            return model.p_feed == model.p_pump

    else:
        def pex_rule(model):
            return model.p_feed == (pm['pex_eff'] * model.final_flow * model.final_pressure + pm['F_feed'] * model.p_pump)/pm['F_feed']
        
    def power_rule(model):
        return model.power == (1/pm['pump_eff']) * pm['F_feed'] * model.p_pump

    model.power_constr = Constraint(rule=power_rule)
    model.pex_constr = Constraint(rule=pex_rule)
    

    def molar_power_rule(model):
        return model.molar_power == model.power / model.recovery

    model.molar_power_rule = Constraint(rule=molar_power_rule)


    # Multi-objective constraints
    if 'recovery' in constraints:
        def mo_rec(model):
            return model.recovery >= constraints['recovery']

        model.mo_rec = Constraint(rule=mo_rec)

    if 'separation_factor' in constraints:
        def mo_sf_rule(model):
            return model.separation_factor >= constraints['separation_factor']

        model.mo_sf = Constraint(rule=mo_sf_rule)
    
    if 'molar_power' in constraints:
        def mo_sp_rule(model):
            return model.molar_power <= constraints['molar_power']

        model.mo_sp = Constraint(rule=mo_sp_rule)

    if 'final_concentration' in constraints:
        def mo_fc_rule(model):
            return model.final_concentration[0] >= constraints['final_concentration']

        model.mo_fc = Constraint(rule=mo_fc_rule)


    # Objective
    if objective =='separation_factor':
        model.obj = Objective(expr=(-(model.final_concentration[0]) / ((model.final_concentration[1]))))
    elif objective == 'recovery':
        model.obj = Objective(expr=(1-model.recovery))
    elif objective =='molar_power':
        model.obj = Objective(expr=(model.molar_power))
    elif objective == 'final_concentration':
        model.obj = Objective(expr=(-model.final_concentration[0]))

    return model


def model_xrd(constraints, objective, pm, membrane_model = 'sd', model_x = False, config='3s'):
    model = ConcreteModel()
    
    if membrane_model == 'sd':
        import models.optimization_sd as opt
    elif membrane_model == 'sdc':
        import models.optimization_sdc as opt
    elif membrane_model == 'pf':
        import models.optimization_pf as opt
    else:
        print('Unknown model')
        return

    opt.initialize_model(pm)
    # Problem-level variables

    model.recovery = Var(within=NonNegativeReals, bounds=(0,1), initialize = 0.0)
    model.separation_factor = Var(within=NonNegativeReals, bounds=(1,100), initialize = 1.0)
    model.solvent_recovery = Var(within=NonNegativeReals, bounds=(0,1), initialize = 0.0)
    model.p_pump = Var(within=NonNegativeReals, bounds = (0,pm['dp_max']), initialize = 0.0) # pressure generated by pump
    model.p_feed = Var(within=NonNegativeReals, bounds = (0,pm['dp_max']), initialize = 0.0) # pressure generated by pressure exchanger after the pump
    model.power = Var(within=NonNegativeReals, bounds = (0,2000), initialize = 1500.0) # power necessary
    model.molar_power = Var(within=NonNegativeReals, bounds = (0,3000), initialize = 1500.0)
    model.F_dilution = Var(pm['stages'],within=NonNegativeReals, bounds=(0,pm['F_dil_feed']), initialize = 0.0)
    model.omega = Var(pm['stages'],within=NonNegativeReals, bounds=(0,1), initialize = 1)
    model.recirc_power = Var(pm['stages'],within=NonNegativeReals, bounds=(0,2000), initialize = 0.0)
    model.power_dilution = Var(pm['stages'],within=NonNegativeReals, bounds = (0,2000), initialize = 0.0)

    model.final_flow = Var(within=NonNegativeReals, bounds=(0,pm['F_lim']), initialize = pm['F_feed'])
    model.final_pressure = Var(within=NonNegativeReals, bounds = (0,pm['dp_max']), initialize = 0.0)
    model.final_concentration = Var(pm['solutes'],within=NonNegativeReals, bounds = lambda b, i: pm['bounds']['Cr'][i], initialize = pm['c_feed_dict'])


    # Create pm['stages'] as Pyomo Blocks
    model.stages = Block(pm['stages'], rule=lambda b: opt.stage_rule(b,pm))


    # Define mass balance and pressure constraints for each membrane element in each stage
    if config == '3s':
        def omb_0_rule(model):
            return model.stages[0].F0 == pm['F_feed'] + model.F_dilution[0] + model.omega[1] * model.stages[1].Fp
        def pre_0_rule(model):
            return model.stages[0].p0 == model.p_feed
        def cmb_0_rule(model,s):
            return model.stages[0].F0 * model.stages[0].C0[s] == pm['F_feed'] * pm['c_feed_dict'][s] + model.omega[1] * model.stages[1].Fp * model.stages[1].Cp[s]

        def omb_1_rule(model):
            return model.stages[1].F0 == model.stages[0].Fr + model.F_dilution[1] + model.omega[2] * model.stages[2].Fp
        def pre_1_rule(model):
            return model.stages[1].p0 == model.stages[0].pr
        def cmb_1_rule(model,s):
            return model.stages[1].F0 * model.stages[1].C0[s] == model.stages[0].Fr * model.stages[0].Cr[s] + model.omega[2] * model.stages[2].Fp * model.stages[2].Cp[s]
        def recirc_1_rule(model):
            return model.recirc_power[1] == (1/pm['pump_eff']) * model.stages[1].Fp * model.omega[1] * (model.stages[0].p0 - model.stages[1].pp)

        def omb_2_rule(model):
            return model.stages[2].F0 == model.stages[1].Fr + model.F_dilution[2]
        def pre_2_rule(model):
            return model.stages[2].p0 == model.stages[1].pr
        def cmb_2_rule(model,s):
            return model.stages[2].F0 * model.stages[2].C0[s] == model.stages[1].Fr * model.stages[1].Cr[s]
        def recirc_2_rule(model):
            return model.recirc_power[2] == (1/pm['pump_eff']) * model.stages[2].Fp * model.omega[2] * (model.stages[1].p0 - model.stages[2].pp)

        def final_mix_rule(model):
            return model.final_flow == model.stages[2].Fr
        def final_pressure_rule(model):
            return model.final_pressure == model.stages[2].pr
        def final_component_rule(model,s):
            return model.final_flow * model.final_concentration[s] == model.stages[2].Fr * model.stages[2].Cr[s]
    else:
        print('Unknown config')
        return
    
    # Assignment
    model.omb_0 = Constraint(rule=omb_0_rule)
    model.cmb_0 = Constraint(pm['solutes'],rule=cmb_0_rule)
    model.omb_1 = Constraint(rule=omb_1_rule)
    model.cmb_1 = Constraint(pm['solutes'],rule=cmb_1_rule)
    model.omb_2 = Constraint(rule=omb_2_rule)
    model.cmb_2 = Constraint(pm['solutes'],rule=cmb_2_rule)
    model.pre_0 = Constraint(rule=pre_0_rule)
    model.pre_1 = Constraint(rule=pre_1_rule)
    model.pre_2 = Constraint(rule=pre_2_rule)

    model.recirc_const1 = Constraint(rule=recirc_1_rule)
    model.recirc_const3 = Constraint(rule=recirc_2_rule)

    model.final_mix_constr = Constraint(rule=final_mix_rule)
    model.final_pressure_constr = Constraint(rule=final_pressure_rule)
    model.final_component_constr = Constraint(pm['solutes'],rule=final_component_rule)


    # Performance metrics
    def recovery_rule(model):
        return (model.final_flow * (model.final_concentration[0])) / (pm['F_feed'] * (pm['c_feed_dict'][0])) == model.recovery

    def solvent_recovery_rule(model):
        return 1 - (model.final_flow/(pm['F_feed']+sum(model.F_dilution[i] for i in pm['stages']))) == model.solvent_recovery

    def separation_factor_rule(model):
        return model.separation_factor == ((model.final_concentration[0]) / (model.final_concentration[1])) / ((pm['c_feed_dict'][0]) / (pm['c_feed_dict'][1]))

    model.recovery_constr = Constraint(rule=recovery_rule)
    model.water_recovery_constr = Constraint(rule=solvent_recovery_rule)
    model.separation_factor_constr = Constraint(rule=separation_factor_rule)


    # Constraint on dilution
    def dilution_rule(model):
        return sum(model.F_dilution[i] for i in pm['stages']) <= pm['F_dil_feed']

    model.dilution_constr = Constraint(rule=dilution_rule)


    # Constraints on pressure exchanger and power
    if model_x == False:
        def pex_rule(model):
            return model.p_feed == model.p_pump

    else:
        def pex_rule(model):
            return model.p_feed == (pm['pex_eff'] * model.final_flow * model.final_pressure + pm['F_feed'] * model.p_pump)/pm['F_feed']

    def dilution_power_rule(model,st):
        return model.power_dilution[st] == (1/pm['pump_eff']) * model.F_dilution[st] * model.stages[st].p0
        
    def power_rule(model):
        return model.power == (1/pm['pump_eff']) * pm['F_feed'] * model.p_pump + model.recirc_power[1] +  model.recirc_power[2] + sum(model.power_dilution[i] for i in pm['stages'])
    
    def molar_power_rule(model):
        return model.molar_power == model.power / model.recovery
    
    model.power_constr = Constraint(rule=power_rule)
    model.dilution_power_constr = Constraint(pm['stages'],rule=dilution_power_rule)
    model.molar_power_rule = Constraint(rule=molar_power_rule)
    model.pex_constraint = Constraint(rule=pex_rule)



    # Multi-objective constraints
    if 'recovery' in constraints:
        def mo_rec(model):
            return model.recovery >= constraints['recovery']

        model.mo_rec = Constraint(rule=mo_rec)

    if 'separation_factor' in constraints:
        def mo_sf_rule(model):
            return model.separation_factor >= constraints['separation_factor']

        model.mo_sf = Constraint(rule=mo_sf_rule)
    
    if 'molar_power' in constraints:
        def mo_sp_rule(model):
            return model.molar_power <= constraints['molar_power']

        model.mo_sp = Constraint(rule=mo_sp_rule)

    if 'final_concentration' in constraints:
        def mo_fc_rule(model):
            return model.final_concentration[0] >= constraints['final_concentration']

        model.mo_fc = Constraint(rule=mo_fc_rule)


    # Objective
    if objective =='separation_factor':
        model.obj = Objective(expr=(-(model.final_concentration[0]) / ((model.final_concentration[1]))))
    elif objective == 'recovery':
        model.obj = Objective(expr=(1-model.recovery))
    elif objective =='molar_power':
        model.obj = Objective(expr=(model.molar_power))
    elif objective == 'final_concentration':
        model.obj = Objective(expr=(-model.final_concentration[0]))

    return model

## OPTIMIZATION AND EXTRACTION #################################################################################################################

def opti(model,solver = 'ipopt'):
    # # Solver call - IPOPT
    try:
        if solver == 'ipopt':
            with SolverFactory('ipopt') as opt:
                results = opt.solve(model, tee=True)
        elif solver == 'baron':
            with SolverFactory('baron') as opt:
                results = opt.solve(model, tee=True, options={"MaxTime": -1})

        if (results.solver.status == SolverStatus.ok) and (results.solver.termination_condition == TerminationCondition.optimal):
            optimal = 1
        else:
            optimal = 0
    except ValueError:
        results = {}
        optimal = 0

    return model, results, optimal


def extract_results(model,n_stages):
    results = {}
    try:
        results['p_feed'] = value(model.p_feed)
    except:
        pass
    try:
        results['p_pump'] = value(model.p_pump)
        results['p_feed'] = value(model.p_feed)
    except:
        pass

    p_perm = {}
    for i in range(n_stages):
        p_perm[i] = value(model.stages[i].pp)
    results['p_perm'] = p_perm

    try:
        omegas = {}
        for i in range(n_stages):
            omegas[i] = value(model.omega[i])
        results['rec_split'] = omegas
    except:
        pass

    try:
        dilutions = {}
        for i in range(n_stages):
            dilutions[i] = value(model.F_dilution[i])
        results['dilution'] = dilutions
    except:
        pass

    results['sep_factor'] = value(model.separation_factor)
    results['molar_power'] = value(model.molar_power)
    results['recovery'] = value(model.recovery)
    results['solvent_recovery'] = value(model.solvent_recovery)

    return results


def log_opti(model):
    import logging

    # Get the logger
    logger = logging.getLogger()

    # Set the logging level to include INFO messages
    logger.setLevel(logging.INFO)
    log_infeasible_constraints(model)

    pyomo.util.infeasible.log_close_to_bounds(model)