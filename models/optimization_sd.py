import numpy as np
from pyomo.environ import *

#########

def initialize_model(pm):
    #ns pm['solutes']
    #ne elements in a stage
    #nst pm['stages'] in total
    #nn pm['nodes'] in an element
    pm['A_node'] = pm['A'] / pm['nn']

    pm['V_sp'] = 0.5 * np.pi * (pm['df']**2) * pm['l_mesh']
    pm['V_tot'] = (pm['l_mesh']**2) * pm['h'] * np.sin(pm['theta'])
    pm['epsilon'] = 1 - (pm['V_sp']/pm['V_tot'])
    pm['S_vsp'] = 4 / pm['df']
    pm['dh'] = (4*pm['epsilon']) / ((2/pm['h']) + (1-pm['epsilon'])*pm['S_vsp'])
    pm['v_factor'] = 1 / (pm['b_env']*pm['h']*pm['epsilon']*pm['n_env'])
    pm['Re_factor'] = (pm['rho']*pm['dh'])/pm['eta']

    pm['L'] = np.array(pm['Li_list'])
    pm['D'] = np.array(pm['Di_list'])
    # Sets
    pm['stages'] = range(pm['nst'])
    pm['elems'] = range(pm['ne']) # elements
    pm['red_elems'] = range(pm['ne']-1)
    pm['nodes'] = range(pm['nn'])
    pm['red_nodes'] = range(pm['nn']-1)
    pm['solutes'] = range(pm['ns'])

    # Initializations
    Cr_init = {}
    Cp_init = {}
    Cm_init = {}
    C0_init = {}
    for j in pm['solutes']:
        Cr_init[j] = pm['c_feed_dict'][j]
        Cp_init[j] = pm['c_feed_dict'][j]
        Cm_init[j] = pm['c_feed_dict'][j]
        C0_init[j] = pm['c_feed_dict'][j]

    Cr_bound = {}
    Cp_bound = {}
    Cm_bound = {}
    C0_bound = {}
    for i in pm['solutes']:
        Cr_bound[i] = (0,1000)
        Cp_bound[i] = (0,1000)
        Cm_bound[i] = (0,1000)
        C0_bound[i] = (0,1000)

    J_init = {}
    for i in pm['solutes']:
        J_init[i] = 0

    J_bound = {}
    for i in pm['solutes']:
        J_bound[i] = (0,(pm['F_lim']*Cr_bound[i][1])/(pm['A_node']))

    pm['bounds'] = {'Cr': Cr_bound, 'C0': C0_bound, 'Cp' : Cp_bound, 'Cm' : Cm_bound, 'J': J_bound}
    pm['inits'] = {'Cr': Cr_init, 'C0': C0_init, 'Cp': Cp_init, 'Cm': Cm_init, 'J': J_init}


# BLOCK RULES ################################################################
    
def sd_node_rule(b,pm):     
    b.P = Var(within=NonNegativeReals, bounds = (0,0.008), initialize = 0.004)
    b.k_mass = Var(pm['solutes'], within=NonNegativeReals, bounds = (0,10), initialize = 1.0)
    b.dp = Var(within=NonNegativeReals, bounds = (0,pm['dp_max']), initialize = 0)
    b.Fr = Var(within=NonNegativeReals, bounds = (0,pm['F_lim']), initialize = pm['F_feed'])
    b.Fp = Var(within=NonNegativeReals, bounds = (0,pm['F_lim']), initialize = 0)
    b.J = Var(pm['solutes'], within=NonNegativeReals, bounds = lambda b, i: pm['bounds']['J'][i], initialize = pm['inits']['J'])
    b.Jv = Var(within=NonNegativeReals, bounds = (0,pm['F_lim']/(pm['A_node'])), initialize = 0.0)
    b.F0 = Var(within=NonNegativeReals, bounds = (0,pm['F_lim']), initialize = pm['F_feed'])
    b.C0 = Var(pm['solutes'], within=NonNegativeReals, bounds = lambda b, i: pm['bounds']['C0'][i], initialize = lambda b, i: pm['inits']['C0'][i])
    b.Cm = Var(pm['solutes'], within=NonNegativeReals, bounds = lambda b, i: pm['bounds']['Cm'][i], initialize = lambda b, i: pm['inits']['Cm'][i])
    b.Cr = Var(pm['solutes'], within=NonNegativeReals, bounds = lambda b, i: pm['bounds']['Cr'][i], initialize = lambda b, i: pm['inits']['Cr'][i])
    b.Cp = Var(pm['solutes'], within=NonNegativeReals, bounds = lambda b, i: pm['bounds']['Cp'][i], initialize = lambda b, i: pm['inits']['Cp'][i])
    
    def permeate_boundary_rule(b,s):
        return b.J[s] == b.Jv * b.Cp[s]

    def solute_flux_rule(b, s):
        return b.J[s] == pm['L'][s] * (b.Cm[s] - b.Cp[s])

    def solvent_flux_rule(b):
        return b.Jv == b.P * (b.dp - 1e-5*8.314*pm['T']*(sum(b.Cm[i] for i in pm['solutes']) - sum(b.Cp[j] for j in pm['solutes'])))
    
    def concentration_polarization_rule(b,s):
        return b.Cm[s] == exp(b.Jv/b.k_mass[s]) * (b.Cr[s] - b.Cp[s]) + b.Cp[s]

    def solute_mass_balance_rule(b,s):
        return b.F0 * b.C0[s] == b.Fr * b.Cr[s] + b.Fp * b.Cp[s]
    
    def overall_mass_balance_rule(b):
        return b.F0 == b.Fr + b.Fp
    
    def permeate_stream_rule(b):
        return b.Fp == b.Jv * pm['A_node']
    
    def permeance_rule(b):
        return b.P == pm['P1'] * b.dp + pm['P0']
    
    # def numerical_stability(b):
    #     return sum(b.C[0,i] for i in pm['solutes']) - sum(b.C[pm['nk']-1,j] for j in pm['solutes']) >= 0
    
    b.permeate_boundary = Constraint(pm['solutes'], rule = permeate_boundary_rule)
    b.solute_flux = Constraint(pm['solutes'], rule = solute_flux_rule)
    b.solvent_flux = Constraint(rule = solvent_flux_rule)
    b.solute_mass_balance = Constraint(pm['solutes'], rule = solute_mass_balance_rule)
    b.concentration_polarization = Constraint(pm['solutes'], rule = concentration_polarization_rule)
    b.overall_mass_balance = Constraint(rule = overall_mass_balance_rule)
    b.permeate_stream = Constraint(rule = permeate_stream_rule)
    b.permeance_const = Constraint(rule = permeance_rule)
    # b.stability_const = Constraint(rule = numerical_stability)

def element_rule(b,pm):

    b.p0 = Var(within=NonNegativeReals, bounds = (0,pm['dp_max']), initialize = pm['dp_max'])
    b.pp = Var(within=NonNegativeReals, bounds = (0,pm['dp_max']), initialize = pm['dp_max'])
    b.pdrop = Var(within=NonNegativeReals, bounds = (0,pm['dp_max']), initialize = 0)
    # b.dp = Var(within=NonNegativeReals, bounds = (0,pm['dp_max']), initialize = 0.0)
    b.k_mass = Var(pm['solutes'],within=NonNegativeReals, bounds = (0,10), initialize = 1.0)
    b.nodes = Block(pm['nodes'],rule=lambda b: sd_node_rule(b,pm))
    b.F0 = Var(within=NonNegativeReals, bounds = (0,pm['F_lim']), initialize = pm['F_feed'])
    b.C0 = Var(pm['solutes'], within=NonNegativeReals, bounds = lambda b, i: pm['bounds']['C0'][i], initialize = pm['c_feed_dict'])
    b.Fr = Var(within=NonNegativeReals, bounds = (0,pm['F_lim']), initialize = pm['F_feed'])
    b.Cr = Var(pm['solutes'], within=NonNegativeReals, bounds = lambda b, i: pm['bounds']['Cr'][i], initialize = pm['c_feed_dict'])
    b.Fp = Var(within=NonNegativeReals, bounds = (0,pm['F_lim']), initialize = 0)
    b.Cp = Var(pm['solutes'], within=NonNegativeReals, bounds = lambda b, i: pm['bounds']['Cp'][i], initialize = pm['c_feed_dict'])

    def intermediary_retentate_rule(b,n):
        return b.nodes[n].Fr == b.nodes[n+1].F0

    def intermediary_retentate_concentration_rule(b,n,s):
        return b.nodes[n].Cr[s] == b.nodes[n+1].C0[s]
    
    def feed_rule(b):
        return b.F0 == b.nodes[0].F0
    
    def feed_concentration_rule(b,s):
        return b.C0[s] == b.nodes[0].C0[s]

    def retentate_rule(b):
        return b.Fr == b.nodes[pm['nn']-1].Fr

    def retentate_concentration_rule(b,s):
        return b.Cr[s] == b.nodes[pm['nn']-1].Cr[s]
    
    def permeate_rule(b):
        return b.Fp == sum(b.nodes[i].Fp for i in pm['nodes'])
    
    def permeate_concentration_rule(b,s):
        return b.Fp * b.Cp[s] == sum(b.nodes[i].Fp * b.nodes[i].Cp[s] for i in pm['nodes'])
    
    # def node_pressure_rule(b,n):
    #     return b.dp == b.nodes[n].dp
    
    def node_pressure_rule(b,n):
        return b.nodes[n].dp == b.p0 - b.pp - b.pdrop
    
    def pressure_drop_rule(b):
        return b.pdrop == (1/100000)*(6.23*((pm['Re_factor']*pm['v_factor']*b.F0)**(-0.3))*pm['rho']*((pm['v_factor']*b.F0)**2)*pm['l_module']) / (2*pm['dh']*(3600**2))
    
    # def pressure_drop_rule(b):
    #     return b.pdrop == 0

    if pm['fixed_k_mass']:
        def polarization_rule(b,s):
            return b.k_mass[s] == pm['k_mass_fix']
    else:
        def polarization_rule(b,s):
            return b.k_mass[s] == (pm['D'][s] * (0.065 * ((pm['eta']/(pm['rho']*pm['D'][s]))**(0.25)) * ((pm['Re_factor']*pm['v_factor']*b.F0)**(0.875)))) / pm['dh']
        
    def k_mass_node_rule(b,n,s):
        return b.k_mass[s] == b.nodes[n].k_mass[s]
    
    b.intermediary_retentate = Constraint(pm['red_nodes'], rule = intermediary_retentate_rule)
    b.intermediary_retentate_concentration = Constraint(pm['red_nodes'], pm['solutes'], rule = intermediary_retentate_concentration_rule)
    b.feed = Constraint(rule = feed_rule)
    b.feed_concentration = Constraint(pm['solutes'], rule = feed_concentration_rule)
    b.retentate = Constraint(rule = retentate_rule)
    b.retentate_concentration = Constraint(pm['solutes'], rule = retentate_concentration_rule)
    b.permeate = Constraint(rule = permeate_rule)
    b.permeate_concentration = Constraint(pm['solutes'], rule = permeate_concentration_rule)

    b.node_pressure = Constraint(pm['nodes'], rule = node_pressure_rule)
    b.pressure_drop = Constraint(rule = pressure_drop_rule)
    
    b.polarization = Constraint(pm['solutes'],rule = polarization_rule)
    b.k_mass_node = Constraint(pm['nodes'],pm['solutes'],rule = k_mass_node_rule)


def stage_rule(b,pm):

    b.p0 = Var(within=NonNegativeReals, bounds = (0,pm['dp_max']), initialize = pm['dp_max'])
    b.pp = Var(within=NonNegativeReals, bounds = (0,pm['dp_max']), initialize = pm['dp_max'])
    b.pr = Var(within=NonNegativeReals, bounds = (0,pm['dp_max']), initialize = pm['dp_max'])
    # b.dp = Var(within=NonNegativeReals, bounds = (0,pm['dp_max']), initialize = 0.0)
    b.elems = Block(pm['elems'],rule=lambda b: element_rule(b,pm))
    b.F0 = Var(within=NonNegativeReals, bounds = (0,pm['F_lim']), initialize = pm['F_feed'])
    b.C0 = Var(pm['solutes'], within=NonNegativeReals, bounds = lambda b, i: pm['bounds']['C0'][i], initialize = pm['c_feed_dict'])
    b.Fr = Var(within=NonNegativeReals, bounds = (0,pm['F_lim']), initialize = pm['F_feed'])
    b.Cr = Var(pm['solutes'], within=NonNegativeReals, bounds = lambda b, i: pm['bounds']['Cr'][i], initialize = pm['c_feed_dict'])
    b.Fp = Var(within=NonNegativeReals, bounds = (0,pm['F_lim']), initialize = 0)
    b.Cp = Var(pm['solutes'], within=NonNegativeReals, bounds = lambda b, i: pm['bounds']['Cp'][i], initialize = pm['c_feed_dict'])

    def intermediary_retentate_rule(b,n):
        return b.elems[n].Fr == b.elems[n+1].F0

    def intermediary_retentate_concentration_rule(b,n,s):
        return b.elems[n].Cr[s] == b.elems[n+1].C0[s]
    
    def feed_rule(b):
        return b.F0 == b.elems[0].F0
    
    def feed_concentration_rule(b,s):
        return b.C0[s] == b.elems[0].C0[s]

    def retentate_rule(b):
        return b.Fr == b.elems[pm['ne']-1].Fr

    def retentate_concentration_rule(b,s):
        return b.Cr[s] == b.elems[pm['ne']-1].Cr[s]
    
    def permeate_rule(b):
        return b.Fp == sum(b.elems[i].Fp for i in pm['elems'])
    
    def permeate_concentration_rule(b,s):
        return b.Fp * b.Cp[s] == sum(b.elems[i].Fp * b.elems[i].Cp[s] for i in pm['elems'])
    
    # def elem_pressure_rule(b,n):
    #     return b.dp == b.elems[n].dp
    
    def intermediary_pressure_rule(b,n):
        return b.elems[n+1].p0 == b.elems[n].p0 - b.elems[n].pdrop

    def feed_pressure_rule(b):
        return b.p0 == b.elems[0].p0

    def retentate_pressure_rule(b):
        return b.pr == b.elems[pm['ne']-1].p0 - b.elems[pm['ne']-1].pdrop
    
    def permeate_pressure_rule(b,n):
        return b.pp == b.elems[n].pp
    
    b.intermediary_retentate = Constraint(pm['red_elems'], rule = intermediary_retentate_rule)
    b.intermediary_retentate_concentration = Constraint(pm['red_elems'], pm['solutes'], rule = intermediary_retentate_concentration_rule)
    b.feed = Constraint(rule = feed_rule)
    b.feed_concentration = Constraint(pm['solutes'], rule = feed_concentration_rule)
    b.retentate = Constraint(rule = retentate_rule)
    b.retentate_concentration = Constraint(pm['solutes'], rule = retentate_concentration_rule)
    b.permeate = Constraint(rule = permeate_rule)
    b.permeate_concentration = Constraint(pm['solutes'], rule = permeate_concentration_rule)

    b.feed_pressure = Constraint(rule = feed_pressure_rule)
    b.intermediary_pressure = Constraint(pm['red_elems'], rule = intermediary_pressure_rule)
    b.retentate_pressure = Constraint(rule = retentate_pressure_rule)
    b.permeate_pressure = Constraint(pm['elems'],rule = permeate_pressure_rule)
    # b.elem_pressure = Constraint(pm['elems'], rule = elem_pressure_rule)
    