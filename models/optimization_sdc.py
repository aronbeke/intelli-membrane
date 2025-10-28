import numpy as np
from pyomo.environ import *

#########

def initialize_model(pm):
    #ns pm['solutes']
    #nk pm['coll']ocation
    #ne elements in a stage
    #nst pm['stages'] in total
    #nn pm['nodes'] in an element
    pm['A_node'] = pm['A'] / pm['nn']

    # FYI
    
    '''
    J_shape = ((pm['nk']-1),pm['ns']) #Solute flux matrix, each row is the same
    D_shape = (pm['nk']-1,pm['nk']) #Polynomial derivatives Vandermonde-matrix at x(j = 2 ... k)
    V_shape = (pm['nk'],pm['nk']) #Vandermonde-matrix at x(1 ... k)
    C_shape = (pm['nk'],pm['ns']) #Concentration polynomial coefficients
    C_reduced_shape = (pm['nk']-1,pm['ns'])
    '''

    V = np.vander(np.linspace(0,1,pm['nk']), pm['nk'], increasing=True)
    V_inv = np.linalg.inv(V)
    D_coeff = np.tile(np.array([range(pm['nk'])]),(pm['nk']-1,1))
    D_vander = np.hstack((np.zeros((pm['nk']-1,1)),np.vander(np.linspace(0,1,pm['nk']-1), pm['nk']-1, increasing=True)))
    D = np.multiply(D_coeff,D_vander)

    pm['V_sp'] = 0.5 * np.pi * (pm['df']**2) * pm['l_mesh']
    pm['V_tot'] = (pm['l_mesh']**2) * pm['h'] * np.sin(pm['theta'])
    pm['epsilon'] = 1 - (pm['V_sp']/pm['V_tot'])
    pm['S_vsp'] = 4 / pm['df']
    pm['dh'] = (4*pm['epsilon']) / ((2/pm['h']) + (1-pm['epsilon'])*pm['S_vsp'])
    pm['v_factor'] = 1 / (pm['b_env']*pm['h']*pm['epsilon']*pm['n_env'])
    pm['Re_factor'] = (pm['rho']*pm['dh'])/pm['eta']

    pm['L'] = np.array(pm['Li_list'])
    pm['K'] = np.array(pm['Ki_list'])

    pm['DVi_matrix'] = D @ V_inv

    # Sets
    pm['stages'] = range(pm['nst'])
    pm['elems'] = range(pm['ne']) # elements
    pm['red_elems'] = range(pm['ne']-1)
    pm['nodes'] = range(pm['nn'])
    pm['red_nodes'] = range(pm['nn']-1)
    pm['solutes'] = range(pm['ns'])
    pm['coll'] = range(pm['nk'])
    pm['red_coll'] = range(1,pm['nk'])
    # pm['stages'] = RangeSet(num_pm['stages'])

    # Initializations
    C_init = {}
    for i in pm['coll']:
        for j in pm['solutes']:
            C_init[(i,j)] = pm['c_feed_dict'][j]

    dCdx_init = {}
    for i in pm['red_coll']:
        for j in pm['solutes']:
            dCdx_init[(i,j)] = pm['c_feed_dict'][j]

    C_bound = {}
    for i in pm['solutes']:
        C_bound[i] = (0,1000)

    J_init = {}
    for i in pm['solutes']:
        J_init[i] = 0

    C_bound_mx = {}
    for i in pm['coll']:
        for j in pm['solutes']:
            C_bound_mx[(i,j)] = C_bound[j]

    dCdx_bound_mx = {}
    for i in pm['red_coll']:
        for j in pm['solutes']:
            dCdx_bound_mx[(i,j)] = (-C_bound[j][1]*10,C_bound[j][1]*10)

    J_bound = {}
    for i in pm['solutes']:
        J_bound[i] = (0,(pm['F_lim']*C_bound[i][1])/(pm['A_node']))

    pm['bounds'] = {'C': C_bound, 'Cr': C_bound, 'C_mx' : C_bound_mx, 'dCdx': dCdx_bound_mx, 'J': J_bound}
    pm['inits'] = {'C': C_init, 'dCdx': dCdx_init, 'J': J_init}



# BLOCK RULES ################################################################
    
def sdc_node_rule(b,pm):     
    b.beta = Var(within=NonNegativeReals, bounds = (0,3), initialize = 1.0)
    b.dp = Var(within=NonNegativeReals, bounds = (0,pm['dp_max']), initialize = 0)
    b.Fr = Var(within=NonNegativeReals, bounds = (0,pm['F_lim']), initialize = pm['F_feed'])
    b.Fp = Var(within=NonNegativeReals, bounds = (0,pm['F_lim']), initialize = 0)
    b.J = Var(pm['solutes'], within=NonNegativeReals, bounds = lambda b, i: pm['bounds']['J'][i], initialize = pm['inits']['J'])
    b.Jv = Var(within=NonNegativeReals, bounds = (0,pm['F_lim']/(pm['A_node'])), initialize = 0.0)
    b.C = Var(pm['coll'], pm['solutes'], within=NonNegativeReals, bounds = lambda b, i, j: pm['bounds']['C_mx'][(i,j)], initialize = pm['inits']['C'])
    b.F0 = Var(within=NonNegativeReals, bounds = (0,pm['F_lim']), initialize = pm['F_feed'])
    b.C0 = Var(pm['solutes'], within=NonNegativeReals, bounds = lambda b, i: pm['bounds']['C'][i], initialize = pm['c_feed_dict'])
    b.Cr = Var(pm['solutes'], within=NonNegativeReals, bounds = lambda b, i: pm['bounds']['C'][i], initialize = pm['c_feed_dict'])
    b.dCdx = Var(pm['red_coll'],pm['solutes'], within=Reals, bounds = lambda b, i, j: pm['bounds']['dCdx'][(i,j)],initialize = pm['inits']['dCdx'])
    
    def permeate_boundary_rule(b,s):
        return b.J[s] == b.Jv * b.C[pm['nk']-1,s]

    def solute_flux_rule(b, c, s):
        return b.J[s] == -pm['L'][s] * b.dCdx[c,s] + pm['K'][s] * b.C[c, s] * b.Jv

    def solvent_flux_rule(b):
        return b.Jv == pm['P'] * (b.dp - 1e-5*8.314*pm['T']*(sum(b.C[0,i] for i in pm['solutes']) - sum(b.C[pm['nk']-1,j] for j in pm['solutes'])))
    
    def concentration_polarization_rule(b,s):
        return b.C[0,s] == b.Cr[s] * b.beta + (1-b.beta) * b.C[pm['nk']-1,s]

    def solute_mass_balance_rule(b,s):
        return b.F0 * b.C0[s] == b.Fr * b.Cr[s] + b.Fp * b.C[pm['nk']-1,s]
    
    def overall_mass_balance_rule(b):
        return b.F0 == b.Fr + b.Fp
    
    def permeate_stream_rule(b):
        return b.Fp == b.Jv * pm['A_node']

    def concentration_derivative_rule(b,c,s):
        return b.dCdx[c,s] == sum(pm['DVi_matrix'][c-1][i] * b.C[i,s] for i in pm['coll'])
    
    # def numerical_stability(b):
    #     return sum(b.C[0,i] for i in pm['solutes']) - sum(b.C[pm['nk']-1,j] for j in pm['solutes']) >= 0
    
    b.permeate_boundary = Constraint(pm['solutes'], rule = permeate_boundary_rule)
    b.ion_flux = Constraint(pm['red_coll'],pm['solutes'], rule = solute_flux_rule)
    b.solvent_flux = Constraint(rule = solvent_flux_rule)
    b.solute_mass_balance = Constraint(pm['solutes'], rule = solute_mass_balance_rule)
    b.concentration_polarization = Constraint(pm['solutes'], rule = concentration_polarization_rule)
    b.overall_mass_balance = Constraint(rule = overall_mass_balance_rule)
    b.permeate_stream = Constraint(rule = permeate_stream_rule)
    b.c_derivative = Constraint(pm['red_coll'],pm['solutes'],rule = concentration_derivative_rule)
    b.permeance_const = Constraint(rule = permeance_rule)
    # b.stability_const = Constraint(rule = numerical_stability)

def element_rule(b,pm):

    b.p0 = Var(within=NonNegativeReals, bounds = (0,pm['dp_max']), initialize = pm['dp_max'])
    b.pp = Var(within=NonNegativeReals, bounds = (0,pm['dp_max']), initialize = pm['dp_max'])
    b.pdrop = Var(within=NonNegativeReals, bounds = (0,pm['dp_max']), initialize = 0)
    # b.dp = Var(within=NonNegativeReals, bounds = (0,pm['dp_max']), initialize = 0.0)
    b.beta = Var(within=NonNegativeReals, bounds = (0,3), initialize = 1.0)
    b.nodes = Block(pm['nodes'],rule=lambda b: sdc_node_rule(b,pm))
    b.F0 = Var(within=NonNegativeReals, bounds = (0,pm['F_lim']), initialize = pm['F_feed'])
    b.C0 = Var(pm['solutes'], within=NonNegativeReals, bounds = lambda b, i: pm['bounds']['C'][i], initialize = pm['c_feed_dict'])
    b.Fr = Var(within=NonNegativeReals, bounds = (0,pm['F_lim']), initialize = pm['F_feed'])
    b.Cr = Var(pm['solutes'], within=NonNegativeReals, bounds = lambda b, i: pm['bounds']['C'][i], initialize = pm['c_feed_dict'])
    b.Fp = Var(within=NonNegativeReals, bounds = (0,pm['F_lim']), initialize = 0)
    b.Cp = Var(pm['solutes'], within=NonNegativeReals, bounds = lambda b, i: pm['bounds']['C'][i], initialize = pm['c_feed_dict'])

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
        return b.Fp * b.Cp[s] == sum(b.nodes[i].Fp * b.nodes[i].C[pm['nk']-1,s] for i in pm['nodes'])
    
    # def node_pressure_rule(b,n):
    #     return b.dp == b.nodes[n].dp
    
    def node_pressure_rule(b,n):
        return b.nodes[n].dp == b.p0 - b.pp - b.pdrop
    
    def pressure_drop_rule(b):
        return b.pdrop == (1/100000)*(6.23*((pm['Re_factor']*pm['v_factor']*b.F0)**(-0.3))*pm['rho']*((pm['v_factor']*b.F0)**2)*pm['l_module']) / (2*pm['dh']*(3600**2))
    
    # def pressure_drop_rule(b):
    #     return b.pdrop == 0
    
    def polarization_rule(b):
        return b.beta == exp(pm['b0']*(b.Fp/b.F0))
    
    def beta_node_rule(b,n):
        return b.beta == b.nodes[n].beta
    
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
    
    b.polarization = Constraint(rule = polarization_rule)
    b.beta_node = Constraint(pm['nodes'],rule = beta_node_rule)


def stage_rule(b,pm):

    b.p0 = Var(within=NonNegativeReals, bounds = (0,pm['dp_max']), initialize = pm['dp_max'])
    b.pp = Var(within=NonNegativeReals, bounds = (0,pm['dp_max']), initialize = pm['dp_max'])
    b.pr = Var(within=NonNegativeReals, bounds = (0,pm['dp_max']), initialize = pm['dp_max'])
    # b.dp = Var(within=NonNegativeReals, bounds = (0,pm['dp_max']), initialize = 0.0)
    b.elems = Block(pm['elems'],rule=lambda b: element_rule(b,pm))
    b.F0 = Var(within=NonNegativeReals, bounds = (0,pm['F_lim']), initialize = pm['F_feed'])
    b.C0 = Var(pm['solutes'], within=NonNegativeReals, bounds = lambda b, i: pm['bounds']['C'][i], initialize = pm['c_feed_dict'])
    b.Fr = Var(within=NonNegativeReals, bounds = (0,pm['F_lim']), initialize = pm['F_feed'])
    b.Cr = Var(pm['solutes'], within=NonNegativeReals, bounds = lambda b, i: pm['bounds']['C'][i], initialize = pm['c_feed_dict'])
    b.Fp = Var(within=NonNegativeReals, bounds = (0,pm['F_lim']), initialize = 0)
    b.Cp = Var(pm['solutes'], within=NonNegativeReals, bounds = lambda b, i: pm['bounds']['C'][i], initialize = pm['c_feed_dict'])

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