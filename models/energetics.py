import numpy as np
import pandas as pd
import math
from scipy.integrate import odeint
from scipy.optimize import fsolve
import rdkit.Chem
import rdkit.Chem.Crippen
import rdkit.Chem.AllChem
import rdkit.Chem.Descriptors

# ALL CONCENTRATIONS AND CONCENTRATION RATIOS ARE MOLAR CONCENTRATIONS AND RATIOS
# ALL SI

### HARD-CODED DATA and PARAMETER INITIALIZATION

def initiate_separation_parameters(solvent,permeance,heat_integration_efficiency, sep_type):

    pm = {
        'A': 40.0,  # m2,# module area
        'p0': 3e6, # Pa  # feed pressure
        'l_module': 0.97,  # m # module length
        'l_mesh': 0.00327,  # m # spacer mesh size
        'df': 0.00033,  # m # filament thickness
        'theta': 1.83259571,  # rad # spacer angle
        'n_env': 28, # number of envelopes
        'b_env': 0.85,  # m # envelope width
        'h': 0.0007,  # m # spacer height 
        'T': 303.0,  # K # temperature
        'pump_eff': 0.85,  # pump efficiency
        'stage_cut': 0.75, # approx. stage cut on a separation stage
        'evap_eff': 2.6, # multiple-effect evaporator efficiency
        'n_elements': 3, # elements/modules per stage
        'solute_density': 1300, # kg/m3 assumed
        'nn': 10, # number of simulation nodes per module
        'R': [], # rejection
        'L': [], #m3/m2sPa # solute permeance
        'M': [], #kg/mol # solute molar mass
        'logPhi_water': [],
        'logPhi_heptane': [],
        'nu': [], #m3/mol # solute molar volume 
        'solubility': [], # mol /m3
        'D': [], #m2/s # solute diffusivity
    }
    
    pm['P'] = permeance #m3/m2sPa
    pm['solvent'] = solvent
    pm['heat_integration_eff'] = heat_integration_efficiency
    if sep_type == 'impurity_removal':
        pm['ns'] = 2
    else:
        pm['ns'] = 1
    pm['solvent_recovery'] = False

    return pm


# def solvent_property(property,solvent,solvent_data_path='data/appdata/solvents_appcat.csv'):
#     property_df = pd.read_csv(solvent_data_path)
#     return property_df[property_df['solvent'] == solvent][property].iloc[0]

def solvent_from_smiles(solvent):
    solvent_smiles = {
        'CCO': 'Ethanol',
        'CCCCCCC': 'Heptane',
        'O': 'Water',
        'CC(OCC)=O': 'Ethyl acetate',
        'CO': 'Methanol',
        'CCCCCC': 'Hexane',
        'N#CC': "Acetonitrile",
        'CC(C)=O': "Acetone",
        'O=CN(C)C': 'Dimethylformamide',
        'ClCCl': 'Dichloromethane',
        'CC1CCCO1': '2-Methyltetrahydrofuran',
        'CC1=CC=CC=C1': 'Toluene',
        'CC(O)C': 'Isopropanol',
        'CC(N(C)C)=O': 'Dimethylacetamide',
        'CC(CC)=O': 'Methyl ethyl ketone',
        'CC(C)(C)OC': 'Methyl tert-butyl ether',
        'CC#N': 'Acetonitrile',
        'C1CCCO1': 'Tetrahydrofuran',
        'C1CCCCC1': 'Cyclohexane'
    }
    return solvent_smiles[solvent]

### MOLECULAR CALCULATIONS #####################################################################


def logp_from_smiles(smiles):
    new_mol=rdkit.Chem.MolFromSmiles(smiles)
    val = rdkit.Chem.Crippen.MolLogP(new_mol)
    return val

def calculate_spherical_radius_from_smiles(smiles):
    mol = rdkit.Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    # Compute 3D coordinates
    mol = rdkit.Chem.AddHs(mol)
    rdkit.Chem.AllChem.EmbedMolecule(mol, rdkit.Chem.AllChem.ETKDG())
    rdkit.Chem.AllChem.UFFOptimizeMolecule(mol)

    volume = rdkit.Chem.AllChem.ComputeMolVolume(mol)

    radius = ((3 * volume) / (4 * np.pi))**(1/3)

    # returns radius in Angstrom
    return radius

def molar_mass_from_smiles(smiles):
    # return molar mass in kg/mol
    return rdkit.Chem.Descriptors.ExactMolWt(rdkit.Chem.MolFromSmiles(smiles)) / 1000

def diffusivity_from_smiles(smiles, viscosity, T=298):
    # Stokes-Einstein–Sutherland equation
    # returns m2/s if viscosity in Pa s

    kB = 1.380649e-23 # J⋅K−1 

    try:
        r = calculate_spherical_radius_from_smiles(smiles) * 1e-10 # m

        D = kB * T / (6 * np.pi * viscosity * r)
    except:
        print('Diffusivity exception')
        D = 1e-9 

    return D
    
def solute_permeance_from_rejection(rejection, molar_volume, solvent_permeance, dp):
    # neglecting osmotic pressure
    # SI units
    Rg = 8.314
    T = 298
    L = (solvent_permeance * dp * (1-rejection)) / (1 + np.exp((-molar_volume/(Rg*T))*dp) * (rejection - 1))
    return L

def rejection_from_solute_permeance(solute_permeance, molar_volume, solvent_permeance, dp):
    Rg = 8.314
    T = 298
    rejection = (solvent_permeance * dp + solute_permeance*(np.exp((-molar_volume/(Rg*T))*dp) - 1)) / (solvent_permeance * dp + solute_permeance*(np.exp((-molar_volume/(Rg*T))*dp)))
    return rejection

### MECHANISTIC MODEL EQUATIONS #############################################################################

def spiral_wound_mass_transfer(F0,D_list,h,rho,eta,l_mesh,df,theta,n_env,b_env):
    '''
    Calculates mass transfer coefficients
    '''

    V_sp = 0.5 * np.pi * (df**2) * l_mesh
    V_tot = (l_mesh**2) * h * np.sin(theta)
    epsilon = 1 - (V_sp/V_tot)
    S_vsp = 4 / df
    dh = (4*epsilon) / (2*(1/h) + (1-epsilon)*S_vsp)
    v = F0 / (b_env*h*epsilon*n_env)
    Re = (rho*v*dh)/eta

    k_list = []
    for D in D_list:
        Sc = eta/(rho*D)
        Sh = 0.065*(Sc ** 0.25)*(Re ** 0.875)
        k = (D*Sh)/dh
        k_list.append(k)

    return k_list


def csd_ternary(p,args):
    '''
    Simulating one node with classical solution-diffusion
    '''
    F0,c0_list,A,P,L_list,p0,pp,nu_list,k_list,ns = args

    Rg = 8.314
    T = 298.15
    
    Fr = p[0]
    Fp = p[1]
    cr_list = p[2:(2+ns)]
    cm_list = p[(2+ns):(2+2*ns)]
    cp_list = p[(2+2*ns):(2+3*ns)]

    cr, cm, cp, c0 = np.transpose(np.array(cr_list)), np.transpose(np.array(cm_list)), np.transpose(np.array(cp_list)), np.transpose(np.array(c0_list))
    L, nu, k = np.transpose(np.array(L_list)), np.transpose(np.array(nu_list)), np.transpose(np.array(k_list))

    netdp = (p0 - pp - Rg*T*(np.sum(cm) - np.sum(cp)))

    #####
    eq_flux = L * A * (cm - cp*np.exp((-(1/(Rg*T))*nu)*netdp)) - Fp* cp

    conc_pol = (cm - cp) - np.exp(np.divide((Fp/A),k))*(cr - cp)

    eq_solvent_flow = P * A * netdp - Fp

    eq_omb = Fr + Fp - F0

    eq_mb = F0*c0 - Fr*cr - Fp*cp
    ######

    eq = []
    eq.extend(eq_flux.ravel().tolist())
    eq.extend(conc_pol.ravel().tolist())
    eq.append(float(eq_solvent_flow))
    eq.append(float(eq_omb))
    eq.extend(eq_mb.ravel().tolist())

    return eq


def sd_mesh_module(parameters, constants):
    '''
    Simulating one separation module
    '''
    F0, c0, A, T, nn, ns, p0, pp = parameters
    L_list, nu_list, solubility_list, D_list, P, h, rho, l_module, eta, l_mesh, df,theta,n_env,b_env = constants

    k_list = spiral_wound_mass_transfer(F0,D_list,h,rho,eta,l_mesh,df,theta,n_env,b_env)

    A_bin = A / nn

    F0_bin = F0
    c0_bin = c0

    Cp_matrix = np.zeros((ns,nn))
    Fp_vector = np.zeros((nn,1))

    for i in range(nn):
        Fp_init = P*A_bin*(p0-pp)
        Fr_init = F0_bin - Fp_init
        cr_init = (np.array(c0_bin)).tolist()
        cm_init = (np.array(c0_bin)*1.1).tolist()
        cp_init = (np.array(c0_bin)*0.9).tolist()
        init = [Fr_init,Fp_init]
        init.extend(cr_init)
        init.extend(cm_init)
        init.extend(cp_init)

        args = [F0_bin,c0_bin,A_bin,P,L_list,p0,pp,nu_list,k_list,ns]
        # print(F0_bin,c0_bin)
        sol = fsolve(csd_ternary, init, args=args)
        
        Fr_bin = sol[0]
        Fp_bin = sol[1]
        cr_bin = sol[2:(2+ns)]
        cm_bin = sol[(2+ns):(2+2*ns)]
        cp_bin = sol[(2+2*ns):(2+3*ns)]
        cp = np.array(cp_bin)

        Cp_matrix[:,i] = cp.flatten()
        Fp_vector[i,0] = Fp_bin
        F0_bin = Fr_bin
        c0_bin = cr_bin
    
    Fr = Fr_bin
    cr_final = cr_bin
    Fp = np.sum(Fp_vector)
    cp = (Cp_matrix @ Fp_vector) / Fp
    cp_final = cp_bin

    els = {}
    els['Fr'] = Fr
    els['cr'] = cr_final
    els['F0'] = F0
    els['c0'] = c0
    els['p0'] = p0
    els['Fp'] = Fp
    els['cp'] = cp_final
    els['pp'] = pp
    return els


def sd_stage_sim(n_elements,parameters, constants):
    '''
    Simulating one separation stage consisting of n_elements modules
    '''
    F0, c_feed, A, T, nn, ns, p0, pp = parameters
    L_list, nu_list, solubility_list, D_list, P, h, rho, l_module, eta, l_mesh, df,theta,n_env,b_env = constants

    solutions = []
    stage_solutions = {}
    
    parameters = [F0, c_feed, A, T, nn, ns, p0, pp]
    els = sd_mesh_module(parameters, constants)
    solutions.append(els)

    for i in range(n_elements-1):
        parameters = [solutions[-1]['Fr'], solutions[-1]['cr'], A, T, nn, ns, p0, pp]
        els = sd_mesh_module(parameters, constants)
        solutions.append(els)

    Fp_final = np.sum(i['Fp'] for i in solutions)
    cp_final = []
    for i in range(len(c_feed)):
        cp_final.append(np.sum(j['Fp']*j['cp'][i] for j in solutions)/Fp_final)

    stage_solutions['Fr'] = solutions[-1]['Fr']
    stage_solutions['cr'] = solutions[-1]['cr']
    stage_solutions['F0'] = F0
    stage_solutions['c0'] = c_feed
    stage_solutions['p0'] = p0
    stage_solutions['Fp'] = Fp_final
    stage_solutions['cp'] = cp_final
    stage_solutions['pp'] = pp
    stage_solutions['elements'] = solutions
    return stage_solutions


#### NANOFILTRATION ENERGY CALCULATIONS ##############################################################################################


def check_concentration(c_actual, c_target, pm):
    '''
    Concentration target check
    '''
    if not pm['solvent_recovery']:
        return c_actual < c_target
    else:
        return c_actual > c_target


def targeted_binary_retentate_nf_cascade(c0,ctarget,pm,index=0):
    '''
    Considers only first solute of list
    Molar energy in J/mol
    '''

    ns = 1
    stage_cut, A, T, nn, n_elements, pump_eff = pm['stage_cut'], pm['A'], pm['T'], pm['nn'], pm['n_elements'], pm['pump_eff']
    L_list, nu_list, solubility_list, D_list, P = [pm['L'][index]], [pm['nu'][index]], [pm['solubility'][index]], [pm['D'][index]], pm['P']
    h, l_module, l_mesh, df,theta,n_env,b_env = pm['h'], pm['l_module'], pm['l_mesh'], pm['df'], pm['theta'], pm['n_env'], pm['b_env']
    rho = pm['density']
    eta = pm['viscosity']
    constants = [L_list, nu_list, solubility_list, D_list, P, h, rho, l_module, eta, l_mesh, df,theta,n_env,b_env]
    stages = {}

    if c0 == 0:
        return float('inf'), 0, float('inf'), float('inf'), stages

    if (ctarget <= c0 and not pm['solvent_recovery']) or (ctarget >= c0 and pm['solvent_recovery']):
        return 0, 1, 0, 0, stages

    # calculate approx. feed flow
    p0 = pm['p0'] # Pa
    pp = 0
    
    F0_ref = P*A*n_elements*(p0-pp) / stage_cut
    F0 = F0_ref
    c0_actual = c0

    n_stages = 0
    parallel = []

    try:
        while check_concentration(c0_actual, ctarget, pm) and n_stages <= 19:
            parameters = [F0, [c0_actual], A, T, nn, ns, p0, pp]
            mod_results = sd_stage_sim(n_elements,parameters, constants)
            stages[n_stages] = mod_results
            c0_actual = mod_results['cr'][0]
            Fr_actual = mod_results['Fr']

            if check_concentration(c0_actual, ctarget, pm):
                parallel_factor = round(F0_ref/Fr_actual)
            else:
                parallel_factor = 1
            parallel.append(parallel_factor)

            F0 = parallel_factor*Fr_actual

            n_stages += 1
    except RuntimeWarning:
        return float('inf'), 0, float('inf'), float('inf'), stages
    
    if n_stages == 20 and check_concentration(c0_actual, ctarget, pm):
        return float('inf'), 0, float('inf'), float('inf'), stages
    
    for i in range(n_stages):
        stages[i]['no_in_parallel'] = np.prod(parallel[i:])
    
    power = (1/pump_eff) * np.prod(parallel) * p0 * F0_ref
    total_area = A * n_elements* np.sum(stages[i]['no_in_parallel'] for i in range(n_stages))

    if not pm['solvent_recovery']:
        time = 1/(stages[n_stages-1]['Fr']*stages[n_stages-1]['cr'][0]) # for 1 mol
        molar_energy_demand = power*time
        recovery = (stages[n_stages-1]['Fr'] * stages[n_stages-1]['cr'][0]) / (F0_ref * c0 * np.prod(parallel))
    else:
        time = 1/(stages[n_stages-1]['Fr']) # for 1 m3
        molar_energy_demand = power*time
        recovery = (stages[n_stages-1]['Fr']) / (F0_ref * np.prod(parallel))

    return molar_energy_demand, recovery, total_area, n_stages, stages

        
def targeted_binary_permeate_nf_cascade(c0,ctarget,pm,index=0):
    '''
    Only one solute
    '''

    ns = 1
    stage_cut, A, T, nn, n_elements, pump_eff = pm['stage_cut'], pm['A'], pm['T'], pm['nn'], pm['n_elements'], pm['pump_eff']
    L_list, nu_list, solubility_list, D_list, P = [pm['L'][index]], [pm['nu'][index]], [pm['solubility'][index]], [pm['D'][index]], pm['P']
    h, l_module, l_mesh, df,theta,n_env,b_env = pm['h'], pm['l_module'], pm['l_mesh'], pm['df'], pm['theta'], pm['n_env'], pm['b_env']
    rho = pm['density']
    eta = pm['viscosity']
    constants = [L_list, nu_list, solubility_list, D_list, P, h, rho, l_module, eta, l_mesh, df,theta,n_env,b_env]

    stages = {}

    if c0 == 0:
        return float('inf'), 0, float('inf'), float('inf'), stages

    if (ctarget <= c0 and not pm['solvent_recovery']) or (ctarget >= c0 and pm['solvent_recovery']):
        return 0, 1, 0, 0, stages

    # calculate approx. feed flow / pressure
    p0 = pm['p0']
    pp = 0
    
    F0_ref = P*A*n_elements*(p0-pp) / stage_cut
    F0 = F0_ref
    c0_actual = c0

    n_stages = 0
    parallel = []

    # warnings.simplefilter('error', RuntimeWarning)

    try:
        while check_concentration(c0_actual, ctarget, pm) and n_stages <= 19:
            parameters = [F0, [c0_actual], A, T, nn, ns, p0, pp]
            mod_results = sd_stage_sim(n_elements,parameters, constants)
            stages[n_stages] = mod_results
            c0_actual = mod_results['cp'][0]
            Fp_actual = mod_results['Fp']

            if check_concentration(c0_actual, ctarget, pm):
                parallel_factor = round(F0_ref/Fp_actual)
            else:
                parallel_factor = 1
            parallel.append(parallel_factor)

            F0 = parallel_factor*Fp_actual

            n_stages += 1
    except RuntimeWarning:
        return float('inf'), 0, float('inf'), float('inf'), stages

    if n_stages == 20 and check_concentration(c0_actual, ctarget, pm):
        return float('inf'), 0, float('inf'), float('inf'), stages

    for i in range(n_stages):
        stages[i]['no_in_parallel'] = np.prod(parallel[i:])
    
    power = (1/pump_eff) * np.sum(stages[i]['no_in_parallel'] * stages[i]['F0'] for i in range(n_stages)) * p0
    total_area = A * n_elements* np.sum(stages[i]['no_in_parallel'] for i in range(n_stages))

    if not pm['solvent_recovery']:
        time = 1/(stages[n_stages-1]['Fp']*stages[n_stages-1]['cp'][0]) # for 1 mol
        molar_energy_demand = power*time
        recovery = (stages[n_stages-1]['Fp'] * stages[n_stages-1]['cp'][0]) / (F0_ref * c0 * np.prod(parallel))
    else:
        time = 1/(stages[n_stages-1]['Fp']) # for 1 m3
        molar_energy_demand = power*time
        recovery = (stages[n_stages-1]['Fp']) / (F0_ref * np.prod(parallel))

    return molar_energy_demand, recovery, total_area, n_stages, stages
    

def targeted_solute_concentration_retentate_cascade(F0,c0,ctarget,pm,index=0):
    '''
    Considers only first solute of list
    '''

    assert not pm['solvent_recovery']

    ns = 1
    stage_cut, A, T, nn, n_elements, pump_eff = pm['stage_cut'], pm['A'], pm['T'], pm['nn'], pm['n_elements'], pm['pump_eff']
    L_list, nu_list, solubility_list, D_list, P = [pm['L'][index]], [pm['nu'][index]], [pm['solubility'][index]], [pm['D'][index]], pm['P']
    h, l_module, l_mesh, df,theta,n_env,b_env = pm['h'], pm['l_module'], pm['l_mesh'], pm['df'], pm['theta'], pm['n_env'], pm['b_env']
    Rej = pm['R'][index]
    rho = pm['density']
    eta = pm['viscosity']
    
    stages = {}

    if c0 == 0:
        return float('inf'), 0, float('inf'), float('inf'), stages

    if (ctarget <= c0 and not pm['solvent_recovery']) or (ctarget >= c0 and pm['solvent_recovery']):
        return 0, 1, 0, 0, stages

    p0 = pm['p0'] # Pa
    pp = 0
    
    F0_actual = F0
    c0_actual = c0

    n_stages = 0
    total_area = 0

    try:
        while check_concentration(c0_actual, ctarget, pm) and n_stages <= 19:
            # determine stage cut
            
            theta = min((1/Rej) * (1 - c0_actual/ctarget), 0.8)
            A_current = min(A, (F0_actual*theta)/(P*(p0-pp)))

            parameters = [F0_actual, [c0_actual], A_current, T, nn, ns, p0, pp]
            constants = [L_list, nu_list, solubility_list, D_list, P, h, rho, l_module * (A_current/A), eta, l_mesh, df,theta,n_env,b_env]

            mod_results = sd_stage_sim(n_elements,parameters, constants)
            stages[n_stages] = mod_results
            c0_actual = mod_results['cr'][0]
            F0_actual = mod_results['Fr']

            n_stages += 1
            total_area += A_current

            print(c0_actual, F0_actual, n_stages, total_area)

            if check_concentration(c0_actual, ctarget, pm) and (c0_actual*F0_actual)/(c0*F0) <= 0.01:
                "Too low recovery, return infeasibility"
                return float('inf'), 0, float('inf'), float('inf'), stages

    except RuntimeWarning:
        return float('inf'), 0, float('inf'), float('inf'), stages
    
    if n_stages == 20 and check_concentration(c0_actual, ctarget, pm):
        return float('inf'), 0, float('inf'), float('inf'), stages
    
    power = (1/pump_eff) * p0 * F0
    time = 1/(stages[n_stages-1]['Fr']*stages[n_stages-1]['cr'][0]) # for 1 mol
    molar_energy_demand = power*time
    recovery = (stages[n_stages-1]['Fr'] * stages[n_stages-1]['cr'][0]) / (F0 * c0)

    return molar_energy_demand, recovery, total_area, n_stages, stages


def targeted_solute_concentration_permeate_cascade(F0,c0,ctarget,pm,index=0):
    '''
    Considers only first solute of list
    '''

    assert not pm['solvent_recovery']

    ns = 1
    stage_cut, A, T, nn, n_elements, pump_eff = pm['stage_cut'], pm['A'], pm['T'], pm['nn'], pm['n_elements'], pm['pump_eff']
    L_list, nu_list, solubility_list, D_list, P = [pm['L'][index]], [pm['nu'][index]], [pm['solubility'][index]], [pm['D'][index]], pm['P']
    h, l_module, l_mesh, df,theta,n_env,b_env = pm['h'], pm['l_module'], pm['l_mesh'], pm['df'], pm['theta'], pm['n_env'], pm['b_env']
    Rej = pm['R'][index]
    rho = pm['density']
    eta = pm['viscosity']
    stages = {}

    if c0 == 0:
        return float('inf'), 0, float('inf'), float('inf'), stages

    if (ctarget <= c0 and not pm['solvent_recovery']) or (ctarget >= c0 and pm['solvent_recovery']):
        return 0, 1, 0, 0, stages

    p0 = pm['p0'] # Pa
    pp = 0
    
    F0_actual = F0
    c0_actual = c0

    n_stages = 0
    total_area = 0

    try:
        while check_concentration(c0_actual, ctarget, pm) and n_stages <= 19:
            # determine stage cut
            
            theta = max((1/Rej) * (1 - (1-Rej)*(c0_actual/ctarget)), 0.2)
            A_current = min(A, (F0_actual*theta)/(P*(p0-pp)))

            parameters = [F0_actual, [c0_actual], A_current, T, nn, ns, p0, pp]
            constants = [L_list, nu_list, solubility_list, D_list, P, h, rho, l_module * (A_current/A), eta, l_mesh, df,theta,n_env,b_env]

            mod_results = sd_stage_sim(n_elements,parameters, constants)
            stages[n_stages] = mod_results
            c0_actual = mod_results['cp'][0]
            F0_actual = mod_results['Fp']

            n_stages += 1
            total_area += A_current

            if check_concentration(c0_actual, ctarget, pm) and (c0_actual*F0_actual)/(c0*F0) <= 0.01:
                "Too low recovery, return infeasibility"
                return float('inf'), 0, float('inf'), float('inf'), stages

    except RuntimeWarning:
        return float('inf'), 0, float('inf'), float('inf'), stages
    
    if n_stages == 20 and check_concentration(c0_actual, ctarget, pm):
        return float('inf'), 0, float('inf'), float('inf'), stages
    
    power = (1/pump_eff) * np.sum(stages[i]['F0'] for i in range(n_stages)) * p0
    time = 1/(stages[n_stages-1]['Fp']*stages[n_stages-1]['cp'][0]) # for 1 mol
    molar_energy_demand = power*time
    recovery = (stages[n_stages-1]['Fp'] * stages[n_stages-1]['cp'][0]) / (F0 * c0)

    return molar_energy_demand, recovery, total_area, n_stages, stages


### TERNARY

def targeted_ternary_retentate_nf_cascade(cfeed,cratio_target,pm):
    '''
    Separation of two solutes.
    Model algorithm keeps input flow rate of each stage constant.
    '''
    # Assign parameters

    ns = 2
    stage_cut, A, T, nn, n_elements, pump_eff = pm['stage_cut'], pm['A'], pm['T'], pm['nn'], pm['n_elements'], pm['pump_eff']
    L_list, nu_list, solubility_list, D_list, P = pm['L'], pm['nu'], pm['solubility'], pm['D'], pm['P']
    h, l_module, l_mesh, df,theta,n_env,b_env = pm['h'], pm['l_module'], pm['l_mesh'], pm['df'], pm['theta'], pm['n_env'], pm['b_env']
    rho = pm['density']
    eta = pm['viscosity']
    constants = [L_list, nu_list, solubility_list, D_list, P, h, rho, l_module, eta, l_mesh, df,theta,n_env,b_env]

    # Determine which solute will be enriched
    stages = {}
    if pm['R'][0] > pm['R'][1]:
        def c_ratio(c):
            return c[0]/c[1]
    elif pm['R'][0] < pm['R'][1]:
        def c_ratio(c):
            return c[1]/c[0]
        
    # Catch if separation is impossible
    if L_list[0] == L_list[1] or pm['R'][0] == pm['R'][1]:
        return float('inf'), 0, float('inf'), float('inf'), 0, stages

    # Catch if separation is unnecessary
    if cratio_target < c_ratio(cfeed):
        if L_list[0] < L_list[1]:
            c_final = cfeed[0]
        elif L_list[0] > L_list[1]:
            c_final = cfeed[1]
        return 0, 1, 0, 0, c_final, stages

    # Calculate stage feed flow based on approximate stage cut
    p0 = pm['p0']
    pp = 0
    
    F0_ref = P*A*n_elements*(p0-pp) / stage_cut
    F0 = F0_ref

    c0 = cfeed.copy()
    c0_actual = c0.copy()

    # simulating process
    n_stages = 0
    # warnings.simplefilter('error', RuntimeWarning)

    conc_lost = False

    # print('STARTING TERNARY NANOFILTRATION SIMULATION',solubility_list)

    try:
        while c_ratio(c0_actual) < cratio_target and n_stages <= 19:
            # print()
            # print('sim params: ',F0,c0_actual)
            parameters = [F0, c0_actual, A, T, nn, ns, p0, pp]  
            mod_results = sd_stage_sim(n_elements,parameters, constants)

            #Check if we have surpassed solubility limits
            solution_factors = [mod_results['cr'][0]/solubility_list[0], mod_results['cr'][1]/solubility_list[1]]

            if (solution_factors[0] >= 0.95 or solution_factors[1] >= 0.95) and n_stages == 0:
                c0[0] = c0[0] * 0.5
                c0[1] = c0[1] * 0.5
                c0_actual = c0.copy()
            else:
                stages[n_stages] = mod_results
                if c_ratio(mod_results['cr']) < cratio_target:
                    c0_actual[0] = (mod_results['cr'][0] * mod_results['Fr'])/F0
                    c0_actual[1] = (mod_results['cr'][1] * mod_results['Fr'])/F0
                    mod_results['Fd'] = mod_results['Fp'] # dilution rate corresponds to the lost permeate
                else:
                    c0_actual[0] = mod_results['cr'][0]
                    c0_actual[1] = mod_results['cr'][1]
                    mod_results['Fd'] = 0
                # print('stage success','Fr',Fr_actual,'c_ratio',c_ratio(c0_actual),'parallels',parallel,'c',c0_actual)
                n_stages += 1
            
            if c0_actual[0] < c0[0] * 0.001 or c0_actual[1] < c0[1] * 0.001:
                conc_lost = True
                break

    except RuntimeWarning:
        return float('inf'), 0, float('inf'), float('inf'), 0, stages
    
    if (n_stages == 20 and c_ratio(c0_actual) < cratio_target) or conc_lost:
        return float('inf'), 0, float('inf'), float('inf'), 0, stages
    
    # Calculating total area, power
    total_area = A * n_elements * n_stages
    dilution_power = (1/pump_eff) * np.sum(stages[i]['Fd'] for i in range(n_stages)) * p0
    power = (1/pump_eff) * p0 * F0 + dilution_power

    # Calculating recovery, necessary time, final concentration
    if L_list[0] < L_list[1]:
        recovery = (stages[n_stages-1]['Fr'] * stages[n_stages-1]['cr'][0]) / (F0_ref * c0[0])
        time = 1/(stages[n_stages-1]['Fr']*stages[n_stages-1]['cr'][0])
        c_final = stages[n_stages-1]['cr'][0]
    else:
        recovery = (stages[n_stages-1]['Fr'] * stages[n_stages-1]['cr'][1]) / (F0_ref * c0[1])
        time = 1/(stages[n_stages-1]['Fr']*stages[n_stages-1]['cr'][1])
        c_final = stages[n_stages-1]['cr'][1]
    
    molar_energy_demand = power*time

    # print()
    # print('No. of stages:',n_stages,'Recovery:',recovery,'E Demand:',molar_energy_demand)

    return molar_energy_demand, recovery, total_area, n_stages, c_final, stages


def targeted_ternary_permeate_nf_cascade(cfeed, cratio_target, pm):
    '''
    Two solutes
    '''

    ns = 2
    stage_cut, A, T, nn, n_elements, pump_eff = pm['stage_cut'], pm['A'], pm['T'], pm['nn'], pm['n_elements'], pm['pump_eff']
    L_list, nu_list, solubility_list, D_list, P = pm['L'], pm['nu'], pm['solubility'], pm['D'], pm['P']
    h, l_module, l_mesh, df,theta,n_env,b_env = pm['h'], pm['l_module'], pm['l_mesh'], pm['df'], pm['theta'], pm['n_env'], pm['b_env']
    rho = pm['density']
    eta = pm['viscosity']
    constants = [L_list, nu_list, solubility_list, D_list, P, h, rho, l_module, eta, l_mesh, df,theta,n_env,b_env]

    stages = {}
    if pm['R'][0] > pm['R'][1]:
        def c_ratio(c):
            return c[1]/c[0]
    elif pm['R'][0] < pm['R'][1]:
        def c_ratio(c):
            return c[0]/c[1]
    
    if L_list[0] == L_list[1] or pm['R'][0] == pm['R'][1]:
        return float('inf'), 0, float('inf'), float('inf'), 0, stages

    if cratio_target < c_ratio(cfeed):
        if L_list[0] < L_list[1]:
            c_final = cfeed[1]
        elif L_list[0] > L_list[1]:
            c_final = cfeed[0]
        return 0, 1, 0, 0, c_final, stages

    # calculate approx. feed flow / pressure
    p0 = pm['p0']
    pp = 0
    
    F0_ref = P*A*n_elements*(p0-pp) / stage_cut
    F0 = F0_ref

    c0 = cfeed.copy()
    c0_actual = c0.copy()

    n_stages = 0
    conc_lost = False
    # warnings.simplefilter('error', RuntimeWarning)
    try:
        while c_ratio(c0_actual) < cratio_target and n_stages <= 19:
            parameters = [F0, c0_actual, A, T, nn, ns, p0, pp]
            mod_results = sd_stage_sim(n_elements,parameters, constants)

            solution_factors = [mod_results['cp'][0]/solubility_list[0], mod_results['cp'][1]/solubility_list[1]]

            if (solution_factors[0] >= 0.95 or solution_factors[1] >= 0.95) and n_stages == 0:
                c0[0] = c0[0] * 0.5
                c0[1] = c0[1] * 0.5
                c0_actual = c0.copy()
            else:
                stages[n_stages] = mod_results
                if c_ratio(mod_results['cp']) < cratio_target:
                    c0_actual[0] = (mod_results['cp'][0] * mod_results['Fp'])/F0
                    c0_actual[1] = (mod_results['cp'][1] * mod_results['Fp'])/F0
                    mod_results['Fd'] = mod_results['Fr'] # dilution rate corresponds to the lost retentate
                else:
                    c0_actual[0] = mod_results['cp'][0]
                    c0_actual[1] = mod_results['cp'][1]
                    mod_results['Fd'] = 0
                # print('stage success','Fr',Fr_actual,'c_ratio',c_ratio(c0_actual),'parallels',parallel,'c',c0_actual)
                n_stages += 1
                # print(c_ratio(c0_actual))
            
            if c0_actual[0] < c0[0] * 0.001 or c0_actual[1] < c0[1] * 0.001:
                conc_lost = True
                break

    except RuntimeWarning:
        return float('inf'), 0, float('inf'), float('inf'), 0, stages
    
    if (n_stages == 20 and c_ratio(c0_actual) < cratio_target) or conc_lost:
        return float('inf'), 0, float('inf'), float('inf'), 0, stages
    
    dilution_power = (1/pump_eff) * np.sum(stages[i]['Fd'] for i in range(n_stages)) * p0
    power = (1/pump_eff) * n_stages * F0 * p0 + dilution_power
    total_area = A * n_elements * n_stages

    if L_list[0] < L_list[1]:
        recovery = (stages[n_stages-1]['Fp'] * stages[n_stages-1]['cp'][1]) / (F0_ref * c0[1])
        time = 1/(stages[n_stages-1]['Fp']*stages[n_stages-1]['cp'][1])
        c_final = stages[n_stages-1]['cp'][1]
    else:
        recovery = (stages[n_stages-1]['Fp'] * stages[n_stages-1]['cp'][0]) / (F0_ref * c0[0])
        time = 1/(stages[n_stages-1]['Fp']*stages[n_stages-1]['cp'][0])
        c_final = stages[n_stages-1]['cp'][0]
    molar_energy_demand = power*time

    # print(molar_energy_demand, recovery, total_area, n_stages, c_final, stages)
    return molar_energy_demand, recovery, total_area, n_stages, c_final, stages


##### EVAPORATION ENERGY CALCULATION ###################################################################################


def evaporation_energy(c0,ctarget,pm):
    '''
    Only considers first solute
    c: mol/m3
    '''
    if (ctarget <= c0 and not pm['solvent_recovery']) or (ctarget >= c0 and pm['solvent_recovery']):
        return 0
    
    if c0 == 0 and pm['solvent_recovery'] == False:
        return float('inf')
    elif c0 == 0 and pm['solvent_recovery']:
        return 0
    
    rho = pm['density']
    M = pm['solvent_molar_mass']
    dH = pm['solvent_heat_of_evaporation']

    try:
        if pm['solvent_recovery'] == False:
            molar_energy_demand = (1-pm['heat_integration_eff'])*(1/pm['evap_eff'])*(rho/(M))*((1/(c0))-(1/(ctarget)))*(dH)
        elif pm['solvent_recovery']:
            molar_energy_demand = (1-pm['heat_integration_eff'])*(1/pm['evap_eff'])*(rho/(M))*(dH)
        return molar_energy_demand
    except RuntimeWarning:
        return float('inf')
    

def _antoine_T_from_P(P, A, B, C):
    """
    Solve Antoine: log10(P) = A - B / (T + C)
    """
    if P <= 0:
        raise ValueError("Pressure must be > 0 for Antoine equation.")
    return B / (A - math.log10(P)) - C

def evaporation_energy_vacuum(c0, ctarget, pm):
    """
    Estimate energy demand (J per mol of solute) for vacuum distillation of a single-solute solution.
    - c0, ctarget in mol / m^3 (like your original).
    - pm: parameter dict with required keys:
        - density : rho, kg / m^3 (solvent density)
        - solvent_molar_mass : M, kg / mol
        - solvent_heat_of_evaporation : dH_latent, J / mol  (latent heat at boiling temp; if given per kg convert outside)
        - evap_eff : evaporation efficiency (0-1)
        - heat_integration_eff : fraction recovered by heat integration (0-1)
        - feed_temp : feed temperature in K  (T_feed)
        - cp : solvent heat capacity in J / (mol K)  (used for sensible heating)
    Optional (but recommended for vacuum calculation):
        - operating_pressure : Pa (absolute). If omitted, atmospheric boiling point used (pm['bp'])
        - antoine_A, antoine_B, antoine_C : Antoine coefficients for log10(P)=A - B/(T+C)
          These coefficients usually expect P in Pa and T in K. If your coefficients use other units,
          convert operating_pressure to the correct units before calling (or provide 'antoine_P_units' key).
        - antoine_P_units : 'mmHg' or 'Pa' or 'bar' (default 'Pa').
        - bp : normal boiling point in K (fallback if Antoine not provided)
        - solvent_recovery : boolean (same meaning as your original).
    Returns:
        energy_per_mol_solute (float): J per mol of solute (like your previous "molar_energy_demand")
        or float('inf') for infeasible / division-by-zero cases.
    """
    # ----------- Input validation and setup -----------
    solvent_recovery = bool(pm.get('solvent_recovery', False))
    if (ctarget <= c0 and not solvent_recovery) or (ctarget >= c0 and solvent_recovery):
        return 0.0, 0.0, 0.0
    if c0 == 0 and not solvent_recovery:
        return float('inf'), float('inf'), float('inf')
    if c0 == 0 and solvent_recovery:
        return 0.0, 0.0, 0.0

    # required params
    rho = pm['density']               # kg / m^3
    M = pm['solvent_molar_mass']      # kg / mol
    dH_latent = pm['solvent_heat_of_evaporation']  # J / mol (latent heat at boil)
    evap_eff = pm.get('evap_eff', 1.0)
    heat_int = pm.get('heat_integration_eff', 0.0)
    T_feed = pm.get('feed_temp', 298.15)   # K
    cp = pm.get('cp', None)                # J / (mol K) -- recommended
    P_oper = pm.get('operating_pressure', 101325.0)  # Pa
    P_atm = 101325.0              # Pa reference
    pump_eff = pm.get('pump_eff', 0.7)   # mechanical efficiency
    k_vap = pm.get('vap_heat_capacity_ratio', 1.3)

    if evap_eff <= 0:
        return float('inf'), float('inf'), float('inf')
    
    # ----------- Boiling temperature from Antoine or bp -----------
    # find boiling temperature at operating pressure (degC)
    T_b_degC = None
    if all(k in pm for k in ('antoine_A','antoine_B','antoine_C')):
        P_oper = pm.get('operating_pressure', 101325.0)  # Pa by default
        antoine_A = pm['antoine_A']
        antoine_B = pm['antoine_B']
        antoine_C = pm['antoine_C']
        P_units = pm.get('antoine_P_units', 'Pa')
        # convert Pa->mmHg if coeffs expect mmHg (common)
        if P_units == 'mmHg':
            P_ant = P_oper / 133.3223684211
        elif P_units == 'Pa':
            P_ant = P_oper
        elif P_units == 'bar':
            P_ant = P_oper * 1e-5
        else:
            print('Pascal fallback for unknown Antoine P_units')
            P_ant = P_oper
        try:
            T_b_K = _antoine_T_from_P(P_ant, antoine_A, antoine_B, antoine_C)
        except Exception:
            T_b_K = None

    if T_b_K is None:
        # fallback: use provided boiling point 'bp' in K
        if 'bp' in pm:
            T_b_K = pm['bp']  # K
        else:
            # if no info, assume normal boiling point from latent-heat estimate not advised
            raise KeyError("Antoine coefficients or 'bp' (boiling point in K) required for vacuum calculation.")

    # ----------- Solvent evaporated mass -----------
    # Determine mass of solvent evaporated per 1 m^3 feed
    V0 = 1.0  # we compute per 1 m^3 initial volume, as in your original function
    n_solute = c0 * V0   # mol of solute in initial 1 m^3
    if not solvent_recovery:
        if ctarget == 0:
            return float('inf'), float('inf'), float('inf')
        V_final = n_solute / ctarget  # m^3
        if V_final < 0:
            return float('inf'), float('inf'), float('inf')
        volume_evaporated = max(0.0, V0 - V_final)  # m^3 of solvent removed
        mass_evaporated = volume_evaporated * rho  # kg
    else:
        # evaporate all solvent in 1 m^3 (same behaviour as your previous True branch)
        mass_evaporated = rho * V0  # kg

    # if no solvent evaporated, zero energy
    if mass_evaporated <= 0:
        return 0.0, 0.0, 0.0

    # ----------- Thermal energy (sensible + latent) -----------
    # sensible heating (if cp provided)
    sensible = 0.0
    moles_evaporated = mass_evaporated / M  # mol
    if cp is not None:
        sensible = moles_evaporated * cp * max(0.0, (T_b_K - T_feed))  # J

    # latent heat (use dH_latent given per mol)
    latent = moles_evaporated * dH_latent    # J
    thermal_energy = (sensible + latent) * (1 - heat_int) / evap_eff # J for the evaporation event from 1 m^3 feed

    # ----------- Vacuum pump work (electrical) -----------
    R = 8.314  # J/molK
    T_v = T_b_K
    try:
        compression_ratio = P_atm / P_oper
        if compression_ratio < 1:
            compression_ratio = 1
        w_comp_per_kg = (R * T_v / (M * pump_eff * (k_vap - 1))) * \
                        ((compression_ratio)**((k_vap - 1)/k_vap) - 1)  # J/kg
        pump_work = mass_evaporated * w_comp_per_kg  # J total
    except Exception:
        pump_work = 0.0

    # ----------- Total energy and normalization -----------
    # return per mole of solute
    total_energy = thermal_energy + pump_work
    try:
        energy_per_mol_solute = total_energy / n_solute  # J per mol solute
        thermal_energy_per_mol_solute = thermal_energy / n_solute
        pump_work_per_mol_solute = pump_work / n_solute
    except ZeroDivisionError:
        return float('inf'), float('inf'), float('inf')

    return energy_per_mol_solute, thermal_energy_per_mol_solute, pump_work_per_mol_solute


##### COUPLED ENERGY CALCULATION ###########################################################################################


def coupled_binary_energy(c0,ctarget,pm,c_resolution=10,index=0):
    '''
    Calculates optimal configuration of continuous NF and evaporation in series. The number of NF stages is calculated from the overall c0 and ct.
    Index argument is useful in ternary parameter situations
    '''

    if c0 == 0:
        return float('inf'), 0, float('inf'), 0, float('inf'), float('inf'), float('inf')

    if ctarget <= c0:
        return 0, 1, 0, 0, 0, 0, 0

    conc = np.linspace(c0,ctarget,c_resolution)
    c_shift = c0
    E_list = []

    optimal_E = evaporation_energy(c0,ctarget,pm)
    c_shift = c0

    optimal_E_cnf = 0
    optimal_E_eva = optimal_E
    optimal_area = 0
    optimal_n_stages = 0

    optimal_recovery = 1

    opt_config = 'cnf+evap'

    for c in conc:
        if pm['R'][index] > 0:
            nf_molar_energy, nf_recovery, total_area, n_stages, elements = targeted_binary_retentate_nf_cascade(c0,c,pm,index=index)
            evap_molar_energy = evaporation_energy(c,ctarget,pm)
            total_energy = nf_molar_energy + evap_molar_energy
        else:
            nf_molar_energy, nf_recovery, total_area, n_stages, elements = targeted_binary_permeate_nf_cascade(c0,c,pm,index=index)
            evap_molar_energy = evaporation_energy(c,ctarget,pm)
            total_energy = nf_molar_energy + evap_molar_energy

        E_list.append(total_energy)

        if total_energy < optimal_E:
            optimal_E = total_energy
            c_shift = c
            optimal_E_cnf = nf_molar_energy
            optimal_E_eva = evap_molar_energy
            optimal_recovery = nf_recovery
            optimal_area = total_area
            optimal_n_stages = n_stages

    return optimal_E, optimal_recovery, optimal_area, c_shift, optimal_E_cnf, optimal_E_eva, optimal_n_stages


def coupled_binary_energy_after_ternary(E_ternary,c1,c_target,solute_idx,pm,c_resolution=10):
    '''
    Calculates optimal configuration of continuous NF and evaporation in series. The number of NF stages is calculated from the overall c0 and ct.
    solute_idx: solute to examine from parameters in pm
    '''

    if c1 == 0:
        return float('inf'), float('inf'), float('inf'), float('inf'), 0, float('inf'), float('inf'), 0

    if c_target <= c1:
        return 0, E_ternary, 0, 0, 1, 0, 0, 0

    conc = np.linspace(c1,c_target,c_resolution)
    c_shift = c1
    E_list = []
    recovery = 1
    n_stages = 0
    area = 0

    # Initialization
    if c_target < c1 or pm['R'][solute_idx] == 0:
        pass
    else:
        E_eva_min = evaporation_energy(c1,c_target,pm)
        E_nf_min = 0
        recovery = 1
        c_shift = c1
        n_stages = 0
        area = 0

    if c_target < c1:
        E_nf_min = 0
        E_eva_min = 0
        recovery = 1
        c_shift = c1
        n_stages = 0
        area = 0
    elif pm['R'][solute_idx] == 0:
        E_nf_min = 0
        E_eva_min = evaporation_energy(c1,c_target,pm)
        recovery = 1
        c_shift = c1
        n_stages = 0
        area = 0
    else:
        E_min = E_ternary/recovery + E_eva_min
        for c in conc:
            if pm['R'][solute_idx] > 0:
                E_nf, recovery_nf, total_area_nf, n_stages_nf, _ = targeted_binary_retentate_nf_cascade(c1,c,pm,index=solute_idx)
                E_eva = evaporation_energy(c,c_target,pm)
                if recovery_nf == 0:
                    E = float('inf')
                else:
                    E = E_ternary/recovery_nf + E_nf + E_eva
            else:
                E_nf, recovery_nf, total_area_nf, n_stages_nf, _ = targeted_binary_permeate_nf_cascade(c1,c,pm,index=solute_idx)
                E_eva = evaporation_energy(c,c_target,pm)
                if recovery_nf == 0:
                    E = float('inf')
                else:
                    E = E_ternary/recovery_nf + E_nf + E_eva
            E_list.append(E) 
            if E < E_min:
                E_nf_min = E_nf
                E_eva_min = E_eva
                recovery = recovery_nf
                c_shift = c
                E_min = E
                n_stages = n_stages_nf
                area = total_area_nf
    
    E_binary_min = E_nf_min + E_eva_min
    corrected_ternary_E = E_ternary/recovery
    nanofiltration_E = E_nf_min
    evaporation_E = E_eva_min

    return E_binary_min, corrected_ternary_E, nanofiltration_E, evaporation_E, recovery, n_stages, area, c_shift


#### EXTRACTION CALCULATIONS ####################################################################################

def extraction(case,c0,c_ratio_t,KA,KB):
    '''
    K is defined as:
    K = c(extractor)/c(original)

    A-standard: A prefers original solution better
    B-standard: B prefers original solution better
    '''
    cA_0 = c0[0]
    cB_0 = c0[1]
    c_ratio_0 = cA_0 / cB_0

    # Determine target ratios
    if case == 'A-standard':
        c_ratio_t_1 = c_ratio_t
        c_ratio_t_2 = 1/c_ratio_t
    elif case == 'B-standard':
        c_ratio_t_1 = 1/c_ratio_t
        c_ratio_t_2 = c_ratio_t   

    extraction_constant = (1+KB)/(1+KA)
    n = np.ceil(np.log(c_ratio_t_1/c_ratio_0)/np.log(extraction_constant))
    try:
        cA_1_final = cA_0/np.power((1+KA),n)
        cB_1_final = cB_0/np.power((1+KB),n)
    except:
        cA_1_final = 0
        cB_1_final = 0

    m = np.ceil(np.log(c_ratio_t_2/c_ratio_0)/np.log((KA/KB)*extraction_constant))
    try:
        cA_2_final = np.power((KA),m)*cA_0/np.power((1+KA),m)
        cB_2_final = np.power((KB),m)*cB_0/np.power((1+KB),m)
    except:
        cA_2_final = 0
        cB_2_final = 0        

    if case == 'A-standard':
        if cB_1_final != 0:
            c_ratio_1_final = cA_1_final/cB_1_final
        elif cA_1_final !=0 and cB_1_final ==0:
            c_ratio_1_final = c_ratio_t
        else:
            c_ratio_1_final = 0
        if cA_2_final != 0:
            c_ratio_2_final = cB_2_final/cA_2_final
        elif cB_2_final !=0 and cA_2_final ==0:
            c_ratio_2_final = c_ratio_t
        else:
            c_ratio_2_final = 0
        recovery_A = cA_1_final/cA_0
        recovery_B = cB_2_final/cA_0

    elif case == 'B-standard':
        if cA_1_final != 0:
            c_ratio_1_final = cB_1_final/cA_1_final
        elif cB_1_final != 0 and cA_1_final ==0:
            c_ratio_1_final = c_ratio_t
        else:
            c_ratio_1_final = 0
        if cB_2_final != 0:
            c_ratio_2_final = cA_2_final/cB_2_final
        elif cA_2_final != 0 and cB_2_final ==0:
            c_ratio_2_final = c_ratio_t
        else:
            c_ratio_2_final = 0
        recovery_A = cA_2_final/cA_0
        recovery_B = cB_1_final/cA_0

    return cA_1_final, cB_1_final, c_ratio_1_final, cA_2_final, cB_2_final, c_ratio_2_final, recovery_A, recovery_B


def counter_current_extraction(case,c0,c_ratio_t,KA,KB,just_impurity=False):
    '''
    Continuous counter current extraction based on Perry's handbook
    Looks like it doesnt work when both solutes prefer the original solvent -> there is an inherent limit
    Therefore if both solutes prefer the same solvent it will ruin one of the extraction branches
    1. n stages of counter current extraction
    2. m stages where the final extractor solution stream is the feed and original solvent the extractor

    If extraction is for purity removal only then solute on index 0 (solute A) is considered the main / target solute
    Ratio of flow rates is 1.
    '''

    cA_0 = c0[0]
    cB_0 = c0[1]

    cA_n = cA_0
    cB_n = cB_0
    cA_m = cA_0
    cB_m = cB_0
    n = 0
    m = 0
    c_ratio_n = cA_0/cB_0
    c_ratio_m = cB_0/cA_0
    rec_A = 1
    rec_B = 1

    if just_impurity and case=='A-standard':
        while cA_n/cB_n < c_ratio_t:
            n += 1
            cA_n = cA_0*(KA-1)/(np.power(KA,(n+1))-1)
            cB_n = cB_0*(KB-1)/(np.power(KB,(n+1))-1)
        c_ratio_n = cA_n/cB_n
        rec_A = cA_n/cA_0
    elif just_impurity and case=='B-standard':
        cA_ext = KA*cA_0/(1+KA)  # conc. of A in the extracting solvent after 1 step
        cB_ext = KB*cB_0/(1+KB)
        cA_m = cA_ext
        cB_m = cB_ext
        while cA_m/cB_m < c_ratio_t:
            m += 1
            cA_m = cA_ext*(1/KA-1)/(np.power(1/KA,(m+1))-1)
            cB_m = cB_ext*(1/KB-1)/(np.power(1/KB,(m+1))-1)
        c_ratio_m = cA_m/cB_m
        rec_A = cA_m/cA_0

    if just_impurity == False and case == 'A-standard':
        while cA_n/cB_n < c_ratio_t:
            n += 1
            cA_n = cA_0*(KA-1)/(np.power(KA,(n+1))-1)
            cB_n = cB_0*(KB-1)/(np.power(KB,(n+1))-1)
            if cB_n == 0 and cA_n != 0:
                c_ratio_n == float('inf')
                break
            elif cB_n == 0 and cA_n == 0:
                c_ratio_n = 0
            else:
                c_ratio_n = cA_n/cB_n

        cA_ext = cA_0 - cA_n
        cB_ext = cB_0 - cB_n

        cA_m = cA_ext
        cB_m = cB_ext

        if cA_m != 0:
            while cB_m/cA_m < c_ratio_t:
                m += 1
                cA_m = cA_ext*(1/KA-1)/(np.power(1/KA,(m+1))-1)
                cB_m = cB_ext*(1/KB-1)/(np.power(1/KB,(m+1))-1)
                if cA_m == 0 and cB_m != 0:
                    c_ratio_m == float('inf')
                    break
                elif cA_m == 0 and cB_m == 0:
                    c_ratio_m = 0
                else:
                    c_ratio_m = cB_m/cA_m
        else:
            c_ratio_m == float('inf')

        rec_A = cA_n/cA_0
        rec_B = cB_m/cB_0

    elif just_impurity == False and case == 'B-standard':
        while cB_n/cA_n < c_ratio_t:
            n += 1
            cA_n = cA_0*(KA-1)/(np.power(KA,(n+1))-1)
            cB_n = cB_0*(KB-1)/(np.power(KB,(n+1))-1)
            if cA_n == 0 and cB_n != 0:
                c_ratio_n == float('inf')
                break
            elif cB_n == 0 and cA_n == 0:
                c_ratio_n = 0
            else:
                c_ratio_n = cB_n/cA_n

        cA_ext = cA_0 - cA_n
        cB_ext = cB_0 - cB_n

        cA_m = cA_ext
        cB_m = cB_ext
        
        if cB_m != 0:
            while cA_m/cB_m < c_ratio_t:
                m += 1
                cA_m = cA_ext*(1/KA-1)/(np.power(1/KA,(m+1))-1)
                cB_m = cB_ext*(1/KB-1)/(np.power(1/KB,(m+1))-1)
                if cB_m == 0 and cA_m != 0:
                    c_ratio_m == float('inf')
                    break
                elif cB_m == 0 and cA_m == 0:
                    c_ratio_m = 0
                else:
                    c_ratio_m = cA_m/cB_m
        else:
            c_ratio_m == float('inf')

        rec_A = cA_m/cA_0
        rec_B = cB_n/cB_0

    c_ratio_1_final = c_ratio_n
    c_ratio_2_final = c_ratio_m
    cA_1_final = cA_n
    cB_1_final = cB_n
    cA_2_final = cA_m
    cB_2_final = cB_m
    recovery_A = rec_A
    recovery_B = rec_B

    return cA_1_final, cB_1_final, c_ratio_1_final, cA_2_final, cB_2_final, c_ratio_2_final, recovery_A, recovery_B


def extraction_wrapper(c0,c_ratio_t,A_logPhi,B_logPhi,just_impurity=False):
    '''
    Wrapper function that chooses the appropriate extraction process (multi-stage batch or continuous counter-current)
    For impurities main solute is in index 0

    K = c(extractor)/c(original)
    '''

    # Choose case
    if A_logPhi == B_logPhi:
        return 'Null-standard', 0, 0, 0, 0, 0, 0, 0, 0

    if A_logPhi < B_logPhi:
        case = 'A-standard'
    elif B_logPhi < A_logPhi:
        case = 'B-standard'
    else:
        return 'Null-standard', 0, 0, 0, 0, 0, 0, 0, 0
    
    KA = np.power(10,A_logPhi)
    KB = np.power(10,B_logPhi)

    try:
        if A_logPhi * B_logPhi < 0:
            cA_1_final, cB_1_final, c_ratio_1_final, cA_2_final, cB_2_final, c_ratio_2_final, recovery_A, recovery_B = counter_current_extraction(case,c0,c_ratio_t,KA,KB,just_impurity=just_impurity)
        elif just_impurity and A_logPhi*B_logPhi > 0 and KA > 1 and case == 'A-standard':
            cA_1_final, cB_1_final, c_ratio_1_final, cA_2_final, cB_2_final, c_ratio_2_final, recovery_A, recovery_B = counter_current_extraction(case,c0,c_ratio_t,KA,KB,just_impurity=just_impurity)
        elif just_impurity and A_logPhi*B_logPhi > 0 and KB < 1 and case == 'B-standard':
            cA_1_final, cB_1_final, c_ratio_1_final, cA_2_final, cB_2_final, c_ratio_2_final, recovery_A, recovery_B = counter_current_extraction(case,c0,c_ratio_t,KA,KB,just_impurity=just_impurity)
        else:
            cA_1_final, cB_1_final, c_ratio_1_final, cA_2_final, cB_2_final, c_ratio_2_final, recovery_A, recovery_B = extraction(case,c0,c_ratio_t,KA,KB)
    except RuntimeWarning:
        cA_1_final, cB_1_final, c_ratio_1_final, cA_2_final, cB_2_final, c_ratio_2_final, recovery_A, recovery_B = 0, 0, 1, 0, 0, 1, 0, 0
    
    if math.isnan(cA_1_final):
        cA_1_final = 0
    if math.isnan(cB_1_final):
        cB_1_final = 0
    if math.isnan(c_ratio_1_final):
        c_ratio_1_final = 1
    if math.isnan(cA_2_final):
        cA_2_final = 0
    if math.isnan(cB_2_final):
        cB_2_final = 0
    if math.isnan(c_ratio_2_final):
        c_ratio_2_final = 1
    if math.isnan(recovery_A):
        recovery_A = 0
    if math.isnan(recovery_B):
        recovery_B = 0

    return case, cA_1_final, cB_1_final, c_ratio_1_final, cA_2_final, cB_2_final, c_ratio_2_final, recovery_A, recovery_B


### MAIN FUNCTIONS ####################################################################################################

def solute_separation_energy(cfeed,c_ratio_t,pm,target_is_max=True):
    '''
    Index 0: solute A
    Index 1: solute B
    '''
    extractor = pm['extractor']

    if extractor == 'Heptane':
        A_logPhi = pm['logPhi_heptane'][0]
        B_logPhi = pm['logPhi_heptane'][1]
    elif extractor == 'Water':
        A_logPhi = pm['logPhi_water'][0]
        B_logPhi = pm['logPhi_water'][1]
    
    # Extraction for ternary separation
    case, cA_1_final, cB_1_final, c_ratio_1_final, cA_2_final, cB_2_final, c_ratio_2_final, ext_recovery_A, ext_recovery_B = extraction_wrapper(cfeed,c_ratio_t,A_logPhi,B_logPhi,just_impurity=False)
    
    # Nanofiltration for ternary separation
    tnf_molar_energy_ret, tnf_recovery_ret, tnf_area_ret, tnf_n_stages_ret, tnf_c_final_ret, tnf_stages_ret = targeted_ternary_retentate_nf_cascade(cfeed,c_ratio_t,pm)
    tnf_molar_energy_per, tnf_recovery_per, tnf_area_per, tnf_n_stages_per, tnf_c_final_per, tnf_stages_per = targeted_ternary_permeate_nf_cascade(cfeed,c_ratio_t,pm)

    # Linking solutes and streams
    if case == 'A-standard':
        cA_from_ext = cA_1_final
        cB_from_ext = cB_2_final
    elif case == 'B-standard':
        cA_from_ext = cA_2_final
        cB_from_ext = cB_1_final
    
    if pm['R'][0] > pm['R'][1]:
        cA_from_tnf = tnf_c_final_ret
        cB_from_tnf = tnf_c_final_per
        nA_tnf = tnf_n_stages_ret
        nB_tnf = tnf_n_stages_per
        areaA_tnf = tnf_area_ret
        areaB_tnf = tnf_area_per
        E_ternaryA_tnf = tnf_molar_energy_ret
        E_ternaryB_tnf = tnf_molar_energy_per
        recoveryA_tnf = tnf_recovery_ret
        recoveryB_tnf = tnf_recovery_per
    elif pm['R'][0] < pm['R'][1]:
        cA_from_tnf = tnf_c_final_per
        cB_from_tnf = tnf_c_final_ret
        nA_tnf = tnf_n_stages_per
        nB_tnf = tnf_n_stages_ret
        areaA_tnf = tnf_area_per
        areaB_tnf = tnf_area_ret
        E_ternaryA_tnf = tnf_molar_energy_per
        E_ternaryB_tnf = tnf_molar_energy_ret
        recoveryA_tnf = tnf_recovery_per
        recoveryB_tnf = tnf_recovery_ret

    # Binary concentration target
    if target_is_max:
        c_target_A = min(max(cA_from_ext,cB_from_ext,cA_from_tnf,cB_from_tnf),pm['solubility'][0])
        c_target_B = min(c_target_A,pm['solubility'][1])
    else:
        c_target_A = cfeed[0]
        c_target_B = cfeed[1]

    # print('Targets:',c_target_A,c_target_B)
    # print('After tnf:',cA_from_tnf,cB_from_tnf)

    # Concentration processes after extraction
    molar_energy_evap_after_ext = evaporation_energy(cA_from_ext,c_target_A,pm) + evaporation_energy(cB_from_ext,c_target_B,pm)

    molar_energy_A_cpld_after_ext, recovery_A_cpld_after_ext, area_A_cpld_after_ext, c_shift_A_cpld_after_ext, nf_molar_energy_A_cpld_after_ext, eva_molar_energy_A_cpld_after_ext, n_stages_A_cpld_after_ext = coupled_binary_energy(cA_from_ext,c_target_A,pm,index=0)
    molar_energy_B_cpld_after_ext, recovery_B_cpld_after_ext, area_B_cpld_after_ext, c_shift_B_cpld_after_ext, nf_molar_energy_B_cpld_after_ext, eva_molar_energy_B_cpld_after_ext, n_stages_B_cpld_after_ext = coupled_binary_energy(cB_from_ext,c_target_B,pm,index=1)

    if pm['R'][0] > 0:
        molar_energy_A_nf_after_ext, recovery_A_nf_after_ext, area_A_nf_after_ext, n_stages_A_nf_after_ext, _ = targeted_binary_retentate_nf_cascade(cA_from_ext,c_target_A,pm,index=0)
    else:
        molar_energy_A_nf_after_ext, recovery_A_nf_after_ext, area_A_nf_after_ext, n_stages_A_nf_after_ext, _ = targeted_binary_permeate_nf_cascade(cA_from_ext,c_target_A,pm,index=0)
    if pm['R'][1] > 0:
        molar_energy_B_nf_after_ext, recovery_B_nf_after_ext, area_B_nf_after_ext, n_stages_B_nf_after_ext, _ = targeted_binary_retentate_nf_cascade(cB_from_ext,c_target_B,pm,index=1)
    else:
        molar_energy_B_nf_after_ext, recovery_B_nf_after_ext, area_B_nf_after_ext, n_stages_B_nf_after_ext, _ = targeted_binary_permeate_nf_cascade(cB_from_ext,c_target_B,pm,index=1)
    
    # Concentration processes after ternary nanofiltration

    molar_energy_A_eva_after_nf = evaporation_energy(cA_from_tnf,c_target_A,pm)
    molar_energy_B_eva_after_nf = evaporation_energy(cB_from_tnf,c_target_B,pm)

    molar_energy_A_cpld_after_nf, _, nf_molar_energy_A_cpld_after_nf, eva_molar_energy_A_cpld_after_nf, recovery_A_cpld_after_nf, n_stages_A_cpld_after_nf, area_A_cpld_after_nf, c_shift_A = coupled_binary_energy_after_ternary(E_ternaryA_tnf,cA_from_tnf,c_target_A,0,pm)
    molar_energy_B_cpld_after_nf, _, nf_molar_energy_B_cpld_after_nf, eva_molar_energy_B_cpld_after_nf, recovery_B_cpld_after_nf, n_stages_B_cpld_after_nf, area_B_cpld_after_nf, c_shift_B = coupled_binary_energy_after_ternary(E_ternaryB_tnf,cB_from_tnf,c_target_B,1,pm)

    try:
        if pm['R'][0] > 0:
            molar_energy_A_nf_after_nf, recovery_A_nf_after_nf, area_A_nf_after_nf, n_stages_A_nf_after_nf, _ = targeted_binary_retentate_nf_cascade(cA_from_tnf,c_target_A,pm,index=0)
        else:
            molar_energy_A_nf_after_nf, recovery_A_nf_after_nf, area_A_nf_after_nf, n_stages_A_nf_after_nf, _ = targeted_binary_permeate_nf_cascade(cA_from_tnf,c_target_A,pm,index=0)
    except:
        # print('here')
        molar_energy_A_nf_after_nf = float('inf')
        recovery_A_nf_after_nf = 0
        area_A_nf_after_nf = float('inf')
        n_stages_A_nf_after_nf = float('inf')
    
    try:
        if pm['R'][1] > 0:
            molar_energy_B_nf_after_nf, recovery_B_nf_after_nf, area_B_nf_after_nf, n_stages_B_nf_after_nf, _ = targeted_binary_retentate_nf_cascade(cB_from_tnf,c_target_B,pm,index=1)
        else:
            molar_energy_B_nf_after_nf, recovery_B_nf_after_nf, area_B_nf_after_nf, n_stages_B_nf_after_nf, _ = targeted_binary_permeate_nf_cascade(cB_from_tnf,c_target_B,pm,index=1)
    except:
        # print('here')
        molar_energy_B_nf_after_nf = float('inf')
        recovery_B_nf_after_nf = 0
        area_B_nf_after_nf = float('inf')
        n_stages_B_nf_after_nf = float('inf')

    # Summarizing energies

    molar_energies = {}

    molar_energies['ext-eva'] = molar_energy_evap_after_ext

    molar_energies['ext-nf']  = molar_energy_A_nf_after_ext + molar_energy_B_nf_after_ext

    molar_energies['ext-cpld']  = molar_energy_A_cpld_after_ext + molar_energy_B_cpld_after_ext
    molar_energies['ext-cpld (nf)']  = nf_molar_energy_A_cpld_after_ext + nf_molar_energy_B_cpld_after_ext
    molar_energies['ext-cpld (eva)'] = eva_molar_energy_A_cpld_after_ext + eva_molar_energy_B_cpld_after_ext

    molar_energies['nf_ternary'] = E_ternaryA_tnf + E_ternaryB_tnf
    molar_energies['nf-eva'] = E_ternaryA_tnf + E_ternaryB_tnf + molar_energy_A_eva_after_nf + molar_energy_B_eva_after_nf

    if recovery_A_nf_after_nf == 0 or recovery_B_nf_after_nf == 0:
        # print('here2')
        molar_energies['nf-nf'] = float('inf')
        molar_energies['nf-nf (ternary)'] = float('inf')
    else:
        molar_energies['nf-nf'] = E_ternaryA_tnf/recovery_A_nf_after_nf + E_ternaryB_tnf/recovery_B_nf_after_nf + molar_energy_A_nf_after_nf + molar_energy_B_nf_after_nf
        molar_energies['nf-nf (ternary)'] = E_ternaryA_tnf/recovery_A_nf_after_nf + E_ternaryB_tnf/recovery_B_nf_after_nf

    if recovery_A_cpld_after_nf == 0 or recovery_B_cpld_after_nf == 0:
        # print('here3')
        molar_energies['nf-cpld'] = float('inf')
        molar_energies['nf-cpld (ternary)'] = float('inf')
        molar_energies['nf-cpld (nf)'] = float('inf')
        molar_energies['nf-cpld (eva)'] = eva_molar_energy_A_cpld_after_nf + eva_molar_energy_B_cpld_after_nf
    else:
        molar_energies['nf-cpld'] = E_ternaryA_tnf/recovery_A_cpld_after_nf + E_ternaryB_tnf/recovery_B_cpld_after_nf + molar_energy_A_cpld_after_nf + molar_energy_B_cpld_after_nf
        molar_energies['nf-cpld (ternary)'] = E_ternaryA_tnf/recovery_A_cpld_after_nf + E_ternaryB_tnf/recovery_B_cpld_after_nf
        molar_energies['nf-cpld (nf)'] = E_ternaryA_tnf/recovery_A_cpld_after_nf + nf_molar_energy_A_cpld_after_nf + E_ternaryB_tnf/recovery_B_cpld_after_nf + nf_molar_energy_B_cpld_after_nf
        molar_energies['nf-cpld (eva)'] = eva_molar_energy_A_cpld_after_nf + eva_molar_energy_B_cpld_after_nf

    # Summarizing average recoveries

    recoveries = {}

    recoveries['ext-eva'] = (ext_recovery_A + ext_recovery_B)/2
    recoveries['ext-nf'] = (ext_recovery_A*recovery_A_nf_after_ext + ext_recovery_B*recovery_B_nf_after_ext)/2
    recoveries['ext-cpld'] = (ext_recovery_A*recovery_A_cpld_after_ext + ext_recovery_B*recovery_B_cpld_after_ext)/2

    recoveries['nf-eva'] = (recoveryA_tnf + recoveryB_tnf)/2
    recoveries['nf-nf'] = (recoveryA_tnf*recovery_A_nf_after_nf + recoveryB_tnf*recovery_B_nf_after_nf)/2
    recoveries['nf-cpld'] = (recoveryA_tnf*recovery_A_cpld_after_nf + recoveryB_tnf*recovery_B_cpld_after_nf)/2

    # Summarizing stages

    no_of_stages = {}

    no_of_stages['ext-eva'] = 0
    no_of_stages['ext-nf'] = n_stages_A_nf_after_ext + n_stages_B_nf_after_ext
    no_of_stages['ext-cpld'] = n_stages_A_cpld_after_ext + n_stages_B_cpld_after_ext

    no_of_stages['nf-eva'] = nA_tnf + nB_tnf
    no_of_stages['nf-nf'] = nA_tnf + nB_tnf + n_stages_A_nf_after_nf + n_stages_B_nf_after_nf
    no_of_stages['nf-cpld'] = nA_tnf + nB_tnf + n_stages_A_cpld_after_nf + n_stages_B_cpld_after_nf

    # Areas

    areas = {}

    areas['ext-eva'] = 0
    areas['ext-nf'] = area_A_nf_after_ext + area_B_nf_after_ext
    areas['ext-cpld'] = area_A_cpld_after_ext + area_B_cpld_after_ext
    areas['nf-eva'] = areaA_tnf + areaB_tnf
    areas['nf-nf'] = areaA_tnf + areaB_tnf + area_A_nf_after_nf + area_B_nf_after_nf
    areas['nf-cpld'] = areaA_tnf + areaB_tnf + area_A_cpld_after_nf + area_B_cpld_after_nf

    for key in molar_energies:
        if molar_energies[key] == float('nan'):
            molar_energies[key] == float('inf')
    for key in recoveries:
        if recoveries[key] == float('nan'):
            recoveries[key] == 0
    for key in no_of_stages:
        if no_of_stages[key] == float('nan'):
            no_of_stages[key] == float('inf')
    for key in areas:
        if areas[key] == float('nan'):
            areas[key] == float('inf')

    return molar_energies, recoveries, no_of_stages, areas


def impurity_removal_energy(cfeed,c_ratio_t,pm,target_is_max=True):
    '''
    Index 0 / A: MAIN solute
    Index 1 / B: IMPURITY solute
    '''
    extractor = pm['extractor']

    if extractor == 'Heptane':
        # print('Heptane extractor')
        A_logPhi = pm['logPhi_heptane'][0]
        B_logPhi = pm['logPhi_heptane'][1]
    elif extractor == 'Water':
        # print('Water extractor')
        A_logPhi = pm['logPhi_water'][0]
        B_logPhi = pm['logPhi_water'][1]
    
    # Extraction for ternary separation
    case, cA_1_final, cB_1_final, c_ratio_1_final, cA_2_final, cB_2_final, c_ratio_2_final, ext_recovery_A, ext_recovery_B = extraction_wrapper(cfeed,c_ratio_t,A_logPhi,B_logPhi,just_impurity=True)
    #print('case, cA_1_final, cB_1_final, c_ratio_1_final, cA_2_final, cB_2_final, c_ratio_2_final, ext_recovery_A, ext_recovery_B ')
    #print(case, cA_1_final, cB_1_final, c_ratio_1_final, cA_2_final, cB_2_final, c_ratio_2_final, ext_recovery_A, ext_recovery_B)

    #print(cfeed)
    # Nanofiltration for ternary separation
    if pm['R'][0] >= pm['R'][1]:
        tnf_molar_energy, tnf_recovery, tnf_area, tnf_n_stages, tnf_c_final, tnf_stages = targeted_ternary_retentate_nf_cascade(cfeed,c_ratio_t,pm)
    elif pm['R'][1] > pm['R'][0]:
        tnf_molar_energy, tnf_recovery, tnf_area, tnf_n_stages, tnf_c_final, tnf_stages = targeted_ternary_permeate_nf_cascade(cfeed,c_ratio_t,pm)
    else:
        tnf_molar_energy, tnf_recovery, tnf_area, tnf_n_stages, tnf_c_final, tnf_stages = targeted_ternary_retentate_nf_cascade(cfeed,c_ratio_t,pm)
    #print(cfeed)

    # Linking solutes and streams
    if case == 'A-standard':
        cA_from_ext = cA_1_final
    elif case == 'B-standard':
        cA_from_ext = cA_2_final
    else:
        cA_from_ext = 0.0
    
    cA_from_tnf = tnf_c_final
    nA_tnf = tnf_n_stages
    areaA_tnf = tnf_area
    E_ternaryA_tnf = tnf_molar_energy
    recoveryA_tnf = tnf_recovery

    # Binary concentration target
    if target_is_max:
        c_target_A = min(max(cA_from_ext,cA_from_tnf),pm['solubility'][0])
    else:
        c_target_A = cfeed[0]
        
    #print(c_target_A)

    # Concentration processes after extraction
    molar_energy_evap_after_ext = evaporation_energy(cA_from_ext,c_target_A,pm)

    molar_energy_A_cpld_after_ext, recovery_A_cpld_after_ext, area_A_cpld_after_ext, c_shift_A_cpld_after_ext, nf_molar_energy_A_cpld_after_ext, eva_molar_energy_A_cpld_after_ext, n_stages_A_cpld_after_ext = coupled_binary_energy(cA_from_ext,c_target_A,pm,index=0)

    if pm['R'][0] > 0:
        molar_energy_A_nf_after_ext, recovery_A_nf_after_ext, area_A_nf_after_ext, n_stages_A_nf_after_ext, _ = targeted_binary_retentate_nf_cascade(cA_from_ext,c_target_A,pm,index=0)
    else:
        molar_energy_A_nf_after_ext, recovery_A_nf_after_ext, area_A_nf_after_ext, n_stages_A_nf_after_ext, _ = targeted_binary_permeate_nf_cascade(cA_from_ext,c_target_A,pm,index=0)
    
    # Concentration processes after ternary nanofiltration

    molar_energy_A_eva_after_nf = evaporation_energy(cA_from_tnf,c_target_A,pm)

    molar_energy_A_cpld_after_nf, _, nf_molar_energy_A_cpld_after_nf, eva_molar_energy_A_cpld_after_nf, recovery_A_cpld_after_nf, n_stages_A_cpld_after_nf, area_A_cpld_after_nf, c_shift_A = coupled_binary_energy_after_ternary(E_ternaryA_tnf,cA_from_tnf,c_target_A,0,pm)

    try:
        if pm['R'][0] > 0:
            molar_energy_A_nf_after_nf, recovery_A_nf_after_nf, area_A_nf_after_nf, n_stages_A_nf_after_nf, _ = targeted_binary_retentate_nf_cascade(cA_from_tnf,c_target_A,pm,index=0)
        else:
            molar_energy_A_nf_after_nf, recovery_A_nf_after_nf, area_A_nf_after_nf, n_stages_A_nf_after_nf, _ = targeted_binary_permeate_nf_cascade(cA_from_tnf,c_target_A,pm,index=0)
    except:
        molar_energy_A_nf_after_nf = float('inf')
        recovery_A_nf_after_nf = 0
        area_A_nf_after_nf = float('inf')
        n_stages_A_nf_after_nf = float('inf')
    

    # Summarizing energies

    molar_energies = {}

    molar_energies['ext-eva'] = molar_energy_evap_after_ext

    molar_energies['ext-nf']  = molar_energy_A_nf_after_ext

    molar_energies['ext-cpld']  = molar_energy_A_cpld_after_ext
    molar_energies['ext-cpld (nf)']  = nf_molar_energy_A_cpld_after_ext
    molar_energies['ext-cpld (eva)'] = eva_molar_energy_A_cpld_after_ext

    molar_energies['nf_ternary'] = E_ternaryA_tnf
    molar_energies['nf-eva'] = E_ternaryA_tnf + molar_energy_A_eva_after_nf

    if recovery_A_nf_after_nf == 0:
        molar_energies['nf-nf'] = float('inf')
        molar_energies['nf-nf (ternary)'] = float('inf')
    else:
        molar_energies['nf-nf'] = E_ternaryA_tnf/recovery_A_nf_after_nf + molar_energy_A_nf_after_nf
        molar_energies['nf-nf (ternary)'] = E_ternaryA_tnf/recovery_A_nf_after_nf

    if recovery_A_cpld_after_nf == 0: 
        molar_energies['nf-cpld'] = float('inf')
        molar_energies['nf-cpld (ternary)'] = float('inf')
        molar_energies['nf-cpld (nf)'] = float('inf')
        molar_energies['nf-cpld (eva)'] = eva_molar_energy_A_cpld_after_nf 
    else: 
        molar_energies['nf-cpld'] = E_ternaryA_tnf/recovery_A_cpld_after_nf + molar_energy_A_cpld_after_nf
        molar_energies['nf-cpld (ternary)'] = E_ternaryA_tnf/recovery_A_cpld_after_nf
        molar_energies['nf-cpld (nf)'] = E_ternaryA_tnf/recovery_A_cpld_after_nf + nf_molar_energy_A_cpld_after_nf
        molar_energies['nf-cpld (eva)'] = eva_molar_energy_A_cpld_after_nf 

    # Summarizing average recoveries

    recoveries = {}

    recoveries['ext-eva'] = ext_recovery_A
    recoveries['ext-nf'] = ext_recovery_A*recovery_A_nf_after_ext
    recoveries['ext-cpld'] = ext_recovery_A*recovery_A_cpld_after_ext

    recoveries['nf-eva'] = recoveryA_tnf
    recoveries['nf-nf'] = recoveryA_tnf*recovery_A_nf_after_nf
    recoveries['nf-cpld'] = recoveryA_tnf*recovery_A_cpld_after_nf

    # Summarizing stages

    no_of_stages = {}

    no_of_stages['ext-eva'] = 0
    no_of_stages['ext-nf'] = n_stages_A_nf_after_ext
    no_of_stages['ext-cpld'] = n_stages_A_cpld_after_ext

    no_of_stages['nf-eva'] = nA_tnf
    no_of_stages['nf-nf'] = nA_tnf + n_stages_A_nf_after_nf
    no_of_stages['nf-cpld'] = nA_tnf + n_stages_A_cpld_after_nf

    # Areas

    areas = {}

    areas['ext-eva'] = 0
    areas['ext-nf'] = area_A_nf_after_ext
    areas['ext-cpld'] = area_A_cpld_after_ext
    areas['nf-eva'] = areaA_tnf
    areas['nf-nf'] = areaA_tnf + area_A_nf_after_nf
    areas['nf-cpld'] = areaA_tnf + area_A_cpld_after_nf

    for key in molar_energies:
        if molar_energies[key] == float('nan'):
            molar_energies[key] == float('inf')
    for key in recoveries:
        if recoveries[key] == float('nan'):
            recoveries[key] == 0
    for key in no_of_stages:
        if no_of_stages[key] == float('nan'):
            no_of_stages[key] == float('inf')
    for key in areas:
        if areas[key] == float('nan'):
            areas[key] == float('inf')

    return molar_energies, recoveries, no_of_stages, areas


def impurity_removal_energy_only_tnf(cfeed,c_ratio_t,pm):
    '''
    Index 0 / A: MAIN solute
    Index 1 / B: IMPURITY solute
    '''

    electricity_usd = {
        'EUR': 0.449,
        'USA': 0.146,
        'IND': 0.117,
        'CHN': 0.088
    } # USD / kWh
    cost_membrane = 100 # USD/m2
    membrane_lifetime = 8766  # h

    catalyst_cost = pm['catalyst_cost'] # USD / mol

    if pm['R'][0] >= pm['R'][1]:
        tnf_molar_energy, tnf_recovery, tnf_area, tnf_n_stages, tnf_c_final, tnf_stages = targeted_ternary_retentate_nf_cascade(cfeed,c_ratio_t,pm)
        if tnf_n_stages == float('inf'):
            mol_flow = 0
        else:
            mol_flow = tnf_stages[tnf_n_stages-1]['cr'][0]*tnf_stages[tnf_n_stages-1]['Fr'] * 3600 # mol/h
    elif pm['R'][1] > pm['R'][0]:
        tnf_molar_energy, tnf_recovery, tnf_area, tnf_n_stages, tnf_c_final, tnf_stages = targeted_ternary_permeate_nf_cascade(cfeed,c_ratio_t,pm)
        if tnf_n_stages == float('inf'):
            mol_flow = 0
        else:
            mol_flow = tnf_stages[tnf_n_stages-1]['cp'][0]*tnf_stages[tnf_n_stages-1]['Fp'] * 3600 # mol/h
    else:
        tnf_molar_energy, tnf_recovery, tnf_area, tnf_n_stages, tnf_c_final, tnf_stages = targeted_ternary_retentate_nf_cascade(cfeed,c_ratio_t,pm)
        if tnf_n_stages == float('inf'):
            mol_flow = 0
        else:
            mol_flow = tnf_stages[tnf_n_stages-1]['cr'][0]*tnf_stages[tnf_n_stages-1]['Fr'] * 3600 # mol/h
    #print(cfeed)

    costs = {}
    waste_cost = None
    membrane_cost = None
    if mol_flow == 0 or tnf_n_stages == float('inf') or tnf_area == float('inf'):
        productivity = 0
        for key in electricity_usd.keys():
            costs[key] = float('inf')
        waste_cost = float('inf')
        membrane_cost = float('inf')
    else:
        productivity = mol_flow / tnf_area # mol/m2/h
        membrane_cost_per_mol_product = (cost_membrane * tnf_area) / (membrane_lifetime * mol_flow)

        waste_catalyst_cost_per_hour = catalyst_cost * (1 - tnf_recovery) * tnf_stages[0]['c0'][0]*tnf_stages[0]['F0'] * 3600 # USD/h
        waste_catalyst_cost_per_mol_product = waste_catalyst_cost_per_hour / mol_flow

        membrane_cost = membrane_cost_per_mol_product
        waste_cost = waste_catalyst_cost_per_mol_product

        for key in electricity_usd.keys():
            costs[key] = (electricity_usd[key] / 3.6e6) * tnf_molar_energy + membrane_cost + waste_cost
            if costs[key] == float('nan'):
                costs[key] = float('inf')

    if tnf_molar_energy == float('nan'):
        tnf_molar_energy = float('inf')
    if tnf_recovery == float('nan'):
        tnf_recovery = 0
    if tnf_n_stages == float('nan'):
        tnf_n_stages = float('inf')
    if tnf_area == float('nan'):
        tnf_area = float('inf')

    return tnf_molar_energy, tnf_recovery, tnf_n_stages, tnf_area, tnf_c_final, productivity, costs, membrane_cost, waste_cost


def concentration_only_bnf(c0,ctarget,pm):
    '''
    Index 0 / A: MAIN solute
    '''

    electricity_usd = {
        'EUR': 0.449,
        'USA': 0.146,
        'IND': 0.117,
        'CHN': 0.088
    } # USD / kWh
    cost_membrane = 100 # USD/m2
    membrane_lifetime = 8766  # h

    catalyst_cost = pm['catalyst_cost'] # USD / mol

    if pm['R'][0] >= 0:
        molar_energy_demand, recovery, total_area, n_stages, stages = targeted_binary_retentate_nf_cascade(c0,ctarget,pm,index=0)
        if n_stages == float('inf'):
            mol_flow = 0
        else:
            mol_flow = stages[n_stages-1]['cr'][0]*stages[n_stages-1]['Fr'] * 3600 # mol/h
    else:
        molar_energy_demand, recovery, total_area, n_stages, stages = targeted_binary_permeate_nf_cascade(c0,ctarget,pm,index=0)
        if n_stages == float('inf'):
            mol_flow = 0
        else:
            mol_flow = stages[n_stages-1]['cp'][0]*stages[n_stages-1]['Fp'] * 3600 # mol/h
    #print(cfeed)

    costs = {}
    membrane_cost = None
    waste_cost = None
    if mol_flow == 0 or n_stages == float('inf') or total_area == float('inf'):
        productivity = 0
        for key in electricity_usd.keys():
            costs[key] = float('inf')
        membrane_cost = float('inf')
        waste_cost = float('inf')
    else:
        productivity = mol_flow / total_area # mol/m2/h
        membrane_cost_per_mol_product = (cost_membrane * total_area) / (membrane_lifetime * mol_flow)

        waste_catalyst_cost_per_hour = catalyst_cost * (1 - recovery) * stages[0]['c0'][0]*stages[0]['F0'] * 3600 # USD/h
        waste_catalyst_cost_per_mol_product = waste_catalyst_cost_per_hour / mol_flow

        membrane_cost = membrane_cost_per_mol_product
        waste_cost = waste_catalyst_cost_per_mol_product

        for key in electricity_usd.keys():
            costs[key] = (electricity_usd[key] / 3.6e6) * molar_energy_demand + membrane_cost + waste_cost
            if costs[key] == float('nan'):
                costs[key] = float('inf')

    if molar_energy_demand == float('nan'):
        molar_energy_demand = float('inf')
    if recovery == float('nan'):
        recovery = 0
    if n_stages == float('nan'):
        n_stages = float('inf')
    if total_area == float('nan'):
        total_area = float('inf')

    return molar_energy_demand, recovery, n_stages, total_area, productivity, costs, membrane_cost, waste_cost