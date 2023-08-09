# %%
import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pypsa

path_file = os.path.dirname(__file__)
sys.path.append(os.path.join(path_file, '..'))
from source_to_pypsa_csv import write_pypsa_network_csvs

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


# %%
# IMPORT MY CASE
# Also build  and visualize network
ss_length = 24
ss_start = 0
snapshots = pd.date_range(start='1/1/2020', end='1/1/2021', freq='H')[ss_start:ss_start+ss_length]
network_data = write_pypsa_network_csvs(snapshots, unit_commitment=False, save_path=os.path.join(path_file, '..', 'rts-gmlc/'))
n = pypsa.Network(snapshots=snapshots)
n.import_from_csv_folder(os.path.join(path_file, '..', 'rts-gmlc/'))

# plot network
n.plot(geomap=False)
plt.show()

# plot timeseries
def plot_series(series, title, legend=False):
    for ts in series:
        plt.plot(series[ts], alpha=0.25, label=ts)
    plt.title(f'{title} profiles (snapshots={len(snapshots)})')
    if legend:
        plt.legend()
    plt.show()

plot_series(n.loads_t.p_set, 'Load')
plot_series(n.generators_t.p_max_pu[[col for col in n.generators_t.p_max_pu if 'HYDRO' in col]], 'Hydro generators max pu', False)
plot_series(n.generators_t.p_max_pu, 'Generators max pu', False)


# %%
# Try optimizing overall
n = pypsa.Network(snapshots=snapshots)
n.import_from_csv_folder(os.path.join(path_file, '..', 'rts-gmlc/'))
n.optimize.create_model()
n.optimize.solve_model()
print(n.model)
print('Consistencies:', n.consistency_check())
print(n.buses_t.marginal_price)

# %%
# figure out load and generation by day

# get maximum generation by generator by hour
maxdf = pd.DataFrame(columns=list(snapshots) + ['bus'])
for g in n.generators.index:
    p_nom = n.generators.loc[g, 'p_nom']
    p_max_pu = n.generators.loc[g, 'p_max_pu']
    if (g in n.generators_t['p_max_pu'].columns):
        ser = n.generators_t['p_max_pu'][g]*p_nom
    else:
        ser = pd.Series(np.ones(len(snapshots))*p_nom*p_max_pu, name=g)
    ser['bus'] = n.generators.loc[g, 'bus']
    maxdf.loc[g] = ser

# get minimum generation by generator by hour
mindf = pd.DataFrame(columns=list(snapshots) + ['bus'])
for g in n.generators.index:
    p_nom = n.generators.loc[g, 'p_nom']
    p_min_pu = n.generators.loc[g, 'p_min_pu']
    ser = pd.Series(np.ones(len(snapshots))*p_min_pu*p_nom, name=g)
    ser['bus'] = n.generators.loc[g, 'bus']
    mindf.loc[g] = ser

# get maximum generation by generators with no minimum
maxdf_nomin = pd.DataFrame(columns=list(snapshots) + ['bus'])
for g in n.generators.index:
    p_nom = n.generators.loc[g, 'p_nom']
    p_max_pu = n.generators.loc[g, 'p_max_pu']
    if n.generators.loc[g, 'p_min_pu'] <= 0:
        if (g in n.generators_t['p_max_pu'].columns):
            ser = n.generators_t['p_max_pu'][g]*p_nom
        else:
            ser = pd.Series(np.ones(len(snapshots))*p_nom*p_max_pu, name=g)
        ser['bus'] = n.generators.loc[g, 'bus']
        maxdf_nomin.loc[g] = ser

# get maximum generation by bus
plt.plot(maxdf.groupby(['bus']).sum().sum(), label='gen p_max')
plt.plot(maxdf_nomin.groupby(['bus']).sum().sum(), label='gen p_max no min')
plt.plot(mindf.groupby('bus').sum().sum(), label='gen p_min')
plt.plot(n.loads_t.p_set.sum(axis=1), label='load p_set')
plt.title('Energy by hour')
plt.legend()
plt.show()



# %%
# QUESTION: Does hydro produce at quantity beneath max capacity?
print(n.generators.loc[n.generators.type == 'HYDRO', 'p_nom'])
print(n.generators_t.p[[col for col in n.generators_t.p.columns if 'HYDRO' in col]])
# ANSWER: Yes! So this explains why there was initially no feasible optimal solution
# QUESTION: What about rooftop photovoltaics?
print(n.generators.loc[n.generators.type == 'RTPV', 'p_nom'])
print(n.generators_t.p[[col for col in n.generators_t.p.columns if 'RTPV' in col]])
# ANSWER: Also yes!


# %%
# TROUBLESHOOT: Does model run without branch constraints
# Test optimization without Kirchoff (transport model)
n = pypsa.Network(snapshots=snapshots)
n.import_from_csv_folder(os.path.join(path_file, '..', 'rts-gmlc/'))
n.optimize.create_model()
n.model.constraints.remove("Kirchhoff-Voltage-Law")
n.optimize.solve_model()
print(n.model)
print('Consistencies:', n.consistency_check())
n.buses_t.marginal_price


# %%
# TROUBLESHOOT: Does model run without load requirements
n = pypsa.Network(snapshots=snapshots)
n.import_from_csv_folder(os.path.join(path_file, '..', 'rts-gmlc/'))
n.optimize.create_model()
n.optimize.add_load_shedding(marginal_cost=1e2, sign=-1)
n.optimize.solve_model()
print(n.model)
print('Consistencies:', n.consistency_check())
n.buses_t.marginal_price


# %%
# TROUBLESHOOT: Can we 2-step it? First solve MILP, then fix generator status and optimize?
n = pypsa.Network(snapshots=snapshots)
n.import_from_csv_folder(os.path.join(path_file, '..', 'rts-gmlc/'))
# 1. Solve MILP
model = pypsa.opf.network_lopf_build_model(n, n.snapshots)
opt = pypsa.opf.network_lopf_prepare_solver(n, solver_name='glpk')
opt.solve(model).write()
# 2. fix mixed integer results and optimize power flow
model.generator_status.fix()
n.results = opt.solve(model)
pypsa.opf.extract_optimisation_results(n, n.snapshots)
n.buses_t.marginal_price
