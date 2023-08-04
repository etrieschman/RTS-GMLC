# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pypsa

from source_to_pypsa_csv import write_pypsa_network_csvs

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


# %%
# Build network
snapshots = 2
start_index = 6
network = pypsa.Network(snapshots=np.arange(snapshots))
network_data = write_pypsa_network_csvs(snapshots, start_index, unit_commitment=False)
network.import_from_csv_folder('../../FormattedData/pypsa/rts-gmlc')



# %%
# Visualize network
network.plot(geomap=False)
plt.show()

def plot_series(series, title, legend=False):
    for ts in series:
        plt.plot(series[ts], alpha=0.5, label=ts)
    plt.title(f'{title} profiles (snapshots={snapshots})')
    if legend:
        plt.legend()
    plt.show()

plot_series(network.loads_t.p_set, 'Load')
plot_series(network.generators_t.p_max_pu[[col for col in network.generators_t.p_max_pu if 'HYDRO' in col]], 'Hydro generators max pu', False)
plot_series(network.generators_t.p_min_pu[[col for col in network.generators_t.p_min_pu if 'HYDRO' in col]], 'Hydro generators min pu', False)
plot_series(network.generators_t.p_max_pu, 'Generators max pu', False)
plot_series(network.generators_t.p_min_pu, 'Generators min pu', False)


# %%
print('Min generation', (network.generators.p_min_pu * network.generators.p_nom).sum())
print('Load by snapshot', network.loads_t.p_set.sum(axis=1))
network.generators.p_nom



# %%
# TROUBLESHOOT: Does model run without branch constraints
# Test optimization without Kirchoff (transport model)
network.optimize.create_model()
network.model.constraints.remove("Kirchhoff-Voltage-Law")
network.optimize.solve_model()

# %%
# TROUBLESHOOT: Does model run without load requirements
network.optimize.create_model()
network.optimize.add_load_shedding(marginal_cost=1, sign=1)
network.optimize.solve_model()


# %%
# Try optimizing overall
network.optimize()


# %%
# TROUBLESHOOT: Can we 2-step it? First solve MILP, then fix generator status and optimize?
# 1. Solve MILP
model = pypsa.opf.network_lopf_build_model(network, network.snapshots)
opt = pypsa.opf.network_lopf_prepare_solver(network, solver_name='glpk')
opt.solve(model).write()
# 2. fix mixed integer results and optimize power flow
model.generator_status.fix()
network.results = opt.solve(model)
pypsa.opf.extract_optimisation_results(network, network.snapshots)
network.generators_t.p

