# %%
import sys, os
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
# Build network
ss_length = 24*7
ss_start = 0
snapshots = pd.date_range(start='1/1/2020', end='1/1/2021', freq='H')[ss_start:ss_start+ss_length]
network_data = write_pypsa_network_csvs(snapshots, unit_commitment=True, save_path=os.path.join(path_file, '..', 'rts-gmlc/'))
n = pypsa.Network(snapshots=snapshots)
n.import_from_csv_folder(os.path.join(path_file, '..', 'rts-gmlc/'))


# %%
# TRIM VARIABLES
n.buses = n.buses[['v_nom', 'area', 'x', 'y']]
n.lines = n.lines[['bus0', 'bus1', 'r', 'x', 'b', 's_nom', 'length']]

# %%
# CLUSTER NETWORK
nc = n.cluster.cluster_by_busmap(n.buses.area)
nc.plot()

# %%
# OPTIMIZE
nc.optimize()
print(nc.model)
print(nc.buses_t.marginal_price)

# %%
# 2-step optimization to get all results. First solve MILP, then fix generator status and optimize?
# 1. Solve MILP
model = pypsa.opf.network_lopf_build_model(nc, nc.snapshots)
opt = pypsa.opf.network_lopf_prepare_solver(nc, solver_name='glpk')
opt.solve(model).write()
# 2. fix mixed integer results and optimize power flow
model.generator_status.fix()
nc.results = opt.solve(model)
pypsa.opf.extract_optimisation_results(nc, nc.snapshots)
nc.buses_t.marginal_price
# %%
