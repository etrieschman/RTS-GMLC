# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pypsa

from source_to_pypsa_csv import write_pypsa_network_csvs

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


# %%
# Build  and visualize network
snapshots = 12
start_index = 0
network_data = write_pypsa_network_csvs(snapshots, start_index, unit_commitment=True)
n = pypsa.Network(snapshots=np.arange(snapshots))
n.import_from_csv_folder('../../FormattedData/pypsa/rts-gmlc')


# %%
n.buses = n.buses[['v_nom', 'area', 'x', 'y']]
n.lines = n.lines[['bus0', 'bus1', 'r', 'x', 'b', 's_nom', 'length']]

# %%
nc = n.cluster.cluster_by_busmap(n.buses.area)
nc.plot()

# %%
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
