# NOTE: Solved network comes from Max Parzen through thread here
# https://groups.google.com/g/pypsa/c/0hlPf903Nrg/m/omvfs0miAgAJ?utm_medium=email&utm_source=footer
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
# pd.set_option('display.max_rows', None)

# %%
# IMPORT SOLVED CASE
nc = pypsa.Network()
nc.import_from_netcdf('./solved_network_RTS_GMLC_base.nc')

# %%
# IMPORT MY CASE
# Also build  and visualize network
snapshots = pd.date_range(start='1/1/2020', end='1/1/2021', freq='H')[:-1]
network_data = write_pypsa_network_csvs(snapshots, unit_commitment=False, save_path=os.path.join(path_file, '..', 'rts-gmlc/'))
n = pypsa.Network(snapshots=snapshots)
n.import_from_csv_folder(os.path.join(path_file, '..', 'rts-gmlc/'))

# %%
# COMPARE OVERALL
print('Solved:\n', nc)
print('Mine:\n', n)
# Will address 1 by 1
# * Can ignore global constraints in solved network. These relate to CO2
# * Can ignore carriers in solved network. These relate to expansion scenarios
# * Need to 

# %%
# COMPARE: BUSES
# Finding, nominal voltage is the same, v_mag_pu_set is different
mb = pd.merge(left=nc.buses, right=n.buses, how='outer', right_index=True, left_index=True, suffixes=('_sol', '_me'))
# v_nom is same!
assert(len(mb.loc[mb.v_nom_sol - mb.v_nom_me != 0]) == 0)
# v_mag_pu_set is not the same
plt.scatter(x=mb.v_mag_pu_set_sol, y=mb.v_mag_pu_set_me)
plt.xlabel('Solved')
plt.ylabel('Mine')
plt.show()
mb.loc[mb.v_mag_pu_set_sol - mb.v_mag_pu_set_me != 0, ['v_mag_pu_set_sol', 'v_mag_pu_set_me', 'v_nom_sol', 'v_nom_me']]

# %%
# COMPARE: LINES
# FINDING: solved model moves any lines connecting buses of different voltages to the transformer dataset
n.lines.loc[n.lines.v_nom0 != n.lines.v_nom1].shape

# %%
# COMPARE: LOAD
# FINDING: solved model drops any loads == 0. I implemented this
assert(len(nc.loads) == len(n.loads))

# %%
# COMPARE: LOAD TIMESERIES
# FINDING: They are the same, but indexed slightly differently
print('Solved:\n', nc.loads_t.p_set)
print('Mine:\n', n.loads_t.p_set)

# %%
# COMPARE: GENERATORS

# %%
