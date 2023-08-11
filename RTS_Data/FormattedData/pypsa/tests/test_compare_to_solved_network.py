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
# FINDING: Many more generators in the solved network, but I think these are added hydrogen generators not in RTS-GMLC
# FINDING: Excluding hydrogen generators, the solved network is missing generator 113_CT_1. Otherwise, generator indexs are the same
print('Solved gens:\n', len(nc.generators))
print('My gens:\n', len(n.generators))
print('Solved gens w/o hydrogen:\n', len(nc.generators.loc[nc.generators.carrier != 'hydrogen']))
print('Solved gens not in mine:\n', nc.generators.loc[~nc.generators.index.isin(n.generators.index)].index) # only fuel cells
print('My gens not in solved:\n', n.generators.loc[~n.generators.index.isin(nc.generators.index)].index) # 113_CT_1
n.generators.loc['113_CT_1']

# %%
# COMPARE: GENERATORS
# FINDING: Solved network has no per-unit settings, only nominal. This must be a relic of the older network version
# FINDING: Solved netwrok has no minimum settings on thermal generators. RTS-GMLC however provides these
# FINDING: p_nom set to 0 for several values of the solved network. These are the same values where TRS-GMLC are set to zero
#          because these values have timeseries p_nom_max/min. This does not seem to be a problem since the newer version relies on p_max_pu instead of p_nom_max
# FINDING: Solved network does not include start-up costs. My network does
# FINDING: Marginal costs differ; mine are ~2x the values of the solved network. It looks like solved network might just use HR_AVG_0 and not HR_INC_2, which I use
print('Solved max/min p (nominal)\nMax:\n', nc.generators.p_nom_max.value_counts(), '\nMin:\n', nc.generators.p_nom_min.value_counts())
print('Solved max/min p (pu)\nMax:\n', nc.generators.p_max_pu.value_counts(), '\nMin:\n', nc.generators.p_min_pu.value_counts())

# compare p_nom
gen_compare = pd.merge(left=nc.generators, right=n.generators, how='inner', left_index=True, right_index=True)
plt.scatter(gen_compare.p_nom_x, gen_compare.p_nom_y)
plt.axline(xy1=(0,0), slope=1)
plt.xlabel('solved network p_nom')
plt.ylabel('my network p_nom')
plt.show()

# compare marginal costs
plt.scatter(gen_compare.marginal_cost_x, gen_compare.marginal_cost_y)
plt.axline(xy1=(0,0), slope=1)
plt.xlabel('solved network p_nom')
plt.ylabel('my network p_nom')
plt.show()

# compare startup costs
plt.scatter(gen_compare.start_up_cost_x, gen_compare.start_up_cost_y)
plt.axline(xy1=(0,0), slope=1)
plt.xlabel('solved network p_nom')
plt.ylabel('my network p_nom')
plt.show()


# %%
# COMPARE: GENERATOR TIMESERIES
# FINDING: Solved network only has p_max_pu. No p_min_pu
# FINDING: My network and the solved network both have the same generators
# FINDING: max_pu values are close enough, but not the same
assert(len(nc.generators_t.p_max_pu.columns) == len(n.generators_t.p_max_pu.columns))
assert(len([col for col in n.generators_t.p_max_pu.columns if col not in nc.generators_t.p_max_pu.columns]) == 0)
assert(len([col for col in nc.generators_t.p_max_pu.columns if col not in n.generators_t.p_max_pu.columns]) == 0)

# what about values?
for col in n.generators_t.p_max_pu.columns:
    if 'rtpv' in col.lower():
        plt.plot(nc.generators_t.p_max_pu[col], n.generators_t.p_max_pu[col], color='red', linewidth=1, alpha=0.15, label='rtpv')
    elif '_pv' in col.lower():
        plt.plot(nc.generators_t.p_max_pu[col], n.generators_t.p_max_pu[col], color='yellow', linewidth=1, alpha=0.15, label='upv')
    elif 'hydro' in col.lower():
        plt.plot(nc.generators_t.p_max_pu[col], n.generators_t.p_max_pu[col], color='blue', linewidth=1, alpha=0.5, label='hydro')
    elif 'wind' in col.lower():
        plt.plot(nc.generators_t.p_max_pu[col], n.generators_t.p_max_pu[col], color='grey', linewidth=1, alpha=0.5, label='wind')
plt.show()

# %%
# COMPARE: OPTIMIZATION RESULTS


# %%
n.generators_t.p_max_pu.columns
# %%
n.transformers.s_nom
# %%
