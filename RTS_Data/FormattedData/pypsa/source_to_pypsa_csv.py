# %%
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import pypsa

DIGITS = 5
miles_to_km = 1.60934
baseMVA = 100.

# NOTE: our objective is to use pypsa's import_from_csv_folder method
def _read_csv(table, clean_cols=True):
    start_path = os.path.join("..", "..", "SourceData")
    df =  pd.read_csv(os.path.join(start_path, table))
    if not clean_cols:
        return df
    df.columns = df.columns.str.lower().str.replace(' ', '')
    return df

# %%
def create_buses():
    # Bus data
    column_map_bus = {
        'busid':'name',
        'basekv':'v_nom',
        'bustype':'type',
        'lng':'x',
        'lat':'y',
        False:'carrier', # all AC
        False:'unit', # not used
        'vmag':'v_mag_pu_set',
        False:'v_mag_pu_min', # not used in pypsa
        False:'v_mag_pu_max' # not used in pypsa
    }
    column_dtype_bus = {
        'name':'str'
    }
    busdata = _read_csv('bus.csv')
    busdata = (busdata
               .drop(columns=[col for col in busdata.columns if col not in column_map_bus.keys()])
               .rename(columns=column_map_bus)
               .astype(column_dtype_bus))
    
    busdata = busdata.set_index('name')
    busdata.loc[busdata.type == 'Ref', 'type'] = 'Slack'
    # make names numeric
    return busdata

bd = create_buses()

# %%
# TODO: confirm that we dont' need to worry about Tr Ratio (none of the other mappings handle it)
def create_branches():
    # Branch data
    column_map_branch = {
        'uid':'name',
        'frombus':'bus0',
        'tobus':'bus1',
        False:'type',
        'x':'x',
        'r':'r',
        False:'g',
        'b':'b',
        'contrating':'s_nom', # options included Cont Rating, LTE Rating, and STE Rating
        False:'s_nom_extendable', # for capacity expansion
        False:'s_nom_min', # for capacity expansion
        False:'s_nom_max', # for capacity expansion
        False:'s_max_pu', # for capacity expansion
        False:'capital_cost', # for capacity expansion
        False:'build_year',
        False:'lifetime',
        'length':'length', # defined below
        False:'carrier',
        False:'terrain_factor', # for capacity expansion
        False:'num_parallel', # ignored when type isn't set
        False:'v_ang_min', # not currently in use by pypsa
        False:'v_ang_max' # not currently in use by pypsa
    }
    column_dtype_branch = {
        'name':'str',
        'bus0':'str',
        'bus1':'str'
    }
    branchdata = _read_csv('branch.csv')
    branchdata = (branchdata
                  .drop(columns=[col for col in branchdata.columns if col not in column_map_branch.keys()])
                  .rename(columns=column_map_branch)
                #   .astype(column_dtype_branch)
                  )
    # merge on bus v_noms
    branchdata = pd.merge(left=branchdata, 
                          right=(_read_csv('bus.csv')[['busid', 'basekv']]
                                 .rename(columns={'busid':'bus0', 'basekv':'v_nom0'})), 
                          on='bus0')
    branchdata = pd.merge(left=branchdata, 
                          right=(_read_csv('bus.csv')[['busid', 'basekv']]
                                 .rename(columns={'busid':'bus1', 'basekv':'v_nom1'})), 
                          on='bus1')
    branchdata = branchdata.set_index('name')
    branchdata['length'] = branchdata.length * miles_to_km

    # TEMP: temporarily set linetypes
    branchdata = branchdata.loc[branchdata.index != 'C35']
    # branchdata.s_nom *= 100

    branchdata.x *= branchdata.v_nom0**2 / baseMVA
    branchdata.r *= branchdata.v_nom0**2 / baseMVA
    branchdata.b /= branchdata.v_nom0**2
    return branchdata

bd = create_branches()
bd


# %%
def create_series(pointers:pd.DataFrame, parameter_map:dict, merge_cols:list, scale=True ):
    # helper to pull and merge all datasets into appropriate dataframes (separate by parameter)
    series_dict = {}
    parameters = pointers['parameter'].drop_duplicates().values
    for p in parameters:
        df = pd.DataFrame(columns=merge_cols)
        filepaths = pointers.loc[pointers.parameter == p, 'datafile'].drop_duplicates().values
        for fp in filepaths:
            f = _read_csv(fp, clean_cols=False)
            df = pd.merge(left=df, right=f, how='outer', on=merge_cols)
        series_dict[parameter_map[p]] = df

    # normalize to per-unit values
    if scale:
        for i, p in pointers.iterrows():
            series_dict[parameter_map[p.parameter]][p.object] /= p.scalingfactor 

    return series_dict


# %%
def create_genseries():
    # get time-varying pointers and restrict to day ahead generation
    pointers = _read_csv('timeseries_pointers.csv')
    pointers = pointers.loc[(pointers.simulation == 'DAY_AHEAD') & 
                            (pointers.category == 'Generator') & 
                            (pointers.object != '212_CSP_HEAD_STORAGE')]
    parameter_map = {'PMax MW':'p_max_pu', 'PMin MW':'p_min_pu'}
    merge_cols = ['Year', 'Month', 'Day', 'Period']
    genseries = create_series(pointers, parameter_map, merge_cols)
    
    return genseries

genseries = create_genseries()
genseries

# %%
def create_gens(unit_commitment):
    # Create generator data
    column_map_gen = {
        'genuid':'name',
        'busid':'bus',
        False:'control', # defined by bus (below)
        'unittype':'type',
        'mwinj':'p_nom', # real power injection set-point
        False:'p_nom_extendable', # for capacity expansion
        False:'p_nom_min', # for capacity expansion
        False:'p_nom_max', # for capacity expansion
        False:'p_min_pu', # time series data
        False:'p_max_pu', # time series data
        False:'p_set', # TODO: do we need this?
        False:'q_set', # TODO: do we need this?
        False:'sign', # generation is positive, load is negative
        'fuel':'carrier',
        False:'marginal_cost', # calculated below
        False:'marginal_cost_quadratic', # not used
        False:'build_year', # not necessary
        False:'lifetime', # not necessary
        False:'capital_cost', # unavailable
        False:'efficiency', # providing electrical values directly
        False:'committable', # defined below
        False:'start_up_cost', # defined below
        False:'shut_down_cost', # defined below
        False:'stand_by_cost', # unavailable
        'minuptimehr':'min_up_time',
        'mindowntimehr':'min_down_time',
        False:'up_time_before', # assume nothing at start
        False:'down_time_before', # assume nothing at start
        False:'ramp_limit_up', # defined below
        False:'ramp_limit_down', # not available
        False:'ramp_limit_start_up', # not available
        False:'ramp_limit_shut_down', # not available
        False:'weight' # used for aggregation
    }
    gendata = _read_csv('gen.csv')
    gendata = (gendata
            #    .drop(columns=[col for col in gendata.columns if col not in column_map_gen.keys()])
               .rename(columns=column_map_gen))
    # NOTE: Assume generator control is set by bus
    # NOTE: Dropping synchronous condensers, of which there are 3
    # NOTE: Dropping CSP, of which there is 1
    # non-storage generation
    gendata = gendata.loc[gendata.carrier.isin(['NG', 'Oil', 'Coal', 'Nuclear', 'Hydro', 'Solar', 'Wind'])].copy()
    gendata = gendata.loc[gendata.category != 'CSP']
    gendata = pd.merge(left=gendata, 
                    right=_read_csv('bus.csv')[['busid', 'bustype']].rename(columns={'busid':'bus','bustype':'control'}), 
                        on='bus').set_index('name')
    gendata.loc[gendata.control == 'Ref', 'control'] = 'Slack'
    # define per-unit max/min universally. this will hopefully get overridden by timeseries data
    mask_pnom = np.where(gendata.p_nom == 0, False, True)
    gendata['p_max_pu'] = 1.
    gendata.loc[mask_pnom, 'p_max_pu'] = gendata.loc[mask_pnom, 'pmaxmw'] / gendata.loc[mask_pnom, 'p_nom']
    gendata['p_min_pu'] = 0.
    gendata.loc[mask_pnom, 'p_min_pu'] = gendata.loc[mask_pnom, 'pminmw'] / gendata.loc[mask_pnom, 'p_nom']
    # NOTE: Assume hourly snapshot
    gendata['ramp_limit_up'] = np.nan
    gendata.loc[mask_pnom, 'ramp_limit_up'] = gendata.loc[mask_pnom, 'rampratemw/min'] / gendata.loc[mask_pnom, 'p_nom']*60
    # unit commitments
    gendata['committable'] = unit_commitment
    gendata.loc[gendata.carrier.isin(['Solar', 'Wind']), 'committable'] = False
    # NOTE: Assuming warm starts for all machines (options are cold, warm, and hot)
    gendata['start_up_cost'] = gendata['nonfuelstartcost$'] + gendata.startheatwarmmbtu*gendata['fuelprice$/mmbtu'] 
    gendata['shut_down_cost'] = gendata['nonfuelshutdowncost$']

    #TEMP
    # gendata.p_nom *= 1000
    
    # assume nuclear is always running
    gendata.loc[gendata.carrier == 'Nuclear', 'up_time_before'] = gendata.loc[gendata.carrier == 'Nuclear', 'min_up_time']
    gendata.loc[gendata.carrier != 'Nuclear', 'up_time_before'] = 0
    # assume constant marginal cost at output_pct_2 instead of one based on production relative to max capacity
    # Also and convert $/mmbtu to $/MWh
    # TODO: Future alternative solution: split each generator into separate generators with different costs
    gendata['marginal_cost'] = (
        gendata['fuelprice$/mmbtu'] * (1/1e6) * (gendata.hr_avg_0 + gendata.hr_incr_2) * (1e3))

    # get generator series
    genseries = create_genseries()

    return gendata, genseries

gd, gs = create_gens(True)
gd


# %%
# TODO -- storage data
# storgendata = gendata.loc[gendata.carrier == 'Storage'].copy()
# storgendata


# %%
def create_loadseries():
    # get time-varying pointers and restrict to day ahead generation
    pointers = _read_csv('timeseries_pointers.csv')
    pointers = pointers.loc[(pointers.simulation == 'DAY_AHEAD') & (pointers.category == 'Area')]
    parameter_map = {'MW Load':'p_set',}
    merge_cols = ['Year', 'Month', 'Day', 'Period']
    loadseries = create_series(pointers, parameter_map, merge_cols, scale=True)
    loadseries = loadseries['p_set']
    return loadseries


# %%
def create_loads():
    # Load data
    column_map_load = {
        False:'name', # defined below	
        'busid':'bus', 
        False:'carrier', # defined below	
        False:'type', # not used
        False:'p_set',	# set in timeseries
        False:'q_set',	# not used
        False:'sign', # use default, that load is negative
        'mwload':'mwload', # used to distribute load
        'area':'area' # used to distribute load
    }
    column_dtype_load = {
            'bus':'str',
            'area':'str'
        }
    loaddata = _read_csv('bus.csv')
    loaddata = (loaddata
                .drop(columns=[col for col in loaddata.columns if col not in column_map_load.keys()])
                .rename(columns=column_map_load)
                .astype(column_dtype_load)
                .set_index('bus', drop=False))
    loaddata.index.rename('name', inplace=True)
    loaddata['carrier'] = 'AC'

    loadseries = create_loadseries()
    for i, b in loaddata.iterrows():
        loadseries[b.bus] = loadseries[b.area] * b.mwload

    loadseries = loadseries.drop(columns=['1', '2', '3', 'Year', 'Month', 'Day', 'Period'])

    return loaddata, loadseries

ld, ls = create_loads()
ld

# %%
def write_pypsa_network_csvs(snapshots, start_index, unit_commitment):
    start_path = os.path.join('..', "..", "FormattedData", 'pypsa', 'rts-gmlc')
    # buses
    create_buses().to_csv(os.path.join(start_path, 'buses.csv'))
    # lines
    create_branches().to_csv(os.path.join(start_path, 'lines.csv'))
    # generators
    gendata, genseries = create_gens(unit_commitment)
    gendata.to_csv(os.path.join(start_path, 'generators.csv'))
    for k in genseries.keys():
        (genseries[k]
         .iloc[start_index:start_index+snapshots]
         .drop(columns=['Year', 'Month', 'Day', 'Period'])
         .to_csv(os.path.join(start_path, f'generators-{k}.csv'), index=True))
    # loads
    loaddata, loadseries = create_loads()
    loaddata.to_csv(os.path.join(start_path, 'loads.csv'))
    loadseries.iloc[start_index:start_index+snapshots].to_csv(os.path.join(start_path, 'loads-p_set.csv'), index=True)
    
    return

