# %%
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import datetime as dt

import pypsa

DIGITS = 5
miles_to_km = 1.60934
baseMVA = 100.

# NOTE: our objective is to use pypsa's import_from_csv_folder method
def _read_csv(table, clean_cols=True):
    start_path = os.path.join(os.path.dirname(__file__), '..', "..", "SourceData")
    df =  pd.read_csv(os.path.join(start_path, table))
    if not clean_cols:
        return df
    df.columns = df.columns.str.lower().str.replace(' ', '')
    return df

# %%
def create_buses():
    # Bus data
    column_map_bus = {
        'busname':'name',
        'busid':'busid',
        'basekv':'v_nom',
        'bustype':'type',
        'lng':'x',
        'lat':'y',
        False:'carrier', # all AC
        False:'unit', # not used
        'vmag':'v_mag_pu_set',
        False:'v_mag_pu_min', # not used in pypsa
        False:'v_mag_pu_max', # not used in pypsa
        'mwload':'mwload', # for distributing load
        'area':'area' # for clustering
    }
    column_dtype_bus = {
        'name':'str',
        'busid':'str'
    }
    busdata = _read_csv('bus.csv')
    busdata = (busdata
               .drop(columns=[col for col in busdata.columns if col not in column_map_bus.keys()])
               .rename(columns=column_map_bus)
               .astype(column_dtype_bus))
    busdata.name = busdata.name.str.upper()
    busdata = busdata.set_index('name')
    busdata.loc[busdata.type == 'Ref', 'type'] = 'Slack'
    # make names numeric
    return busdata

bd = create_buses()
bd

# %%
def create_branches():
    # Branch data
    column_map_branch = {
        'uid':'name',
        'frombus':'busid0',
        'tobus':'busid1',
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
        'busid0':'str',
        'busid1':'str'
    }
    branchdata = _read_csv('branch.csv')
    branchdata = (branchdata
                  .drop(columns=[col for col in branchdata.columns if col not in column_map_branch.keys()])
                  .rename(columns=column_map_branch)
                  .astype(column_dtype_branch)
                  )
    # merge on bus v_noms
    busdata = create_buses()
    branchdata = pd.merge(left=branchdata, 
                          right=busdata.reset_index()[['name','busid', 'v_nom']]
                                 .rename(columns={'busid':'busid0', 'v_nom':'v_nom0', 'name':'bus0'}), 
                          on='busid0')
    branchdata = pd.merge(left=branchdata, 
                          right=busdata.reset_index()[['name','busid', 'v_nom']]
                                 .rename(columns={'busid':'busid1', 'v_nom':'v_nom1', 'name':'bus1'}), 
                          on='busid1')
    
    branchdata = branchdata.set_index('name').drop(columns=['busid0', 'busid1'])
    branchdata['length'] = branchdata.length * miles_to_km

    
    branchdata.x *= branchdata.v_nom0**2 / baseMVA
    branchdata.r *= branchdata.v_nom0**2 / baseMVA
    branchdata.b /= branchdata.v_nom0**2 * baseMVA
    # split into transformers and lines
    # drop line C35
    branchdata = branchdata.loc[branchdata.index != 'C35']
    mask_trans = branchdata.v_nom0 != branchdata.v_nom1
    linedata = branchdata.loc[~mask_trans]
    transdata = branchdata.loc[mask_trans]
    return linedata, transdata

ld, td = create_branches()
ld


# %%
dt.datetime(12, 5, 7, 12).time()

# %%
def create_series(pointers:pd.DataFrame, parameter_map:dict, scale=True ):
    # helper to pull and merge all datasets into appropriate dataframes (separate by parameter)
    date_cols = ['Year', 'Month', 'Day', 'Period']
    series_dict = {}
    parameters = pointers['parameter'].drop_duplicates().values
    for p in parameters:
        df = pd.DataFrame(columns=date_cols)
        filepaths = pointers.loc[pointers.parameter == p, 'datafile'].drop_duplicates().values
        for fp in filepaths:
            f = _read_csv(fp, clean_cols=False)
            df = pd.merge(left=df, right=f, how='outer', on=date_cols)

        df.Period -=1
        df.index = pd.to_datetime(df[date_cols].rename(columns={'Period':'Hour'}))
        df.index.name = 'snapshots'
        series_dict[parameter_map[p]] = df.drop(columns=['Year', 'Month', 'Day', 'Period'])

    # normalize to per-unit values
    if scale:
        for i, p in pointers.iterrows():
            series_dict[parameter_map[p.parameter]][p.object] /= p.scalingfactor 

    return series_dict


# %%
def create_genseries():
    # get time-varying pointers and restrict to day ahead generation
    pointers = _read_csv('timeseries_pointers.csv')
    pointers = pointers.loc[(pointers.simulation == 'DAY_AHEAD')
                            & (pointers.category == 'Generator')
                            & (pointers.object != '212_CSP_HEAD_STORAGE')
                            # & (pointers.parameter != 'PMin MW') # network doesn't optimize when we include this
                            ]
    parameter_map = {'PMax MW':'p_max_pu', 'PMin MW':'p_min_pu'}
    genseries = create_series(pointers, parameter_map)
    
    return genseries

gs = create_genseries()
gs

# %%
def create_gens(unit_commitment):
    # Create generator data
    column_map_gen = {
        'genuid':'name',
        'busid':'busid',
        False:'control', # defined by bus (below)
        'unittype':'type',
        False:'p_nom', # set below
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
               .rename(columns=column_map_gen)
               .astype({'busid': 'str'}))
    # NOTE: Assume generator control is set by bus
    # NOTE: Dropping synchronous condensers, of which there are 3. Other programs also do this
    # NOTE: Dropping CSP, of which there is 1
    # non-storage generation
    intermittent_list = ['Solar', 'Wind']
    nonintermittent_list = ['NG', 'Oil', 'Coal', 'Nuclear', 'Hydro']
    gendata = gendata.loc[gendata.carrier.isin(intermittent_list + nonintermittent_list)].copy()
    gendata = gendata.loc[gendata.category != 'CSP']
    gendata = pd.merge(left=gendata, 
                    right=create_buses().reset_index()[['name', 'busid', 'type']].rename(columns={'name':'bus','type':'control'}), 
                        on='busid').set_index('name')
    gendata.loc[gendata.control == 'Ref', 'control'] = 'Slack'
    # define p_nom
    gendata.loc[gendata.carrier.isin(intermittent_list), 'p_nom'] = gendata.pmaxmw
    gendata.loc[gendata.carrier.isin(nonintermittent_list), 'p_nom'] = gendata.mwinj
    gendata['p_max_pu'] = gendata.pmaxmw / gendata.p_nom
    # NOTE: network can't find feasible solution when pminmw exists
    # gendata['p_min_pu'] = gendata.pminmw / gendata.p_nom

    # NOTE: Assume hourly snapshot
    gendata['ramp_limit_up'] = np.nan
    gendata.loc[gendata.carrier.isin(nonintermittent_list), 'ramp_limit_up'] = (
        gendata.loc[gendata.carrier.isin(nonintermittent_list), 'rampratemw/min'] / 
        gendata.loc[gendata.carrier.isin(nonintermittent_list), 'p_nom']*60)
    # unit commitments
    gendata['committable'] = unit_commitment
    gendata.loc[gendata.carrier.isin(intermittent_list), 'committable'] = False
    # NOTE: Assuming warm starts for all machines (options are cold, warm, and hot)
    gendata['start_up_cost'] = gendata['nonfuelstartcost$'] + gendata.startheatwarmmbtu*gendata['fuelprice$/mmbtu'] 
    gendata['shut_down_cost'] = gendata['nonfuelshutdowncost$']

    # assume nuclear is always running (since RTS-GMLC starttime is set to 9999 and also has no startup costs)
    gendata['up_time_before'] = 0
    gendata.loc[gendata.carrier == 'Nuclear', 'up_time_before'] = gendata.loc[gendata.carrier == 'Nuclear', 'min_up_time']
    # assume constant marginal cost at output_pct_2 instead of one based on production relative to max capacity
    # Also and convert $/mmbtu to $/MWh
    # TODO: Future alternative solution: split each generator into separate generators with different costs
    gendata['marginal_cost'] = (
        gendata['fuelprice$/mmbtu'] * (1/1e6) * (gendata.hr_avg_0 + gendata.hr_incr_2) * (1e3))
    
    # clean up columns
    keep_cols = ['busid','bus','control', 'type', 
                 'p_nom', 'p_min_pu', 'p_max_pu', 
                'carrier','marginal_cost', 'marginal_cost_quadratic', 
                'committable', 'start_up_cost', 'shut_down_cost', 'stand_by_cost', 
                'min_up_time', 'min_down_time', 'up_time_before', 'down_time_before', 
                'ramp_limit_up', 'ramp_limit_down', 'ramp_limit_start_up', 'ramp_limit_shut_down']
    gendata = gendata.drop(columns=[col for col in gendata.columns if col not in keep_cols])

    # get generator series
    genseries = create_genseries()

    return gendata, genseries

gd, gs = create_gens(True)
gd


# %%
# TODO -- storage data
# storgendata = gendata.loc[gendata.carrier == 'Storage'].copy()

# %%
def create_loads():

    loaddata = create_buses().astype({'area':'str'})
    loaddata['bus'] = loaddata.index
    loaddata.index.rename('name', inplace=True)
    loaddata['carrier'] = 'AC'
    loaddata['pct_areaload'] = loaddata.mwload / loaddata.groupby('area')['mwload'].transform('sum')
    # clean up columns and drop 0-load
    loaddata = loaddata.loc[loaddata.pct_areaload > 0, ['bus', 'busid', 'carrier', 'area', 'pct_areaload']]
    # load series data
    # get time-varying pointers and restrict to day ahead generation
    pointers = _read_csv('timeseries_pointers.csv')
    pointers = pointers.loc[(pointers.simulation == 'DAY_AHEAD') & (pointers.category == 'Area')]
    parameter_map = {'MW Load':'p_set',}
    loadseries = create_series(pointers, parameter_map, scale=False)['p_set']
    for i, b in loaddata.iterrows():
        loadseries[b.bus] = loadseries[b.area] * b.pct_areaload

    loadseries = loadseries.drop(columns=['1', '2', '3'])

    return loaddata, loadseries

ld, ls = create_loads()
ld.shape


# %%
def write_pypsa_network_csvs(snapshots, unit_commitment, save_path):
    # buses
    create_buses().drop(columns='busid').to_csv(os.path.join(save_path, 'buses.csv'))
    # lines
    lines, trans = create_branches()
    lines.to_csv(os.path.join(save_path, 'lines.csv'))
    trans.to_csv(os.path.join(save_path, 'transformers.csv'))
    # generators
    gendata, genseries = create_gens(unit_commitment)
    gendata.drop(columns='busid').to_csv(os.path.join(save_path, 'generators.csv'))
    for k in genseries.keys():
        (genseries[k]
         .loc[snapshots]
         .to_csv(os.path.join(save_path, f'generators-{k}.csv'), index=True))
    # loads
    loaddata, loadseries = create_loads()
    loaddata.drop(columns='busid').to_csv(os.path.join(save_path, 'loads.csv'))
    loadseries.loc[snapshots].to_csv(os.path.join(save_path, 'loads-p_set.csv'), index=True)
    
    return

# %%