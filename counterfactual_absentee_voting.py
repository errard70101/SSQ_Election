# -*- coding: utf-8 -*-
"""
Created on Fri Jun  1 11:28:20 2018

@author: Shih-Yang Lin

This script calculates the counterfactual market share when absentee voting
is allowed in all markets.
"""

import numpy as np
import pandas as pd
from BLP_market_share import calculate_n_consumer_market_share as cal_mkt_shr
import matplotlib.pyplot as plt
import sys

# Read data
# =============================================================================

if sys.platform == 'darwin':
    file_path = "/Users/errard/Dropbox/SSQ_Election/RawData20180515.csv"
    market_population_file_name = "/Users/errard/Dropbox/SSQ_Election/adj_census_pop_T.csv"
    xi_file_name = "/Users/errard/Dropbox/SSQ_Election/xi.csv"
else:
    file_path = "C:/SSQ_Election/RawData20180515.csv"
    market_population_file_name = "C:/SSQ_Election/adj_census_pop_T.csv"
    xi_file_name = "C:/SSQ_Election/xi.csv"

def read_csv(file_path):
    '''
    This function read csv files to pandas dataframe.

    input:
        file_path: str.
    output:
        dta: pandas dataframe
    '''
    f = open(file_path)
    dta = pd.read_csv(f)
    f.close()
    return(dta)

dta = read_csv(file_path)
dta.loc[:, 'const'] = np.ones((len(dta), 1))

market_population = read_csv(market_population_file_name)
market_population = market_population.drop(columns = ['depcity', 'arrcity'])

xi = np.loadtxt(xi_file_name)
dta = dta.assign(xi = xi)

dta = pd.merge(dta, market_population,
               left_on = ['depcity', 'arrcity'],
               right_on = ['depcity_code', 'arrcity_code'],
               how = 'left')

# Set up hyper parameters
# =============================================================================
n_consumers = int(1e4)

estimates = [-5.246357, -.6285018, -.8258803, -1.604927, 1.448219,
              0.7957358,  1.093028, -0.2235259, -0.1370856, -6.126835, 1]
sigma = [2.816481]

data = dta[['mkt', 'priced', 'timed', 'pop_dep', 'moninc', 'edu',
            'unemploymentrate_arr', 'unemploymentrate_dep',
            'poll', 'mayor_arr', 'const', 'xi', 'census_pop', 'brand',
            'marketshare1']].copy()

end_var = [0]

seed = 654781324

# =============================================================================
# Create new product -- absentee voting

abs_vot = dta[['mkt', 'priced', 'timed', 'pop_dep', 'moninc', 'edu',
            'unemploymentrate_arr', 'unemploymentrate_dep',
            'poll', 'mayor_arr', 'const', 'xi', 'census_pop', 
            'brand', 'marketshare1']].groupby(['mkt']).agg({
               'priced': 'mean', 'timed': 'mean', 'pop_dep': 'mean',
               'moninc': 'mean', 'edu': 'mean', 'unemploymentrate_arr': 'mean',
               'unemploymentrate_dep': 'mean', 'poll': 'mean', 'mayor_arr': 'mean',
               'const': 'mean', 'xi': 'median', 'census_pop': 'mean', 'brand': 'mean',
               'marketshare1': 'mean'})

abs_vot.loc[:, 'priced'] = 0
abs_vot.loc[:, 'timed'] = 0
abs_vot.loc[:, 'brand'] = 0
abs_vot.loc[:, 'marketshare1'] = 0
abs_vot = abs_vot.reset_index()

# Add convenient voting options to each markets
data = data.append(abs_vot, ignore_index = True)

mkt = data['mkt'].copy()

#%% Calculating counterfactual market share.
data.loc[:, 'c_mkt_shr'] = cal_mkt_shr(estimates, 
            sigma, data.drop(columns = ['mkt', 'census_pop', 'brand', 'marketshare1']), 
            end_var, mkt, n_consumers, seed = seed)

#%%
total_pop = sum(market_population['census_pop'])
observed_voters = sum(dta['marketshare1'] * dta['census_pop'])
counterfactual_n_voters = sum(data['c_mkt_shr']* data['census_pop'])

#%% Calculate traffic volume change for voting
n_voters_difference = round(counterfactual_n_voters - observed_voters)
turnout_difference = round(n_voters_difference*100/total_pop, 2)
observed_turnout = round(observed_voters*100/total_pop, 2)
counterfactual_turnout = observed_turnout + turnout_difference
print('')
print('=========================================================')
print('If convenience voting is allowed:')
print('The number of voters will increase by:')
print(str(n_voters_difference))

print('The turnout rate will increase by:')
print(str(turnout_difference) + '%')

print('The turnout rate before convenience voting is allowed:')
print(str(observed_turnout) + '%')
print('The turnout rate after convenience voting is allowed:')
print(str(counterfactual_turnout) + '%')
print('=========================================================')

#%%
data['b_brand'] = 0
data.loc[(data['brand'] >= 201) & (data['brand'] <= 203), 'b_brand'] = 2 
data.loc[(data['brand'] > 203) & (data['brand'] <= 303), 'b_brand'] = 3
data.loc[data['brand'] == 401, 'b_brand'] = 4 
data.loc[data['brand'] == 502, 'b_brand'] = 5

#%%
data = data.assign(predicted_voters = round(data['c_mkt_shr'] * data['census_pop']),
                   observed_voters = round(data['marketshare1'] * data['census_pop']))
#%% Produce Table xxx
summary_1 = data.loc[:, ['b_brand', 'predicted_voters', 'observed_voters']].groupby(by = 'b_brand').sum()
summary_1['difference'] = summary_1['predicted_voters'] - summary_1['observed_voters']
summary_1 = summary_1.assign(mkt_shr_before = round(summary_1['observed_voters']*100/total_pop, 2),
                             mkt_shr_after = round(summary_1['predicted_voters']*100/total_pop, 2))

print('The number of non-voters before convenience voting is allowed is:')
print(str(total_pop - sum(summary_1['observed_voters'])))

print('The number of non-voters after convenience voting is allowed is:')
print(str(total_pop - sum(summary_1['predicted_voters'])))

#%% Produce Table xxx

city_location_dict = {1: 'N', 2: 'N', 3: 'N', 8: 'N',
                      4: 'C', 11: 'C', 12: 'C', 13: 'C', 14: 'C',
                      5: 'S', 6: 'S', 9: 'S', 16: 'S'}
dep_region = data['mkt'].copy()
arr_region = data['mkt'].copy()

for i in range(len(dep_region)):
    if len(str(dep_region[i])) == 3:
        dep_region[i] = str(dep_region[i])[0]
        arr_region[i] = str(arr_region[i])[1:3]
    else:
        dep_region[i] = str(dep_region[i])[0:2]
        arr_region[i] = str(arr_region[i])[2:4]

dep_region = dep_region.replace(city_location_dict)
arr_region = arr_region.replace(city_location_dict)

data = data.assign(dep_region = dep_region, arr_region = arr_region)

summary_2 = data.loc[:, ['dep_region', 'arr_region', 
                         'predicted_voters', 'observed_voters']].groupby(by = ['dep_region', 'arr_region']).agg({
                         'predicted_voters': 'sum', 'observed_voters': 'sum'})
    
census_pop_by_region = data.loc[:, ['mkt', 'dep_region', 'arr_region', 'census_pop']].copy()
census_pop_by_region = census_pop_by_region.groupby(by = ['mkt', 'dep_region', 'arr_region']).mean().reset_index()
census_pop_by_region = census_pop_by_region.loc[:, ['dep_region', 'arr_region', 'census_pop']].groupby(by = ['dep_region', 'arr_region']).sum()

summary_2['census_pop'] = census_pop_by_region['census_pop']   

summary_2['difference'] = summary_2['predicted_voters'] - summary_2['observed_voters']
summary_2['difference%'] = round(summary_2['difference']*100/summary_2['observed_voters'], 2)

summary_2['turnout_after'] = round(summary_2['predicted_voters']*100/summary_2['census_pop'], 2)
summary_2['turnout_before'] = round(summary_2['observed_voters']*100/summary_2['census_pop'], 2)
summary_2['turnout_diff'] = summary_2['turnout_after'] - summary_2['turnout_before'] 