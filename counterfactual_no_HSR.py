# -*- coding: utf-8 -*-
"""
Created on Tue May 22 08:50:54 2018

@author: Shih-Yang Lin

This script calculates the counterfactual market share when High Speed Rail
(HSR) does not exist.
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
dta['const'] = np.ones((len(dta), 1))

xi = np.loadtxt(xi_file_name)
dta = dta.assign(xi = xi)

market_population = read_csv(market_population_file_name)
market_population = market_population.drop(columns = ['depcity', 'arrcity'])

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

data = dta[['priced', 'timed', 'pop_dep', 'moninc', 'edu',
            'unemploymentrate_arr', 'unemploymentrate_dep',
            'poll', 'mayor_arr', 'const', 'xi']]

mkt = dta['mkt']

end_var = [0]

seed = 654781324
# =============================================================================
#%% Calculate the market share when nothing changes. This calculation costs
#   about 3.5 mins.

data = dta[['priced', 'timed', 'pop_dep', 'moninc', 'edu',
            'unemploymentrate_arr', 'unemploymentrate_dep',
            'poll', 'mayor_arr', 'const', 'xi']]
    
work = 'Calculating the market share when nothing changes.'
congrats = 'Finish calculating.'
    
dta = dta.assign(simulated_mkt_shr = cal_mkt_shr(estimates, sigma, data, end_var, 
                                                 mkt, n_consumers, work, congrats, seed))
#%% Calculate counterfactual market share when HSR does not exist. This calculation
#   costs about 3.5 mins.

counterfactual_data = dta[['priced', 'timed', 'pop_dep', 'moninc', 'edu',
                           'unemploymentrate_arr', 'unemploymentrate_dep',
                           'poll', 'mayor_arr', 'const', 'xi', 'brand', 'mkt']]
    
counterfactual_data = counterfactual_data[counterfactual_data['brand'] > 203]

mkt = counterfactual_data['mkt']

data = counterfactual_data.drop(columns = ['brand', 'mkt'])

work = 'Calculating the market share when HSR does not exist.'

counterfactual_data = counterfactual_data.assign(c_mkt_shr = 
                                                 cal_mkt_shr(estimates, sigma, 
                                                             data, end_var, mkt, 
                                                             n_consumers, work, 
                                                             congrats, seed))

del work, congrats, data, mkt
#%% Analyze the result

# remove irrelevant independent variables from counterfactual_data
counterfactual_data = counterfactual_data[['mkt', 'brand', 'c_mkt_shr']]

# add the counterfactual result in dta
dta = pd.merge(dta, counterfactual_data, left_on = ['mkt', 'brand'], 
               right_on = ['mkt', 'brand'], how = 'left')

# Since we do not have the counterfactual market share for HSR, the 
# counterfactual market shares for HSR are missing values.
# This line fills these missing values with 0s.
dta = dta.fillna(value = {'c_mkt_shr': 0})

# Find the markets where HSR exists.
HSR_mkt = dta.loc[dta['brand'] <= 203, 'mkt'].unique()

# HSR_mkt_data is the dataset that countains the markes where HSR exists.
select = list()
for i in range(len(dta)):
    select.append(dta.loc[i, 'mkt'] in HSR_mkt)
    
HSR_mkt_data = dta[select]

# Calculate the number of observed_voters, the number of simulated voters,
# and the number of counterfactual voters.
HSR_mkt_data = HSR_mkt_data.assign(observed_voters = HSR_mkt_data['marketshare1'] * HSR_mkt_data['census_pop'],
                                   simulated_voters = HSR_mkt_data['simulated_mkt_shr'] * HSR_mkt_data['census_pop'],
                                   c_voters = HSR_mkt_data['c_mkt_shr'] * HSR_mkt_data['census_pop'])

# Calculate counterfactual voters difference = the number of counterfactual
# voters - the number of simulated voters.
HSR_mkt_data['c_voters_diff'] = HSR_mkt_data['c_voters'] - HSR_mkt_data['simulated_voters']

# Create a variable, big brand:
# if brand = 201, 202, 203, b_brand = 2
# if brand = 301, 302, 303, b_brand = 3
# if brand = 401, b_brand = 4
# if brand = 502, b_brand = 5
 
HSR_mkt_data['b_brand'] = 0
HSR_mkt_data.loc[HSR_mkt_data['brand'] <= 203, 'b_brand'] = 2 
HSR_mkt_data.loc[(HSR_mkt_data['brand'] > 203) & (HSR_mkt_data['brand'] <= 303), 'b_brand'] = 3
HSR_mkt_data.loc[HSR_mkt_data['brand'] == 401, 'b_brand'] = 4 
HSR_mkt_data.loc[HSR_mkt_data['brand'] == 502, 'b_brand'] = 5 

# Create a variable that indicates the available transportation modes in each 
# markets.
HSR_mkt_data['available_modes'] = 0
for i in HSR_mkt_data.index:
    HSR_mkt_data.loc[i, 'available_modes'] = len(HSR_mkt_data[HSR_mkt_data['mkt'] == HSR_mkt_data.loc[i, 'mkt']])

#%% 
analysis_1 = HSR_mkt_data.groupby(by = ['mkt', 'b_brand']).agg({'observed_voters': 'sum',
                                                                'c_voters_diff': 'sum',
                                                                'available_modes': 'mean'})
    
analysis_1 = analysis_1.reset_index()

substitution_share = list()
observed_HSR_voter = list()
for i in range(len(analysis_1)):
    numerator = analysis_1.loc[i, 'c_voters_diff']
    denumerator = abs(analysis_1.loc[
                      (analysis_1['mkt'] == analysis_1.loc[i, 'mkt']) & 
                      (analysis_1['b_brand'] == 2), 'c_voters_diff'].values)
    substitution_share.append(numerator/denumerator[0])
    observed_HSR_voter.append(analysis_1.loc[
                              (analysis_1['mkt'] == analysis_1.loc[i, 'mkt']) & 
                              (analysis_1['b_brand'] == 2), 'observed_voters'].values[0])
    del numerator, denumerator

analysis_1 = analysis_1.assign(sub_shr = substitution_share,
                               obs_HSR_voter = observed_HSR_voter)

analysis_1 = analysis_1.assign(predicted_voter_change = analysis_1['sub_shr'] * analysis_1['obs_HSR_voter'])

#%% Produce Table xxx
HSR_observed_voters = analysis_1[analysis_1['b_brand'] == 2]
HSR_observed_voters = HSR_observed_voters.groupby(by = ['available_modes']).agg({'observed_voters': 'sum'})

summary_result = analysis_1.groupby(by = ['available_modes']).agg({'observed_voters': 'sum',
                                                                   'predicted_voter_change': 'sum'})

summary_mkt_count = analysis_1.groupby(by = ['available_modes', 'mkt']).count().reset_index()
summary_mkt_count = summary_mkt_count.groupby(by = ['available_modes']).count()

summary_result = summary_result.assign(HSR_observed_voters = HSR_observed_voters['observed_voters'],
                                       mkt_count = summary_mkt_count['mkt'])

summary_result = summary_result.assign(avg_obs_voters = round(summary_result['observed_voters']/summary_result['mkt_count']),
                                       avg_pre_voters_diff = round(summary_result['predicted_voter_change']/summary_result['mkt_count']),
                                       avg_HSR_obs_voters = round(summary_result['HSR_observed_voters']/summary_result['mkt_count']))

summary_result = summary_result.assign(avg_pre_voters_diff_p = round(summary_result['avg_pre_voters_diff']*100/summary_result['avg_obs_voters'], 2))

# Calculate averages
print('The average observed HSR voters:')
print(round(sum(summary_result['HSR_observed_voters'])/sum(summary_result['mkt_count'])))
print('The average observed voters:')
print(round(sum(summary_result['observed_voters'])/sum(summary_result['mkt_count'])))
print('The average predicted voter change:')
print(round(sum(summary_result['predicted_voter_change'])/sum(summary_result['mkt_count'])))

#%% Produce Table xxx

city_location_dict = {1: 'N', 2: 'N', 3: 'N', 8: 'N',
                      4: 'C',
                      5: 'S', 6: 'S', 9: 'S'}
dep_region = analysis_1['mkt'].copy()
arr_region = analysis_1['mkt'].copy()

for i in range(len(dep_region)):
    dep_region[i] = str(dep_region[i])[0]
    arr_region[i] = str(arr_region[i])[2]

dep_region = dep_region.replace(city_location_dict)
arr_region = arr_region.replace(city_location_dict)

analysis_2 = analysis_1.assign(dep_region = dep_region, arr_region = arr_region)

HSR_observed_voters = analysis_2[analysis_2['b_brand'] == 2]
HSR_observed_voters = HSR_observed_voters.groupby(by = ['dep_region', 'arr_region']).agg({'observed_voters': 'sum'})

summary_result_2 = analysis_2.groupby(by = ['dep_region', 'arr_region']).agg({'observed_voters': 'sum',
                                                                              'predicted_voter_change': 'sum'})

summary_mkt_count_2 = analysis_2.groupby(by = ['dep_region', 'arr_region', 'mkt']).count().reset_index()
summary_mkt_count_2 = summary_mkt_count_2.groupby(by = ['dep_region', 'arr_region']).count()

#%%
summary_result_2 = summary_result_2.assign(HSR_observed_voters = HSR_observed_voters['observed_voters'],
                                           mkt_count = summary_mkt_count_2['mkt'])

summary_result_2 = summary_result_2.assign(avg_obs_voters = round(summary_result_2['observed_voters']/summary_result_2['mkt_count']),
                                           avg_pre_voters_diff = round(summary_result_2['predicted_voter_change']/summary_result_2['mkt_count']),
                                           avg_HSR_obs_voters = round(summary_result_2['HSR_observed_voters']/summary_result_2['mkt_count']),
                                           HSR_observed_voter_mkt_shr = round(summary_result_2['HSR_observed_voters']*100/summary_result_2['observed_voters'], 2))

summary_result_2 = summary_result_2.assign(avg_pre_voters_diff_p = round(summary_result_2['avg_pre_voters_diff']*100/summary_result_2['avg_obs_voters'], 2))

# Calculate averages
print(round(sum(summary_result_2['HSR_observed_voters'])/sum(summary_result_2['mkt_count'])))
print(round(sum(summary_result_2['observed_voters'])/sum(summary_result_2['mkt_count'])))
print(round(sum(summary_result_2['predicted_voter_change'])/sum(summary_result_2['mkt_count'])))