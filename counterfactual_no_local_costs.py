# -*- coding: utf-8 -*-
"""
Created on Mon Jun 25 14:12:12 2018

@author: Shih-Yang Lin

This script calculates the counterfactual market share when there are no local
travel pecuniary and time costs.
"""

import numpy as np
import pandas as pd
from BLP_market_share import calculate_n_consumer_market_share as cal_mkt_shr
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

data = dta[['price', 'fritime', 'pop_dep', 'moninc', 'edu',
            'unemploymentrate_arr', 'unemploymentrate_dep',
            'poll', 'mayor_arr', 'const', 'xi']]

mkt = dta['mkt']

end_var = [0]

seed = 654781324

# =============================================================================

#%% Calculating counterfactual market share.

work = 'Calculating the counterfactual market share'
congrats = 'Finish calculating.'
dta.loc[:, 'c_mkt_shr'] = cal_mkt_shr(estimates, 
            sigma, data, end_var, mkt, n_consumers, work, congrats, seed)
counterfactual_n_voters = sum(dta['c_mkt_shr']* dta['census_pop'])

#%%
total_pop = sum(market_population['census_pop'])
observed_voters = sum(dta['marketshare1'] * dta['census_pop'])

#%%
print(round((counterfactual_n_voters - observed_voters)*100/total_pop, 3))
