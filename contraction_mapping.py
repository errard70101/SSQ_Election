#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 26 16:20:26 2018

@author: errard

This scripts performs the contraction mapping described in BLP (1995) paper.
"""

import numpy as np
import pandas as pd
from BLP_market_share import calculate_n_consumer_market_share as cal_mkt_shr
from BLP_market_share import calculate_utility as cal_uti
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

market_population = read_csv(market_population_file_name)
market_population = market_population.drop(columns = ['depcity', 'arrcity'])

dta = pd.merge(dta, market_population,
               left_on = ['depcity', 'arrcity'],
               right_on = ['depcity_code', 'arrcity_code'],
               how = 'left')

# Set up hyper parameters
# =============================================================================
n_consumers = int(1)

estimates = [-5.246357, -.6285018, -.8258803, -1.604927, 1.448219,
              0.7957358,  1.093028, -0.2235259, -0.1370856, -6.126835]
sigma = [2.816481]

data = dta[['priced', 'timed', 'pop_dep', 'moninc', 'edu',
            'unemploymentrate_arr', 'unemploymentrate_dep',
            'poll', 'mayor_arr', 'const']]

mkt = dta['mkt']

end_var = [0]

seed = 654781324
# =============================================================================
#%% Calculate the market share when nothing changes. This calculation costs
#   about 3.5 mins.

data = dta[['priced', 'timed', 'pop_dep', 'moninc', 'edu',
            'unemploymentrate_arr', 'unemploymentrate_dep',
            'poll', 'mayor_arr', 'const']]

mkt_shr = dta['marketshare1'].copy()
mkt_shr[mkt_shr == 0] = 1e-100
mkt_shr = np.array(mkt_shr)

#%%
def contraction_mapping(mkt_shr, estimates, sigma, data, end_var, mkt, 
                        xi = None,
                        work = 'Calculating the market share...',
                        congrats = 'Finished!!', 
                        n_consumers = 500, tolerance = 1e-15, max_iter = 500):
    """
    input:
        mkt_shr: n x 1 numpy ndarray, the observed market share.
        
    output:
    
    """
    difference = 1
    n_iter = 1 
    
    delta_old = cal_uti(estimates, sigma, data, end_var, np.zeros(len(sigma)))
    
    if type(xi) == type(None):
        xi = np.zeros(len(data))
    
    data = data.assign(xi = xi)
    est = estimates.copy()
    est.append(1)
    
    delta = cal_uti(est, sigma, data, end_var, np.zeros(len(sigma)))

    while (difference > tolerance) & (n_iter <= max_iter):
        print('*****Contraction Mapping*****')
        print('<Iteration: ' + str(n_iter) + '>')
        n_iter += 1
        
        diff = np.log(mkt_shr) - np.log(cal_mkt_shr(est, 
                                      sigma, data, end_var, mkt, n_consumers, 
                                      work, congrats, seed))
        
        delta = delta + diff.reshape(len(delta), 1)
    
        difference = np.max(np.abs(diff))
        
        print('The difference is ' + str(difference))
        print('')
        
        xi = delta - delta_old
        data = data.assign(xi = xi)
        
   
    return(delta, xi)

#%%
xi = np.loadtxt(xi_file_name)

mean_uti, xi = contraction_mapping(mkt_shr, estimates, sigma, data, end_var, 
                                   mkt, xi = xi, n_consumers = int(1e4), max_iter = 700)

#%%

np.savetxt(file_path + 'xi.csv', xi, delimiter = ',')