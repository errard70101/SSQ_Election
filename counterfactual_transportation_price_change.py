# -*- coding: utf-8 -*-
"""
Created on Tue Jun 26 09:36:15 2018

@author: Shih-Yang Lin

This script calculates the counterfactual market share when the prices of a 
specific transportation mode change.
"""

import numpy as np
import pandas as pd
from BLP_market_share import calculate_n_consumer_market_share as cal_mkt_shr
import matplotlib.pyplot as plt
import sys

# Read data
# =============================================================================

if sys.platform == 'darwin':
    save_path = "/Users/errard/Dropbox/SSQ_Election/"
    file_path = "/Users/errard/Dropbox/SSQ_Election/RawData20180515.csv"
    market_population_file_name = "/Users/errard/Dropbox/SSQ_Election/adj_census_pop_T.csv"
    xi_file_name = "/Users/errard/Dropbox/SSQ_Election/xi.csv"
else:
    save_path = "C:/SSQ_Election/"
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

#%%
dta['b_brand'] = 0
dta.loc[(dta['brand'] >= 201) & (dta['brand'] <= 203), 'b_brand'] = 2 
dta.loc[(dta['brand'] > 203) & (dta['brand'] <= 303), 'b_brand'] = 3
dta.loc[dta['brand'] == 401, 'b_brand'] = 4 
dta.loc[dta['brand'] == 502, 'b_brand'] = 5

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

#%%
total_pop = sum(market_population['census_pop'])
observed_voters = sum(dta['marketshare1'] * dta['census_pop'])

#%% Calculating counterfactual market share.
price_adjustment = np.array(range(-5, 6)) * 0.1


def calculate_counterfactual(price_adjustment, target_mode):
    '''
    input:
        price_adjustment: a numpy array which records the percentage of price change
        target_mode: integer in [2, 3, 4, 5] which indicates the transportation mode
        
    output:
        counterfactual_n_voters: pandas dataframe
        n_voters_difference: pandas dataframe
        n_voters_difference_percentage: pandas dataframe
    '''
    assert type(target_mode) == int, 'target_mode must be 2, 3, 4, or 5.'
    assert target_mode in [2, 3, 4, 5], 'target_mode must be 2, 3, 4, or 5.'
    
    
    mode_list = ['HSR', 'train', 'bus', 'car']
    
    counterfactual_n_voters = pd.DataFrame(data = np.zeros([len(price_adjustment), 4]),
                                           columns = mode_list,
                                           index = price_adjustment)

    for i in range(len(price_adjustment)):
        data = dta[['priced', 'timed', 'pop_dep', 'moninc', 'edu',
                    'unemploymentrate_arr', 'unemploymentrate_dep',
                    'poll', 'mayor_arr', 'const', 'xi']].copy()

        data.loc[:, 'priced'] = (dta['priced'] 
                                * (1 + (dta['b_brand'] == target_mode)
                                * price_adjustment[i]))
        
        work = ('Total setting of prices is ' + str(len(price_adjustment)) + '.' +
                ' Calculating the ' + str(i + 1) + 'th one.')
        congrats = 'Finish calculating.'
        
        
        c_label = '_' + mode_list[target_mode - 2] + '_p_'
        
        dta.loc[:, (str(price_adjustment[i]) + c_label + 'c_mkt_shr')] = cal_mkt_shr(estimates, 
                sigma, data, end_var, mkt, n_consumers, work, congrats, seed)
    
    
        for n_modes in range(2, 6):
            counterfactual_n_voters.loc[price_adjustment[i], mode_list[n_modes - 2]] = sum(
               dta.loc[dta['b_brand'] == n_modes, str(price_adjustment[i]) + c_label + 'c_mkt_shr']* 
               dta.loc[dta['b_brand'] == n_modes, 'census_pop'])
    
    # Calculate traffic volume change for voting
    n_voters_difference = counterfactual_n_voters - counterfactual_n_voters.loc[0, :]
    n_voters_difference_percentage = round(n_voters_difference*100/counterfactual_n_voters.loc[0, :], 2)

    return(counterfactual_n_voters, n_voters_difference, n_voters_difference_percentage)


counterfactual_n_voters_HSR, n_voters_difference_HSR, n_voters_difference_percentage_HSR = \
    calculate_counterfactual(price_adjustment, 2)

counterfactual_n_voters_train, n_voters_difference_train, n_voters_difference_percentage_train = \
    calculate_counterfactual(price_adjustment, 3)
    
counterfactual_n_voters_bus, n_voters_difference_bus, n_voters_difference_percentage_bus = \
    calculate_counterfactual(price_adjustment, 4)
    
counterfactual_n_voters_car, n_voters_difference_car, n_voters_difference_percentage_car = \
    calculate_counterfactual(price_adjustment, 5)
    
#%% Save the results

counterfactual_n_voters_HSR.to_csv(
        save_path + 'results/counterfactual_n_voters_HSR.csv', index = False)
n_voters_difference_HSR.to_csv(
        save_path + 'results/counterfactual_n_voters_difference_HSR.csv', index = False)
n_voters_difference_percentage_HSR.to_csv(
        save_path + 'results/counterfactual_n_voters__difference_percentage_HSR.csv', index = False)

counterfactual_n_voters_train.to_csv(
        save_path + 'results/counterfactual_n_voters_train.csv', index = False)
n_voters_difference_train.to_csv(
        save_path + 'results/counterfactual_n_voters_difference_train.csv', index = False)
n_voters_difference_percentage_train.to_csv(
        save_path + 'results/counterfactual_n_voters__difference_percentage_train.csv', index = False)

counterfactual_n_voters_bus.to_csv(
        save_path + 'results/counterfactual_n_voters_bus.csv', index = False)
n_voters_difference_bus.to_csv(
        save_path + 'results/counterfactual_n_voters_difference_bus.csv', index = False)
n_voters_difference_percentage_bus.to_csv(
        save_path + 'results/counterfactual_n_voters__difference_percentage_bus.csv', index = False)

counterfactual_n_voters_car.to_csv(
        save_path + 'results/counterfactual_n_voters_car.csv', index = False)
n_voters_difference_car.to_csv(
        save_path + 'results/counterfactual_n_voters_difference_car.csv', index = False)
n_voters_difference_percentage_car.to_csv(
        save_path + 'results/counterfactual_n_voters__difference_percentage_car.csv', index = False)

#%% Load results

def load_csv(f_path):
    f = open(f_path)
    output = pd.read_csv(f_path)
    f.close()
    return(output)
    
n_voters_difference_HSR = load_csv(
        save_path + 'results/counterfactual_n_voters_difference_HSR.csv')
n_voters_difference_train = load_csv(
        save_path + 'results/counterfactual_n_voters_difference_train.csv')
n_voters_difference_bus = load_csv(
        save_path + 'results/counterfactual_n_voters_difference_bus.csv')
n_voters_difference_car = load_csv(
        save_path + 'results/counterfactual_n_voters_difference_car.csv')

n_voters_difference_percentage_HSR = load_csv(
        save_path + 'results/counterfactual_n_voters__difference_percentage_HSR.csv')
n_voters_difference_percentage_train = load_csv(
        save_path + 'results/counterfactual_n_voters__difference_percentage_train.csv')
n_voters_difference_percentage_bus = load_csv(
        save_path + 'results/counterfactual_n_voters__difference_percentage_bus.csv')
n_voters_difference_percentage_car = load_csv(
        save_path + 'results/counterfactual_n_voters__difference_percentage_car.csv')

#%% Make figures

fig, axes = plt.subplots(nrows = 2, ncols = 2, figsize = (10, 15))

def make_subplot(row, col, df, mode_name, y_label):
    axes[row, col].plot(df.index, df['HSR'], '-', label = 'HSR')
    axes[row, col].plot(df.index, df['train'], '-.', label = 'Train')
    axes[row, col].plot(df.index, df['bus'], '--', label = 'Bus')
    axes[row, col].plot(df.index, df['car'], '-o', label ='Car')
    axes[row, col].legend()
    axes[row, col].set_xticks(df.index)
    axes[row, col].set_xticklabels(['-50', '-40', '-30', '-20', '-10', '0', '10', '20', '30', '40', '50'])
    axes[row, col].set_xlabel(mode_name + ' price change %')
    axes[row, col].set_ylabel(y_label)

make_subplot(0, 0, n_voters_difference_percentage_HSR, 'HSR', 
             'Traffic volume change for voting %')
make_subplot(0, 1, n_voters_difference_percentage_train, 'Train',
             'Traffic volume change for voting %')
make_subplot(1, 0, n_voters_difference_percentage_bus, 'Bus', 
             'Traffic volume change for voting %')
make_subplot(1, 1, n_voters_difference_percentage_car, 'Car', 
             'Traffic volume change for voting %')

fig.savefig(save_path + 'fig/fig4a.png')    

#%% Make figures

fig, axes = plt.subplots(nrows = 2, ncols = 2, figsize = (10, 15))

make_subplot(0, 0, n_voters_difference_HSR, 'HSR', 
             'Traffic volume change for voting')
make_subplot(0, 1, n_voters_difference_train, 'Train',
             'Traffic volume change for voting')
make_subplot(1, 0, n_voters_difference_bus, 'Bus',
             'Traffic volume change for voting')
make_subplot(1, 1, n_voters_difference_car, 'Car',
             'Traffic volume change for voting')

fig.savefig(save_path + 'fig/fig4b.png')