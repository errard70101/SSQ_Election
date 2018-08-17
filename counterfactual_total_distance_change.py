# -*- coding: utf-8 -*-
"""
Created on Mon Jul 16 12:39:48 2018

@author: Shih-Yang Lin

This script calculates the counterfactual market share when the distance of all
transportation modes change.
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

#%% Calculating counterfactual market share.
distance_adjustment = np.array([-10, -5, -3, -1, 0, 1, 3, 5, 10])
price_adjustment = distance_adjustment * (29.97*1.6/9.7/1000) #avg. fuel price:29.97
time_adjustment = distance_adjustment * (1.6/40) #avg. speed = 40 km/hr
counterfactual_n_voters = list()
counterfactual_n_voters_HSR = list()
counterfactual_n_voters_train = list()
counterfactual_n_voters_bus = list()
counterfactual_n_voters_car = list()

for i in range(len(time_adjustment)):
    data = dta[['priced', 'timed', 'pop_dep', 'moninc', 'edu',
            'unemploymentrate_arr', 'unemploymentrate_dep',
            'poll', 'mayor_arr', 'const', 'xi']].copy()

    data.loc[:, 'priced'] = dta['priced'] + price_adjustment[i]
    data.loc[:, 'timed'] = dta['timed'] + time_adjustment[i]
    work = ('Total setting of prices is ' + str(len(time_adjustment)) + '.' +
            ' Calculating the ' + str(i + 1) + 'th one.')
    congrats = 'Finish calculating.'

    dta.loc[:, (str(distance_adjustment[i]) + 'p_' + 'c_mkt_shr')] = cal_mkt_shr(estimates, 
            sigma, data, end_var, mkt, n_consumers, work, congrats, seed)
    counterfactual_n_voters.append(sum(dta[str(distance_adjustment[i]) + 'p_' + 'c_mkt_shr']* dta['census_pop']))
    counterfactual_n_voters_HSR.append(
            sum(dta.loc[(dta['brand'] == 201) | (dta['brand'] == 202) | (dta['brand'] == 203),
                        str(distance_adjustment[i]) + 'p_' + 'c_mkt_shr']
              * dta.loc[(dta['brand'] == 201) | (dta['brand'] == 202) | (dta['brand'] == 203),
                        'census_pop']))
    counterfactual_n_voters_train.append(
            sum(dta.loc[(dta['brand'] == 301) | (dta['brand'] == 302) | (dta['brand'] == 303),
                        str(distance_adjustment[i]) + 'p_' + 'c_mkt_shr']
              * dta.loc[(dta['brand'] == 301) | (dta['brand'] == 302) | (dta['brand'] == 303),
                        'census_pop']))
    counterfactual_n_voters_bus.append(
            sum(dta.loc[(dta['brand'] == 401), str(distance_adjustment[i]) + 'p_' + 'c_mkt_shr']
              * dta.loc[(dta['brand'] == 401), 'census_pop']))
    counterfactual_n_voters_car.append(
            sum(dta.loc[(dta['brand'] == 502), str(distance_adjustment[i]) + 'p_' + 'c_mkt_shr']
              * dta.loc[(dta['brand'] == 502), 'census_pop']))

#%%
total_pop = sum(market_population['census_pop'])
observed_voters = sum(dta['marketshare1'] * dta['census_pop'])
observed_nonvoters = total_pop - observed_voters

#%% Calculate traffic volume change for voting
n_voters_difference = list()
n_voters_difference_percentage = list()
n_voters_difference_HSR = list()
n_voters_difference_percentage_HSR = list()
n_voters_difference_train = list()
n_voters_difference_percentage_train = list()
n_voters_difference_bus = list()
n_voters_difference_percentage_bus = list()
n_voters_difference_car = list()
n_voters_difference_percentage_car = list()

for i in range(len(counterfactual_n_voters)):
    n_voters_difference.append(round(counterfactual_n_voters[i] - counterfactual_n_voters[4]))
    n_voters_difference_percentage.append(round(n_voters_difference[i]*100/total_pop, 2))
    n_voters_difference_HSR.append(
            round(counterfactual_n_voters_HSR[i] - counterfactual_n_voters_HSR[4]))
    n_voters_difference_percentage_HSR.append(
            round(n_voters_difference_HSR[i]*100/total_pop, 2))
    n_voters_difference_train.append(
            round(counterfactual_n_voters_train[i] - counterfactual_n_voters_train[4]))
    n_voters_difference_percentage_train.append(
            round(n_voters_difference_train[i]*100/total_pop, 2))
    n_voters_difference_bus.append(
            round(counterfactual_n_voters_bus[i] - counterfactual_n_voters_bus[4]))
    n_voters_difference_percentage_bus.append(
            round(n_voters_difference_bus[i]*100/total_pop, 2))
    n_voters_difference_car.append(
            round(counterfactual_n_voters_car[i] - counterfactual_n_voters_car[4]))
    n_voters_difference_percentage_car.append(
            round(n_voters_difference_car[i]*100/total_pop, 2))
    

counterfactual_result = pd.DataFrame(data = {'All traffic volume change for voting': n_voters_difference,
                                             'All distance change mile': ['-10', '-5', '-3', '-1', '0', '1', '3', '5', '10'],
                                             'All traffic volume change for voting %': n_voters_difference_percentage,
                                             'HSR traffic volume change for voting': n_voters_difference_HSR,
                                             'HSR traffic volume change for voting %': n_voters_difference_percentage_HSR,
                                             'train traffic volume change for voting': n_voters_difference_train,
                                             'train traffic volume change for voting %': n_voters_difference_percentage_train,
                                             'bus traffic volume change for voting': n_voters_difference_bus,
                                             'bus traffic volume change for voting %': n_voters_difference_percentage_bus,
                                             'car traffic volume change for voting': n_voters_difference_car,
                                             'car traffic volume change for voting %': n_voters_difference_percentage_car})

counterfactual_result = counterfactual_result[['All distance change mile', 
                                               'All traffic volume change for voting', 
                                               'All traffic volume change for voting %',
                                               'HSR traffic volume change for voting', 
                                               'HSR traffic volume change for voting %',
                                               'train traffic volume change for voting', 
                                               'train traffic volume change for voting %',
                                               'bus traffic volume change for voting', 
                                               'bus traffic volume change for voting %',
                                               'car traffic volume change for voting', 
                                               'car traffic volume change for voting %']]
print(counterfactual_result)

#%% Produce Figure 3
plt.figure(figsize = (10, 10))
plt.plot(distance_adjustment, n_voters_difference, '-', label = 'Voters')
plt.ylabel('All traffic volume change for voting')
plt.xlabel('All distance change (mile)')
plt.yticks([-80000, -60000, -40000, -20000, 0, 
            20000, 40000, 60000, 80000])

plt.grid(True)
plt.savefig(fname = save_path + "fig/fig3b.eps", format = 'eps')
plt.show()
#%% Save the counterfactual result
counterfactual_result.to_csv(save_path + "results/counterfactual_total_distance_change.csv", index = False)

