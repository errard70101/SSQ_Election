# -*- coding: utf-8 -*-
"""
Created on Mon May 21 17:28:20 2018

@author: Shih-Yang Lin
"""

import numpy as np
import pandas as pd
import time
from BLP_market_share import calculate_n_consumer_market_share as cal_mkt_shr
import matplotlib.pyplot as plt

# Read data
# =============================================================================
file_path = "C:/SSQ_Election/RawData20180515.csv"
market_population_file_name = "C:/SSQ_Election/adj_census_pop_T.csv"

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
n_consumers = int(1e4)

estimates = [-5.246357, -.6285018, -.8258803, -1.604927, 1.448219,
              0.7957358,  1.093028, -0.2235259, -0.1370856, -6.126835]
sigma = [2.816481]

data = dta[['priced', 'timed', 'pop_dep', 'moninc', 'edu',
            'unemploymentrate_arr', 'unemploymentrate_dep',
            'poll', 'mayor_arr', 'const']]

mkt = dta['mkt']

end_var = [0]

v_set = np.random.RandomState(seed = 654781324).normal(size = (n_consumers, len(sigma)))
# =============================================================================

#%% Calculating counterfactual market share.
price_adjustment = np.array(range(5, 16)) * 0.1
counterfactual_n_voters = list()

tic = time.time()
for i in range(len(price_adjustment)):
    data = dta[['priced', 'timed', 'pop_dep', 'moninc', 'edu',
            'unemploymentrate_arr', 'unemploymentrate_dep',
            'poll', 'mayor_arr', 'const']]
    
    data['priced'] = dta['priced']*price_adjustment[i]
    work = ('Total setting of prices is ' + str(len(price_adjustment)) + '.' +
            ' Calculating the ' + str(i + 1) + 'th one.')
    congrats = 'Finish calculating.'
    
    dta[str(price_adjustment[i]) + 'p_' + 'c_mkt_shr'] = cal_mkt_shr(estimates, sigma, data, end_var, v_set, mkt, n_consumers, work, congrats)
    counterfactual_n_voters.append(sum(dta[str(price_adjustment[i]) + 'p_' + 'c_mkt_shr']* dta['cencus_pop']))
    
toc = time.time()

print('')
print('Total time elaped is ' + str(round((toc - tic)/60, 2)) + ' mins.')
#%%
total_pop = sum(market_population['cencus_pop'])
observed_voters = sum(dta['marketshare1'] * dta['cencus_pop'])

#%% Calculate traffic volume change for voting
n_voters_difference = list()
n_voters_difference_percentage = list()
i = 0
for n_voters in counterfactual_n_voters:
    n_voters_difference.append(round(n_voters - counterfactual_n_voters[1]))
    n_voters_difference_percentage.append(round(n_voters_difference[i]*100/total_pop, 2))
    i += 1
    
counterfactual_result = pd.DataFrame(data = {'All traffic volume change for voting': n_voters_difference,
                                             'All price change %': ['-50%', '-40%', '-30%', '-20%', '-10%', '0%', '10%', '20%', '30%', '40%', '50%'],
                                             'All traffic volume change for voting %': n_voters_difference_percentage})

counterfactual_result = counterfactual_result[['All price change %', 'All traffic volume change for voting', 'All traffic volume change for voting %']]
print(counterfactual_result)
#%% Produce Figure 3
    
plt.plot(price_adjustment, n_voters_difference)
plt.ylabel('All traffic volume change for voting')
plt.xlabel('All price change %')
plt.yticks([-60000, -40000, -20000, 0, 20000, 40000, 60000, 80000])
plt.xticks(price_adjustment, 
           ['-50%', '-40%', '-30%', '-20%', '-10%', '0%', '10%', '20%', '30%', '40%', '50%'])
plt.grid(True)
plt.show()
    

#%% Save the counterfactual result
counterfactual_result.to_csv("counterfactual_total_price_change_temp.csv", index = False)