# -*- coding: utf-8 -*-
"""
Created on Mon May 21 09:58:58 2018

@author: Shih-Yang Lin

This script calculate the market share.
"""
import numpy as np
from progress import progress as pg

def calculate_utility(estimates, sigma, data, end_var, v):
    '''
    Calculate mean utility of each products base on the estimates and data.
    input:
        estimates: list, the estimates of BLP.
        sigma: list, the estimated standard errors of random coefficients.
        data: n x k numpy ndarray, the independent variables include constant 
              term.
        end_var: list, the position of endogenous variables.
        v: (len(sigma), ) numpy ndarray, consumer taste.
    output:
        uti: numpy array, the mean utility of each products.
    '''

    # Examine the number of variables
    Worning_Message_0 = ('The length of estimates is ' + str(len(estimates)) +
                         ', but the number of columns of data is ' +
                         str(data.shape[1]) + '.')

    assert len(estimates) == data.shape[1], Worning_Message_0

    Worning_Message_1 = ('The length of sigma is ' + str(len(sigma)) +
                         ', but the number of endogenous variables is ' +
                         str(len(end_var)) + '.')

    assert len(sigma) == len(end_var), Worning_Message_1

    Worning_Message_2 = ('The length of sigma is ' + str(len(sigma)) +
                         ', but the number of consumer taste variables is ' +
                         str(len(v)) + '.')

    assert len(v) == len(sigma), Worning_Message_2

    del Worning_Message_0, Worning_Message_1, Worning_Message_2

    # Data preparation
    estimates = np.array(estimates).reshape(len(estimates), 1)
    sigma = np.array(sigma).reshape(len(sigma), 1)
    endogenous_variables = data.iloc[:, end_var]

    # Calculate utilities
    uti = np.dot(data, estimates) + np.dot(endogenous_variables, sigma*v)

    return(uti)

def calculate_one_consumer_market_share(uti, mkt):
    '''
    Calculate simulated market share of each products in each markets
    base on their utilities.
    input:
        uti: n x 1 numpy array, the utilities of each products in each markets.
        mkt: n x 1 numpy array, the market id of each products in each markets.
    output:
        mkt_shr: n x 1 numpy array, the market share of each products in each
            markets.
    '''
    # Generate new market id
    mkt_list =  np.unique(mkt)
    new_mkt = np.zeros((len(mkt), ))
    new_mkt_id = 0
    for mkt_id in mkt_list:
        new_mkt[mkt == mkt_id] = new_mkt_id
        new_mkt_id += 1

    # Calculate the denominators of market shares in each markets
    denominator = list()
    for i in range(len(mkt_list)):
        denominator.append(1 + np.sum(np.exp(uti[new_mkt == i])))

    assert len(denominator) == len(mkt_list)
    
    denominator = np.array(denominator)
    denominator = denominator[new_mkt.astype(int)]

    # Calculate the market share
    mkt_shr = np.exp(uti.reshape(len(uti), ))/denominator

    return(mkt_shr)

def calculate_n_consumer_market_share(estimates, sigma, data, end_var, v_set, mkt, n_consumers, work = 'Calculating the market share.', congrats = 'Finished.'):
    '''
    Calculate mean utility of each products base on the estimates and data.
    input:
        estimates: list, the estimates of BLP.
        sigma: list, the estimated standard errors of random coefficients.
        data: n x k numpy ndarray, the independent variables include constant 
              term.
        end_var: list, the position of endogenous variables.
        v_set: n_consumers x len(sigma) numpy ndarray, a set of consumer tastes.
        mkt: n x 1 numpy array, the market id of each products in each markets.
        n_consumers: int, number of simulated consumers in each market.
    output:
        mkt_shr: numpy array, the average market share of n consumers.
    '''
    mkt_shr = np.zeros((len(data), ))
    for i in range(n_consumers):
        uti = calculate_utility(estimates, sigma, data, end_var, v_set[i])
        mkt_shr += calculate_one_consumer_market_share(uti, mkt)
        del uti
        pg(i + 1, n_consumers, work, congrats)
    mkt_shr = mkt_shr/n_consumers

    return(mkt_shr)