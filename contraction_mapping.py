#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 26 16:20:26 2018

@author: errard

This scripts perform the contraction mapping described in BLP (1995) paper.
"""

import numpy as np
import pandas as pd
from BLP_market_share import calculate_n_consumer_market_share as cal_mkt_shr
from BLP_market_share import calculate_utility as cal_uti
import matplotlib.pyplot as plt



def contraction_mapping(estimates, sigma, data, end_var, n_consumers = 500):
    """
    input:
        
    output:
    
    """
    
    cal_uti(estimates, sigma, data, end_var, v):
    
    
    delta + np.log(s) - log(s)