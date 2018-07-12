# SSQ_Election
The program codes to produce the result in SSQ artcle,
"Estimating the Willingness to Pay for Voting when Absentee Voting is not Allowed."


## A. Program Version

1. Python: 3.6
  python packages:
    - numpy: 1.14.3
    - pandas: 0.23.0
    - matplotlib: 2.2.2

2. Stata: SE 13

## B. The program files:

1. counterfactual_total_price_change.py: calculate the counterfactual market share when the prices of all transportation modes change.

2. counterfactual_absentee_voting.py: calculate the counterfactual market share when absentee voting is allowed.

3. counterfactual_no_HSR.py: calculates the counterfactual market share when High Speed Rail (HSR) does not exist.

4. counterfactual_no_local_cost.py: calculates the counterfactual market share when there are no local travel pecuniary and time costs.

5. counterfactual_transportation_price_change.py: calculates the counterfactual market share when the prices of a specific transportation mode change.

6. BLP_market_share.py: a script to calculate predicted market share.

7. progress.py: a script to show progress bar by Vladimir Ignatev.

8. contraction_mapping.py: performs the contraction mapping described in BLP (1995) paper

## C. Data files:
1. adj_census_pop_T.csv: the population data based on the 2010 Population Census conducted by the Ministry of the Interior.

2. RawData20180515.csv: the data required for BLP model.
