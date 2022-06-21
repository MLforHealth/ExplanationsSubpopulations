import os
import sys
import hashlib
import json
from pathlib import Path
import copy

import numpy as np
import pandas as pd
from sklearn import metrics

# GLOBAL VARS
# group settings
list_groups = [
    '[ 0. nan]',
    '[ 1. nan]',
    '[nan  0.]',
    '[nan  1.]',
    '[nan  2.]',
    '[nan  3.]',
    '[nan  4.]'
    
    
]

# the two sensitive groups of interest for all 4 datasets
# NB: first item in tuple is sex, second is race
# so here sex is considered the sensitive attribute for
# Adult and MIMIC and race for COMPAS and LSAC
list_groups_dict={}
list_groups_dict['adult']=[
    '[ 0. nan]',
    '[ 1. nan]']
list_groups_dict['mimic_tab']=[
     '[ 0. nan]',
     '[ 1. nan]']
list_groups_dict['mimic_tab_fair_ds']=[
     '[ 0. nan]',
     '[ 1. nan]']
list_groups_dict['lsac']=[
    '[nan  0.]',
    '[nan  1.]',
    '[nan  2.]',
    '[nan  3.]',
    '[nan  4.]']
list_groups_dict['lsac_cat']=[
    '[nan  0.]',
    '[nan  1.]',
    '[nan  2.]',
    '[nan  3.]',
    '[nan  4.]']

list_groups_dict['compas_balanced']=[
    '[nan  0.]',
    '[nan  1.]']


def meanPairDiff(arr):
    listDiff=[]
    for i in range(len(arr)):
        for j in range(i+1,len(arr)):
            listDiff.append(np.abs(arr[i]-arr[j]))
    return np.nanmean(listDiff)


def compute_fairness_metrics(x, met, 
                             list_groups_vals=list_groups_dict['adult']):
    """Compute 2 fairness metrics described in paper.
    
    Parameters:
    x: pd.DataFrame, dataframe with metric for different groups
       with "all" containing metric over all datapoints
    met: str, metric of interest
        NB: this function works only for metrics with polarity like
        AUROC. For error-based metrics, see results_latex_utils.py
    list_groups_vals: list, list with groups of interest
        
        
    Returns:
    2 metrics
    
    """
    res = {}
    res[f'all'] = x[f'all'].iloc[0]
    res[f'min'] = x[met].min()
        

    # Some datasets have more than 2 self-reported groups annotations available
    res[f'mean_gap'] = meanPairDiff([
        x[x.group==i][met].mean() for i in list_groups_vals
    ])
    all_gaps=[]
    for group in list_groups_vals:
        curr_group=x[x.group==group]
        if curr_group.shape[0]>0:
            res[f'{group}_gap']=res[f'all']-curr_group[met].mean()
            all_gaps.append(res[f'all']-curr_group[met].mean())
        else:
            res[f'{group}_gap']=np.nan
            res[f'{group}']=np.nan
            
    res['max_gap']=np.nanmax(all_gaps)
        
    return res['mean_gap'],res['max_gap']

