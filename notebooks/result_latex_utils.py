import os
import sys
sys.path.append('../')
import hashlib
import json
from pathlib import Path
import copy

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_auc_score, auc, roc_curve
from sklearn import metrics

from compute_metrics import Model
from lib.TabularDataset import dataset_params
from lib import TabularDataset

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



def maxPairDiff(arr):
    listDiff=[]
    for i in range(len(arr)):
        for j in range(i+1,len(arr)):
            listDiff.append(np.abs(arr[i]-arr[j]))
    return np.nanmean(listDiff)

def agg_func(x):
    res = {}
    for met, disp in zip(['ACC', 'AUROC'], ['accuracy', 'auroc']):
        res[f'{disp}_all'] = res[f'{disp}_all'].iloc[0]
        res[f'{disp}_min'] = x[met].min()
        res[f'{disp}_minority'] = x.loc[x['n'].idxmin(), met]
        res[f'{disp}_majority'] = x.loc[x['n'].idxmax(), met]
        res[f'{disp}_gap'] = res[f'{disp}_all']-res[f'{disp}_min']
        
        # Gap in first sensitive attribute, always sex in our setup
        res[f'{disp}_sens1_gap'] = np.abs(x[x.group=='[ 0. nan]'][met].mean()-x[x.group=='[ 1. nan]'][met].mean())
        
        # Some datasets have more than 2 self-reported race annotations available
        if '[nan  2.]' in x.group.unique():
            res[f'{disp}_sens2_gap'] = maxPairDiff([
                x[x.group=='[nan  0.]'][met].mean(),
                x[x.group=='[nan  1.]'][met].mean(),
                x[x.group=='[nan  2.]'][met].mean(),
                x[x.group=='[nan  3.]'][met].mean(),
                x[x.group=='[nan  4.]'][met].mean()
            ])
        else:
            res[f'{disp}_sens2_gap'] = np.abs(x[x.group=='[nan  0.]'][met].mean()-x[x.group=='[nan  1.]'][met].mean())
        
        for group in list_groups:
            curr_group=x[x.group==group]
            if curr_group.shape[0]>0:
                res[f'{disp}_{group}_gap']=res[f'{disp}_all']-curr_group[met].mean()
                # taking negative of the actual numbers to convert to same polarity as gap
                # for a gap-based metric lower is better, similarly for negative of a performance
                # metric, lower is better
                res[f'{disp}_{group}']=-1*curr_group[met].mean()
            else:
                res[f'{disp}_{group}_gap']=np.nan
                res[f'{disp}_{group}']=np.nan

    # this is the residual error, computing separately due to 
    # differing polarity with ACC and AUROC
    for met, disp in zip(['epsilon'], ['epsilon']):
    	# NB: this is an approximatation, we don't use in paper
        res[f'{disp}_all'] = np.nanmean(x[met])
        res[f'{disp}_max'] = x[met].values[np.argmax(np.abs(x[met].values))]
        res[f'{disp}_minority'] = x.loc[x['n'].idxmin(), met]
        res[f'{disp}_majority'] = x.loc[x['n'].idxmax(), met]
        res[f'{disp}_gap'] = res[f'{disp}_all']-res[f'{disp}_max']
        res[f'{disp}_sens1_gap'] = np.abs(x[x.group=='[ 0. nan]'][met].mean()-x[x.group=='[ 1. nan]'][met].mean())
        if '[nan  2.]' in x.group.unique():
            res[f'{disp}_sens2_gap'] = maxPairDiff([
                x[x.group=='[nan  0.]'][met].mean(),
                x[x.group=='[nan  1.]'][met].mean(),
                x[x.group=='[nan  2.]'][met].mean(),
                x[x.group=='[nan  3.]'][met].mean(),
                x[x.group=='[nan  4.]'][met].mean()
            ])
        else:
            res[f'{disp}_sens2_gap'] = np.abs(x[x.group=='[nan  0.]'][met].mean()-x[x.group=='[nan  1.]'][met].mean())
        for group in list_groups:
            curr_group=x[x.group==group]
            if curr_group.shape[0]>0:
                res[f'{disp}_{group}_gap']=res[f'{disp}_all']-curr_group[met].mean()
            else:
                res[f'{disp}_{group}_gap']=np.nan
        
    return pd.Series(res)

def bold_max(s):
    fmts = []
    for grp in s.index.get_level_values(0).unique():
        max_seq = [i for i in s[grp] if isinstance(i, (float, np.float32, np.float64))]
        max_val = max(max_seq) if len(max_seq) else 0
        fmts += ['font-weight: bold' if i == max_val else '' for i in s[grp]]
    return fmts
