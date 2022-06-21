import numpy as np
import pandas as pd
from pathlib import Path
from itertools import product
from lib.TabularDataset import dataset_params

def combinations(grid):
    return list(dict(zip(grid.keys(), values)) for values in product(*grid.values()))
        
def get_hparams(experiment):
    if experiment not in globals():
        raise NotImplementedError
    return globals()[experiment]().get_hparams()    

def get_script_name(experiment):
    if experiment not in globals():
        raise NotImplementedError
    return globals()[experiment].fname
 
    
#### write experiments here 
class rule_depth_vary():
    fname = 'run.py'
    def __init__(self):
        self.hparams = {
            'experiment': ['rule_depth_vary_correct_replication'],
            'dataset': ['adult','lsac','lsac_cat','mimic_tab_fair_ds','mimic_tab','compas_balanced'],
            'blackbox_model': ['lr','nn'],
            'explanation_type': ['global'],
            'explanation_model': ['decision_tree'],
            'n_features': [100],
            'tree_depth':list(range(1,7)),
            'train_grp_clf': [False],
            'balance_groups':[False],
            'grp_clf_attr':['all'],
            'seed': [1,2,3,4,5] ,
            'evaluate_val': [True],
            'C':[1],
            'model_type': ['sklearn']
        }
    def get_hparams(self):
        return combinations(self.hparams)


class gam_max_iter_vary():
    fname = 'run.py'
    def __init__(self):
        self.hparams = {
            'experiment': ['gam_max_iter_vary_correct_replication'],
            'dataset': ['adult','lsac','lsac_cat','mimic_tab_fair_ds','mimic_tab','compas_balanced'],
            'blackbox_model': ['lr','nn'],
            'explanation_type': ['global'],
            'explanation_model': ['factor_gam'],#decision_tree
            'n_features': [100],
            'gam_max_iter':[100,200,500],
            'train_grp_clf': [False],
            'balance_groups':[False,True],
            'seed': [1,2,3,4,5] ,
            'evaluate_val': [True],
            'grp_clf_attr':['all'],
            'C':[1],
            'model_type': ['sklearn']
        }

    def get_hparams(self):
        return combinations(self.hparams)

class rule_balanced_gender():
    fname = 'run.py'
    def __init__(self):
        self.hparams = {
            'experiment': ['rule_balanced_correct_replication'],
            'dataset': ['adult','mimic_tab'],
            'blackbox_model': ['lr','nn'],
            'explanation_type': ['global'],
            'explanation_model':['decision_tree','balanced_tree'],
            'n_features':[100],
            'seed': [1,2,3,4,5],
            'balance_groups':[True,False],
            'balance_group_idx':[0],
            'evaluate_val': [True],
            'C':[1],
            'tree_depth':list(range(1,7)),
            'model_type': ['sklearn'],
            'perturb_sigma':[1]#,5,10,15,20,25,100]
        }

    def get_hparams(self):
        return combinations(self.hparams)


class rule_balanced_race():
    fname = 'run.py'
    def __init__(self):
        self.hparams = {
            'experiment': ['rule_balanced_correct_replication'],
            'dataset': ['lsac','compas_balanced'],
            'blackbox_model': ['lr','nn'],
            'explanation_type': ['global'],
            'explanation_model':['decision_tree','balanced_tree'],
            'n_features':[100],
            'seed': [1,2,3,4,5],
            'balance_groups':[False,True],
            'balance_group_idx':[1],
            'evaluate_val': [True],
            'C':[1],
            'tree_depth':list(range(1,7)),
            'model_type': ['sklearn'],
            'perturb_sigma':[1]#,5,10,15,20,25,100]
        }
    def get_hparams(self):
        return combinations(self.hparams)

class rule_vary_features():
    fname = 'run.py'
    def __init__(self):
        self.hparams = {
            'experiment': ['rule_vary_features_correct_replication'],
            'dataset': ['mimic_tab','adult'],
            'blackbox_model': ['lr','nn'],
            'explanation_type': ['global'],
            'explanation_model': ['factor_gam','decision_tree'],#,'decision_tree'
            'n_features':[2,5,10,15,20,30,40,50,60],
            'seed': [1,2,3,4,5],
            'balance_groups':[False],
            'evaluate_val': [True],
            'tree_depth':[5],
            'gam_max_iter':[100],
            'model_type': ['sklearn'],
            'perturb_sigma':[1]#,5,10,15,20,25,100]
        }
    def get_hparams(self):
        return combinations(self.hparams)

class gam_balanced_gender():
    fname = 'run.py'
    def __init__(self):
        self.hparams = {
            'experiment': ['gam_balanced_correct_replication'],
            'dataset': ['adult','mimic_tab'],
            'blackbox_model': ['lr','nn'],
            'explanation_type': ['global'],
            'explanation_model':['factor_gam'],
            'n_features':[100],
            'seed': [1,2,3,4,5],
            'balance_groups':[True],
            'balance_group_idx':[0],
            'evaluate_val': [True],
            'C':[1],
            'model_type': ['sklearn'],
            'gam_max_iter':[100,200,500],
            'perturb_sigma':[1]#,5,10,15,20,25,100]
        }

    def get_hparams(self):
        return combinations(self.hparams)

class gam_balanced_race():
    fname = 'run.py'
    def __init__(self):
        self.hparams = {
            'experiment': ['gam_balanced_correct_replication'],
            'dataset': ['lsac','compas_balanced'],
            'blackbox_model': ['lr','nn'],
            'explanation_type': ['global'],
            'explanation_model':['factor_gam'],
            'n_features':[100],
            'seed': [1,2,3,4,5],
            'balance_groups':[True],
            'balance_group_idx':[1],
            'evaluate_val': [True],
            'C':[1],
            'model_type': ['sklearn'],
            'gam_max_iter':[100,200,500],
            'perturb_sigma':[1]#,5,10,15,20,25,100]
        }

    def get_hparams(self):
        return combinations(self.hparams)
# LIME
class lime_all():
    fname = 'run.py'
    def __init__(self):
        self.hparams = {
            'experiment': ['lime_all_correct_replication'],
            'dataset': ['lsac_cat','adult','lsac','mimic_tab','compas_balanced'],
            'blackbox_model': ['lr','nn'],
            'explanation_type': ['local'],
            'explanation_model':['shap_blackbox_preprocessed','lime'],
            'n_features':[100],
            'seed': [0,1,2,3,4,5],
            'balance_groups':[False],
            'grp_clf_attr':['all'],
            'evaluate_val': [True],
            'C':[1],
            'model_type': ['sklearn'],
            'perturb_sigma':[1]#,5,10,15,20,25,100]
        }

    def get_hparams(self):
        return combinations(self.hparams)

class lime_balanced_gender():
    fname = 'run.py'
    def __init__(self):
        self.hparams = {
            'experiment': ['lime_balanced_correct_replication'],
            'dataset': ['adult','mimic_tab'],
            'blackbox_model': ['lr','nn'],
            'explanation_type': ['local'],
            'explanation_model':['shap_blackbox_preprocessed','lime'],# ['lime','shap_scaled','shap_not_transformed','shap_transformed'],
            'n_features':[100],
            'seed': [1,2,3,4,5],
            'balance_groups':[True],
            'balance_group_idx':[0],
            'evaluate_val': [True],
            'C':[1],
            'model_type': ['sklearn'],
            'perturb_sigma':[1]#,5,10,15,20,25,100]
        }

    def get_hparams(self):
        return combinations(self.hparams)

class lime_balanced_race():
    fname = 'run.py'
    def __init__(self):
        self.hparams = {
            'experiment': ['lime_balanced_correct_replication'],
            'dataset': ['lsac','recidivism','compas_balanced'],
            'blackbox_model': ['lr','nn'],
            'explanation_type': ['local'],
            'explanation_model':['shap_blackbox_preprocessed','lime'],# ['lime','shap_scaled','shap_not_transformed','shap_transformed'],
            'n_features':[100],
            'seed': [1,2,3,4,5],
            'balance_groups':[True],
            'balance_group_idx':[1],
            'evaluate_val': [True],
            'C':[1],
            'model_type': ['sklearn'],
            'perturb_sigma':[1]#,5,10,15,20,25,100]
        }
    def get_hparams(self):
        return combinations(self.hparams)

class lime_vary_sigma():
    fname = 'run.py'
    def __init__(self):
        self.hparams = {
            'experiment': ['lime_vary_sigma_correct_replication'],
            'dataset': ['adult','lsac','compas_balanced','mimic_tab'],
            'blackbox_model': ['lr','nn'],
            'explanation_type': ['local'],
            'explanation_model':['lime'],
            'n_features':[100],
            'seed': [1,2,3,4,5],
            'balance_groups':[False],
            'grp_clf_attr':['all'],
            'evaluate_val': [True],
            'C':[1],
            'model_type': ['sklearn'],
            'perturb_sigma':[0.001,0.01,0.1,0.5,5,10,20]
        }
    def get_hparams(self):
        return combinations(self.hparams)

class lime_vary_features():
    fname = 'run.py'
    def __init__(self):
        self.hparams = {
            'experiment': ['lime_vary_features_correct_replication'],
            'dataset': ['mimic_tab','adult'],
            'blackbox_model': ['lr','nn'],
            'explanation_type': ['local'],
            'explanation_model': ['lime','shap_blackbox_preprocessed'],
            'n_features':[2,5,10,15,20,30,40,50,60],
            'seed': [1,2,3,4,5],
            'balance_groups':[False],
            'evaluate_val': [True],
            'C':[1],
            'blackbox_train_fair':[False],
            'model_type': ['sklearn'],
            'perturb_sigma':[1]#,5,10,15,20,25,100]
        }

    def get_hparams(self):
        return combinations(self.hparams)

    
    
# JTT
class JTT_hyperparam_search():
    fname = 'run.py'
    def __init__(self):
        self.hparams = {
            'experiment': ['JTT'],
            'dataset': ['adult', 'lsac','mimic_tab','compas_balanced'],
            'blackbox_model': ['lr', 'nn'],
            'explanation_type': ['local'],
            'explanation_model': ['lime'],
            'n_features': [100],
            'seed': list(range(5)),
            'model_type': ['JTT'],
            'jtt_lambda': [5, 10, 50, 20, 100],
            'C':[1],
            'grp_clf_attr':['all'],
            'balance_groups':[False],
            'ignore_lime_weights': [False],
            'evaluate_val': [True]
        }

    def get_hparams(self):
        return combinations(self.hparams)

class JTT_mimic_hyperparam_search():
    fname = 'run.py'
    def __init__(self):
        self.hparams = {
            'experiment': ['JTT_mimic_lr'],
            'dataset': ['mimic_tab'],
            'blackbox_model': ['lr','nn'],
            'explanation_type': ['local'],
            'explanation_model': ['lime'],
            'n_features': [100],
            'seed': list(range(5)),
            'model_type': ['JTT'],
            'jtt_lambda': [5, 10, 50, 20, 100],
            'jtt_thres':[0.5,0.6],
            'C':[1,2,3],
            'grp_clf_attr':['all'],
            'balance_groups':[False],
            'ignore_lime_weights': [False],
            'evaluate_val': [True]
        }

    def get_hparams(self):
        return combinations(self.hparams)

## Distances between subgroups
class DatasetDists():
    fname = 'get_dataset_distances.py'
    def __init__(self):
        base_hparams = {
            'experiment': ['dist'],
            'dataset': list(dataset_params.keys()),
            'dist_method': ['clf', 'mmd', 'wass', 'pca_mmd', 'pca_wass'],
            'fairness_type': ['dp', 'eo_p', 'eo_n']
        }

        self.hparams_grid = combinations(base_hparams)

    def get_hparams(self):
        return self.hparams_grid