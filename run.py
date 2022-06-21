import argparse
from lib import TabularDataset
from lib.train_blackbox import train_clf
from lib.train_blackbox_fair import train_clf_fair
from lib.train_explanation import get_explanation_predictions
import torch
import numpy as np
import pandas as pd
import socket
import sys
import random
from pathlib import Path
import copy
import os
import json
import pandas as pd

from itertools import count
from collections import defaultdict
from imblearn.over_sampling import RandomOverSampler


def program_config(parser):
    parser.add_argument("--dataset", default = 'adult', choices = list(TabularDataset.dataset_params.keys()), 
        type = str)
    parser.add_argument('--blackbox_model',
                        default='xgb', choices = ['lr', 'nn', 'svm_rbf', 'xgb', 'rf'], type=str)
    parser.add_argument('--explanation_type',
                        default='local', type=str)
    parser.add_argument('--explanation_model',
                        default='lime', type=str)
    parser.add_argument('--n_features',
                        default=10, type=int)                    
    parser.add_argument('--model_type', type=str, choices = ['ARL', 'ERM', 'LfF', 'sklearn', 'JTT', 'JointDRO', 'GroupDRO', 'reductionist']) 
    parser.add_argument('--seed', type = int, default = 42)
    parser.add_argument('--experiment', type=str, default = '')
    parser.add_argument('--output_dir', type = Path, required = True)
    parser.add_argument('--ignore_lime_weights', action = 'store_true')
    parser.add_argument('--evaluate_val', action = 'store_true', help = "generate explanations for validation set")
    parser.add_argument('--max_epochs', default = 100, type = int)
    parser.add_argument('--perturb_sigma', type = float, default = 1.)

    # balance training set
    parser.add_argument('--balance_groups', action = 'store_true', help = 'balance groups on training set by supersampling')
    parser.add_argument('--balance_labels', action = 'store_true', help = 'balance labels on training set by supersampling')
    parser.add_argument('--balance_group_idx', type=int, default=0)
    
    # group aware debiasing
    parser.add_argument('--train_grp_clf', action = 'store_true')
    parser.add_argument('--grp_clf_attr', type = str, default='all') # sensitive attribute in data

    # ERM and ARL
    parser.add_argument('--lr', type = float, default = 5e-1)
    parser.add_argument('--C', type = float, default = 1.0)
    parser.add_argument('--batch_size', type = int, default = None)
    parser.add_argument('--debug', action = 'store_true')

    # JTT
    parser.add_argument('--jtt_lambda', type = float, default = 10.)
    parser.add_argument('--jtt_thres', type = float, default = 0.5)

    # JointDRO
    parser.add_argument('--joint_dro_alpha', type=float, default=0.1)  # default=1 in jtt implementation

    # GroupDRO
    parser.add_argument('--groupdro_eta', type = float, default = 1.)

    # reductionist 
    parser.add_argument('--reductionist_type', type = str, choices = ['EO','DP', 'Acc'], default = 'EO')
    parser.add_argument('--reductionist_difference_bound', type = float, default = 0.01)
    parser.add_argument('--reductionist_thres', type = float, default = 0.5)

    # global decision tree
    parser.add_argument('--tree_depth',type=int, default=7)
    
    # parameters for fitting GAMs
    parser.add_argument('--gam_max_iter', type = int, default=100)
    return parser


parser = argparse.ArgumentParser()
parser = program_config(parser)
args = parser.parse_args()

print("Environment:")
print("\tPython: {}".format(sys.version.split(" ")[0]))
print("\tPyTorch: {}".format(torch.__version__))
print("\tCUDA: {}".format(torch.version.cuda))
print("\tDevice: {}".format('cuda' if torch.cuda.is_available() else 'cpu'))
print("\tCUDNN: {}".format(torch.backends.cudnn.version()))
print("\tNumPy: {}".format(np.__version__))
print("\tNode: {}".format(socket.gethostname()))

random.seed(args.seed)
torch.manual_seed(args.seed)

(args.output_dir).mkdir(parents=True, exist_ok=True)

dataset = TabularDataset.Dataset(args.dataset)
X_train, X_train_expl, X_val_expl, X_test, y_train, y_train_expl, y_val_expl, y_test, g_train, g_train_expl, g_val_expl, g_test = dataset.get_data()

assert(args.grp_clf_attr is None or args.grp_clf_attr == 'all' or args.grp_clf_attr in g_train.columns)

random.seed(args.seed)
torch.manual_seed(args.seed)

if args.balance_groups:
    try:
        #assert args.grp_clf_attr in g_train.columns
        assert args.balance_group_idx<=1
        balance_grp=g_train.columns[args.balance_group_idx]
        grp = g_train[balance_grp]
    except AssertionError:
        print('Using all sensitive columns by default!')
        mapping = defaultdict(count().__next__)
        grp_vals=g_train.values
        grp = []
        for element in grp_vals:
            grp.append(mapping[tuple(element)])
        grp = np.array(grp)
    rs = RandomOverSampler(sampling_strategy='not majority')
    inds = rs.fit_resample(np.arange(len(X_train)).reshape(-1, 1), grp)[0].squeeze()    
    X_train = X_train.iloc[inds]
    y_train = y_train[inds]
    g_train = g_train.iloc[inds]
    
if args.balance_labels:
    rs = RandomOverSampler(sampling_strategy='not majority')
    inds = rs.fit_resample(np.arange(len(X_train)).reshape(-1, 1), y_train)[0].squeeze()        
    X_train = X_train.iloc[inds]
    y_train = y_train[inds]
    g_train = g_train.iloc[inds]

if args.debug:
    X_val = X_val.head(128)
    X_test = X_test.head(128)
    y_val = y_val[:128, :]
    y_test = y_test[:128, :]
    g_val = g_val.head(128)
    g_test = g_test.head(128)

clf, blackbox_pred_val, blackbox_pred_test, blackbox_prob_val, blackbox_prob_test = train_clf(args.blackbox_model,
                                X_train, X_val_expl,
                                X_test, y_train,
                                y_val_expl, y_test,
                                cat_cols = TabularDataset.dataset_params[args.dataset].categorical_columns)

if args.model_type in ['GroupDRO', 'reductionist']:
    assert args.train_grp_clf

if args.train_grp_clf:
    for i in g_train.columns:
        assert i not in X_train.columns
    #assert args.grp_clf_attr in g_train.columns

    grp_clf, _, _, _, _ = train_clf('xgb', X_train, X_val_expl, X_test, g_train[args.grp_clf_attr].values,
        g_val_expl[args.grp_clf_attr].values, g_test[args.grp_clf_attr].values, scoring = 'roc_auc_ovr')
else:
    grp_clf = None

final_outputs = []
if args.evaluate_val: ssets = ['test', 'val']
else: ssets = ['test']
    
    
# getting train group info for training fair global models
mapping = defaultdict(count().__next__)

sens_list=g_train_expl.values
sens_list_id = []
for element in sens_list:
    sens_list_id.append(mapping[tuple(element)])
sens_list_id = np.array(sens_list_id)

assert args.balance_group_idx<=1
balance_grp=g_train.columns[args.balance_group_idx]
balance_grp_expl= g_train_expl[balance_grp].values

for sset, mat, blackbox_pred, blackbox_prob, ground_truth in zip(ssets, [X_test, X_val_expl],
         [blackbox_pred_test, blackbox_pred_val],
         [blackbox_prob_test, blackbox_prob_val],
         [y_test, y_val_expl]):
    expl_predictions = get_explanation_predictions(
        args.explanation_type, args.explanation_model,
        X_train_expl,
        mat, y_train_expl,
        clf, hparams = vars(args), n_feat = args.n_features,
        model_type = args.model_type,
        set_name = sset,
        grp_clf = grp_clf,
        perturb_sigma = args.perturb_sigma,
        reductionist_type=args.expl_reductionist_type,
        reductionist_difference_bound=args.expl_reductionist_difference_bound,
        thresh=args.expl_thresh,
        grp_train = balance_grp_expl
        )
    curr_test = mat.copy()
    curr_test['blackbox_pred'] = blackbox_pred
    curr_test['blackbox_prob'] = blackbox_prob[:,1]
    curr_test['expl_pred'] = expl_predictions
    curr_test['groundtruth'] = ground_truth
    curr_test['set'] = sset
    final_outputs.append(curr_test)


final_outputs = pd.concat(final_outputs, ignore_index = True)
final_outputs.to_csv(args.output_dir/'{}_{}_{}.csv'.format(args.blackbox_model,
                                        args.explanation_type,
                                        args.explanation_model), index = False)

# creating outputs
with (args.output_dir/'args.json').open('w') as f:
    temp = copy.deepcopy(vars(args))
    for i in temp:
        if isinstance(temp[i], Path):
            temp[i] = os.fspath(temp[i])
    json.dump(temp, f, indent = 4)
    print(json.dumps(temp, indent = 4, sort_keys = True))

with (args.output_dir/'done').open('w') as f:
    f.write('done')
    
