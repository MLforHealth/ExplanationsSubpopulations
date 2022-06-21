import numpy as np
import pandas as pd
import argparse
from lib import TabularDataset
from lib.train_blackbox import get_model
from pathlib import Path
import torch
import random
import socket
import copy
import json
import os
import sys
from xgboost.sklearn import XGBClassifier
from lib.distances import SinkhornDistance, MMD
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default = 'adult', choices = list(TabularDataset.dataset_params.keys()), 
        type = str)
parser.add_argument('--dist_method', type=str, choices = ['clf', 'mmd', 'wass', 'pca_mmd', 'pca_wass'], default = 'clf') 
parser.add_argument('--pca_dim', type=int, default = 5)
parser.add_argument('--wass_eps', type=float, default = 0.1)
parser.add_argument('--fairness_type', choices = ['dp', 'eo_p', 'eo_n'], default = 'dp')
parser.add_argument('--seed', type = int, default = 42)
parser.add_argument('--experiment', type=str, default = '')
parser.add_argument('--output_dir', type = Path, required = True)

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

with (args.output_dir/'args.json').open('w') as f:
    temp = copy.deepcopy(vars(args))
    for i in temp:
        if isinstance(temp[i], Path):
            temp[i] = os.fspath(temp[i])
    json.dump(temp, f, indent = 4)
    print(json.dumps(temp, indent = 4, sort_keys = True))

dataset = TabularDataset.Dataset(args.dataset)
X_train, X_val, X_test, y_train, y_val, y_test, g_train, g_val, g_test = dataset.get_data()

if args.fairness_type == 'eo_p':
    subset_y = 1
elif args.fairness_type == 'eo_n':
    subset_y = 0

if args.fairness_type.startswith('eo'): # we do not use validation data here
    X_train = X_train[y_train == subset_y]
    g_train = g_train[y_train == subset_y]
    X_test = X_test[y_test == subset_y]
    g_test = g_test[y_test == subset_y]

def ohe_to_df(enc, col, name):
    return pd.DataFrame(enc.transform(col.values.reshape(-1, 1)),
        columns = [f'{name}=={i}'for i in enc.categories_[0]],
        index = col.index)

for col in dataset.categorical_columns:
    X_train[col] = X_train[col].fillna(-1)
    X_test[col] = X_test[col].fillna(-1)
    enc = OneHotEncoder(sparse = False, handle_unknown = 'ignore').fit(X_train[col].values.reshape(-1, 1))
    X_train = pd.concat((X_train, ohe_to_df(enc, X_train[col], col)), axis = 1)
    X_test = pd.concat((X_test, ohe_to_df(enc, X_test[col], col)), axis = 1)

X_train = X_train.drop(columns = dataset.categorical_columns)
X_test = X_test.drop(columns = dataset.categorical_columns)

scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

results = {}
for attr in dataset.sensitive_attributes:
    results[attr] = {}
    if args.dist_method == 'clf':
        xgb = GridSearchCV(XGBClassifier(), param_grid = {'max_depth': list(range(7))}, n_jobs = -1, 
                        refit = True,
                        cv = 5,
                        scoring = 'roc_auc_ovr').fit(X_train, g_train[attr])
        pred= xgb.predict_proba(X_test)
        for grp in sorted(g_train[attr].unique()):
            results[attr][grp] = roc_auc_score(g_test[attr] == grp, pred[:, int(grp)])
    else: # only use train from here on
        if args.dist_method.startswith('pca'):
            X_train = PCA(n_components = args.pca_dim).fit_transform(X_train)
        for grp in sorted(g_train[attr].unique()):
            if args.dist_method.endswith('mmd'):
                results[attr][grp] = MMD()(torch.from_numpy(X_train), torch.from_numpy(X_train[g_train[attr] == grp, :])).item()
            elif args.dist_method.endswith('wass'):
                with torch.no_grad():
                    results[attr][grp] = SinkhornDistance(eps=args.wass_eps, max_iter=100)(torch.from_numpy(X_train), 
                        torch.from_numpy(X_train[g_train[attr] == grp, :]))[0].item()
torch.save(results, os.path.join(args.output_dir, "results.pkl"))    

with open(os.path.join(args.output_dir, 'done'), 'w') as f:
    f.write('done')