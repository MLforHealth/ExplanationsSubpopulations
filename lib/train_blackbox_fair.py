"""Script for training blackbox model."""

import numpy as np
import pandas as pd

from fairlearn.reductions import DemographicParity, EqualizedOdds, ErrorRateParity, ExponentiatedGradient

# feature selection
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, LabelBinarizer
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import warnings
from itertools import count
from collections import defaultdict
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
from aif360.sklearn.inprocessing import AdversarialDebiasing

warnings.filterwarnings("ignore")

def scale_data(all_col_indices, cat_col_indices = []):
    steps=[('scaler',StandardScaler())]
    if len(cat_col_indices) > 0:
        numeric_features=list(set(all_col_indices)-set(cat_col_indices))
        numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())])

        categorical_transformer = OneHotEncoder(handle_unknown='ignore',sparse=False)

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, cat_col_indices)])

        steps=[
               ('scaler', preprocessor)
            ]

    return steps

def get_model_nn_debiased(x,y,grp_train,
        adversary_loss_weight=2,n_units=100,
        scoring = 'roc_auc', cat_cols = []):

    assert isinstance(x, pd.DataFrame)

    cat_col_idx=[list(x.columns).index(i) for i in cat_cols]
    cat_cols=cat_col_idx
    x=x.values
    x=pd.DataFrame(x)
    all_cols=x.columns
    if y.ndim == 2:
        y = y.squeeze()

    scaler_steps=scale_data(all_col_indices=all_cols, cat_col_indices = cat_cols)
    scaler = Pipeline(scaler_steps)
    x = scaler.fit_transform(x)

    adv_deb = AdversarialDebiasing(prot_attr=grp_train.columns,adversary_loss_weight=adversary_loss_weight,classifier_num_hidden_units=n_units)
    grp_ind = pd.MultiIndex.from_frame(grp_train, names=grp_train.columns)
    y_all = pd.Series(y)
    y_all.index=grp_ind
    x=pd.DataFrame(x)
    x.index=grp_ind
    adv_deb.fit(x,y_all)
    scaler_steps.extend([('clf',adv_deb)])
    return Pipeline(scaler_steps)


def get_model(x, y, model_name, grp_train=[], 
    train_fair=False,
    reductionist_type='DP',
    reductionist_difference_bound=0.01,
    fair_blackbox_c=1,
    scoring = 'roc_auc', cat_cols = []):
    """Compute correlation between a feature and score.

    Parameters:
    x: np.array, feature values
    y: np.array, score values
    model_name: str, name of sklearn model
    cat_cols: list of str, categorical columns in x. Ignored if model_name is 'rf' or 'xgb'

    Outputs:
    clf: sklearn estimator, fit model

    """
    if len(cat_cols):
        assert isinstance(x, pd.DataFrame)
        cat_col_indices = [list(x.columns).index(i) for i in cat_cols]
    else:
        cat_col_indices = []

    if isinstance(x, pd.DataFrame):
        x = x.values

    if y.ndim == 2:
        y = y.squeeze()

    gs_args = {
        'n_jobs': -1, 
        'refit': True,
        'cv': 5,
        'scoring': scoring
    }
    
    # get scaling steps for categorical and numeric features separately
    scaler_steps=scale_data(all_col_indices=list(range(x.shape[1])), cat_col_indices = cat_col_indices)
    scaler = Pipeline(scaler_steps)
    x = scaler.fit_transform(x)

    """
    model_dict = {}
    model_dict["lr"] = LogisticRegression()#GridSearchCV(LogisticRegression(solver = 'liblinear'), param_grid = {'C': 10.**np.linspace(-5, 1, 25)}, **gs_args)
    model_dict["nn"] = MLPClassifier()#GridSearchCV(MLPClassifier(), param_grid = {'hidden_layer_sizes': [(n_hidden, ) for n_hidden in [50, 100, 200]]}, **gs_args)
    """
    
    if train_fair:
        if reductionist_type == 'DP':
            cons = DemographicParity(difference_bound=reductionist_difference_bound)
        elif reductionist_type == 'EO':
            cons = EqualizedOdds(difference_bound=reductionist_difference_bound)
        elif reductionist_type == 'Acc':
            cons = ErrorRateParity(difference_bound=reductionist_difference_bound)
        else:
            raise NotImplementedError
        est = LogisticRegression(C=fair_blackbox_c)#model_dict[model_name]
        # getting train group info for training fair global models
        # this creates a list of groups, accounting for intersectional
        # groups defined by sensitive attributes
        mapping = defaultdict(count().__next__)
        sens_list=grp_train.values
        sens_list_id = []
        for element in sens_list:
            sens_list_id.append(mapping[tuple(element)])
        grp = np.array(sens_list_id)
        grp=sens_list
        try:
            x=x.toarray()
        except AttributeError:
            print('x already in array form!')
        eg = ExponentiatedGradient(est, cons, eps = reductionist_difference_bound).fit(x, y, sensitive_features = grp)
        clf = eg.predictors_.iloc[-1]
        scaler_steps.extend([('clf',clf)])
        return Pipeline(scaler_steps)
    else:
        print('Please use function from train_blackbox.py!')
        raise NotImplementedError
                                    
    return clf


def preprocess_pipeline(X_train, y_train, X_test):
    """Carry out all steps in preprocessing for classification.

    Parameters:
    X_train: np.array, train features
    X_test: np.array, test features
    y_train: np.array, train labels
    num_features: int, number of features to select

    Outputs:
    X_train: np.array, train features
    X_test: np.array, test features

    """
    # setting np.inf to np.nan
    X_train[~np.isfinite(X_train)] = np.nan
    X_test[~np.isfinite(X_test)] = np.nan

    return X_train, y_train, X_test


def train_clf_fair(model_name, x_train, x_val,
              x_test, y_train,
              y_val, y_test, 
              grp_train=[], 
              train_fair=2,
              reductionist_type='DP',
              reductionist_difference_bound=0.01,
              fair_blackbox_c=1,
              adversary_loss_weight=2,n_units=100,
              scoring = 'roc_auc', cat_cols = []):
    """Train+test prediction results with fair blackbox models."""
    if train_fair==2:
        clf=get_model_nn_debiased(x_train, y_train, grp_train,
                adversary_loss_weight=adversary_loss_weight,
                n_units=n_units,
                scoring = scoring, cat_cols = cat_cols)
        scaler=clf['scaler']
        clf_model=clf['clf']
        x_val=x_val.values
        x_test=x_test.values
        x_val = pd.DataFrame(scaler.transform(pd.DataFrame(x_val)))
        x_test = pd.DataFrame(scaler.transform(pd.DataFrame(x_test)))
        val_pred=clf_model.predict(x_val)
        test_pred=clf_model.predict(x_test)
        val_prob=clf_model.predict_proba(x_val)
        test_prob=clf_model.predict_proba(x_test)
        return clf, val_pred, test_pred, val_prob, test_prob

        
    else:
        clf = get_model(x_train, y_train, model_name, grp_train=grp_train, train_fair = train_fair,
            reductionist_type=reductionist_type,
            reductionist_difference_bound=reductionist_difference_bound,
            fair_blackbox_c=fair_blackbox_c,
            scoring = scoring, cat_cols = cat_cols)
        if isinstance(x_test, pd.DataFrame):
            x_val = x_val.values
            x_test = x_test.values

        return clf, clf.predict(x_val), clf.predict(x_test), clf.predict_proba(x_val), clf.predict_proba(x_test)

