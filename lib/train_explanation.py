"""Get predictions from explanation model."""

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from functools import partial
import fairlime
import fairlime.lime_tabular
from rulefit import RuleFit
from lib.algorithms import ERM, ARL, JTT, LfF, JointDRO, GroupDRO, Reductionist
from tqdm import trange
import shap
from pygam import LogisticGAM, s, f, terms
from fairlearn.reductions import DemographicParity, EqualizedOdds, ErrorRateParity, ExponentiatedGradient
from lib import utils
import random
from lib.TabularDataset import dataset_params
from imblearn.over_sampling import RandomOverSampler
import math
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, LabelBinarizer
from sklearn.feature_selection import SelectKBest, f_regression

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

def select_top_k_categorical(train_x, train_y, n_features=100):
    if train_x.shape[1]<n_features:
        n_features = train_x.shape[1]

    categorical_features_idx=[]
    for i in range(train_x.shape[1]):
        if len(np.unique(train_x[:,i])) <= 2:
            categorical_features_idx.append(i)

    if len(categorical_features_idx)>0:
        score_func = partial(mutual_info_regression,
            discrete_features=categorical_features_idx)
    else:
        score_func = mutual_info_regression

    selector = SelectKBest(score_func,k=n_features)
    selector.fit(train_x, train_y)
    top_k_indices = np.argpartition(selector.scores_, -1*n_features)[-1*n_features:]
    return top_k_indices

def preprocess_select_k(train_x, test_x, train_y, cat_col_indices, 
    n_features=100):
    all_col_indices = np.arange(train_x.shape[1])
    steps = [
        ('scaler', StandardScaler())
    ]
    if len(cat_col_indices) > 0:
        numeric_features=list(set(all_col_indices)-set(cat_col_indices))
        numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
        ])

        categorical_transformer = OneHotEncoder(handle_unknown='ignore')

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, cat_col_indices)])

        steps=[
               ('scaler', preprocessor),
            ]

    preprocessor=Pipeline(steps)

    if isinstance(train_x, pd.DataFrame):
        train_x = train_x.values
        test_x = test_x.values

    preprocessor.fit(train_x)    
    train_x_scaled = preprocessor.transform(train_x)
    test_x_scaled = preprocessor.transform(test_x)

    try:
        train_x_scaled=train_x_scaled.toarray()
        test_x_scaled=test_x_scaled.toarray()
    except AttributeError:
        print('Already in csr state')

    if train_x_scaled.shape[1]<n_features:
        n_features = train_x_scaled.shape[1]

    categorical_features_idx = []
    for i in range(train_x_scaled.shape[1]):
        if len(np.unique(train_x_scaled[:,i])) <= 2:
            categorical_features_idx.append(i)

    if len(categorical_features_idx)>0:
        score_func = partial(mutual_info_regression,
            discrete_features=categorical_features_idx)
    else:
        score_func = mutual_info_regression

    selector = SelectKBest(score_func,k=n_features)
    selector.fit(train_x_scaled, train_y)
    train_x_transformed = selector.transform(train_x_scaled.copy())
    test_x_transformed = selector.transform(test_x_scaled.copy())

    transformed_categorical_features_idx = []
    transformed_numerical_features_idx = []
    for i in range(train_x_transformed.shape[1]):
        if len(np.unique(train_x_transformed[:,i])) <= 2:
            transformed_categorical_features_idx.append(i)
        else:
            transformed_numerical_features_idx.append(i)

    top_k_indices = np.argpartition(selector.scores_, -1*n_features)[-1*n_features:]

    return (train_x_scaled, test_x_scaled, train_x_transformed, test_x_transformed, 
        transformed_categorical_features_idx, 
        transformed_numerical_features_idx,top_k_indices)



def get_explanation_predictions(
        explanation_type,
        explanation_model,
        X_train, 
        X_test, y_train,
        clf, hparams, 
        n_feat = 10, 
        model_type = None,
        set_name = 'test',
        grp_clf = None,
        perturb_sigma = 1.,
        reductionist_type='Acc',
        reductionist_difference_bound=0.01,
        thresh=0.5,
        grp_train=None):
    """Get explanation.

    Parameters:
    explanation_type: str, local or global
    explanation_model: str, explanation model name
    X_train: pd.DataFrame, original unscaled data for training
        explanation models
    X_test:pd.DataFrame, original unscaled data for testing
        explanation models
    y_train: np.array, true groundtruth label
        NB: not used any where except for getting counts for 
        LIME
    clf: sklearn Pipeline, fit preprocessing and 
    grid-searched model pipeline
    n_feat: int, number of features to be used in explanation
    model_type: str, sklearn or not
    set_name: str, test or val
    grp_clf: sklearn model, predicts group information,
    perturb_sigma: float, sampling variance for LIME
    reductionist_type: str, for training fair decision tree
    thresh: decision making threshold.

    """
    
    if utils.has_checkpoint(suffix = set_name):
        chkpt = utils.load_checkpoint(suffix = set_name)
        expl_pred = chkpt['expl_pred']
        np.random.set_state(chkpt['numpy_state'])
        random.setstate(chkpt['random_state'])
        init = len(expl_pred)
        print("Loaded checkpoint with %s predictions." % init)
    else:
        expl_pred = []
        init = 0
    if hparams['dataset']=='adult':
        # this value is unseen in blackbox train set so 
        # could lead to indexing errors
        all_vals=np.array(X_train.NativeCountry.values)
        ind_delete = np.arange(len(all_vals))[all_vals==15]
        all_keep = list(set(np.arange(len(all_vals)))-set(ind_delete))
        X_train = X_train.iloc[all_keep]
        y_train = y_train[all_keep]
        if grp_train is not None:
            grp_train = grp_train[all_keep]

    categorical_features=[list(X_train.columns).index(i) for i in dataset_params[hparams['dataset']].categorical_columns]
    (train_x_scaled, test_x_scaled,
        train_x_transformed, test_x_transformed, 
        transformed_categorical_features_idx, 
        transformed_numerical_features_idx,top_k_indices)= preprocess_select_k(X_train, 
        X_test,
        clf.predict_proba(X_train)[:,1],  
        categorical_features,
        n_features=n_feat)

    if explanation_type == 'local' and explanation_model == 'lime':
        explainer = fairlime.lime_tabular.LimeTabularExplainer(
            X_train.values,
            feature_names=X_train.columns,
            categorical_features=[list(X_train.columns).index(i) for i in dataset_params[hparams['dataset']].categorical_columns],
            class_names=np.unique(y_train),
            discretize_continuous=False)

        if model_type == 'ERM':
            model_regressor = ERM(hparams)
        elif model_type == 'ARL':
            model_regressor = ARL(hparams)
        elif model_type == 'JTT':
            model_regressor = JTT(hparams)
        elif model_type == 'LfF':
            model_regressor = LfF(hparams)
        elif model_type is None or model_type == 'sklearn':
            model_regressor = None
        elif model_type == 'JointDRO':
            model_regressor = JointDRO(hparams)
        elif model_type == 'GroupDRO':
            model_regressor = GroupDRO(hparams)
        elif model_type == 'reductionist':
            model_regressor = Reductionist(hparams)
        else:
            raise NotImplementedError(model_type)

        for i in trange(init, X_test.shape[0]):
            exp = explainer.explain_instance(
                X_test.iloc[i].values, clf.predict_proba,
                num_features=n_feat,
                labels = (0,),
                model_regressor = model_regressor,
                grp_clf = grp_clf,
                perturb_sigma = perturb_sigma)
            expl_pred.append(1 - exp.local_pred[0])

            if i % 100 == 0 or i == X_test.shape[0] - 1:
                utils.save_checkpoint({'expl_pred': expl_pred,
                                        'numpy_state': np.random.get_state(),
                                        'random_state': random.getstate()},
                                        suffix = set_name)
        
    elif explanation_type == 'global' and explanation_model == "decision_tree":
        y_target = clf.predict(X_train.values)
        print(np.unique(y_target))
        rf = DecisionTreeClassifier(max_depth=hparams['tree_depth'])
        rf.fit(train_x_transformed, y_target)
        expl_pred = rf.predict_proba(test_x_transformed)[:,1]

    elif explanation_type == 'global' and explanation_model == "balanced_tree":
        rf = DecisionTreeClassifier(max_depth=hparams['tree_depth'])
        y_target = clf.predict(X_train.values)

        rs = RandomOverSampler(sampling_strategy='not majority')
        # NB: This step is necessary because some datasets are
        # converted to csr matrix during select k best features
        # step. This happens mostly with the adult dataset
        try:
            train_x_transformed=train_x_transformed.toarray()
            test_x_transformed=test_x_transformed.toarray()
        except AttributeError:
            # this happens when transformed values are already in
            # full or non-csr form
            print('Array already in full form for model training!')

        inds = rs.fit_resample(np.arange(train_x_transformed.shape[0]).reshape(-1, 1), grp_train)[0].squeeze()        
        x_train = train_x_transformed[inds,:]
        y_target = y_target[inds]

        rf.fit(x_train, y_target)
        expl_pred = rf.predict_proba(test_x_transformed)[:,1] 


    elif explanation_type == 'global' and explanation_model == "factor_gam":
        print('GAM')

        y_target = clf.predict(X_train.values)
        term_list = []
        for idx in transformed_categorical_features_idx:
            term_list.append(f(idx))

        for idx in transformed_numerical_features_idx:
            term_list.append(s(idx))

        joint_term_list = term_list[0]
        for term_item in term_list[1:]:
            joint_term_list+=term_item
        
        # NB: This step is necessary because some datasets are
        # converted to csr matrix during select k best features
        # step. This happens mostly with the adult dataset
        try:
            train_x_transformed=train_x_transformed.toarray()
            test_x_transformed=test_x_transformed.toarray()
        except AttributeError:
            # this happens when transformed values are already in 
            # full or non-csr form
            print('Array already in full form for GAM training!')
        
        # check that conversion to CSR array is complete
        assert len(term_list)==train_x_transformed.shape[1]

        gam = LogisticGAM(joint_term_list,
                max_iter=hparams["gam_max_iter"]).gridsearch(train_x_transformed,
                y_target)
        expl_pred = gam.predict_proba(test_x_transformed)


   
    elif explanation_type == 'local' and explanation_model == 'shap_blackbox_preprocessed':
        expl_pred = []
        K = 50
        blackbox_train_x = clf.best_estimator_['scaler'].transform(X_train.values)
        blackbox_test_x = clf.best_estimator_['scaler'].transform(X_test.values)
        
        # NB: This step is necessary because some datasets are
        # converted to csr matrix during select k best features
        # step. This happens mostly with the adult dataset
        try:
            # NB: Never reaches this step, but leaving it here
            blackbox_train_x=blackbox_train_x.toarray()
            blackbox_test_x=blackbox_test_x.toarray()
        except AttributeError:
            # this happens when transformed values are already in
            # full or non-csr form
            print('Array already in full form for GAM training!')
        
        explainer = shap.KernelExplainer(
            clf.best_estimator_['clf'].predict_proba,
            shap.sample(blackbox_train_x, K))

        data = blackbox_test_x
        shap_top_k=select_top_k_categorical(blackbox_train_x, 
                clf.predict_proba(X_train)[:,1], n_features=n_feat)
        print(data)
        # Shapley predictions is the sum of shapley values
        for i in data:
            try:
                shap_values = explainer.shap_values(
                    i, nsamples=50)
                expl_pred.append(explainer.expected_value[1] + np.sum(shap_values[1][shap_top_k]))
            except:
                if hparams['dataset']!='adult':
                    raise ValueError('SHAP explanation generation failed!')
                else:
                    explainer = shap.KernelExplainer(
                    clf.best_estimator_['clf'].predict_proba,
                    shap.sample(blackbox_train_x, 2*K))
                    shap_values = explainer.shap_values(
                    i, nsamples=50)
                    expl_pred.append(explainer.expected_value[1] + np.sum(shap_values[1]))
                    print('selecting top-k did not work!')

    else:
        raise NotImplementedError

    return np.array(expl_pred)

