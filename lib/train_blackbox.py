"""Script for training blackbox model."""

import numpy as np
import pandas as pd

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

warnings.filterwarnings("ignore")

def add_scaler(clf, all_col_indices, cat_col_indices = []):
    steps = [
        ('scaler', StandardScaler()),
        ('clf', clf)
    ]
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
               ('scaler', preprocessor),
               ('clf', clf)
            ]

    return Pipeline(steps)


def get_model(x, y, model_name, scoring = 'roc_auc_ovr', cat_cols = []):
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

    model_dict = {}
    model_dict["lr"] = GridSearchCV(add_scaler(
        LogisticRegression(solver = 'liblinear'), cat_col_indices = cat_col_indices, all_col_indices=list(range(x.shape[1]))), param_grid = {'clf__C': 10.**np.linspace(-5, 1, 25)}, **gs_args)
    model_dict["svm_rbf"] = GridSearchCV(add_scaler(SVC(kernel='rbf', probability=True), all_col_indices=list(range(x.shape[1])), cat_col_indices = cat_col_indices),
        param_grid = {'clf__C': 10.**np.linspace(-5, 1, 25)}, **gs_args)
    model_dict["nn"] = GridSearchCV(add_scaler(MLPClassifier(), 
        cat_col_indices = cat_col_indices, 
        all_col_indices=list(range(x.shape[1]))), param_grid = {'clf__hidden_layer_sizes': [(n_hidden, ) for n_hidden in [50, 100, 200]]}, **gs_args)
    model_dict["xgb"] = GridSearchCV(XGBClassifier(), param_grid = {'max_depth': list(range(7))}, **gs_args)
    model_dict["rf"] = GridSearchCV(RandomForestClassifier(), param_grid = {'max_depth': list(range(7))}, **gs_args)

    clf = model_dict[model_name]

    clf.fit(x, y)
                                    
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


def train_clf(model_name, x_train, x_val,
              x_test, y_train,
              y_val, y_test, scoring = 'roc_auc', cat_cols = []):
    """Train+test prediction results. """
    clf = get_model(x_train, y_train, model_name, scoring = scoring, cat_cols = cat_cols)
    if isinstance(x_test, pd.DataFrame):
        x_val = x_val.values
        x_test = x_test.values

    return clf, clf.predict(x_val), clf.predict(x_test), clf.predict_proba(x_val), clf.predict_proba(x_test)
