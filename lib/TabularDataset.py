import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from lib.datasets import params_mimic_tab_fair,params_recidivism,params_compas_balanced,params_adult, params_lsac, params_mimic_tab

dataset_params = {
    'adult': params_adult,
    'recidivism': params_recidivism,
    'lsac': params_lsac,
    'mimic_tab': params_mimic_tab,
    'compas_balanced':params_compas_balanced,
    'mimic_tab_fair_ds':params_mimic_tab_fair,
}

"""
try:
    from lib.datasets import params_mimic
    dataset_params['mimic'] = params_mimic
except:
    print("MIMIC notes data not available!")
"""
class Dataset:
    def __init__(self, ds_name):
        opts = dataset_params[ds_name]

        self.link = opts.link
        self.columns = opts.columns
        self.train_cols = opts.train_cols
        self.label = opts.label
        self.sensitive_attributes = opts.sensitive_attributes
        self.use_sensitive = opts.use_sensitive
        self.already_split = opts.already_split
        self.categorical_columns = opts.categorical_columns
        self.has_header = opts.has_header


        if self.use_sensitive:
            for i in self.sensitive_attributes:
                if i not in self.train_cols:
                    self.train_cols.append(i)
        else:
            for i in self.sensitive_attributes:
                if i in self.train_cols:
                    self.train_cols.remove(i)

    def get_data(self, retain_cols=False):
        df = pd.read_csv(
            self.link,
            header=0 if self.has_header else None)
        if not self.has_header:
            df.columns = self.columns

        label = self.label
        train_cols = self.train_cols
        cat_cols_all=self.categorical_columns + [i for i in self.sensitive_attributes if isinstance(df[i].iloc[0], str)]
        
        if retain_cols:
            cat_cols_all=self.categorical_columns
                
        for col in cat_cols_all:
            # NB: Choosing OrdinalEncoder here instead of OneHot because:
            # preserves alphabetical/numerical order to help in result interpretation.
            # The exact encoding for training ML models is done in the `train_blackbox.py`
            # script with OneHotEncoding.
            enc = OrdinalEncoder()
            df[col] = enc.fit_transform(
                df[col].values.reshape(-1, 1))
        
        if not self.already_split:
            X = df[train_cols]
            # Turning response into 0 and 1
            y = df[label].values
            # Fixing the train/test split across experiments
            seed = 1
            X_train, X_test, y_train, y_test, g_train, g_test = train_test_split(
                X, y, df[self.sensitive_attributes], test_size=0.5, random_state=seed)
            X_val, X_test, y_val, y_test, g_val, g_test = train_test_split(
                X_test, y_test, g_test, test_size=0.2, random_state=seed)
            X_train_expl, X_val_expl, y_train_expl, y_val_expl, g_train_expl, g_val_expl = train_test_split(
                    X_val, y_val, g_val, test_size=0.25, random_state=seed)
        else:
            df_train = df[df.fold_id =='train']
            df_val = df[df.fold_id =='eval']
            df_test = df[df.fold_id =='test']

            X_train, X_val, X_test = df_train[train_cols], df_val[train_cols], df_test[train_cols]
            y_train, y_val, y_test = df_train[label].values, df_val[label].values, df_test[label].values
            g_train, g_val, g_test = df_train[self.sensitive_attributes], df_val[self.sensitive_attributes], df_test[self.sensitive_attributes]
            seed=1
            X_train_expl, X_val_expl, y_train_expl, y_val_expl, g_train_expl, g_val_expl = train_test_split(
                    X_val, y_val, g_val, test_size=0.25, random_state=seed)
        
        enc = OrdinalEncoder()
        y_train = enc.fit_transform(
            y_train.reshape(-1, 1))
        y_train_expl = enc.transform(y_train_expl.reshape(-1, 1))
        y_val_expl = enc.transform(y_val_expl.reshape(-1, 1))
        y_test = enc.transform(y_test.reshape(-1, 1))

        return X_train, X_train_expl, X_val_expl, X_test, y_train, y_train_expl, y_val_expl, y_test, g_train, g_train_expl, g_val_expl, g_test
