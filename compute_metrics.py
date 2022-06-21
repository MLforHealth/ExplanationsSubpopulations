import numpy as np
from collections import namedtuple
import glob
import pandas as pd
from sklearn import metrics
from sklearn.metrics import roc_auc_score, auc, roc_curve,precision_recall_curve
from lib.metrics import compute_fairness_metrics_diff


def perf_measure(y_actual, y_pred):
    y_hat = y_pred >= 0.5

    ACC = metrics.accuracy_score(y_actual, y_hat)

    try:
        fpr, tpr, thresholds = roc_curve(y_actual, y_pred, pos_label=1)
        roc = metrics.auc(fpr, tpr)
        # NB: Commenting out, but verified that both methods produced same 
        # results
#         roc = roc_auc_score(y_actual, y_pred)

    except ValueError:
        roc = np.nan

    return {
        'AUROC': roc,
        'ACC': ACC,
    }


class Model:

    def __init__(self, pred_proba, label, label_proba):
        self.pred_proba = pred_proba
        self.pred = (pred_proba >= 0.5).astype(int)
        self.label = label
        self.label_proba = label_proba

    def logits(self):
        raw_logits = np.clip(np.log(self.pred / (1 - self.pred)), -100, 100)
        return raw_logits

    def num_samples(self):
        return len(self.pred)

    def base_rate(self):
        """
        Percentage of samples belonging to the positive class
        """
        return np.mean(self.label)

    def accuracy(self):
        return self.accuracies().mean()

    def precision(self):
        return (self.label[self.pred.round() == 1]).mean()

    def recall(self):
        return (self.label[self.label == 1].round()).mean()

    def tpr(self):
        """
        True positive rate
        """
        return np.mean(np.logical_and(self.pred.round() == 1, self.label == 1))

    def fpr(self):
        """
        False positive rate
        """
        return np.mean(np.logical_and(self.pred.round() == 1, self.label == 0))

    def tnr(self):
        """
        True negative rate
        """
        return np.mean(np.logical_and(self.pred.round() == 0, self.label == 0))

    def fnr(self):
        """
        False negative rate
        """
        return np.mean(np.logical_and(self.pred.round() == 0, self.label == 1))

    def fn_cost(self):
        """
        Generalized false negative cost
        """
        return 1 - self.pred[self.label == 1].mean()

    def fp_cost(self):
        """
        Generalized false positive cost
        """
        return self.pred[self.label == 0].mean()

    def accuracies(self):
        return self.pred.round() == self.label

    def eq_odds(self, othr, mix_rates=None):
        has_mix_rates = not (mix_rates is None)
        if not has_mix_rates:
            mix_rates = self.eq_odds_optimal_mix_rates(othr)
        sp2p, sn2p, op2p, on2p = tuple(mix_rates)

        self_fair_pred = self.pred.copy()
        self_pp_indices, = np.nonzero(self.pred.round())
        self_pn_indices, = np.nonzero(1 - self.pred.round())
        np.random.shuffle(self_pp_indices)
        np.random.shuffle(self_pn_indices)

        n2p_indices = self_pn_indices[:int(len(self_pn_indices) * sn2p)]
        self_fair_pred[n2p_indices] = 1 - self_fair_pred[n2p_indices]
        p2n_indices = self_pp_indices[:int(len(self_pp_indices) * (1 - sp2p))]
        self_fair_pred[p2n_indices] = 1 - self_fair_pred[p2n_indices]

        othr_fair_pred = othr.pred.copy()
        othr_pp_indices, = np.nonzero(othr.pred.round())
        othr_pn_indices, = np.nonzero(1 - othr.pred.round())
        np.random.shuffle(othr_pp_indices)
        np.random.shuffle(othr_pn_indices)

        n2p_indices = othr_pn_indices[:int(len(othr_pn_indices) * on2p)]
        othr_fair_pred[n2p_indices] = 1 - othr_fair_pred[n2p_indices]
        p2n_indices = othr_pp_indices[:int(len(othr_pp_indices) * (1 - op2p))]
        othr_fair_pred[p2n_indices] = 1 - othr_fair_pred[p2n_indices]

        fair_self = Model(self_fair_pred, self.label)
        fair_othr = Model(othr_fair_pred, othr.label)

        if not has_mix_rates:
            return fair_self, fair_othr, mix_rates
        else:
            return fair_self, fair_othr

    def __repr__(self):
        return '\n'.join([
            'Accuracy:\t%.3f' % self.accuracy(),
            'F.P. cost:\t%.3f' % self.fp_cost(),
            'F.N. cost:\t%.3f' % self.fn_cost(),
            'Base rate:\t%.3f' % self.base_rate(),
            'Avg. score:\t%.3f' % self.pred.mean(),
        ])

    def compute(self):
        result_dict = perf_measure(self.label, self.pred_proba)
        result_dict['epsilon'] = np.mean(self.label_proba - self.pred_proba)
        return result_dict


if __name__ == '__main__':
    """
    To run the demo:

    ```
    python eq_odds.py <path_to_model_predictions.csv>
    ```

    `<path_to_model_predictions.csv>` should contain the following columns for the VALIDATION set:

    - `prediction` (a score between 0 and 1)
    - `label` (ground truth - either 0 or 1)
    - `group` (group assignment - either 0 or 1)

    Try the following experiments, which were performed in the paper:
    ```
    python eq_odds.py data/income.csv
    python eq_odds.py data/health.csv
    python eq_odds.py data/criminal_recidivism.csv
    ```
    """

    group_names = ['Race', 'Gender']
    df_res = []
    df_res_fair = []
    col_names = ['Accuracy', 'TPR', 'FNR', 'TNR', 'FPR']

    for comp_file in glob.glob('pred_dir/*.csv'):
        model_name = comp_file.split('/')[-1].split('.')[0]
        df = pd.read_csv(comp_file)
        unique_groups = df[group_names].drop_duplicates()
        for group in range(unique_groups.shape[0]):
            sel_rows = df[(df[group_names].values ==
                           unique_groups.iloc[group].values).all(1)]

            group_0_val_model = Model(np.array(sel_rows['expl_pred'].values > 0.5, dtype=np.int8), sel_rows[
                                      'blackbox_pred'].values)
            df_curr = pd.Series(group_0_val_model.compute()).to_frame().T
            df_curr['group'] = str(unique_groups.iloc[group].values)
            df_curr['model'] = model_name.split('_')[0]
            df_curr['expl'] = model_name.split('_')[-1]
            df_curr['sample_size'] = sel_rows.shape[0]

            df_res.append(df_curr)
    df_res = pd.concat(df_res).reset_index()
    df_res.to_csv('all_results.csv', index=False)

    # computing fairness metrics as https://arxiv.org/pdf/2106.13346.pdf
    # NB: Currently valid only for binary sensitive attributes
    for comp_file in glob.glob('pred_dir/*.csv'):
        df = pd.read_csv(comp_file)
        df_curr_fair = []
        for group in group_names:
            if df[group].nunique() == 2:
                dp, di_fp, di_fn = compute_fairness_metrics_diff(df, group)
            else:
                # currently these metrics only work for 2 groups
                dp, di_fp, di_fn = np.nan, np.nan, np.nan
            df_curr_fair.append(
                {'group': group, 'dp': [dp], 'di_fp': di_fp, 'di_fn': di_fn})
        df_curr_fair = pd.concat(df_curr_fair).reset_index()
        df_curr_fair['file'] = comp_file
        df_res_fair.append(df_curr_fair)
    pd.concat(df_res_fair).reset_index().to_csv(
        'all_fair_results.csv', index=False)
