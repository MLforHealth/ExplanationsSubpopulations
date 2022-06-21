# Source: LAFTR codebase
import numpy as np
import pandas as pd
import glob

eps = 1e-12


def pos(Y):
    return np.sum(np.round(Y)).astype(np.float32)


def neg(Y):
    return np.sum(np.logical_not(np.round(Y))).astype(np.float32)


def PR(Y):
    return pos(Y) / (pos(Y) + neg(Y))


def NR(Y):
    return neg(Y) / (pos(Y) + neg(Y))


def TP(Y, Ypred):
    return np.sum(np.multiply(Y, np.round(Ypred))).astype(np.float32)


def FP(Y, Ypred):
    return np.sum(np.multiply(np.logical_not(Y), np.round(Ypred))).astype(np.float32)


def TN(Y, Ypred):
    return np.sum(np.multiply(np.logical_not(Y), np.logical_not(np.round(Ypred)))).astype(np.float32)


def FN(Y, Ypred):
    return np.sum(np.multiply(Y, np.logical_not(np.round(Ypred)))).astype(np.float32)


def TPR(Y, Ypred):
    return TP(Y, Ypred) / pos(Y)


def FPR(Y, Ypred):
    return FP(Y, Ypred) / neg(Y)


def TNR(Y, Ypred):
    return TN(Y, Ypred) / neg(Y)


def FNR(Y, Ypred):
    return FN(Y, Ypred) / pos(Y)


def calibPosRate(Y, Ypred):
    return TP(Y, Ypred) / pos(Ypred)


def calibNegRate(Y, Ypred):
    return TN(Y, Ypred) / neg(Ypred)


def errRate(Y, Ypred):
    return (FP(Y, Ypred) + FN(Y, Ypred)) / float(Y.shape[0])


def accuracy(Y, Ypred):
    return 1 - errRate(Y, Ypred)


def subgroup(fn, mask, Y, Ypred=None):
    m = np.greater(mask, 0.5).flatten()
    Yf = Y.flatten()
    if not Ypred is None:  # two-argument functions
        Ypredf = Ypred.flatten()
        return fn(Yf[m], Ypredf[m])
    else:  # one-argument functions
        return fn(Yf[m])


def DI_FP(Y, Ypred, A):
    fpr1 = subgroup(FPR, A, Y, Ypred)
    fpr0 = subgroup(FPR, 1 - A, Y, Ypred)
    return abs(fpr1 - fpr0)


def DI_FN(Y, Ypred, A):
    fnr1 = subgroup(FNR, A, Y, Ypred)
    fnr0 = subgroup(FNR, 1 - A, Y, Ypred)
    return abs(fnr1 - fnr0)


def DI(Y, Ypred, A):
    return (DI_FN(Y, Ypred, A) + DI_FP(Y, Ypred, A)) * 0.5


def DP(Ypred, A):  # demographic disparity
    return abs(subgroup(PR, A, Ypred) - subgroup(PR, 1 - A, Ypred))


def compute_fairness_metrics_diff(df, sensitive_attribute='Gender'):
    """df contains sensitive attribute and all predictions+true label"""

    # demographic parity difference between blackbox and expl model
    dp_diff = DP(df['blackbox_pred'],
                 df[sensitive_attribute].values) - \
        DP(df['expl_pred'], df[sensitive_attribute].values)

    # parity metrics
    di_fp_diff = DI_FP(df['groundtruth'],
                       df['blackbox_pred'],
                       df[sensitive_attribute].values) - \
        DI_FP(df['groundtruth'],
              df['expl_pred'],
              df[sensitive_attribute].values)

    di_fn_diff = DI_FN(df['groundtruth'],
                       df['blackbox_pred'],
                       df[sensitive_attribute].values) - \
        DI_FN(df['groundtruth'],
              df['expl_pred'],
              df[sensitive_attribute].values)

    return dp_diff, di_fp_diff, di_fn_diff
