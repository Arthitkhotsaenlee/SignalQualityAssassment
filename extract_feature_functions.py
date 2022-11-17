import numpy as np
import pandas as pd
from scipy import stats

def Hjorth_params(data):
    def activity(x):
        return np.var(x)

    def mobility(x):
        x_first = np.diff(x, 1)
        ratio = activity(x_first) / activity(x)
        return np.sqrt(ratio)

    def complexity(x):
        x_first = np.diff(x, 1)
        return mobility(x_first) / mobility(x)

    return {"activity": activity(data),
            "mobility": mobility(data),
            "complexity": complexity(data)}


def second_power(data):
    """
    Not in used
    :param data:
    :return:
    """
    # calculation the second order power
    d = np.diff(data)
    pa = np.sqrt(np.power(d[0:-1],2)+np.power(d[1::],2))
    # score generations session
    # calculate maximum fluctuation
    fu = np.abs(np.max(pa)-min(pa))
    # calculate the mean and variance
    pm = np.mean(pa)
    vb = np.var(pa)


def statistical_features(data):
    def mean(x):
        return np.mean(x)
    def std(x):
        return np.std(x)
    def ptp(x):
        return np.ptp(x)
    def minim(x):
        return np.min(x)
    def maxim(x):
        return np.max(x)
    def rms(x):
        return np.sqrt(np.mean(np.power(x,2)))
    def abs_diff(x):
        return np.sum(np.abs(np.diff(x)))
    def skewness(x):
        return stats.skew(x)
    def kurtosis(x):
        return stats.kurtosis(x)
    return {"mean":mean(data), "std":std(data), "ptp": ptp(data), "minim": minim(data), "manim": maxim(data),
            "rms": rms(data), "abs_diff": abs_diff(data), "skewness": skewness(data), "kurtosis": kurtosis(data)}

def extract_features(epochs):
    fea_df = pd.DataFrame()
    for i in range(0, len(epochs.shape[0])):
        hdf = Hjorth_params(epochs[i, 0, :])
        sdf = statistical_features(epochs[i, 0, :])
        df_fea_new = pd.DataFrame([hdf.update(sdf)])
        fea_df = pd.concat([fea_df, df_fea_new],axis=0, ignore_index=True)
    return fea_df



