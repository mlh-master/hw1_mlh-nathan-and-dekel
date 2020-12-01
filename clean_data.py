# -*- coding: utf-8 -*-
"""
Created on Sun Jul 21 17:14:23 2019

@author: smorandv
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def rm_ext_and_nan(CTG_features, extra_feature):
    """

    :param CTG_features: Pandas series of CTG features
    :param extra_feature: A feature to be removed
    :return: A dictionary of clean CTG called c_ctg
    """
    # ------------------ IMPLEMENT YOUR CODE HERE:------------------------------
    c_ctg = CTG_features.drop(axis=1, labels=[extra_feature]).apply(lambda x: pd.to_numeric(x, errors='coerce')).dropna()
    # --------------------------------------------------------------------------
    return c_ctg


def nan2num_samp(CTG_features, extra_feature):
    """

    :param CTG_features: Pandas series of CTG features
    :param extra_feature: A feature to be removed
    :return: A pandas dataframe of the dictionary c_cdf containing the "clean" features
    """
    c_cdf = {}
    # ------------------ IMPLEMENT YOUR CODE HERE:------------------------------
    c_cdf = CTG_features.drop(axis=1, labels=[extra_feature]).apply((lambda x: pd.to_numeric(x, errors='coerce')))
    for feature in c_cdf:
        summ = c_cdf.loc[:, feature].value_counts().sort_index().cumsum()
        cdf = summ/(summ.max())
        idx = np.where(c_cdf.loc[:, feature].isna())[0]
        s = np.random.uniform(0, 1, idx.size)
        i = 0
        while i < idx.size:
            nanval = (s[i]-cdf).abs().idxmin()
            c_cdf.loc[idx[i] + 1, feature] = nanval
            i += 1
    # -------------------------------------------------------------------------
    return pd.DataFrame(c_cdf)


def sum_stat(c_feat):
    """

    :param c_feat: Output of nan2num_cdf
    :return: Summary statistics as a dictionary of dictionaries (called d_summary) as explained in the notebook
    """
    # ------------------ IMPLEMENT YOUR CODE HERE:------------------------------
    d_summary = {}
    for feat in c_feat:
        d_summary[feat] = c_feat.describe()[feat].to_dict()
    # -------------------------------------------------------------------------
    return d_summary


def rm_outlier(c_feat, d_summary):
    """

    :param c_feat: Output of nan2num_cdf
    :param d_summary: Output of sum_stat
    :return: Dataframe of the dictionary c_no_outlier containing the feature with the outliers removed
    """
    c_no_outlier = {}
    # ------------------ IMPLEMENT YOUR CODE HERE:------------------------------
    temp = {}
    for feat in c_feat:
        temp[feat] = {}
        IQ = d_summary[feat]['75%'] - d_summary[feat]['25%']
        LF = d_summary[feat]['25%'] - (3/2)*IQ
        UF = d_summary[feat]['75%'] + (3/2)*IQ
        temptemp = c_feat.copy()[feat]
        temptemp[(temptemp < LF)] = np.nan
        temptemp[(temptemp > UF)] = np.nan
        temp[feat] = temptemp
        c_no_outlier = temp
    # -------------------------------------------------------------------------
    return pd.DataFrame(c_no_outlier)


def phys_prior(c_cdf, feature, thresh):
    """

    :param c_cdf: Output of nan2num_cdf
    :param feature: A string of your selected feature
    :param thresh: A numeric value of threshold
    :return: An array of the "filtered" feature called filt_feature
    """
    # ------------------ IMPLEMENT YOUR CODE HERE:-----------------------------
    # UC cannot be more then 60 contractions per 30 min. Min duration of contraction is 30 sec.
    # UC cannot be negative.
    temp = c_cdf[feature]
    filt_feature = temp[(temp <= thresh) & (temp >= 0)]
    # -------------------------------------------------------------------------
    return filt_feature


def norm_standard(CTG_features, selected_feat, mode='none', flag=False):
    """

    :param CTG_features: Pandas series of CTG features
    :param selected_feat: A two elements tuple of strings of the features for comparison
    :param mode: A string determining the mode according to the notebook
    :param flag: A boolean determining whether or not plot a histogram
    :return: Dataframe of the normalized/standardazied features called nsd_res
    """
    # ------------------ IMPLEMENT YOUR CODE HERE:------------------------------
    n_bins = 100
    standard = {}
    nsd_res = {}
    if mode == 'none':
        nsd_res = CTG_features.copy()
    elif mode == 'MinMax':
        for feature in CTG_features:
            nsd_res[feature] = {}
            standard[feature] = {}
            standard[feature]['min']= CTG_features.loc[:, feature].min()
            standard[feature]['max']= CTG_features.loc[:, feature].max()
            nsd_res[feature] = (CTG_features.loc[:, feature] - standard[feature]['min']).div(standard[feature]['max']-standard[feature]['min'])
    elif mode == 'mean':
        for feature in CTG_features:
            nsd_res[feature] = {}
            standard[feature] = {}
            standard[feature]['mean']= CTG_features.loc[:, feature].mean()
            standard[feature]['min']= CTG_features.loc[:, feature].min()
            standard[feature]['max']= CTG_features.loc[:, feature].max()
            nsd_res[feature] = (CTG_features.loc[:, feature] - standard[feature]['mean']).div(standard[feature]['max']-standard[feature]['min'])
    elif mode == 'standard':
        for feature in CTG_features:
            nsd_res[feature] = {}
            standard[feature] = {}
            standard[feature]['mean']= CTG_features.loc[:, feature].mean()
            standard[feature]['std']= CTG_features.loc[:, feature].std()
            nsd_res[feature] = (CTG_features.loc[:, feature] - standard[feature]['mean']).div(standard[feature]['std'])

    if flag == True:
        x, y = selected_feat
        plt.figure()
        plt.hist(nsd_res[x], bins=n_bins)
        plt.hist(nsd_res[y], bins=n_bins)
        plt.legend([x, y])
        plt.title([mode])

    # -------------------------------------------------------------------------
    return pd.DataFrame(nsd_res)
