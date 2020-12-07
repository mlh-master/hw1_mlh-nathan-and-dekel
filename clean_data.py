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
    c_ctg={x: pd.to_numeric(CTG_features[x],errors="coerce").dropna() for x in CTG_features.drop(columns=[extra_feature])}

    return c_ctg



def nan2num_samp(CTG_features, extra_feature):
    """

    :param CTG_features: Pandas series of CTG features
    :param extra_feature: A feature to be removed
    :return: A pandas dataframe of the dictionary c_cdf containing the "clean" features
    """
    # ------------------ IMPLEMENT YOUR CODE HERE:------------------------------
    c_cdf={}

    for x in CTG_features.drop(columns=[extra_feature]):
        c_cdf[x]=pd.to_numeric(CTG_features[x],errors="coerce")
        forprob=c_cdf[x].dropna()
        #c_cdf[x]=c_cdf[x].fillna(np.random.choice(forprob.unique(),p=list(forprob.value_counts(normalize=True))))
        c_cdf[x] = c_cdf[x].apply(lambda x: np.random.choice(forprob) if (np.isnan(x)) else x)

    # -------------------------------------------------------------------------
    return pd.DataFrame(c_cdf)


def sum_stat(c_feat):
    """

    :param c_feat: Output of nan2num_cdf
    :return: Summary statistics as a dicionary of dictionaries (called d_summary) as explained in the notebook
    """
    # ------------------ IMPLEMENT YOUR CODE HERE:------------------------------
    d_summary={x:c_feat[x].describe()[3:].to_dict() for x in c_feat}
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
    for x in c_feat:
        dx=d_summary[x]
        outlierup = dx['75%']+1.5*(dx['75%']-dx['25%'])
        outlierdown = dx['25%']-1.5*(dx['75%']-dx['25%'])
        c_no_outlier[x] = c_feat[x].apply(lambda x: np.nan if ((x>outlierup) or (x<outlierdown)) else x)
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
    filt_feature=c_cdf[feature].loc[lambda s: s < thresh].to_numpy()
    # -------------------------------------------------------------------------
    return filt_feature


def norm_standard(CTG_features, selected_feat=('LB', 'ASTV'), mode='none', flag=False):
    """

    :param CTG_features: Pandas series of CTG features
    :param selected_feat: A two elements tuple of strings of the features for comparison
    :param mode: A string determining the mode according to the notebook
    :param flag: A boolean determining whether or not plot a histogram
    :return: Dataframe of the normalized/standardazied features called nsd_res
    """

    # ------------------ IMPLEMENT YOUR CODE HERE:------------------------------
    nsd_res=CTG_features.copy()
    if mode!='none':
        for columns in nsd_res:
            min, max=nsd_res[columns].min(),nsd_res[columns].max()
            mean,std=nsd_res[columns].mean(),nsd_res[columns].std()
            if mode=='mean':
                nsd_res[columns]=nsd_res[columns].apply(lambda x: (x-mean)/(max-min))
            elif mode=='MinMax':
                nsd_res[columns]=nsd_res[columns].apply(lambda x: (x - min) / (max - min))
            else:
                nsd_res[columns]=nsd_res[columns].apply(lambda x: (x - mean) /std)

    if flag==True:
        x, y = selected_feat
        n_bins = 100
        if mode == 'none':
            xlbl = ['%','beats/min']
            axarr = nsd_res.hist(column=[x,y], bins=n_bins, layout=(1, 2), figsize=(10, 5))
            for i, ax in enumerate(axarr.flatten()):
                ax.set_xlabel(xlbl[i])
                ax.set_ylabel("Count")
        else:
            plt.hist(nsd_res[x], bins=n_bins)
            plt.hist(nsd_res[y], bins=n_bins)
            plt.legend([x, y])
            plt.ylabel("Count")
            plt.title([mode])

        plt.show()


    # -------------------------------------------------------------------------
    return nsd_res
