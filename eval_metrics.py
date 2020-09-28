#!/usr/bin/env python
# coding=utf-8
# **********************************************************************
# Created Time: 2019-05-06 16:34:10
# File Name   : evaluation_metrics.py
# Author      : Siyuan Dong
# Mail        : dongsiyuan@dip-ai.com
# Description :
# **********************************************************************

import copy
import numpy as np
from scipy import stats
from sklearn import preprocessing, metrics
from sklearn.metrics import r2_score

def r_square(ori, pre):
    """
    Task:    To compute the R-square value
    Input:   ori    Vector with original labels
             pre    Vector with predicted labels
    Output:  r2   R-square value
    """
    r2 = r2_score(ori, pre)
    return r2

def rmse(ori, pre):
    """
    Task:    To compute root mean squared error (RMSE)
    Input:   ori    Vector with original labels
             pre    Vector with predicted labels
    Output:  rmse_value   RSME
    """
    rmse_value = np.sqrt(((ori - pre)**2).mean(axis=0))
    return rmse_value


def pearson(ori, pre):
    """
    Task:    To compute Pearson correlation coefficient
    Input:   ori      Vector with original labels
             pre      Vector with predicted labels
    Output:  pearson_value  Pearson correlation coefficient
    """
    pearson_value = np.corrcoef(ori, pre)[0, 1]
    return pearson_value


def spearman(ori, pre):
    """
    Task:    To compute Spearman's rank correlation coefficient
    Input:   ori      Vector with original labels
             pre      Vector with predicted labels
    Output:  spearman_value     Spearman's rank correlation coefficient
    """
    spearman_value = stats.spearmanr(ori, pre)[0]
    return spearman_value


def find_ci(ori, pre):
    """
    Task:    To compute concordance index (CI)
    Input:   ori      Vector with original labels
             pre      Vector with predicted labels
    Output:  ci_value     CI
    """
    ind = np.argsort(ori)
    ori = ori[ind]
    pre = pre[ind]
    i = len(ori)-1
    j = i-1
    z_value = 0
    s_value = 0
    while i > 0:
        while j >= 0:
            if ori[i] > ori[j]:
                z_value = z_value + 1
                u_value = pre[i] - pre[j]
                if u_value > 0:
                    s_value = s_value + 1
                elif u_value == 0:
                    s_value = s_value + 0.5
            j = j - 1
        i = i - 1
        j = i-1
    ci_value = s_value/z_value
    return ci_value


def find_f1(ori, pre, threshold=7.0):
    """
    Task:    To compute F1 score using the threshold of (7) M
             to binarize pKd's into true class labels.
    Input:   ori      Vector with original labels
             pre      Vector with predicted labels
    Output:  f1     F1 score
    """
    ori_binary = copy.deepcopy(ori)
    ori_binary = preprocessing.binarize(np.array(ori_binary).reshape(1, -1),
                                        threshold=threshold, copy=False)[0]
    pre_binary = copy.deepcopy(pre)
    pre_binary = preprocessing.binarize(np.array(pre_binary).reshape(1, -1),
                                        threshold=threshold, copy=False)[0]
    f1_value = metrics.f1_score(ori_binary, pre_binary)
    return f1_value


def average_auc(ori, pre, lower=6, upper=8, interval=10):
    """
    Task:    To compute average area under the ROC curves (AUC) given ten
             interaction threshold values from the pKd interval [6 M, 8 M]
             to binarize pkd into true class labels.
    Input:   ori      Vector with original labels
             pre      Vector with predicted labels
    Output: av_auc   average AUC


    Important thing:
        When plug in the data, the original data should be included in all part of the interval.
        For instance, if lower bound = 6, upper bound = 8 and interval = 2.
        The data should in four interval [., 6], [6, 7], [7, 8], [8, .].
        Otherwise, the function of average_auc will return 'nan'.
    """
    thr = np.linspace(lower, upper, interval)
    auc = np.empty(np.shape(thr))
    auc[:] = np.nan
    for i, thr_v in enumerate(thr):
        ori_binary = copy.deepcopy(ori)
        ori_binary = preprocessing.binarize(np.array(ori_binary).reshape(1, -1),
                                            threshold=thr_v, copy=False)[0]
        fpr, tpr, _ = metrics.roc_curve(ori_binary, pre, pos_label=1)
        auc[i] = metrics.auc(fpr, tpr)
    av_auc = np.mean(auc)
    return av_auc

def get_confusion_matrix(ori, pre, threshold):
    """
    compute the confusion matrix
    return: matrix in a list 
    """
    tp, fp, fn, tn = 0., 0., 0., 0.
    for i in range(len(ori)):
        if ori[i] >= threshold and pre[i] >= threshold:
            tp += 1
        elif ori[i] < threshold and pre[i] >= threshold:
            fp += 1
        elif ori[i] >= threshold and pre[i] < threshold:
            fn += 1
        else:
            tn += 1
    presision = tp / (tp + fp)
    recall = tp / (tp + fn)
    acc = (tp + tn) / (tp+fp+fn+tn)
    sp = tn / (tn + fp)
    f1 = 2 * presision * recall / (presision + recall)
    print('presision: %s' %presision)
    print('recall: %s' %recall)
    print('acc: %s' %acc)
    print('specificity: %s' %sp)
    print('f1: %s' %f1)
    print( [tp, fp, fn, tn])
    #return [tp, fp, fn, tn]

def print_all_evaluation(ori, pre, f1_threshold=7.0, auc_lower=6, auc_upper=8, auc_interval=2):
    """
    Task:   Print all evaluation values.
    """
    rmse_value = rmse(ori, pre)
    pearson_value = pearson(ori, pre)
    spearman_value = spearman(ori, pre)
    #ci_value = find_ci(ori, pre)
    f1_value = find_f1(ori, pre, f1_threshold)
    auc_value = average_auc(ori, pre, auc_lower, auc_upper, auc_interval)
    r2 = r_square(ori, pre)
    R2 = pearson_value ** 2 

    print('rmse value: %s' %rmse_value)
    print('pearson value: %s' %pearson_value)
    print('spearman value: %s' %spearman_value)
    #print('ci value: %s' %ci_value)
    print('f1 value: %s' %f1_value)
    print('average_auc value: %s' %auc_value)
    print('r-square value: %s' %r2)
    print('R2 value: %s' %R2)
    return [rmse_value, pearson_value, spearman_value, f1_value, auc_value, r2, R2]
