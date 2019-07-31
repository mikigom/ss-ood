# Reference
# https://github.com/ShiyuLiang/odin-pytorch
# Thanks for @ShiyuLiang

# -*- coding: utf-8 -*-
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""
Created on Sat Sep 19 20:55:56 2015
@author: liangshiyu

Modified on April 30 2018
@author: Junghoon Seo
"""

from __future__ import print_function

import os
import numpy as np

from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve, auc, precision_recall_curve


def read_score_file(file_normal_score, file_anomaly_score):
    normal = np.loadtxt(file_normal_score, delimiter=',')
    anomaly = np.loadtxt(file_anomaly_score, delimiter=',')
    normal_score = normal[:, 0]
    anomaly_score = anomaly[:, 0]
    normal_time = normal[:, 1]
    anomaly_time = anomaly[:, 1]
    start = np.min([np.min(normal_score), np.min(anomaly_score)])
    end = np.max([np.max(normal_score), np.max(anomaly_score)])
    gap = (end - start) / 100000
    return normal_score, anomaly_score, normal_time, anomaly_time, start, end, gap


def tpr95(normal_score, anomaly_score, start, end, gap, return_threshold_at_tpr95):
    # calculate the false-positive error when tpr is 95%
    total = 0.0
    fpr = 0.0
    threshold_at_tpr95 = None
    for delta in np.arange(start, end, gap):
        tpr = np.sum(np.sum(normal_score >= delta)) / np.float(len(normal_score))
        error2 = np.sum(np.sum(anomaly_score > delta)) / np.float(len(anomaly_score))
        if tpr <= 0.952 and tpr >= 0.948:
            fpr += error2
            total += 1
        else:
            threshold_at_tpr95 = delta
    fprBase = fpr / total

    if return_threshold_at_tpr95 is True:
        return fprBase, threshold_at_tpr95
    else:
        return fprBase


def auroc(normal_score, anomaly_score, start=None, end=None, gap=None):
    # calculate the AUROC
    """
    aurocBase = 0.0
    fprTemp = 1.0
    for delta in np.arange(start, end, gap):
        tpr = np.sum(np.sum(normal_score >= delta)) / np.float(len(normal_score))
        fpr = np.sum(np.sum(anomaly_score > delta)) / np.float(len(anomaly_score))
        aurocBase += (-fpr + fprTemp) * tpr
        fprTemp = fpr
    aurocBase += fpr * tpr
    """

    truth = np.concatenate((np.zeros_like(anomaly_score), np.ones_like(normal_score)))
    preds = np.concatenate((anomaly_score, normal_score))
    fpr, tpr, roc_thresholds = roc_curve(truth, preds)
    aurocBase = auc(fpr, tpr)

    return aurocBase


def auprIn(normal_score, anomaly_score, start=None, end=None, gap=None):
    # calculate the AUPR-In
    """
    precisionVec = []
    recallVec = []

    auprBase = 0.0
    recallTemp = 1.0
    for delta in np.arange(start, end, gap):
        tp = np.sum(np.sum(normal_score >= delta)) / np.float(len(normal_score))
        fp = np.sum(np.sum(anomaly_score >= delta)) / np.float(len(anomaly_score))
        if tp + fp == 0: continue
        precision = tp / (tp + fp)
        recall = tp
        precisionVec.append(precision)
        recallVec.append(recall)
        auprBase += (recallTemp - recall) * precision
        recallTemp = recall
    auprBase += recall * precision
    """
    # pr curve where "normal" is the positive class
    truth = np.concatenate((np.zeros_like(anomaly_score), np.ones_like(normal_score)))
    preds = np.concatenate((anomaly_score, normal_score))
    precision_norm, recall_norm, pr_thresholds_norm = precision_recall_curve(truth, preds)
    auprBase = auc(recall_norm, precision_norm)

    return auprBase


def auprOut(normal_score, anomaly_score, start, end, gap):
    # calculate the AUPR-Out
    """
    auprBase = 0.0
    recallTemp = 1.0
    for delta in np.arange(end, start, -gap):
        fp = np.sum(np.sum(normal_score < delta)) / np.float(len(normal_score))
        tp = np.sum(np.sum(anomaly_score < delta)) / np.float(len(anomaly_score))
        if tp + fp == 0: break
        precision = tp / (tp + fp)
        recall = tp
        auprBase += (recallTemp - recall) * precision
        recallTemp = recall
    auprBase += recall * precision
    """
    truth = np.concatenate((np.zeros_like(anomaly_score), np.ones_like(normal_score)))
    preds = np.concatenate((anomaly_score, normal_score))
    # pr curve where "anomaly" is the positive class
    precision_anom, recall_anom, pr_thresholds_anom = precision_recall_curve(truth, -preds, pos_label=0)
    auprBase = auc(recall_anom, precision_anom)
    return auprBase


def detection(normal_score, anomaly_score, start, end, gap):
    # calculate the minimum detection error
    errorBase = 1.0
    for delta in np.arange(start, end, gap):
        tpr = np.sum(np.sum(normal_score < delta)) / np.float(len(normal_score))
        error2 = np.sum(np.sum(anomaly_score > delta)) / np.float(len(anomaly_score))
        errorBase = np.minimum(errorBase, (tpr + error2) / 2.0)
    return errorBase


def get_total_time(normal_time, anomaly_time):
    timeBase = np.sum(normal_time) + np.sum(anomaly_time)
    return timeBase


def metric(file_normal_score, file_anomaly_score, epoch, return_threshold_at_tpr95=True,):
    normal_score, anomaly_score, normal_time, anomaly_time, start, end, gap = \
        read_score_file(file_normal_score, file_anomaly_score)

    fprReturns = tpr95(normal_score, anomaly_score, start, end, gap, return_threshold_at_tpr95)
    if isinstance(fprReturns, tuple):
        fprBase, threshold_at_tpr95 = fprReturns
    else:
        fprBase = fprReturns
        threshold_at_tpr95 = None

    errorBase = detection(normal_score, anomaly_score, start, end, gap)
    aurocBase = auroc(normal_score, anomaly_score, start, end, gap)
    auprinBase = auprIn(normal_score, anomaly_score, start, end, gap)
    auproutBase = auprOut(normal_score, anomaly_score, start, end, gap)
    timeBase = get_total_time(normal_time, anomaly_time)

    print("Epoch: %s", epoch)
    print("")
    print("{:20}{:13.1f}%".format("FPR at TPR 95%:", fprBase * 100))
    print("{:20}{:13.1f}%".format("Detection error:", errorBase * 100))
    print("{:20}{:13.1f}%".format("AUROC:", aurocBase * 100))
    print("{:20}{:13.1f}%".format("AUPR In:", auprinBase * 100))
    print("{:20}{:13.1f}%".format("AUPR Out:", auproutBase * 100))
    print("{:20}{:13.1f}m".format("Time (min):", timeBase/60))

    metric_dic = dict()
    metric_dic['FPR@TPR95'] = fprBase
    metric_dic['Detection-error'] = errorBase
    metric_dic['AUROC'] = aurocBase
    metric_dic['AUPR-In'] = auprinBase
    metric_dic['AUPR-Out'] = auproutBase
    metric_dic['Time'] = timeBase/60
    metric_dic['Threshold@TPR95'] = threshold_at_tpr95

    return metric_dic

