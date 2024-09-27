# All metrics should return values in terms of unfairness, so that a higher value is worse. The
# returned value should have a minimum possible value of 0 and a maximum of 1.
# This paper (page 13) has 6 definitions: https://arxiv.org/pdf/1703.09207.pdf
# This paper discusses historical definitions: https://doi.org/10.1145/3287560.3287600
import pandas as pd
from sklearn import metrics
import numpy as np
from collections import defaultdict
from statistics import mean
#from fairlearn.metrics import equalized_odds_difference

UNFAIRNESS_METRICS = ['demographic_parity', 'equalized_odds', 'positive_predictive_parity_difference','balanced_accuracy_difference']


# def demographic_parity(s,label_pre):
#     y_test_1 = []
#     y_test_0 = []
#     for i in range(len(s)):
#         if s[i] == 1:
#             y_test_1.append(label_pre[i])
#         else:
#             y_test_0.append(label_pre[i])

#     ED1 = sum(y_test_1) / len(y_test_1)
#     ED0 = sum(y_test_0) / len(y_test_0)
#     return abs(ED1-ED0)

def demographic_parity(s, label_pre):
    y_test_1 = []
    y_test_0 = []
    for i in range(len(s)):
        if s[i] == 1:
            y_test_1.append(label_pre[i])
        else:
            y_test_0.append(label_pre[i])

    if not y_test_1 or not y_test_0:
        return 0.0  # Return 0 if any of the lists is empty
    
    ED1 = sum(y_test_1) / len(y_test_1)
    ED0 = sum(y_test_0) / len(y_test_0)
    return abs(ED1 - ED0)



def positive_predictive_parity_difference(s,label_pre,label):
    y_test_1 = []
    y_test_0 = []
    for i in range(len(s)):
        if label[i] == 1:
            if s[i] == 1:
                y_test_1.append(label_pre[i])
            else:
                y_test_0.append(label_pre[i])

    ED1 = sum(y_test_1) / len(y_test_1)
    ED0 = sum(y_test_0) / len(y_test_0)
    return abs(ED1-ED0)

def equalized_odds(s,label_pre,label):
    tpy_test_1 = []
    tpy_test_0 = []
    fpy_test_1 = []
    fpy_test_0 = []
    for i in range(len(s)):
        if label[i] == 1:
            if s[i] == 1:
                tpy_test_1.append(label_pre[i])
            else:
                tpy_test_0.append(label_pre[i])
        else:
            if s[i] == 1:
                fpy_test_1.append(label_pre[i])
            else:
                fpy_test_0.append(label_pre[i])


    TP_ED1 = sum(tpy_test_1) / len(tpy_test_1)
    TP_ED0 = sum(tpy_test_0) / len(tpy_test_0)
    TP = abs(TP_ED1-TP_ED0)

    FP_ED1 = sum(fpy_test_1) / len(fpy_test_1)
    FP_ED0 = sum(fpy_test_0) / len(fpy_test_0)
    FP = abs(FP_ED1 - FP_ED0)

    return max(TP,FP)

def balanced_accuracy_difference(s,label_pre,label):
    tpy_test_1 = []
    tpy_test_0 = []
    tny_test_1 = []
    tny_test_0 = []
    for i in range(len(s)):
        if label[i] == 1:
            if s[i] == 1:
                tpy_test_1.append(label_pre[i])
            else:
                tpy_test_0.append(label_pre[i])
        else:
            if s[i] == 1:
                tny_test_1.append(label_pre[i])
            else:
                tny_test_0.append(label_pre[i])

    TP_ED1 = sum(tpy_test_1) / len(tpy_test_1)
    TP_ED0 = sum(tpy_test_0) / len(tpy_test_0)
    TP = abs(TP_ED1 - TP_ED0)

    TN_ED1 = 1-sum(tny_test_1) / len(tny_test_1)
    TN_ED0 = 1-sum(tny_test_0) / len(tny_test_0)
    TN = abs(TN_ED1 - TN_ED0)

    return (TP+TN)/2


def calc_fairness(s,label_pre,label,unfairness_metric):
    """
    Args:
        Data: test set
        S_test: sensitive feature in Data
        Test_pred: predictive label in Data
        Y_test: True classification labels in Data

    Returns:
        float: Unfairness measure in [0, 1], where 0 means perfect fairness
    """
    measurement = None  # Set a default value
    if unfairness_metric == 'demographic_parity':
        measurement = demographic_parity(s,label_pre)
    elif unfairness_metric == 'equalized_odds':
        measurement = equalized_odds( s, label_pre,label)
    elif unfairness_metric == 'positive_predictive_parity_difference':
        measurement = positive_predictive_parity_difference( s, label_pre, label)
    elif unfairness_metric == 'balanced_accuracy_difference':
        measurement = balanced_accuracy_difference( s, label_pre,label)

    return measurement


