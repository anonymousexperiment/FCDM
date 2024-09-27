import numpy as np
import pandas as pd
import data_loader
import Fairness_metrics
##############################################################################


def run_experiment(datasetname):
    
    if datasetname == 'CelebA':
        label = data_loader.get_celebA_data()['label']
        label_pre = data_loader.get_celebA_data()['label_pre']
        s = data_loader.get_celebA_data()['s']
       

    for unfairness_metric in Fairness_metrics.UNFAIRNESS_METRICS:
        print('Unfairness metric:', unfairness_metric)
        cal_acc_fair(s, label_pre, label, unfairness_metric)

def cal_acc_fair(s, label_pre, label, unfairness_metric):
    fair_result = Fairness_metrics.calc_fairness(s, label_pre, label, unfairness_metric)
    print("fair_result", fair_result)

run_experiment('CelebA')


















