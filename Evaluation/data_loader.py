import numpy as np
import pandas as pd
import typing

from sklearn.svm import SVC

from sklearn.model_selection import train_test_split

def get_celebA_data():
    data = pd.read_csv('/root/CFDM/CFDM_celebAHQ_pre_Chubby_vgg16.csv')
    label = data['Chubby'].values
    label_pre = data['pre_Chubby'].values
    #s = data['Male'].values
    #s = data['Big_Nose'].values
    #s = data['Big_Nose'].values
    #s = data['Mouth_Slightly_Open'].values
    s = data['Young'].values
    #s = data['Mouth_Slightly_Open'].values

    return {
        'label': label,
        'label_pre': label_pre,
        's': s
    }



get_celebA_data()
