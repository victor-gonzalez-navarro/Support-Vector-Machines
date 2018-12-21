import pandas as pd
import numpy as np


def read_keyboard():
    print('\n' + '\033[1m' + 'Which C value do you want to use?' + '\033[0m')
    C = int(input('Insert a number between 1-20: '))
    print('\n' + '\033[1m' + 'Which kernel function do you want to use?' + '\033[0m' + '\n1: Linear\n2: Radial Basis '
                                'Function\n3: Plynomial\n4: Custom kernel')
    dist = int(input('Insert a number between 1-4: '))
    if dist == 1:
        kernel = 'linear'
    elif dist == 2:
        kernel = 'rbf'
    elif dist == 3:
        kernel = 'poly'
    elif dist == 4:
        kernel = 'my_knl'

    return(C, kernel)


def trn_tst_idxs(ref_data_dic, dataset):
    trn_tst_dic = {}

    for key, fold_data in dataset.items():

        foldata_trn = pd.DataFrame(fold_data[0])
        foldata_trn = foldata_trn.fillna('nonna')
        foldata_trn = foldata_trn.values

        foldata_tst = pd.DataFrame(fold_data[1])
        foldata_tst = foldata_tst.fillna('nonna')
        foldata_tst = foldata_tst.values

        trn_idxs = [ref_data_dic[str(sample)] for sample in foldata_trn]
        tst_idxs = [ref_data_dic[str(sample)] for sample in foldata_tst]
        trn_tst_dic[key] = []
        trn_tst_dic[key].append(trn_idxs)
        trn_tst_dic[key].append(tst_idxs)
    return trn_tst_dic