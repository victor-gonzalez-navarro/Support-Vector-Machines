import pandas as pd
import numpy as np


def read_keyboard():
    print('\n' + '\033[1m' + 'Which K value do you want to use?' + '\033[0m')
    k = int(input('Insert a number between 1-20: '))
    print('\n' + '\033[1m' + 'Which distance function do you want to use?' + '\033[0m' + '\n1: Euclidean\n2: Manhattan'
          '\n3: Canberra\n4: HVDM')
    dist = int(input('Insert a number between 1-4: '))
    if dist == 1:
        metric = 'euclidean'
    elif dist == 2:
        metric = 'manhattan'
    elif dist == 3:
        metric = 'canberra'
    elif dist == 4:
        metric = 'hvdm'
    print('\n' + '\033[1m' + 'Which voting policy do you want to use?' + '\033[0m' + '\n1: Most voted solution\n'
                '2: Modified Plurality\n3: Borda Count')
    voting_policy = int(input('Insert a number between 1-3: '))
    print('')
    if voting_policy == 1:
        voting_policy = 'most_voted'
    elif voting_policy == 2:
        voting_policy = 'modified_plurality'
    elif voting_policy == 3:
        voting_policy = 'borda_count'

    return(k, metric, voting_policy)


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