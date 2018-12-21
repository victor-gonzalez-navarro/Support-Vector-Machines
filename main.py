import os
import re
import time

from scipy.io import arff

from preproc.preprocess import Preprocess
from algorithms.auxiliary_methods import *
from algorithms.SVM_Algorithm import SVM_Algorithm
from sklearn.preprocessing.label import LabelEncoder


# -------------------------------------------------------------------------------------------------------- Read datasets
def obtain_arffs(path):
    # Read all the datasets
    processed = []
    arffs_dic = {}

    for folder in os.listdir(path):
        folds_dic = {}
        for filename in os.listdir(path + folder + '/'):
            if re.match('(.*).fold.(\d*).(train|test).arff', filename) and filename not in processed:
                row = int(re.sub('(\w*).(\d*).(\w*)', r'\2', filename))
                trn_file = re.sub('(train|test)', 'train', filename)
                tst_file = re.sub('(train|test)', 'test', filename)
                folds_dic[row] = []
                folds_dic[row].append(arff.loadarff(path + folder + '/' + trn_file)[0])
                folds_dic[row].append(arff.loadarff(path + folder + '/' + tst_file)[0])
                processed.append(trn_file)
                processed.append(tst_file)
        arffs_dic[folder] = folds_dic
    return arffs_dic


# ----------------------------------------------------------------------------------------------------------------- Main
def main():
    print('\033[1m' + 'Loading all the datasets...' +'\033[0m')
    arffs_dic = obtain_arffs('./datasetsSelected/')

    # Extract an specific database
    dataset_name = 'sick' #sick # nursery
    dataset = arffs_dic[dataset_name]

    # ------------------------------------------------------------------------------------ Compute indices for each fold
    # Use folder 0 of that particular dataset to find indices of train and test for each fold
    ref_data = np.concatenate((dataset[0][0], dataset[0][1]), axis=0)
    df_aux = pd.DataFrame(ref_data)
    df_aux = df_aux.fillna('nonna').values
    ref_data_dic = {}
    for i in range(df_aux.shape[0]):
        ref_data_dic[str(df_aux[i, :])] = i

    trn_tst_dic = trn_tst_idxs(ref_data_dic, dataset)

    # --------------------------------------------------------------------------------- Reading parameters from keyboard
    C, kernel, decision_function = read_keyboard()

    # ------------------------------------------------------------------------------------------------------- Preprocess
    df1 = pd.DataFrame(ref_data)
    groundtruth_labels = df1[df1.columns[len(df1.columns) - 1]].values  # original labels in a numpy array
    df1 = df1.drop(df1.columns[len(df1.columns) - 1], 1)
    if dataset_name == 'sick':
        df1 = df1.drop('TBG', 1)  # This column only contains NaNs so does not add any value to the clustering

    data1 = df1.values  # original data in a numpy array without labels
    load = Preprocess()

    # ---------------------------------------------------------------------------------------- Encode groundtruth labels
    le = LabelEncoder()
    le.fit(np.unique(groundtruth_labels))
    groundtruth_labels = le.transform(groundtruth_labels)

    data_x = load.preprocess_method(data1)
    # -------------------------------------------------------------------------------------------- Supervised classifier
    # Compute accuracy for each fold
    accuracies = []
    fold_number = 0
    start_time = time.time()
    for trn_idxs, tst_idxs in trn_tst_dic.values():
        fold_number = fold_number + 1
        print('Computing accuracy for fold number '+str(fold_number))
        trn_data = data_x[trn_idxs]
        trn_labels = groundtruth_labels[trn_idxs]
        tst_data = data_x[tst_idxs]
        tst_labels = groundtruth_labels[tst_idxs]

        svecm = SVM_Algorithm(C, kernel, decision_function)
        acc = svecm.algorithm(trn_data, trn_labels, tst_data, tst_labels)
        accuracies.append(acc)

    mean_accuracies = str(round(np.mean(accuracies), 3))
    std_accuracies = str(round(np.std(accuracies), 2))
    print('\n\033[1m'+'The mean accuracy of classification in the test set is: ' + mean_accuracies + ' ± ' +
          std_accuracies+'\033[0m')
    print('\033[1mRunning time for the 10 folds: %s seconds\033[0m' % round(time.time() - start_time, 4))


# ----------------------------------------------------------------------------------------------------------------- Init
if __name__ == '__main__':
    main()
