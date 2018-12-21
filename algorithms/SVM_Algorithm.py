import matplotlib.pyplot as plt
import numpy as np

from sklearn.svm import SVC

class SVM_Algorithm():

    trn_data = None
    trn_labels = None
    tst_labels = None
    dic_feats = {}

    def __init__(self, C=1, kernel='rbf'):
        self.C = C
        self.kernel_fcn = kernel

    def algorithm(self, trn_data, trn_labels, tst_data, tst_labels):
        acc = self.main_function(trn_data, trn_labels, tst_data, tst_labels, self.C, self.kernel_fcn)
        return acc

    def main_function(self, X_train, y_train, X_test, y_test, C, kernel_function):
        # Write here your SVM code and choose a linear kernel
        if kernel_function != 'my_knl':
            svm = SVC(C = C, kernel=kernel_function)
        else:
            svm = SVC(C=C, kernel=self.my_kernel)
        svm.fit(X_train, y_train)
        scores = svm.predict(X_test)
        # Print on the console the number of correct predictions and the total of predictions
        num_correct_predict = sum([x == y for x, y in zip(y_test, scores)])
        accuracy = num_correct_predict / len(y_test)
        return accuracy

    def my_kernel(self, X, Y):
        return np.dot(X**2, Y.T)
