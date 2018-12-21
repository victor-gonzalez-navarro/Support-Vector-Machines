import matplotlib.pyplot as plt
import numpy as np

from sklearn.svm import SVC

class SVM_Algorithm():

    trn_data = None
    trn_labels = None
    tst_labels = None
    dic_feats = {}

    def __init__(self, C=1, kernel='rbf', decision_function='ovo'):
        self.C = C
        self.kernel_fcn = kernel
        self.decision_function = decision_function

    def algorithm(self, trn_data, trn_labels, tst_data, tst_labels):
        acc = self.main_function(trn_data, trn_labels, tst_data, tst_labels, self.C, self.kernel_fcn)
        return acc

    def main_function(self, X_train, y_train, X_test, y_test, C, kernel_function):
        if kernel_function != 'my_knl':
            svm = SVC(C=C, kernel=kernel_function, decision_function_shape=self.decision_function)
        else:
            svm = SVC(C=C, kernel=self.my_kernel, decision_function_shape=self.decision_function)

        svm.fit(X_train, y_train)
        scores = svm.predict(X_test)
        # Print on the console the number of correct predictions and the total of predictions
        num_correct_predict = sum([x == y for x, y in zip(y_test, scores)])
        accuracy = num_correct_predict / len(y_test)
        return accuracy

    def my_kernel(self, X, Y):
        return ((np.dot(X, Y.T))**4 +(np.dot(X, Y.T))**2)
