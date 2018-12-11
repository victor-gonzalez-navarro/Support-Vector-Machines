﻿#############################################################
#############################################################
#############################################################


import numpy as np
import cvxopt
import cvxopt.solvers
from sklearn.svm import SVC
import matplotlib.pyplot as plt


if __name__ == "__main__":
    import pylab as pl

    def generate_data_set1():
        # generate training data in the 2-d case
        mean1 = np.array([0, 2])
        mean2 = np.array([2, 0])
        cov = np.array([[0.8, 0.6], [0.6, 0.8]])
        X1 = np.random.multivariate_normal(mean1, cov, 100)
        y1 = np.ones(len(X1))
        X2 = np.random.multivariate_normal(mean2, cov, 100)
        y2 = np.ones(len(X2)) * -1
        return X1, y1, X2, y2

    def generate_data_set2():
        mean1 = [-1, 2]
        mean2 = [1, -1]
        mean3 = [4, -4]
        mean4 = [-4, 4]
        cov = [[1.0,0.8], [0.8, 1.0]]
        X1 = np.random.multivariate_normal(mean1, cov, 50)
        X1 = np.vstack((X1, np.random.multivariate_normal(mean3, cov, 50)))
        y1 = np.ones(len(X1))
        X2 = np.random.multivariate_normal(mean2, cov, 50)
        X2 = np.vstack((X2, np.random.multivariate_normal(mean4, cov, 50)))
        y2 = np.ones(len(X2)) * -1
        return X1, y1, X2, y2

    def generate_data_set3():
        # generate training data in the 2-d case
        mean1 = np.array([0, 2])
        mean2 = np.array([2, 0])
        cov = np.array([[1.5, 1.0], [1.0, 1.5]])
        X1 = np.random.multivariate_normal(mean1, cov, 100)
        y1 = np.ones(len(X1))
        X2 = np.random.multivariate_normal(mean2, cov, 100)
        y2 = np.ones(len(X2)) * -1
        return X1, y1, X2, y2

    def split_train(X1, y1, X2, y2):
        X1_train = X1[:90]
        y1_train = y1[:90]
        X2_train = X2[:90]
        y2_train = y2[:90]
        X_train = np.vstack((X1_train, X2_train))
        y_train = np.hstack((y1_train, y2_train))
        return X_train, y_train

    def split_test(X1, y1, X2, y2):
        X1_test = X1[90:]
        y1_test = y1[90:]
        X2_test = X2[90:]
        y2_test = y2[90:]
        X_test = np.vstack((X1_test, X2_test))
        y_test = np.hstack((y1_test, y2_test))
        return X_test, y_test

    def run_svm_dataset1():
        X1, y1, X2, y2 = generate_data_set1()
        X_train, y_train = split_train(X1, y1, X2, y2)
        X_test, y_test = split_test(X1, y1, X2, y2)

         #### 
        # Write here your SVM code and choose a linear kernel
        svm = SVC(kernel='linear')
        svm.fit(X_train, y_train)

        plt.plot(svm.support_vectors_, linestyle='None', marker='.')
        plt.show()

        scores = svm.predict(X_test)

        accuracy = sum([x == y for x, y in zip(y_test, scores)]) / len(y_test)
        print(accuracy)

        idx1 = y_train == -1
        idx2 = y_train == 1
        plt.plot(X_train[idx1, :], y_train[idx1], c='r', linestyle='None', marker='.')
        plt.plot(X_train[idx2, :], y_train[idx2], c='b', linestyle='None', marker='.')
        plt.show()

        idx1 = y_test == -1
        idx2 = y_test == 1
        plt.figure()
        plt.plot(X_test[idx1, :], y_test[idx1], c='r', linestyle='None', marker='.')
        plt.plot(X_test[idx2, :], y_test[idx2], c='b', linestyle='None', marker='.')
        plt.show()

        plt.figure()
        plt.plot(svm.support_vectors_, linestyle='None', marker='.')
        plt.show()

        # plot the graph with the support_vectors_
        # print on the console the number of correct predictions and the total of predictions
        ####


    def run_svm_dataset2():
        X1, y1, X2, y2 = generate_data_set2()
        X_train, y_train = split_train(X1, y1, X2, y2)
        X_test, y_test = split_test(X1, y1, X2, y2)

        #### 
        # Write here your SVM code and choose a linear kernel with the best C pparameter
        # plot the graph with the support_vectors_
        # print on the console the number of correct predictions and the total of predictions
        ####



    def run_svm_dataset3():
        X1, y1, X2, y2 = generate_data_set3()
        X_train, y_train = split_train(X1, y1, X2, y2)
        X_test, y_test = split_test(X1, y1, X2, y2)

        #### 
        # Write here your SVM code and use a gaussian kernel 
        # plot the graph with the support_vectors_
        # print on the console the number of correct predictions and the total of predictions
        ####



#############################################################
#############################################################
#############################################################

# EXECUTE SVM with THIS DATASETS      
    run_svm_dataset1()   # data distribution 1
    run_svm_dataset2()   # data distribution 2
    run_svm_dataset3()   # data distribution 3

#############################################################
#############################################################
#############################################################


#import numpy as np
#import matplotlib.pyplot as plt
#from sklearn import svm, datasets
#
#
#def make_meshgrid(x, y, h=.02):
#   """Create a mesh of points to plot in
#
#   Parameters
#   ----------
#   x: data to base x-axis meshgrid on
#   y: data to base y-axis meshgrid on
#   h: stepsize for meshgrid, optional
#
#   Returns
#   -------
#   xx, yy : ndarray
#   """
#   x_min, x_max = x.min() - 1, x.max() + 1
#   y_min, y_max = y.min() - 1, y.max() + 1
#   xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
#                        np.arange(y_min, y_max, h))
#   return xx, yy
#
#
#def plot_contours(ax, clf, xx, yy, **params):
#   """Plot the decision boundaries for a classifier.
#
#   Parameters
#   ----------
#   ax: matplotlib axes object
#   clf: a classifier
#   xx: meshgrid ndarray
#   yy: meshgrid ndarray
#   params: dictionary of params to pass to contourf, optional
#   """
#   Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
#   Z = Z.reshape(xx.shape)
#   out = ax.contourf(xx, yy, Z, **params)
#   return out
#
#
## import some data to play with
#iris = datasets.load_iris()
## Take the first two features. We could avoid this by using a two-dim dataset
#X = iris.data[:, :2]
#y = iris.target
#
## we create an instance of SVM and fit out data. We do not scale our
## data since we want to plot the support vectors
#C = 1.0  # SVM regularization parameter
#models = (svm.SVC(kernel='linear', C=C),
#         svm.LinearSVC(C=C),
#         svm.SVC(kernel='rbf', gamma=0.7, C=C),
#         svm.SVC(kernel='poly', degree=3, C=C))
#models = (clf.fit(X, y) for clf in models)
#
## title for the plots
#titles = ('SVC with linear kernel',
#         'LinearSVC (linear kernel)',
#         'SVC with RBF kernel',
#         'SVC with polynomial (degree 3) kernel')
#
## Set-up 2x2 grid for plotting.
#fig, sub = plt.subplots(2, 2)
#plt.subplots_adjust(wspace=0.4, hspace=0.4)
#
#X0, X1 = X[:, 0], X[:, 1]
#xx, yy = make_meshgrid(X0, X1)
#
#for clf, title, ax in zip(models, titles, sub.flatten()):
#   plot_contours(ax, clf, xx, yy,
#                 cmap=plt.cm.coolwarm, alpha=0.8)
#   ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
#   ax.set_xlim(xx.min(), xx.max())
#   ax.set_ylim(yy.min(), yy.max())
#   ax.set_xlabel('Sepal length')
#   ax.set_ylabel('Sepal width')
#   ax.set_xticks(())
#   ax.set_yticks(())
#   ax.set_title(title)
#
#plt.show()
#