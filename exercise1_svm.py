#############################################################
#############################################################
#############################################################


import numpy as np
import cvxopt.solvers
from sklearn.svm import SVC
import matplotlib.pyplot as plt


if __name__ == "__main__":

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

    def generate_data_set2_victor():
        mean1 = [-1, 2]
        mean2 = [1, -1]
        mean3 = [4, -4]
        mean4 = [-4, 4]
        cov = [[1.0,0.8], [0.8, 1.0]]
        X1 = np.random.multivariate_normal(mean1, cov, 50)
        X1 = np.vstack((X1, np.random.multivariate_normal(mean3, cov, 50)))
        np.random.shuffle(X1)
        y1 = np.ones(len(X1))
        X2 = np.random.multivariate_normal(mean2, cov, 50)
        X2 = np.vstack((X2, np.random.multivariate_normal(mean4, cov, 50)))
        np.random.shuffle(X2)
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

        ####
        # Write here your SVM code and choose a linear kernel
        # plot the graph with the support_vectors_
        # Print on the console the number of correct predictions and the total of predictions
        ####

        X1, y1, X2, y2 = generate_data_set1()
        C = 1
        dat_number = 1
        for krn_fnct in ['linear', 'rbf', 'my_knl']:
            main_function(X1, y1, X2, y2, C, krn_fnct, True, dat_number)
            print('-------------------------------------------------')



    def run_svm_dataset2():

        ####
        # Write here your SVM code and choose a linear kernel with the best C parameter
        # plot the graph with the support_vectors_
        # print on the console the number of correct predictions and the total of predictions
        ####

        X1, y1, X2, y2 = generate_data_set2()
        dat_number = 2
        for i in [0.1, 0.5, 1, 5]:
            C = i
            main_function(X1, y1, X2, y2, C, 'linear', False, dat_number)
            print('-------------------------------------------------')

        C = 1
        for krn_fnct in ['linear', 'rbf', 'poly']:
            main_function(X1, y1, X2, y2, C, krn_fnct, True, dat_number)
            print('-------------------------------------------------')


    def run_svm_dataset2_victor():

        ####
        # Write here your SVM code and choose a linear kernel with the best C parameter
        # plot the graph with the support_vectors_
        # print on the console the number of correct predictions and the total of predictions
        ####

        X1, y1, X2, y2 = generate_data_set2_victor()
        dat_number = 2
        for i in [0.1, 1, 10, 500]:
            C = i
            main_function(X1, y1, X2, y2, C, 'linear', False, dat_number)
            print('-------------------------------------------------')


    def run_svm_dataset3():

        ####
        # Write here your SVM code and use a gaussian kernel
        # plot the graph with the support_vectors_
        # print on the console the number of correct predictions and the total of predictions
        ####

        X1, y1, X2, y2 = generate_data_set3()
        C = 1
        dat_number = 3
        for krn_fnct in ['linear', 'rbf', 'poly']:
            main_function(X1, y1, X2, y2, C, krn_fnct, True, dat_number)
            print('-------------------------------------------------')


    def main_function(X1, y1, X2, y2, C, kernel_function, plot_boolean, dat_number):
        if plot_boolean:
            plt.figure(1)
            plt.subplot(231)
            plt.plot(X1[:, 0], X1[:, 1], c='r', linestyle='None', marker='.')
            plt.plot(X2[:, 0], X2[:, 1], c='b', linestyle='None', marker='.')
            plt.xlabel('X1')
            plt.ylabel('X2')
            plt.title('Labeled dataset (Train + Test)')
            # plt.show()

        X_train, y_train = split_train(X1, y1, X2, y2)
        X_test, y_test = split_test(X1, y1, X2, y2)

        if plot_boolean:
            plt.subplot(232)
            idx1 = y_train == 1
            idx2 = y_train == -1
            plt.plot(X_train[idx1, 0], X_train[idx1, 1], c='r', linestyle='None', marker='.')
            plt.plot(X_train[idx2, 0], X_train[idx2, 1], c='b', linestyle='None', marker='.')
            plt.xlabel('X1')
            plt.ylabel('X2')
            plt.title('Labeled Training set')
            #plt.show()

        # Write here your SVM code and choose a linear kernel
        if kernel_function != 'my_knl':
            svm = SVC(C = C, kernel=kernel_function)
        else:
            svm = SVC(C=C, kernel=my_kernel)
        svm.fit(X_train, y_train)

        if plot_boolean:
            # Plot the graph with the support_vectors_
            plt.subplot(233)
            plt.plot(X_train[idx1, 0], X_train[idx1, 1], c='r', linestyle='None', marker='.')
            plt.plot(X_train[idx2, 0], X_train[idx2, 1], c='b', linestyle='None', marker='.')
            plt.plot(X_train[svm.support_, 0], X_train[svm.support_, 1], c='g', linestyle='None', marker='*')
            plt.xlabel('X1')
            plt.ylabel('X2')
            plt.title('Labeled Training set with support vectors')
            #plt.show()

            # Plot decidecision boundaries
            plt.subplot(234)
            xmin, xmax = X_train[:, 0].min() - 0.5, X_train[:, 0].max() + 0.5
            ymin, ymax = X_train[:, 1].min() - 0.5, X_train[:, 1].max() + 0.5
            x, y = np.meshgrid(np.arange(xmin, xmax, 0.01), np.arange(ymin, ymax, 0.01))
            z = svm.predict(np.c_[x.ravel(), y.ravel()])
            z = z.reshape(x.shape)
            plt.contourf(x, y, z, alpha=0.2, colors=['blue', 'red'])
            plt.plot(X_train[idx1, 0], X_train[idx1, 1], c='r', linestyle='None', marker='.')
            plt.plot(X_train[idx2, 0], X_train[idx2, 1], c='b', linestyle='None', marker='.')
            plt.plot(X_train[svm.support_, 0], X_train[svm.support_, 1], c='g', linestyle='None', marker='*')
            plt.xlabel('X1')
            plt.ylabel('X2')
            plt.title('Labeled Training set with support vectors')
            #plt.show()

        scores = svm.predict(X_test)

        if plot_boolean:
            plt.subplot(235)
            idx1 = y_test == 1
            idx2 = y_test == -1
            plt.plot(X_test[idx1, 0], X_test[idx1, 1], c='r', linestyle='None', marker='.')
            plt.plot(X_test[idx2, 0], X_test[idx2, 1], c='b', linestyle='None', marker='.')
            plt.xlabel('X1')
            plt.ylabel('X2')
            plt.title('Testing set with target labels')
            #plt.show()

            plt.subplot(236)
            idx1 = scores == 1
            idx2 = scores == -1
            plt.plot(X_test[idx1, 0], X_test[idx1, 1], c='r', linestyle='None', marker='.')
            plt.plot(X_test[idx2, 0], X_test[idx2, 1], c='b', linestyle='None', marker='.')
            plt.xlabel('X1')
            plt.ylabel('X2')
            plt.title('Testing set with predicted labels')

            plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25,
                                wspace=0.35)
            plt.suptitle('[Dataset '+str(dat_number)+'] Parameters: '+ kernel_function + ' kernel and C = ' +str(C),
                         fontweight='bold')
            plt.show()


        # Print on the console the number of correct predictions and the total of predictions
        num_correct_predict = sum([x == y for x, y in zip(y_test, scores)])
        print('Number of correct predictions: ' + str(num_correct_predict))
        print('Total number of predictions: ' + str(len(y_test)))
        accuracy = num_correct_predict / len(y_test)
        print('Accuracy with kernel = '+ str(kernel_function)+' and C = '+ str(C)+' : ' +'\033[1m' + str(accuracy)+'\033[0m')


    def my_kernel(X, Y):
        return (np.dot(X, Y.T))
#############################################################
#############################################################
#############################################################

# EXECUTE SVM with THIS DATASETS
    print('\033[1m'+'Daraset 1: Results using a SVM with different kernel functions'+'\033[0m')
    run_svm_dataset1()   # data distribution 1
    print('\n\033[1m'+'Dataset 2: Results using a SVM with different C values and kernel functions'+'\033[0m')
    run_svm_dataset2()   # data distribution 2
    # print('\n\033[1m'+'Modified dataset 2: Results using a SVM with a linear kernel and the best parameter of
    # C'+'\033['                                                                                          '0m')
    #run_svm_dataset2_victor()   # data distribution 2 (modified)
    print('\n\033[1m'+'Dataset 3: Results using a SVM with a different kernel functions'+'\033[0m')
    run_svm_dataset3()   # data distribution 3

#############################################################
#############################################################
#############################################################
