import numpy as np
import pandas as pd

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import mutual_info_classif
import sklearn_relief as relief
from sklearn.feature_selection import GenericUnivariateSelect




class Preprocess:


    def preprocess_method(self, data, metric, groundtruth_labels, w):
        features_del = []

        # Normalize the data as long as the metric distance is not HVDM
        if (metric != 'hvdm') and (w == False):
            for feature in range(data.shape[1]):

                # Numerical Features
                if type(data[0, feature]) in [float, np.float64]:

                    # Calculate the mean of this feature of feed NaNs with it
                    mean_v = np.nanmean(data[:, feature], dtype=float)

                    # Calculate the max and min to normalize numerical data between 0 and 1
                    max_v = np.nanmax(data[:, feature])
                    min_v = np.nanmin(data[:, feature])

                    for sample in range(data.shape[0]):
                        if np.isnan(data[sample, feature]):
                            data[sample, feature] = mean_v
                        if max_v != 0:
                            data[sample, feature] = (data[sample, feature] - min_v) / (max_v - min_v)

            return data

        # In this case the data is not normalized, it is divided by 4*standard deviation
        elif (metric == 'hvdm') and (w == False):
            for feature in range(data.shape[1]):

                # Numerical Features
                if type(data[0, feature]) in [float, np.float64]:

                    # Calculate the mean of this feature without considering NaNs
                    mean_v = np.nanmean(data[:, feature], dtype=float)

                    # Calculate the standard deviation of this feature without considering NaNs
                    std = np.nanstd(data[:, feature], dtype=float)

                    for sample in range(data.shape[0]):
                        if np.isnan(data[sample, feature]):
                            data[sample, feature] = mean_v/(4*std)
                        else:
                            data[sample, feature] = (data[sample, feature])/(4*std)

            return data

        elif (w == True):
            features_del = []

            for feature in range(data.shape[1]):

                # Numerical Features
                if type(data[0, feature]) in [float, np.float64]:

                    # Calculate the mean of this feature of feed NaNs with it
                    mean_v = np.nanmean(data[:, feature], dtype=float)

                    # Calculate the max and min to normalize numerical data between 0 and 1
                    max_v = np.nanmax(data[:, feature])
                    min_v = np.nanmin(data[:, feature])

                    for sample in range(data.shape[0]):
                        if np.isnan(data[sample, feature]):
                            data[sample, feature] = mean_v
                        if max_v != 0:
                            data[sample, feature] = (data[sample, feature] - min_v) / (max_v - min_v)

                # Categorical Features
                if type(data[0, feature]) is bytes:
                    # Calculate the mode of this feature
                    cat_values = np.unique(data[:, feature])
                    moda = max(cat_values, key=lambda x: data[:, feature].tolist().count(x))

                    # Assign the mode to NaNs
                    cond_nan = np.where(data[:, feature] == '?'.encode('utf8'))
                    data[cond_nan, feature] = moda

                    # OneHotEncoding
                    data1 = np.array(pd.get_dummies(data[:, feature]))
                    data = np.concatenate((data, data1), axis=1)

                    features_del.append(feature)

            # Delete categorical feature
            data = np.delete(data, features_del, 1)
            wmethod = 3

            # ------------------W1: Mutual_info_classif-------------------------
            if wmethod == 1:
                # a = mutual_info_classif(data, groundtruth_labels)
                # data = data * a
                trans = GenericUnivariateSelect(score_func=mutual_info_classif, mode='percentile', param=50)
                data = trans.fit_transform(data, groundtruth_labels)

            # ------------------W2: Tree-based feature selection----------------
            elif wmethod == 2:
                clf = ExtraTreesClassifier(n_estimators=50)
                clf = clf.fit(data, groundtruth_labels)
                # a = clf.feature_importances_
                # data = data * a
                model = SelectFromModel(clf, prefit=True)
                data = model.transform(data) # selects some features, the n_estimator is a different parameter

            # ------------------W3: Relief--------------------------------------
            elif wmethod == 3:
                # r = relief.Relief(n_features=3)
                # r.fit(data, groundtruth_labels)
                # a = r.w_
                # data = data * a
                r = relief.Relief(n_features=30)
                data = r.fit_transform(data, groundtruth_labels)

            return data