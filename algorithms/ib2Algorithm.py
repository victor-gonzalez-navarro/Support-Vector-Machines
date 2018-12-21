from algorithms.distances import *
from algorithms.voting_policies import *


class ib2Algorithm():

    trn_data = None
    trn_labels = None
    tst_labels = None
    dic_feats = {}

    def __init__(self, k=1, metric='euclidean', voting_policy = 'most_voted'):
        self.k = k
        if metric == 'euclidean':
            self.d = euclidean
        elif metric == 'manhattan':
            self.d = manhattan
        elif metric == 'canberra':
            self.d = canberra
        elif metric == 'hvdm':
            self.d = hvdm_2

        if voting_policy == 'most_voted':
            self.vp = most_voted
        elif voting_policy == 'modified_plurality':
            self.vp = modified_plurality
        elif voting_policy == 'borda_count':
            self.vp = borda_count

    def fit(self, trn_data, labels):
        trn_data_keep = trn_data[0,:].reshape(1,len(trn_data[0,:]))
        labels_keep = np.array(labels[0]).reshape(1)
        for j in range(1,trn_data.shape[0]):
            neighbor = np.argpartition([euclidean(trn_data[j,:], trn_sample, 0, 0) for trn_sample in trn_data_keep],
                                       kth=0)[:1]
            if labels[j] != labels_keep[neighbor]:
                trn_data_concat = trn_data[j,:].reshape(1,len(trn_data[j,:]))
                trn_data_keep = np.concatenate((trn_data_keep,trn_data_concat),axis=0)
                labels_keep = np.concatenate((labels_keep, np.array(labels[j]).reshape(1)))
        self.trn_data = trn_data_keep
        self.trn_labels = labels_keep

        if self.d == hvdm_2:

            for i in range(self.trn_data.shape[1]):
                if type(self.trn_data[0,i]) not in [float, np.float64]:
                    dic_vals = {}
                    for value in np.unique(trn_data[:,i]):
                        if value != b'?':
                            dic_class = {}
                            for label in np.unique(labels):
                                dic_class[label] = np.sum((self.trn_data[:,i] == value) * (self.trn_labels == label))
                            dic_vals[value] = dic_class
                    self.dic_feats[i] = dic_vals

    def classify(self, tst_data):
        self.tst_labels = np.zeros((tst_data.shape[0], 1))
        for i in range(tst_data.shape[0]):
            #neighbor_idxs = np.argpartition([self.d(tst_data[i, :], trn_samp, self.trn_data, self.trn_labels)
            #                                 for trn_samp in self.trn_data], kth=self.k - 1)[:self.k]#
            neighbor_idxs = np.argpartition([self.d(tst_data[i,:], trn_samp, self.trn_labels, self.dic_feats)
                            for trn_samp in self.trn_data], kth=self.k-1)[:self.k]
            order_labels = self.trn_labels[neighbor_idxs]
            self.tst_labels[i] = self.vp(order_labels, self.k)
