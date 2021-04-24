# -*- coding: utf-8 -*-
# Author: Eirini Papagiannopoulou
# Date: 14/10/2020

"""LV keyword extraction model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import string
import logging

from pke.unsupervised.statistical.lv import LV

import numpy as np

from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest

class OutlierDetection(LV):
    """OutlierDetection keyword extraction model.

    Parameterized example::

        import string
        import pke

        # 1. create a OutlierDetection extractor.
        extractor = pke.unsupervised.OutlierDetection()

        # 2. load the content of the document.
        extractor.load_document(input='path/to/input',
                                language='en',
                                normalization=None)
        # 3. select unigrams as candidates.
        stopword_list = [...] # add stopword list
        extractor.candidate_selection(stopword_list)

        # 4. calculate co-occurrences between each pair of candidate words in a window of 10 words in corporating positional info
        extractor.find_cooccurrences(stopword_list, window=10, position=True)

        # 5. convert co-occurrences dictionary to a co-occurrrence numpy matrix
        extractor.build_cooccurrences_matrix():

        # 6. weight the candidates using the decision function of the outlier detector
        extractor.candidate_weighting()

        # 7. get the 10-highest scored candidates as keywords
        keywords = extractor.get_n_best(n=10)
    """
    def __init__(self):
        """Redefining initializer for OutlierDetection."""

        super(OutlierDetection, self).__init__()

    def candidate_weighting(self, n_estimators = 200, max_samples = 0.5, max_features = 0.75, nu = 0.05, method = 'IF'):
        """....
        Args:
        ...
        """
        ivocab = {v: k for k, v in self.vocab.items()}
          
        if method == 'OCSVM':
            clf = OneClassSVM(nu = nu, kernel = 'rbf', gamma = 'scale').fit(self.cooccurrences_matrix)

            signed_distances = clf.decision_function(self.cooccurrences_matrix)

            for i in range(self.cooccurrences_matrix.shape[0]):
              if signed_distances[i] < 0:
                self.weights[ivocab[i]] = abs(signed_distances[i])

            print('inliers:', signed_distances[np.where(signed_distances >= 0)].shape, 'i.e.', self.cooccurrences_matrix.shape[0]-signed_distances[np.where(signed_distances >= 0)].shape[0])
            print('outliers:', signed_distances[np.where(signed_distances < 0)].shape)
        elif method == 'IF':
            clf = IsolationForest(n_estimators = n_estimators, max_samples = max_samples, max_features = max_features, contamination = nu, random_state = 0).fit(self.cooccurrences_matrix)

            signed_distances = clf.decision_function(self.cooccurrences_matrix)

            for i in range(self.cooccurrences_matrix.shape[0]):
              if signed_distances[i] < 0:
                self.weights[ivocab[i]] = abs(signed_distances[i])

            print('inliers:', signed_distances[np.where(signed_distances >= 0)].shape, 'i.e.', self.cooccurrences_matrix.shape[0]-signed_distances[np.where(signed_distances >= 0)].shape[0])
            print('outliers:', signed_distances[np.where(signed_distances < 0)].shape)
        return signed_distances[np.where(signed_distances < 0)].shape[0]