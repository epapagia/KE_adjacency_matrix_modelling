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

from pke.base import LoadFile

from scipy.spatial.distance import euclidean

import numpy as np
from numpy import mean 

from sklearn.decomposition import PCA


class LV(LoadFile):
    """LV keyword extraction model.

    Parameterized example::

        import string
        import pke

        # 1. create a LV extractor.
        extractor = pke.unsupervised.LV()

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
        extractor.build_cooccurrences_matrix()

        # 6. weight the candidates using a the euclidean distance from the mean vector
        extractor.candidate_weighting()

        # 7. get the 10-highest scored candidates as keywords
        keywords = extractor.get_n_best(n=10)
    """
    def __init__(self):
        """Redefining initializer for LV."""

        super(LV, self).__init__()
        
        self.dict_cooccurrences = {}

        self.cooccurrences_matrix = np.zeros(shape=(1, 1))

        self.word_to_sentID = {}
        
        self.vocab = {}

    def candidate_selection(self, stopwords_list):
        """Select unigrams as keyword candidates.

        Args:
            stopwords_list (list): the stoplist for filtering candidates

        """
        text = []
        vocab_index = 0
        for sent_id, sentence in enumerate(self.sentences):
            for i, word in enumerate(sentence.stems):
                word = word.translate(str.maketrans('', '', string.punctuation))
                text.append(word)
                if word not in self.vocab.keys() and word not in stopwords_list and not str(word).isdigit() and len(str(word)) > 1:
                  self.vocab[word] = vocab_index
                  vocab_index += 1
                  self.word_to_sentID[word] = sent_id
        print("self.word_to_sentID:", self.word_to_sentID)
        return self.vocab

    def candidate_weighting(self):
        """....
        Args:len()
        ...
        """
        ivocab = {v: k for k, v in self.vocab.items()}  

        mean_vector = np.mean(self.cooccurrences_matrix, axis=0)
        for i in range(0, self.cooccurrences_matrix.shape[0]):
          self.weights[ivocab[i]] = euclidean(self.cooccurrences_matrix[i,:], mean_vector)


    def find_cooccurrences(self, stopwords_list, window=10, position=True, weights=True):
        """Build a co-occurrrence matrix among the document's words considering pos tags.
    The number of times two words co-occur in a window is encoded in the corresponding
    cell of the matrix. Sentence boundaries **are not** taken into account in the
    window.
    Args:
        window (int): the window for counting the co-occurrence between two words,
            defaults to 10.
        ...
    """
        text = []
        vocab_index = 0
        for sent_id, sentence in enumerate(self.sentences):
            for i, word in enumerate(sentence.stems):
                word = word.translate(str.maketrans('', '', string.punctuation))
                text.append(word)

        for i, word1 in enumerate(text[:int(len(text))]):
            if word1 in self.vocab.keys():
                if word1 not in self.dict_cooccurrences.keys():
                    self.dict_cooccurrences[word1] = {}
                for j in range(i + 1, min(i + window, len(text[:int(len(text))]))):
                    word2 = text[j]
                    if word1 != word2 and word2 in self.vocab.keys():
                        if weights:
                            if position:
                              if word2 not in self.dict_cooccurrences[word1].keys():
                                  self.dict_cooccurrences[word1][word2] = 1.0*(1.0/(self.word_to_sentID[word1] + self.word_to_sentID[word2] + 1.0))
                              else:
                                  self.dict_cooccurrences[word1][word2] += 1.0*(1.0/(self.word_to_sentID[word1] + self.word_to_sentID[word2] + 1.0))
                            else:
                              if word2 not in self.dict_cooccurrences[word1].keys():
                                  self.dict_cooccurrences[word1][word2] = 1.0
                              else:
                                  self.dict_cooccurrences[word1][word2] += 1.0
                        else:
                            self.dict_cooccurrences[word1][word2] = 1.0

        return self.dict_cooccurrences

    def build_cooccurrences_matrix(self):
        """Convert co-occurrences dictionary to a co-occurrrence numpy matrix among the document's words."""
        self.cooccurrences_matrix = np.zeros(shape=(len(self.vocab.keys()), len(self.vocab.keys())))
        for w1 in self.dict_cooccurrences.keys():
            for w2 in self.dict_cooccurrences[w1].keys():
                self.cooccurrences_matrix[self.vocab[w1], self.vocab[w2]] = self.dict_cooccurrences[w1][w2]
        return self.cooccurrences_matrix


    def pca_projection(self, num_components):
        """ Only for the experimental study """
        pca = PCA(n_components=num_components)
        pca_cooccurrences_matrix = pca.fit_transform(self.cooccurrences_matrix)
        print('explained_variance_ratio_:', pca.explained_variance_ratio_)
        
        return pca_cooccurrences_matrix, pca.explained_variance_ratio_

