# -*- coding: utf-8 -*-

"""Base classes for the pke module."""

from collections import defaultdict

from pke.data_structures import Candidate, Document
from pke.readers import MinimalCoreNLPReader, RawTextReader

from nltk.stem.snowball import SnowballStemmer
from nltk import RegexpParser
from nltk.corpus import stopwords
from nltk.tag.mapping import map_tag

import string
from string import punctuation
import os
import logging
import codecs

from six import string_types

from builtins import str

import numpy as np
from numpy import mean, absolute 

from sklearn.decomposition import PCA
from sklearn.covariance import MinCovDet
from sklearn.decomposition import KernelPCA
from sklearn.svm import OneClassSVM

from copy import deepcopy

import operator

import math

import pickle

from scipy.spatial.distance import euclidean

import networkx as nx

ISO_to_language = {'en': 'english', 'pt': 'portuguese', 'fr': 'french',
                   'es': 'spanish', 'it': 'italian', 'nl': 'dutch',
                   'de': 'german'}

escaped_punctuation = {'-lrb-': '(', '-rrb-': ')', '-lsb-': '[', '-rsb-': ']',
                       '-lcb-': '{', '-rcb-': '}'}


class LoadFile(object):
    """The LoadFile class that provides base functions."""

    def __init__(self):
        """Initializer for LoadFile class."""

        self.input_file = None
        """Path to the input file."""

        self.language = None
        """Language of the input file."""

        self.normalization = None
        """Word normalization method."""

        self.sentences = []
        """Sentence container (list of Sentence objects)."""

        self.candidates = defaultdict(Candidate)
        """Keyphrase candidates container (dict of Candidate objects)."""

        self.weights = {}
        """Weight container (can be either word or candidate weights)."""

        self._models = os.path.join(os.path.dirname(__file__), 'models')
        """Root path of the models."""

        self._df_counts = os.path.join(self._models, "df-semeval2010.tsv.gz")
        """Path to the document frequency counts provided in pke."""

        self.stoplist = None
        """List of stopwords."""

        # self.dict_cooccurrences = {}

        # self.cooccurrences_matrix = np.zeros(shape=(1, 1))


    # def candidate_selection(self, n=3, stoplist=None, **kwargs):
    #     """Select 1-3 grams as keyphrase candidates.

    #     Args:
    #         n (int): the length of the n-grams, defaults to 3.
    #         stoplist (list): the stoplist for filtering candidates, defaults to
    #             `None`. Words that are punctuation marks from
    #             `string.punctuation` are not allowed.

    #     """

    #     # select ngrams from 1 to 3 grams
    #     self.ngram_selection(n=n)

    #     # initialize empty list if stoplist is not provided
    #     if stoplist is None:
    #         stoplist = list(string.punctuation)

    #     # filter candidates containing punctuation marks
    #     self.candidate_filtering(stoplist=stoplist)

    # def LV(self, W, vocab):
    #     """Produce local word embedding (GloVe).
    #     Args:
    #     Wpos (numpy 2d array): the initial word-sentence position_matrix before PCA (required to count word occurrences in document)
    #     W (numpy 2d array): the vectors (rows: the different vectors, cols: the vector dimensions).
    #     """
    #     ivocab = {v: k for k, v in vocab.items()}  

    #     mean_vector = np.mean(W, axis=0)
    #     dist = {}
    #     for i in range(0, W.shape[0]):
    #       dist[ivocab[i]] = euclidean(W[i,:], mean_vector)
        
    #     sorted_outliers = sorted(dist.items(), key=operator.itemgetter(1), reverse=True)
        
        
    #     return sorted_outliers


    def candidate_weighting(self, outlineness, dfi=None, dfo=None, dfi2=None, dfo2=None):
        """Candidate weighting function using document frequencies.

        Args:
            df (dict): document frequencies, the number of documents should be
                specified using the "--NB_DOC--" key.
        """
        weights = {}
        dict_outlineness = dict(outlineness)
        # initialize the number of documents as --NB_DOC-- + 1 (current)
        N = 1 + dfi.get('--NB_DOC--', 0)

        # loop throught the candidates
        for k, v in dict_outlineness.items():
            
            candidate_dfi = 0.0
            # get candidate document frequency
            if k in dfi.keys():
              candidate_dfi = dfi.get(k, 0)

            candidate_dfo = 0.0
            # get candidate document frequency
            if k in dfo.keys():
              candidate_dfo = dfo.get(k, 0)

            if dfi2 != None and dfo2!=None:
              # get candidate document frequency
              if k in dfi2.keys():
                candidate_dfi += dfi2.get(k, 0)

              # get candidate document frequency
              if k in dfo2.keys():
                candidate_dfo += dfo2.get(k, 0)
                 
            # compute the docio score
            # docio = math.sqrt((candidate_dfo+1.0)/(candidate_dfi+1.0))# math.exp((candidate_dfo+1.0)/(candidate_dfi+1.0)) # (candidate_dfo/candidate_dfi)**(0.75)
            docio = math.sqrt((candidate_dfo)/(candidate_dfi+1.0))
            # # compute the idf score
            # docio = math.log(N / candidate_df, 2)
            # if candidate_dfo == 0 and candidate_dfi == 0:
            #   weights[k] = 0
            # else:
            weights[k] = abs(v)*docio

        sorted_weights = sorted(weights.items(), key=operator.itemgetter(1), reverse=True)

        return sorted_weights

    # def document_relative_variance(self, doc_name, drv_path, W, vocab, nu = 0.1, kernel = 'rbf', gamma = 'scale'):

    #     ivocab = {v: k for k, v in vocab.items()}  

    #     clf = OneClassSVM(nu = nu, kernel = kernel, gamma = gamma).fit(W)

    #     signed_distances = clf.decision_function(W)

    #     doc_outliers = {}
    #     doc_inliers = {}
    #     for i in range(W.shape[0]):
    #       if signed_distances[i] > 0:
    #         doc_inliers[ivocab[i]] = 1
    #       else:
    #         doc_outliers[ivocab[i]] = 1

    #     print('inliers:', signed_distances[np.where(signed_distances >= 0)].shape)
    #     print('outliers:', signed_distances[np.where(signed_distances < 0)].shape)

       
    #     with open(drv_path+'/'+doc_name+'_doc_inliers', 'wb') as file_relVar1:
    #       file_relVar1.write(pickle.dumps(doc_inliers))

    #     with open(drv_path+'/'+doc_name+'_doc_outliers', 'wb') as file_relVar2:
    #       file_relVar2.write(pickle.dumps(doc_outliers))

        
        
    #     with open(drv_path+'/'+doc_name+'_doc_inliers', 'rb') as handle1:
    #       inliers = pickle.load(handle1)

    #     with open(drv_path+'/'+doc_name+'_doc_outliers', 'rb') as handle2:
    #       outliers = pickle.load(handle2)

    #     # absolute_values = sorted(dict_points_scores.items(), key=operator.itemgetter(1))

    #     print('inliers:', inliers)
    #     print('outliers:', outliers)
    #     print('inliers+outliers:', len(inliers.keys()), len(outliers.keys()), len(inliers.keys())+len(outliers.keys()))
    #     # print('absolute_values:', absolute_values)

    # def detect_outlying_words(self, W, vocab, nu = 0.1, kernel = 'rbf', gamma = 'scale'):

    #     ivocab = {v: k for k, v in vocab.items()}  
    #     # number_of_outliers = int(nu*W.shape[0])

    #     # mean_vector = np.mean(W, axis=0)
    #     # dist = {}
    #     # for i in range(0, W.shape[0]):
    #     #   dist[ivocab[i]] = euclidean(W[i,:], mean_vector)   
        
    #     # sorted_inliers = dict(sorted(dist.items(), key=operator.itemgetter(1))[:(W.shape[0]-number_of_outliers)])

    #     # W_inliers = np.zeros(shape=(0, W.shape[1]))
    #     # for inlier in sorted_inliers.keys():
    #     #   W_inliers = np.vstack((W_inliers, W[vocab[inlier],:]))
    #     # print('W_inliers.shape:', W_inliers.shape, 'W.shape:', W.shape)
          
    #     clf = OneClassSVM(nu = nu, kernel = kernel, gamma = gamma).fit(W)

    #     signed_distances = clf.decision_function(W)

    #     points_signed_distances = {}
    #     for i in range(W.shape[0]):
    #       if signed_distances[i] < 0:
    #         points_signed_distances[ivocab[i]] = signed_distances[i]

    #     print('inliers:', signed_distances[np.where(signed_distances >= 0)].shape, 'i.e.', W.shape[0]-signed_distances[np.where(signed_distances >= 0)].shape[0])
    #     print('outliers:', signed_distances[np.where(signed_distances < 0)].shape)
    #     print('support_vectors_:', clf.support_vectors_.shape)


    #     # return signed_distances[np.where(signed_distances >= 0)], signed_distances[np.where(signed_distances < 0)]

    #     sorted_words = sorted(points_signed_distances.items(), key=operator.itemgetter(1))

    #     return sorted_words

    # def words_study(self, W, vocab):
    #     """Produce local word embedding (GloVe).
    #     Args:
    #     W (numpy 2d array): the vectors (rows: the different vectors, cols: the vector dimensions).
    #     vocab (dictionary): key->word, value->row number for W array
    #     support_fraction: the proportion of points to be included in the support of the raw MCD estimate. Default is None, which implies that the minimum value of support_fraction will be used within the algorithm: (n_sample + n_features + 1) / 2. The parameter must be in the range (0, 1).
    #     """

    #     ivocab = {v: k for k, v in vocab.items()}

    #     mean_vector = np.mean(W, axis=0)

    #     # Standard deviation for each dimension
    #     dict_std_dim = {}
    #     for dim in range(W.shape[1]):
    #       dict_std_dim[dim] = (mean(W[:,dim])+abs(np.std(W[:,dim])), mean(W[:,dim])-abs(np.std(W[:,dim])))
        
    #     print('dict_std_dim:', dict_std_dim)

    #     points_scores = defaultdict(float)
    #     for point in range(W.shape[0]):
    #       for dim in range(W.shape[1]):
    #         if W[point,dim] <= dict_std_dim[dim][0] and W[point,dim] >= dict_std_dim[dim][1]:
    #           points_scores[ivocab[point]] += 1 
    #     print(points_scores)
        
    #     counter_of_points = defaultdict(int)
    #     for p in points_scores.keys():
    #       if points_scores[p] == W.shape[1]:
    #         counter_of_points['inner points'] += 1 
    #       else:
    #         counter_of_points['outer points'] += 1
    #     print('stats:', counter_of_points)

    #     # sorted_words = sorted(points_rv.items(), key=operator.itemgetter(1), reverse=True)
        
    #     # return sorted_words


    # def words_relative_variance(self, W, vocab, support_fraction = None, minCovDet = True):
    #     """Produce local word embedding (GloVe).
    #     Args:
    #     W (numpy 2d array): the vectors (rows: the different vectors, cols: the vector dimensions).
    #     vocab (dictionary): key->word, value->row number for W array
    #     support_fraction: the proportion of points to be included in the support of the raw MCD estimate. Default is None, which implies that the minimum value of support_fraction will be used within the algorithm: (n_sample + n_features + 1) / 2. The parameter must be in the range (0, 1).
    #     """

    #     ivocab = {v: k for k, v in vocab.items()}
    #     W_support = np.zeros(shape=(1, 1))

    #     if minCovDet:
    #       clf = MinCovDet(assume_centered=False, support_fraction=support_fraction)
    #       clf.fit(W)            
    #       ids_of_robust_observations = clf.support_
    #       vocab_W_support = {}
    #       robust_observations_counter = 0
    #       for i in range(ids_of_robust_observations.shape[0]):
    #         if ids_of_robust_observations[i] == True:
    #           vocab_W_support[robust_observations_counter] = i
    #           robust_observations_counter += 1
    #       W_support = W[ids_of_robust_observations,:]
    #       print('robust_observations_counter:', robust_observations_counter)
    #       print('vocab_W_support:', len(vocab_W_support.keys()), vocab_W_support)
    #     else:
    #       W_support = deepcopy(W)

    #     mean_vector = np.mean(W_support, axis=0)
    #     print('check equality:', mean_vector, clf.location_)

    #     # Standard deviation as radius for rv
    #     std_0 = abs(np.std(W_support[:,0])) + mean(W_support[:,0])
    #     std_1 = abs(np.std(W_support[:,1])) + mean(W_support[:,1])
    #     std_2 = abs(np.std(W_support[:,2])) + mean(W_support[:,2])
    #     print('std_0:', std_0)
    #     print('std_1:', std_1)
    #     print('std_2:', std_2)
        
    #     # basic squared radius for the inner sphere
    #     R_2 = std_0**2 + std_1**2  + std_2**2

    #     points_scores = defaultdict(float)
    #     for point in range(W.shape[0]):
    #       dx = (W[point,0]-mean_vector[0])**2
    #       dy = (W[point,1]-mean_vector[1])**2
    #       dz = (W[point,2]-mean_vector[2])**2
    #       points_scores[ivocab[point]] = dx+dy+dz
    #     print(points_scores)

    #     sorted_words = sorted(points_scores.items(), key=operator.itemgetter(1), reverse=True)    
    #     max_point = vocab[sorted_words[0][0]]
    #     print('max_point', max_point, sorted_words[0][0])
    #     max_point_dx = (W[max_point, 0]-mean_vector[0])**2
    #     max_point_dy = (W[max_point, 1]-mean_vector[1])**2
    #     max_point_dz = (W[max_point, 2]-mean_vector[2])**2
        
    #     # find the number of std-radius annulus exist for the target text
    #     print('math.sqrt(max_point_dx+max_point_dy+max_point_dz)/math.sqrt(R_2)', math.sqrt(max_point_dx+max_point_dy+max_point_dz), math.sqrt(R_2))
    #     num_annulus = int(math.sqrt(max_point_dx+max_point_dy+max_point_dz)/math.sqrt(R_2))+1
    #     print('num_annulus', num_annulus)
        
    #     # Same-centred circles for rv: key->circle id (from inner to outer), value->corresponding squared radius
    #     same_centred_circles_rv_dict = defaultdict(float)
    #     for c in range(1, num_annulus+1, 1):
    #       same_centred_circles_rv_dict[c] = c*math.sqrt(R_2)

    #     points_rv = defaultdict(int)
    #     for point in range(W.shape[0]):
    #       dx = (W[point,0]-mean_vector[0])**2
    #       dy = (W[point,1]-mean_vector[1])**2
    #       dz = (W[point,2]-mean_vector[2])**2
    #       for c in range(1, num_annulus+1, 1):
    #         if math.sqrt(dx+dy+dz) <= same_centred_circles_rv_dict[c]:
    #           points_rv[ivocab[point]] = c
    #           break

        
    #     counter_of_points = defaultdict(int)
    #     for p in points_rv.keys():
    #       counter_of_points[points_rv[p]] += 1 
    #     print('stats:', counter_of_points)

    #     sorted_words = sorted(points_rv.items(), key=operator.itemgetter(1), reverse=True)
        
    #     return sorted_words

    # def document_relative_variance(self, dataset_name, doc_name, W, ivocab, support_fraction = 'default', num_circles = 3):
    #     """Produce local word embedding (GloVe).
    #     Args:
    #     Wpos (numpy 2d array): the initial word-sentence position_matrix before PCA (required to count word occurrences in document)
    #     W (numpy 2d array): the vectors (rows: the different vectors, cols: the vector dimensions).
    #     """
        
    #     if support_fraction == 'default':
    #       support_fraction = ((W.shape[0]+W.shape[1])/2.0)/W.shape[0]
    #     else:
    #       support_fraction = support_fraction

    #     vocab = {v: k for k, v in ivocab.items()}

    #     W_cov = clf.covariance_
    #     W_cov_inv = np.zeros((1,1))
        
    #     if linalg.cond(W_cov) < 1/sys.float_info.epsilon:
    #       clf = MinCovDet(assume_centered=False, support_fraction=support_fraction)
    #       clf.fit(W)
    #       mean_vector = clf.location_
    #       # W_cov = np.cov(W.T)
    #       ids_of_robust_observations = clf.support_
    #       vocab_W_support = {}
    #       robust_observations_counter = 0
    #       for i in range(ids_of_robust_observations.shape[0]):
    #         if ids_of_robust_observations[i] == True:
    #           vocab_W_support[robust_observations_counter] = i
    #           robust_observations_counter += 1
    #       print('robust_observations_counter:', robust_observations_counter)
    #       print('vocab_W_support:', len(vocab_W_support.keys()), vocab_W_support)

    #       ids_of_raw_robust_observations = clf.raw_support_
    #       print('ids_of_robust_observations:', ids_of_robust_observations.shape)
    #       W_support = W[ids_of_robust_observations,:]
    #       print('ids_of_raw_robust_observations:', ids_of_raw_robust_observations.shape)
    #       W_raw_support = W[ids_of_raw_robust_observations,:]
    #       print('W_support:', W_support.shape, 'W_raw_support:', W_raw_support.shape, 'W:', W.shape, W)

    #     mean_vector = np.mean(W_support, axis=0)

    #     # quartiles Q3 
    #     # q3_0 = abs(np.percentile(W[:,0], 50)-mean_vector[0])
    #     # q3_1 = abs(np.percentile(W[:,1], 50)-mean_vector[1])
    #     # print('q3_0:', q3_0)
    #     # print('q3_1:', q3_1)

    #     # # Absolute mean deviation for drv
    #     # mad_0 = mean(absolute(W[:,0] - mean(W[:,0])))
    #     # mad_1 = mean(absolute(W[:,1] - mean(W[:,1])))
    #     # print('mad_0:', mad_0)
    #     # print('mad_1:', mad_1)
    #     # Standard deviation as radius for rv
    #     std_0 = abs(np.std(W_support[:,0])) + mean(W_support[:,0])
    #     std_1 = abs(np.std(W_support[:,1])) + mean(W_support[:,1])
    #     print('std_0:', std_0)
    #     print('std_1:', std_1)
        
    #     # basic squared radius for drv
    #     # R_2 = mad_0**2 + mad_1**2
    #     # basic squared radius for rv
    #     R_2 = std_0**2 + std_1**2
    #     # R_2 = q3_0**2 + q3_1**2
    #     # R_2 = (q3_0 + q3_1)**2

    #     # Same-centred circles for drv: key->circle id (from inner to outer), value->corresponding squared radius
    #     same_centred_circles_dict = {}
    #     for c in range(num_circles+1, 1, -1):
    #       same_centred_circles_dict[c] = (num_circles+2-c)*math.sqrt(R_2)

    #     points_scores = {}
    #     points_scores2 = {}
    #     for point in range(W.shape[0]):
    #       dx = (W[point,0]-mean_vector[0])**2
    #       dy = (W[point,1]-mean_vector[1])**2
    #       for c in range(num_circles+1, 1, -1):
    #         if math.sqrt(dx+dy) <= same_centred_circles_dict[c]:
    #           points_scores[ivocab[point]] = c
    #           points_scores2[ivocab[point]] = dx+dy
    #           break
    #       if ivocab[point] not in points_scores.keys():
    #         points_scores[ivocab[point]] = 1
    #         points_scores2[ivocab[point]] = dx+dy

    #     with open('/content/gdrive/My Drive/Colab_notebooks/LocalRepresentationsKE/pke/pke/models/docs_words_relative_variance_counts/'+dataset_name+'_'+str(num_circles)+'/'+doc_name, 'wb') as file_relVar:
    #       file_relVar.write(pickle.dumps(points_scores))

    #     with open('/content/gdrive/My Drive/Colab_notebooks/LocalRepresentationsKE/pke/pke/models/docs_words_relative_variance_counts/'+dataset_name+'_'+str(num_circles)+'/'+doc_name, 'rb') as handle:
    #       dict_points_scores = pickle.load(handle)

    #     print('dict_points:', dict_points_scores)

    #     counter_of_points = defaultdict(int)
    #     for nc in range(1, num_circles+2, 1):
    #       counter_of_points[nc] = 0 
    #     for p in dict_points_scores.keys():
    #       counter_of_points[dict_points_scores[p]] += 1 

    #     print('stats:', counter_of_points)

    #     sorted_words = sorted(points_scores2.items(), key=operator.itemgetter(1), reverse=True)    
    #     max_point = vocab[sorted_words[0][0]]
    #     print('max_point', max_point, sorted_words[0][0])
    #     max_point_x = W[max_point, 0]
    #     max_point_y = W[max_point, 1]
    #     max_point_dx = (W[max_point, 0]-mean_vector[0])**2
    #     max_point_dy = (W[max_point, 1]-mean_vector[1])**2
    #     # find the number of std-radius annulus exist for the target text
    #     print('math.sqrt(max_point_dx+max_point_dy)/math.sqrt(R_2)', math.sqrt(max_point_dx+max_point_dy), math.sqrt(R_2))
    #     num_annulus = int(math.sqrt(max_point_dx+max_point_dy)/math.sqrt(R_2))+1
    #     print('num_annulus', num_annulus)
        
    #     # Same-centred circles for rv: key->circle id (from inner to outer), value->corresponding squared radius
    #     same_centred_circles_rv_dict = {}
    #     for c in range(1, num_annulus+1, 1):
    #       same_centred_circles_rv_dict[c] = c*math.sqrt(R_2)

    #     points_rv = {}
    #     for point in range(W.shape[0]):
    #       dx = (W[point,0]-mean_vector[0])**2
    #       dy = (W[point,1]-mean_vector[1])**2
    #       for c in range(1, num_annulus+1, 1):
    #         # print('trace0:', point, ivocab[point], math.sqrt(dx+dy), same_centred_circles_rv_dict[c])
    #         if math.sqrt(dx+dy) <= same_centred_circles_rv_dict[c]:
    #           # print('trace1:', point, ivocab[point], c)
    #           points_rv[ivocab[point]] = c
    #           break
        
    #     counter_of_points = defaultdict(int)
    #     for nc in range(1, num_circles+1, 1):
    #       counter_of_points[nc] = 0 
    #     for p in points_rv.keys():
    #       counter_of_points[points_rv[p]] += 1 

    #     print('stats:', counter_of_points)

    #     sorted_words = sorted(points_rv.items(), key=operator.itemgetter(1), reverse=True)
        
    #     return sorted_words



    # def build_cooccurrences(self, stopwords_list, window=10, position=True):
    #     """Build a co-occurrrence matrix among the document's words considering pos tags.
    # The number of times two words co-occur in a window is encoded in the corresponding
    # cell of the matrix. Sentence boundaries **are not** taken into account in the
    # window.
    # Args:
    #     window (int): the window for counting the co-occurrence between two words,
    #         defaults to 10.
    # """
    #     word_to_sentID = {}
    #     vocab = {}
    #     text = []
    #     vocab_index = 0
    #     for sent_id, sentence in enumerate(self.sentences):
    #         for i, word in enumerate(sentence.stems):
    #             word = word.translate(str.maketrans('', '', string.punctuation))
    #             text.append(word)
    #             if word not in vocab.keys() and word not in stopwords_list and not str(word).replace(".", "").replace(",", "").isdigit() and len(str(word)) > 1:
    #               vocab[word] = vocab_index
    #               vocab_index += 1
    #               word_to_sentID[word] = sent_id

    #     for i, word1 in enumerate(text[:int(len(text))]):
    #         if word1 in vocab.keys():
    #             if word1 not in self.dict_cooccurrences.keys():
    #                 self.dict_cooccurrences[word1] = {}
    #             for j in range(i + 1, min(i + window, len(text[:int(len(text))]))):
    #                 word2 = text[j]
    #                 if word1 != word2 and word2 in vocab.keys():
    #                     if position:
    #                       if word2 not in self.dict_cooccurrences[word1].keys():
    #                           self.dict_cooccurrences[word1][word2] = 1.0*(1.0/(word_to_sentID[word1] + word_to_sentID[word2] + 1.0))#1.0*(1.0/(text.index(word1) + text.index(word2))) #((text.index(word1) + text.index(word2))/2.0))
    #                       else:
    #                           self.dict_cooccurrences[word1][word2] += 1.0*(1.0/(word_to_sentID[word1] + word_to_sentID[word2] + 1.0))#1.0*(1.0/(text.index(word1) + text.index(word2))) #((text.index(word1) + text.index(word2))/2.0))
    #                     else:
    #                       if word2 not in self.dict_cooccurrences[word1].keys():
    #                           self.dict_cooccurrences[word1][word2] = 1.0
    #                       else:
    #                           self.dict_cooccurrences[word1][word2] += 1.0

    #     return self.dict_cooccurrences, vocab

    # def get_cooccurrences_matrix(self, vocab):
    #     """Convert co-occurrences dictionary to a co-occurrrence numpy matrix among the document's words.
    # """
    #     self.cooccurrences_matrix = np.zeros(shape=(len(vocab.keys()), len(vocab.keys())))
    #     for w1 in self.dict_cooccurrences.keys():
    #         for w2 in self.dict_cooccurrences[w1].keys():
    #             self.cooccurrences_matrix[vocab[w1], vocab[w2]] = self.dict_cooccurrences[w1][w2]
    #     return self.cooccurrences_matrix

    # def pca_projection(self, cooccurrences_matrix, num_components):
    #     pca = PCA(n_components=num_components)
    #     pca_cooccurrences_matrix = pca.fit_transform(cooccurrences_matrix)
    #     print('explained_variance_ratio_:', pca.explained_variance_ratio_)
        
    #     return pca_cooccurrences_matrix

    # def kpca_projection(self, cooccurrences_matrix, num_components):
    #     pca = KernelPCA(kernel='rbf', n_components=num_components, remove_zero_eig=True, gamma=10)
    #     pca_cooccurrences_matrix = pca.fit_transform(cooccurrences_matrix)
    #     print('lambdas_:', len(pca.lambdas_))
        
    #     return pca_cooccurrences_matrix

    def load_document(self, input, tagger=True, **kwargs):
        """Loads the content of a document/string/stream in a given language.

        Args:
            input (str): input.
            language (str): language of the input, defaults to 'en'.
            encoding (str): encoding of the raw file.
            normalization (str): word normalization method, defaults to
                'stemming'. Other possible values are 'lemmatization' or 'None'
                for using word surface forms instead of stems/lemmas.
        """

        # get the language parameter
        language = kwargs.get('language', 'en')

        # test whether the language is known, otherwise fall back to english
        if language not in ISO_to_language:
            logging.warning(
                "ISO 639 code {} is not supported, switching to 'en'.".format(
                    language))
            language = 'en'

        # initialize document
        doc = Document()

        if isinstance(input, string_types):

            # if input is an input file
            if os.path.isfile(input):

                # an xml file is considered as a CoreNLP document
                if input.endswith('xml'):
                    parser = MinimalCoreNLPReader()
                    doc = parser.read(path=input, **kwargs)
                    doc.is_corenlp_file = True

                # other extensions are considered as raw text
                else:
                    parser = RawTextReader(language=language)
                    encoding = kwargs.get('encoding', 'utf-8')
                    with codecs.open(input, 'r', encoding=encoding) as file:
                        text = file.read()
                    doc = parser.read(text=text, path=input, tagger=tagger, **kwargs)

            # if input is a string
            else:
                parser = RawTextReader(language=language)
                doc = parser.read(text=input, **kwargs)

        elif getattr(input, 'read', None):
            # check whether it is a compressed CoreNLP document
            name = getattr(input, 'name', None)
            if name and name.endswith('xml'):
                parser = MinimalCoreNLPReader()
                doc = parser.read(path=input, **kwargs)
                doc.is_corenlp_file = True
            else:
                parser = RawTextReader(language=language)
                doc = parser.read(text=input.read(), **kwargs)

        else:
            logging.error('Cannot process {}'.format(type(input)))

        # set the input file
        self.input_file = doc.input_file

        # set the language of the document
        self.language = language

        # set the sentences
        self.sentences = doc.sentences

        # initialize the stoplist
        self.stoplist = stopwords.words(ISO_to_language[self.language])

        # word normalization
        self.normalization = kwargs.get('normalization', 'stemming')
        if self.normalization == 'stemming':
            self.apply_stemming()
        elif self.normalization is None:
            for i, sentence in enumerate(self.sentences):
                self.sentences[i].stems = sentence.words

        # lowercase the normalized words
        for i, sentence in enumerate(self.sentences):
            self.sentences[i].stems = [w.lower() for w in sentence.stems]

        # POS normalization
        if getattr(doc, 'is_corenlp_file', False):
            self.normalize_pos_tags()
            self.unescape_punctuation_marks()

    def apply_stemming(self):
        """Populates the stem containers of sentences."""

        if self.language == 'en':
            # create a new instance of a porter stemmer
            stemmer = SnowballStemmer("porter")
        else:
            # create a new instance of a porter stemmer
            stemmer = SnowballStemmer(ISO_to_language[self.language],
                                      ignore_stopwords=True)

        # iterate throughout the sentences
        for i, sentence in enumerate(self.sentences):
            self.sentences[i].stems = [stemmer.stem(w) for w in sentence.words]

    def normalize_pos_tags(self):
        """Normalizes the PoS tags from udp-penn to UD."""

        if self.language == 'en':
            # iterate throughout the sentences
            for i, sentence in enumerate(self.sentences):
                self.sentences[i].pos = [map_tag('en-ptb', 'universal', tag)
                                         for tag in sentence.pos]

    def unescape_punctuation_marks(self):
        """Replaces the special punctuation marks produced by CoreNLP."""

        for i, sentence in enumerate(self.sentences):
            for j, word in enumerate(sentence.words):
                l_word = word.lower()
                self.sentences[i].words[j] = escaped_punctuation.get(l_word,
                                                                     word)

    def is_redundant(self, candidate, prev, minimum_length=1):
        """Test if one candidate is redundant with respect to a list of already
        selected candidates. A candidate is considered redundant if it is
        included in another candidate that is ranked higher in the list.

        Args:
            candidate (str): the lexical form of the candidate.
            prev (list): the list of already selected candidates (lexical
                forms).
            minimum_length (int): minimum length (in words) of the candidate
                to be considered, defaults to 1.
        """

        # get the tokenized lexical form from the candidate
        candidate = self.candidates[candidate].lexical_form

        # only consider candidate greater than one word
        if len(candidate) < minimum_length:
            return False

        # get the tokenized lexical forms from the selected candidates
        prev = [self.candidates[u].lexical_form for u in prev]

        # loop through the already selected candidates
        for prev_candidate in prev:
            for i in range(len(prev_candidate) - len(candidate) + 1):
                if candidate == prev_candidate[i:i + len(candidate)]:
                    return True
        return False

    def get_n_best(self, n=10, redundancy_removal=False, stemming=False):
        """Returns the n-best candidates given the weights.

        Args:
            n (int): the number of candidates, defaults to 10.
            redundancy_removal (bool): whether redundant keyphrases are
                filtered out from the n-best list, defaults to False.
            stemming (bool): whether to extract stems or surface forms
                (lowercased, first occurring form of candidate), default to
                False.
        """

        # sort candidates by descending weight
        best = sorted(self.weights, key=self.weights.get, reverse=True)

        # remove redundant candidates
        if redundancy_removal:

            # initialize a new container for non redundant candidates
            non_redundant_best = []

            # loop through the best candidates
            for candidate in best:

                # test wether candidate is redundant
                if self.is_redundant(candidate, non_redundant_best):
                    continue

                # add the candidate otherwise
                non_redundant_best.append(candidate)

                # break computation if the n-best are found
                if len(non_redundant_best) >= n:
                    break

            # copy non redundant candidates in best container
            best = non_redundant_best

        # get the list of best candidates as (lexical form, weight) tuples
        n_best = [(u, self.weights[u]) for u in best[:min(n, len(best))]]

        # replace with surface forms if no stemming
        if not stemming:
            n_best = [(' '.join(self.candidates[u].surface_forms[0]).lower(),
                       self.weights[u]) for u in best[:min(n, len(best))]]

        if len(n_best) < n:
            logging.warning(
                'Not enough candidates to choose from '
                '({} requested, {} given)'.format(n, len(n_best)))

        # return the list of best candidates
        return n_best

    def add_candidate(self, words, stems, pos, offset, sentence_id):
        """Add a keyphrase candidate to the candidates container.

        Args:
            words (list): the words (surface form) of the candidate.
            stems (list): the stemmed words of the candidate.
            pos (list): the Part-Of-Speeches of the words in the candidate.
            offset (int): the offset of the first word of the candidate.
            sentence_id (int): the sentence id of the candidate.
        """

        # build the lexical (canonical) form of the candidate using stems
        lexical_form = ' '.join(stems)

        # add/update the surface forms
        self.candidates[lexical_form].surface_forms.append(words)

        # add/update the lexical_form
        self.candidates[lexical_form].lexical_form = stems

        # add/update the POS patterns
        self.candidates[lexical_form].pos_patterns.append(pos)

        # add/update the offsets
        self.candidates[lexical_form].offsets.append(offset)

        # add/update the sentence ids
        self.candidates[lexical_form].sentence_ids.append(sentence_id)

    def ngram_selection(self, n=3, pos = None):
        """Select all the n-grams and populate the candidate container.

        Args:
            n (int): the n-gram length, defaults to 3.
        """

        # loop through the sentences
        for i, sentence in enumerate(self.sentences):

            # limit the maximum n for short sentence
            skip = min(n, sentence.length)

            # compute the offset shift for the sentence
            shift = sum([s.length for s in self.sentences[0:i]])

            # added for the experiments
            if pos != None: 
                # generate the ngrams
                for j in range(sentence.length):
                    if sentence.pos[j] in pos:
                        for k in range(j + 1, min(j + 1 + skip, sentence.length + 1)):
                            # add the ngram to the candidate container
                            self.add_candidate(words=sentence.words[j:k],
                                              stems=sentence.stems[j:k],
                                              pos=sentence.pos[j:k],
                                              offset=shift + j,
                                              sentence_id=i)
            else:
                # generate the ngrams
                for j in range(sentence.length):
                    for k in range(j + 1, min(j + 1 + skip, sentence.length + 1)):
                        # add the ngram to the candidate container
                        self.add_candidate(words=sentence.words[j:k],
                                          stems=sentence.stems[j:k],
                                          pos=sentence.pos[j:k],
                                          offset=shift + j,
                                          sentence_id=i)
        return self.candidates
      
    def longest_pos_sequence_selection(self, valid_pos=None):
        self.longest_sequence_selection(
            key=lambda s: s.pos, valid_values=valid_pos)

    def longest_keyword_sequence_selection(self, keywords):
        self.longest_sequence_selection(
            key=lambda s: s.stems, valid_values=keywords)

    def longest_sequence_selection(self, key, valid_values):
        """Select the longest sequences of given POS tags as candidates.

        Args:
            key (func) : function that given a sentence return an iterable
            valid_values (set): the set of valid values, defaults to None.
        """

        # loop through the sentences
        for i, sentence in enumerate(self.sentences):

            # compute the offset shift for the sentence
            shift = sum([s.length for s in self.sentences[0:i]])

            # container for the sequence (defined as list of offsets)
            seq = []

            # loop through the tokens
            for j, value in enumerate(key(self.sentences[i])):

                # add candidate offset in sequence and continue if not last word
                if value in valid_values:
                    seq.append(j)
                    if j < (sentence.length - 1):
                        continue

                # add sequence as candidate if non empty
                if seq:

                    # add the ngram to the candidate container
                    self.add_candidate(words=sentence.words[seq[0]:seq[-1] + 1],
                                       stems=sentence.stems[seq[0]:seq[-1] + 1],
                                       pos=sentence.pos[seq[0]:seq[-1] + 1],
                                       offset=shift + seq[0],
                                       sentence_id=i)

                # flush sequence container
                seq = []

    def grammar_selection(self, grammar=None):
        """Select candidates using nltk RegexpParser with a grammar defining
        noun phrases (NP).

        Args:
            grammar (str): grammar defining POS patterns of NPs.
        """

        # initialize default grammar if none provided
        if grammar is None:
            grammar = r"""
                NBAR:
                    {<NOUN|PROPN|ADJ>*<NOUN|PROPN>} 
                    
                NP:
                    {<NBAR>}
                    {<NBAR><ADP><NBAR>}
            """

        # initialize chunker
        chunker = RegexpParser(grammar)

        # loop through the sentences
        for i, sentence in enumerate(self.sentences):

            # compute the offset shift for the sentence
            shift = sum([s.length for s in self.sentences[0:i]])

            # convert sentence as list of (offset, pos) tuples
            tuples = [(str(j), sentence.pos[j]) for j in range(sentence.length)]

            # parse sentence
            tree = chunker.parse(tuples)

            # find candidates
            for subtree in tree.subtrees():
                if subtree.label() == 'NP':
                    leaves = subtree.leaves()

                    # get the first and last offset of the current candidate
                    first = int(leaves[0][0])
                    last = int(leaves[-1][0])

                    # add the NP to the candidate container
                    self.add_candidate(words=sentence.words[first:last + 1],
                                       stems=sentence.stems[first:last + 1],
                                       pos=sentence.pos[first:last + 1],
                                       offset=shift + first,
                                       sentence_id=i)

    @staticmethod
    def _is_alphanum(word, valid_punctuation_marks='-'):
        """Check if a word is valid, i.e. it contains only alpha-numeric
        characters and valid punctuation marks.

        Args:
            word (string): a word.
            valid_punctuation_marks (str): punctuation marks that are valid
                    for a candidate, defaults to '-'.
        """
        for punct in valid_punctuation_marks.split():
            word = word.replace(punct, '')
        return word.isalnum()

    def candidate_filtering(self,
                            stoplist=None,
                            minimum_length=3,
                            minimum_word_size=2,
                            valid_punctuation_marks='-',
                            maximum_word_number=5,
                            only_alphanum=True,
                            pos_blacklist=None):
        """Filter the candidates containing strings from the stoplist. Only
        keep the candidates containing alpha-numeric characters (if the
        non_latin_filter is set to True) and those length exceeds a given
        number of characters.
            
        Args:
            stoplist (list): list of strings, defaults to None.
            minimum_length (int): minimum number of characters for a
                candidate, defaults to 3.
            minimum_word_size (int): minimum number of characters for a
                token to be considered as a valid word, defaults to 2.
            valid_punctuation_marks (str): punctuation marks that are valid
                for a candidate, defaults to '-'.
            maximum_word_number (int): maximum length in words of the
                candidate, defaults to 5.
            only_alphanum (bool): filter candidates containing non (latin)
                alpha-numeric characters, defaults to True.
            pos_blacklist (list): list of unwanted Part-Of-Speeches in
                candidates, defaults to [].
        """

        if stoplist is None:
            stoplist = []

        if pos_blacklist is None:
            pos_blacklist = []

        # loop through the candidates
        for k in list(self.candidates):

            # get the candidate
            v = self.candidates[k]

            # get the words from the first occurring surface form
            words = [u.lower() for u in v.surface_forms[0]]

            # discard if words are in the stoplist
            if set(words).intersection(stoplist):
                del self.candidates[k]

            # discard if tags are in the pos_blacklist
            elif set(v.pos_patterns[0]).intersection(pos_blacklist):
                del self.candidates[k]

            # discard if containing tokens composed of only punctuation
            elif any([set(u).issubset(set(punctuation)) for u in words]):
                del self.candidates[k]

            # discard candidates composed of 1-2 characters
            elif len(''.join(words)) < minimum_length:
                del self.candidates[k]

            # discard candidates containing small words (1-character)
            elif min([len(u) for u in words]) < minimum_word_size:
                del self.candidates[k]

            # discard candidates composed of more than 5 words
            elif len(v.lexical_form) > maximum_word_number:
                del self.candidates[k]

            # discard if not containing only alpha-numeric characters
            if only_alphanum and k in self.candidates:
                if not all([self._is_alphanum(w, valid_punctuation_marks)
                            for w in words]):
                    del self.candidates[k]
