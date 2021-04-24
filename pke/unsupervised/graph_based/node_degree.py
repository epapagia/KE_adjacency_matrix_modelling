# -*- coding: utf-8 -*-
# Author: Florian Boudin
# Date: 09-11-2018

"""NodeDegree (RAKE) keyphrase extraction model.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import networkx as nx

from pke.unsupervised.graph_based.singlerank import SingleRank

import string

import operator

class NodeDegree(SingleRank):
    """NodeDegree keyphrase extraction model.

    Parameterized example::
        import pke
        # 1. create a NodeDegree extractor.
        extractor = pke.unsupervised.NodeDegree()

        # 2. load the content of the document.
        extractor.load_document(input='path/to/input',
                                language='en',
                                normalization=None)

        # 3. select the longest sequences of nouns and adjectives as candidates.
        extractor.candidate_selection_all(stopwords)

        # 4. weight the candidates using the sum of their word's scores that are
        #    computed using random walk. In the graph, nodes are words of
        #    certain part-of-speech (nouns and adjectives) that are connected if
        #    they occur in a window of 10 words.
        extractor.candidate_weighting(window=10)

        # 5. get the 10-highest scored candidates as keyphrases
        keyphrases = extractor.get_n_best(n=10)

    """

    def __init__(self):
        """Redefining initializer for SingleRank."""
        super(NodeDegree, self).__init__()

    def candidate_weighting(self, window=10, weights=True):
        """Keyphrase candidate ranking using the node degree.

        Args:
            window (int): the window within the sentence for connecting two
                words in the graph, defaults to 10.
            weights (int): whether or not to consider edge weights, defaults to True.
        """
        # build the word graph
        self.build_word_graph_all(window=window, weights=weights)

        w = {}
        if weights:
            # compute the word scores using random walk
            w = self.graph.degree(weight='weight')
        else:
            # compute the word scores using random walk
            w = self.graph.degree(weight=None)

        w_sorted = sorted(dict(w).items(),key=operator.itemgetter(1),reverse=True)

        return w_sorted
