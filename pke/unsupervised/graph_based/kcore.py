# -*- coding: utf-8 -*-
# Author: Eirini Papagiannopoulou
# Date: 14/10/2020

"""K-core keyword extraction model.

Model described in:

* François Rousseau and Michalis Vazirgiannis.
  Main Core Retention on Graph-of-words for Single-Document Keyword Extraction
  *In proceedings of the ECIR 2015*.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import networkx as nx

from pke.unsupervised.graph_based.textrank import TextRank

import string

import igraph


class Kcore(TextRank):
    """K-core keyword extraction model.

    Concept of k-core on the graph-of-words representation of text for single-document keyword extraction,
    retaining only the nodes from the main core as representative terms.  

    Parameterized example::

        import pke

        # define the set of valid Part-of-Speeches
        pos = {'NOUN', 'PROPN', 'ADJ'}

        extractor_kcore = Kcore()
        # load the content of the document
        # the input language is set to English (used for the stoplist)
        # normalization is set to stemming (computed with Porter's stemming algorithm)
        extractor_kcore.load_document(input=doc_text, language="en", normalization='stemming')
        # select the longest sequences of nouns and adjectives as candidates.
        extractor_kcore.candidate_selection(pos=pos)
        # returns the k-core of the graph. In the graph, nodes are words of
        #    certain part-of-speech (nouns and adjectives) that are connected if
        #    they occur in a window of 4 words.
        gof = extractor_kcore.build_word_graph(window=4, pos=pos)


        results_kcore = extractor_kcore.k-core(window=4, pos=pos)

    """

    def __init__(self):
        """Redefining initializer for K-core."""

        super(Kcore, self).__init__()

        self.graph_weighted = igraph.Graph();

        self.vocab = {}

    def candidate_selection_all(self, stopwords_list):
        """Select unigrams as keyword candidates.

        Args:
            stopwords_list (list): the stoplist for filtering candidates

        """
        text = []
        for sent_id, sentence in enumerate(self.sentences):
            for i, word in enumerate(sentence.stems):
                word = word.translate(str.maketrans('', '', string.punctuation))
                text.append(word)
                if word not in stopwords_list and not str(word).isdigit() and len(str(word)) > 1:
                    if word not in self.vocab.keys():
                        self.vocab[word] = 1
                    else:
                        self.vocab[word] += 1
        return self.vocab, text

    # It uses networkx python package that implements the canonical k-core, i.e., without weights
    def build_word_graph_all(self, window=10):#, weights=True):
        """Build an unweighted graph representation of the document in which nodes/vertices
        are words and edges represent co-occurrence relation.
        Co-occurrence relations can be controlled using the distance (window)
        between word occurrences in the document.

        Args:
            window (int): the window for connecting two words in the graph,
                defaults to 10.
        """
        text = []
        # flatten document as a sequence of (word, pass_syntactic_filter) tuples
        text = [(word.translate(str.maketrans('', '', string.punctuation)), word.translate(str.maketrans('', '', string.punctuation)) in self.vocab.keys()) for sentence in self.sentences
                for i, word in enumerate(sentence.stems)]
        text_stems = [word.translate(str.maketrans('', '', string.punctuation)) for sentence in self.sentences
                for i, word in enumerate(sentence.stems)]
        text_stems = set(text_stems) 
        # add nodes to the graph
        self.graph_weighted.add_nodes_from([word for word, valid in text if valid])

        # add edges to the graph
        for i, (node1, is_in_graph1) in enumerate(text):

            # speed up things
            if not is_in_graph1:
                continue

            for j in range(i + 1, min(i + window, len(text))):
                node2, is_in_graph2 = text[j]
                if is_in_graph2 and node1 != node2:
                    # if weights:
                    #     if not self.graph.has_edge(node1, node2):
                    #         self.graph.add_edge(node1, node2, weight=0.0)
                    #     self.graph[node1][node2]['weight'] += 1.0
                    # else:
                    if not self.graph.has_edge(node1, node2):
                        self.graph.add_edge(node1, node2)
        
        return self.graph, len(text_stems)

    # It uses igraph python package that implements the weighted k-core, i.e., with weights
    def build_weighted_word_graph_all(self, window=10, weights=True):
        """Build a weighted graph representation of the document in which nodes/vertices
        are words and edges represent co-occurrence relation.
        Co-occurrence relations can be controlled using the distance (window)
        between word occurrences in the document.

        The number of times two words co-occur in a window is encoded as *edge
        weights*. Sentence boundaries **are not** taken into account in the
        window.

        Args:
            window (int): the window for connecting two words in the graph,
                defaults to 10.
        """
        text = []
        # flatten document as a sequence of (word, pass_syntactic_filter) tuples
        text = [(word.translate(str.maketrans('', '', string.punctuation)), word.translate(str.maketrans('', '', string.punctuation)) in self.vocab.keys()) for sentence in self.sentences
                for i, word in enumerate(sentence.stems)]
        text_stems = [word.translate(str.maketrans('', '', string.punctuation)) for sentence in self.sentences
                for i, word in enumerate(sentence.stems)]
        text_stems = set(text_stems) 
        # add nodes to the graph
        vertices = []
        for word, valid in text:
            if valid:
                vertices.append(word)
        vertices = list(set(vertices))

        self.graph_weighted.add_vertices(vertices)

        # add edges to the graph
        for i, (node1, is_in_graph1) in enumerate(text):

            # speed up things
            if not is_in_graph1:
                continue

            for j in range(i + 1, min(i + window, len(text))):
                node2, is_in_graph2 = text[j]
                if is_in_graph2 and node1 != node2:
                    if self.graph_weighted.get_eid(node1, node2, directed=False, error=False) == -1:
                        self.graph_weighted.add_edge(node1, node2, weight=0)
                    if weights:
                        edge_id = self.graph_weighted.get_eid(node1, node2)
                        edge = self.graph_weighted.es[edge_id]
                        edge["weight"] += 1.0
                   
        
        return self.graph_weighted, len(text_stems)


    def build_word_graph_based_on_syntax(self, window=4, pos=None):
        """Build a graph representation of the document in which nodes/vertices
        are words and edges represent co-occurrence relation. Syntactic filters
        can be applied to select only words of certain Part-of-Speech.
        Co-occurrence relations can be controlled using the distance (window)
        between word occurrences in the document.

        The number of times two words co-occur in a window is encoded as *edge
        weights*. Sentence boundaries **are not** taken into account in the
        window.

        Args:
            window (int): the window for connecting two words in the graph,
                defaults to 4.
            pos (set): the set of valid pos for words to be considered as nodes
                in the graph, defaults to ('NOUN', 'PROPN', 'ADJ').
        """

        if pos is None:
            pos = {'NOUN', 'PROPN', 'ADJ'}

        # flatten document as a sequence of (word, pass_syntactic_filter) tuples
        text = [(word, sentence.pos[i] in pos) for sentence in self.sentences
                for i, word in enumerate(sentence.stems)]
        
        # flatten document as a sequence of (word, pass_syntactic_filter) tuples
        text_stems = [word for sentence in self.sentences
                for i, word in enumerate(sentence.stems)]
        text_stems = set(text_stems)      

        # add nodes to the graph
        self.graph.add_nodes_from([word for word, valid in text if valid])

        # add edges to the graph
        for i, (node1, is_in_graph1) in enumerate(text):

            # speed up things
            if not is_in_graph1:
                continue

            for j in range(i + 1, min(i + window, len(text))):
                node2, is_in_graph2 = text[j]
                if is_in_graph2 and node1 != node2:
                    if not self.graph.has_edge(node1, node2):
                        self.graph.add_edge(node1, node2, weight=0.0)
                    self.graph[node1][node2]['weight'] += 1.0

        return self.graph, len(text_stems)
    
    def k_core(self):
        k_core_words = nx.k_core(self.graph)

        return k_core_words

    def weighted_k_core(self):
        k_core = self.graph_weighted.k_core()

        return k_core