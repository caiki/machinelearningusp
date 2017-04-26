# -*- coding: utf-8 -*-

"""
University of São Paulo
Mathematics and Statistics Institute
Course: MAC0460_5832 - Machine Learning - 2017/1

Student: Juliana Cavalcanti Correa
Assignment: EP1 - W-operators design
"""

def pattern_hash(pattern):
    """
    Receives n-dim numpy array and converts it to a 1-dim tuple (hashable)
    """
    return tuple(pattern.flatten())


def add_to_freqtable(input, result, freqtable):
    pattern = pattern_hash(input)
    if not pattern in freqtable:
        freqtable[pattern] = { True : 0, False : 0 }
    freqtable[pattern][result] += 1


def optimal_decision(pattern, freqtable):
    """
    Returns the value that minimizes MAE for this pattern considering the
    observations given by the frequency table
    """
    return freqtable[pattern][True] > freqtable[pattern][False]


def generate_operator(freqtable):
    """
    Returns the operator, which consists of a list of patterns for which the
    output is estimated to be valued True)
    """
    return filter(lambda x: optimal_decision(x, freqtable), freqtable.keys())
