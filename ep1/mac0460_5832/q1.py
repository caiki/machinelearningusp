# -*- coding: utf-8 -*-
"""
Universidade de São Paulo
Mathematics and Statistics Institute
Course: MAC0460_5832 - Machine Learning - 2017/1
Student: Juliana Cavalcanti Correa
Assignment: EP1, question 1
Goal: Design a 1x3 W-operator by learning from source and target image
"""

import numpy as np


def estimate_pattern_results(src, target):
    """
    Slides the window W through the src image and counts the number of times each
    1x3 pattern shows up with the corresponding value (True or False) for the
    center element in the target image
    Returns a frequency table which serves as an estimator of
    P(pattern,0) and P(pattern,1)
    """

    freqtable = {}

    for i in range(src.shape[0]):
        for j in range(1, src.shape[1]-1): # clips the borders

            pattern = tuple(src[i, j-1:j+2]) # lists are not hashable, tuples are
            result = target[i, j]

            if not pattern in freqtable:
                freqtable[pattern] = { True : 0, False : 0 }
            freqtable[pattern][result] += 1

    return freqtable


def optimal_decision(pattern, freqtable):
    """
    Returns the value that minimizes MAE for this pattern considering the
    observations given by the frequency table
    """
    return freqtable[pattern][True] > freqtable[pattern][False]


def generate_operator(freqtable):
    """
    Returns the operator (list of patterns for which the output is estimated to
    be True)
    """
    return filter(lambda x: optimal_decision(x, freqtable), freqtable.keys())


def apply_operator(src, operator):
    """
    Generates and returns the output image by applying operator to src image
    """
    target = np.zeros_like(src, dtype=bool)
    for i in range(src.shape[0]):
        for j in range(1, src.shape[1]-1):
            pattern = tuple(src[i, j-1:j+2])
            if pattern in operator:
                target[i,j] = True

    return target
