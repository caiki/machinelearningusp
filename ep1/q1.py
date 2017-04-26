# -*- coding: utf-8 -*-

"""
University of São Paulo
Mathematics and Statistics Institute
Course: MAC0460_5832 - Machine Learning - 2017/1

Student: Juliana Cavalcanti Correa
Assignment: EP1, question 1
Goal: Design a 1x3 W-operator by learning from a binary image
"""

import numpy as np
from ep1 import *


def build_pattern_freqs(trainingdata):
    """
    Slides the window W through the src image and counts the number of times each
    1x3 pattern shows up and corresponds to a center element in the target image
    of value True or False
    Returns a frequency table which serves as an estimator for
    P(X | pattern) where X in (True,False)
    """

    freqtable = {}

    for imgpair in trainingdata:
        src = imgpair[0]
        target = imgpair[1]

        for i in range(src.shape[0]):
            for j in range(1, src.shape[1]-1): # clips the borders
                pattern = src[i, j-1:j+2]
                result = target[i, j]
                add_to_freqtable(pattern, result, freqtable)
        return freqtable


def apply_operator(src, operator):
    """
    Generates and returns the output image by applying operator to src image
    """
    target = np.zeros_like(src, dtype=bool)
    for i in range(src.shape[0]):
        for j in range(1, src.shape[1]-1):
            pattern = pattern_hash(src[i, j-1:j+2])
            if pattern in operator:
                target[i,j] = True
    return target
