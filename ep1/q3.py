# -*- coding: utf-8 -*-

"""
University of São Paulo
Mathematics and Statistics Institute
Course: MAC0460_5832 - Machine Learning - 2017/1

Student: Juliana Cavalcanti Correa
Assignment: EP1, question 2
Goal: Design a 3x3 W-operator for border detection by learning from a binary image
"""

import numpy as np
from ep1 import *


def build_pattern_freqs(trainingdata):
    """
    For each image, slides the window W through the src image and counts the
    number of times each 3x3 pattern shows up with the corresponding value
    (True or False) for the center element in the target image

    trainingdata is a list where each entry is a tuple (input img, output img),
    with images being represented as numpy arrays

    Returns a frequency table which serves as an estimator for
    P(X | pattern) where X in (0, 1)
    """

    freqtable = {}

    for imgpair in trainingdata:
        src = imgpair[0]
        target = imgpair[1]

        for i in range(1, src.shape[0]-1):
            for j in range(1, src.shape[1]-1):
                pattern = np.array( src[i-1, j],
                                    src[i, j-1:j+2],
                                    src[i+2, j] )
                result = target[i, j]
                add_to_freqtable(pattern, result, freqtable)

    return freqtable


def apply_operator(src, operator):
    """
    Generates and returns the output image by applying operator to src image
    """
    target = np.zeros_like(src, dtype=bool)
    for i in range(1, src.shape[0]-1):
        for j in range(1, src.shape[1]-1):
            pattern = pattern_hash(src[i-1:i+2, j-1:j+2])
            if pattern in operator:
                target[i,j] = True

    return target
