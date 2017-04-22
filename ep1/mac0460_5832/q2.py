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
from q1 import optimal_decision


def estimate_pattern_results(trainingdata):
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

                p = src[i-1:i+2, j-1:j+2].reshape(9)
                pattern = tuple(p)
                result = target[i, j]

                if not pattern in freqtable:
                    freqtable[pattern] = { True : 0, False : 0 }
                freqtable[pattern][result] += 1

    return freqtable


def generate_operator(freqtable):
    """
    Returns the operator (list of patterns for which the output is estimated to
    be True)
    """
    return filter(lambda x: optimal_decision(x, freqtable), freqtable.keys())
    #return map(lambda x: np.array(x).reshape(3,3), f)



def apply_operator(src, operator):
    """
    Generates and returns the output image by applying operator to src image
    """
    target = np.zeros_like(src, dtype=bool)
    for i in range(1, src.shape[0]-1):
        for j in range(1, src.shape[1]-1):
            pattern = tuple(src[i-1:i+2, j-1:j+2].reshape(9))
            #pattern = src[i-1:i+2, j-1:j+2]
            if pattern in operator:
                target[i,j] = True

    return target
