# -*- coding: utf-8 -*-

"""
University of São Paulo
Mathematics and Statistics Institute
Course: MAC0460_5832 - Machine Learning - 2017/1

Student: Juliana Cavalcanti Correa
Assignment: EP1 - W-operators design
"""

from mac0460_5832.utils import *


def get_mask_borders(se_mask_shape):
    """
    Returns the borders to be cropped from the image
    Mask shape must consist of a pair of odd numbers
    """
    return ( (se_mask_shape[0]-1)/2, (se_mask_shape[1]-1)/2 )


def get_pattern(src, se_mask, i, j):
    """
    Returns the pattern resulting from the structuring element mask
    centered at position (i,j) of the source img
    """
    border = get_mask_borders(se_mask.shape)
    top = i-border[0]
    bottom = i+border[0]
    left = j-border[1]
    right = j+border[1]
    window = src[top:bottom+1, left:right+1]
    return np.logical_and(window, se_mask)


def pattern_hash(pattern):
    """
    Receives n-dim numpy array and converts it to a 1-dim tuple (hashable)
    """
    return tuple(pattern.flatten())


def add_to_freqtable(pattern, result, freqtable):
    pattern = pattern_hash(pattern)
    if not pattern in freqtable:
        freqtable[pattern] = { True : 0, False : 0 }
    freqtable[pattern][result] += 1


def build_pattern_freqs(trainingdata, se_mask):
    """
    Slides the window W through the src image and counts the number of times each
    1x3 pattern shows up and corresponds to a center element in the target image
    of value True or False
    Returns a frequency table which serves as an estimator for
    P(X | pattern) where X in (True,False)
    """

    freqtable = {}
    border = get_mask_borders(se_mask.shape)

    for imgpair in trainingdata:
        src = imgpair[0]
        target = imgpair[1]
        
        for i in range(border[0], src.shape[0]-border[0]):
            for j in range(border[1], src.shape[1]-border[1]):
                pattern = get_pattern(src, se_mask, i, j)
                result = target[i, j]
                add_to_freqtable(pattern, result, freqtable)

    return freqtable


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
