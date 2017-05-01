# -*- coding: utf-8 -*-

"""
University of São Paulo
Mathematics and Statistics Institute
Course: MAC0460_5832 - Machine Learning - 2017/1

Student: Juliana Cavalcanti Correa
Assignment: EP1 - W-operators design
"""

from mac0460_5832.utils import *


def pattern_hash(pattern):
    """
    Receives n-dim numpy array and converts it to a 1-dim tuple (hashable)
    """
    return tuple(pattern.flatten())


class structuring_element:
    """
    structure for the se (cross, rectangle, etc) containing its mask and the
    size of its borders
    """
    def __init__(self, se_mask):
        self.mask = se_mask
        self.border = ( (se_mask.shape[0]-1)/2, (se_mask.shape[1]-1)/2 )


class w_operator:

    def __init__(self, se_mask=None, trainingdata=[]):
        if se_mask is not None:
            self.struct_elem = structuring_element(se_mask)
        self.trainingdata = trainingdata
        self.freqtable = {}
        self.operator = []
        self.ein = []
        self.eout = []


    def add_training_example(self, srcimg, destimg):
        """
        scans a new example and adds its data to the frequency table
        """
        self.trainingdata.append((srcimg, destimg))
        if self.struct_elem is not None:
            self.scan_example(srcimg, destimg)


    def slide_window(self, src, i, j):
        """
        Returns the pattern resulting from the structuring element mask
        centered at position (i,j) of the source img
        """
        (bi, bj) = self.struct_elem.border # half-window size
        window = src[i-bi:i+bi+1, j-bj:j+bj+1]
        return np.logical_and(window, self.struct_elem.mask)


    def add_to_freqtable(self, pattern, result):
        pattern = pattern_hash(pattern)
        if not pattern in self.freqtable:
            self.freqtable[pattern] = { True : 0, False : 0 }
        self.freqtable[pattern][result] += 1


    def scan_example(self, src, target):
        """
        Slides a window with the structuring element as a mask through the src
        image and returns a frequency table which serves as an estimator for
        P(X | pattern) where X is the value of the corresponding position (i,j),
        in the target image, of the center of the window
        """
        bi, bj = self.struct_elem.border # half-window size
        for i in range(bi, src.shape[0]-bi):
            for j in range(bj, src.shape[1]-bj):
                pattern = self.slide_window(src, i, j)
                self.add_to_freqtable(pattern, target[i, j])


    def optimal_decision(self, pattern):
        """
        Returns the value that minimizes MAE for this pattern considering the
        observations given by the frequency table
        """
        return self.freqtable[pattern][True] > self.freqtable[pattern][False]


    def learn(self):
        """
        Builds the operator, which consists of a list of patterns for which the
        output is estimated to be True
        """
        if self.trainingdata is not None and len(self.freqtable) == 0:
            self.train()
        if len(self.freqtable) > 0:
            patterns = self.freqtable.keys()
            self.operator = filter(lambda x: self.optimal_decision(x), patterns)


    def train(self):
        """
        Scan all examples in the training data and build the full freq table
        Not needed if examples were added one by one with add_training_example,
        because, in that case, freq table is built while examples are added
        """
        for (src, target) in self.trainingdata:
            self.scan_example(src, target)


    def apply_operator(self, src):
        """
        Generates and returns the output image by applying the operator to src image
        """
        target = np.zeros_like(src, dtype=bool)
        bi, bj = self.struct_elem.border # half-window size
        for i in range(bi, src.shape[0]-bi):
            for j in range(bj, src.shape[1]-bj):
                if pattern_hash(self.slide_window(src, i, j)) in self.operator:
                    target[i,j] = True
        return target
