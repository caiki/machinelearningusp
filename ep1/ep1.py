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
    receives n-dim numpy array and converts it to a 1-dim tuple (hashable)
    """
    return tuple(pattern.flatten())


def img_dist(img1, img2):
    """
    returns the percentage of pixels that are different between img1 and img2
    both images must be bidimensional numpy arrays with the same shape
    """
    correctpixels = sum(sum(img1 == img2))
    totalpixels = img1.shape[0] * img1.shape[1]
    return  1 - (float(correctpixels) / totalpixels)


def mean_dist(imglist):
    """
    finds the mean error (distance) in a list containing pairs of images
    """
    dist = [img_dist(x[0],x[1]) for x in imglist]
    return float(sum(dist)) / len(dist)


class structuring_element:
    """
    structuring element containing its mask and the size of its borders
    """
    def __init__(self, se_mask):
        self.mask = se_mask
        self.border = ( (se_mask.shape[0]-1)/2, (se_mask.shape[1]-1)/2 )

    def imgborders (self, img):
        bi, bj =  self.border
        return ( bi, img.shape[0]-bi, bj, img.shape[1]-bj )

    def cropborders (self, img):
        top, bottom, left, rigth = self.imgborders(img)
        return img[top:bottom,left:rigth]


class w_operator:

    def __init__(self, se_mask=se_box(1), trainingdata=[]):
        self.se = structuring_element(se_mask)
        self.trainingdata = trainingdata
        self.freqtable = {}
        self.operator = []


    def add_training_example(self, srcimg, destimg):
        """
        scans a new example and adds its data to the frequency table
        """
        self.trainingdata.append((srcimg, destimg))
        if self.se is not None:
            self.scan_example(srcimg, destimg)
            self.update_model()


    def slide_window(self, src, i, j):
        """
        returns the pattern resulting from centering the mask window with the
        structuring element at position (i,j) of the source img
        """
        (bi, bj) = self.se.border # half-window size
        window = src[i-bi:i+bi+1, j-bj:j+bj+1]
        return np.logical_and(window, self.se.mask)


    def add_to_freqtable(self, pattern, result):
        pattern = pattern_hash(pattern)
        if not pattern in self.freqtable:
            self.freqtable[pattern] = { True : 0, False : 0 }
        self.freqtable[pattern][result] += 1


    def scan_example(self, src, target):
        """
        slides a window with the se through the src img and builds a frequency
        table for estimating P(X | pattern), where X corresponds to the value of
        position (i,j) in the target image
        """
        top, bottom, left, rigth = self.se.imgborders(src)
        for i in range(top, bottom):
            for j in range(left, rigth):
                pattern = self.slide_window(src, i, j)
                self.add_to_freqtable(pattern, target[i, j])


    def train(self):
        """
        scan all examples in the training data and build the full freq table
        Not needed if examples were added one by one with add_training_example
        """
        for (src, target) in self.trainingdata:
            self.scan_example(src, target)


    def optimal_decision(self, pattern):
        """
        returns the value that minimizes MAE for this pattern considering the
        observations given by the frequency table
        """
        freq = self.freqtable.get(pattern)
        return freq and freq[True] > freq[False]


    def update_model(self):
        """
        generates an operator based on the freq table
        """
        if len(self.freqtable) > 0:
            self.operator = filter(lambda x: self.optimal_decision(x), self.freqtable)


    def learn(self):
        """
        builds the operator, which consists of a list of patterns for which the
        output is estimated to be True
        """
        if self.trainingdata is not None and len(self.freqtable) == 0:
            self.train()
        self.update_model()


    def error(self, imgpairs):
        """
        returns the mean distance between target images and output images
        generated with current operator
        """
        self.update_model()
        v = []
        for (srcimg, targetimg) in imgpairs:
            target = self.se.cropborders(targetimg)
            output = self.se.cropborders(self.apply_operator(srcimg))
            v.append((target, output))
        return mean_dist(v)


    def error_in_sample(self):
        return self.error(self.trainingdata)


    def apply_operator(self, src):
        """
        generates and returns the output image by applying the operator to src image
        """
        target = np.zeros_like(src, dtype=bool)
        top, bottom, left, rigth = self.se.imgborders(src)
        for i in range(top, bottom):
            for j in range(left, rigth):
                pattern = pattern_hash(self.slide_window(src, i, j))
                #if pattern_hash(self.slide_window(src, i, j)) in self.operator:
                if self.optimal_decision(pattern) :
                    target[i,j] = True
        return target
