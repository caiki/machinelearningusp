# -*- coding: utf-8 -*-
"""
unit tests for ep1.py
"""

import numpy as np
import unittest
from ep1 import *
from mac0460_5832.utils import *

class TestEP(unittest.TestCase):

    def setUp(self):
        onescol = np.ones((4,1), dtype=bool)
        zeroscol = np.zeros((4,1), dtype=bool)
        self.img = np.concatenate((onescol,zeroscol,onescol,zeroscol,onescol), axis=1)
        self.mask_horizontal = np.ones((1,3), dtype=bool)
        self.mask_square =  np.ones((3,3), dtype=bool)
        self.mask_cross = se_cross(1)
        self.trainingdata = [(self.img, self.img),(self.img.T, self.img.T)]

    def test_pattern_hash(self):
        result = pattern_hash(np.ones((3,3), dtype=bool))
        expected = tuple([True for i in range(9)])
        self.assertEquals(expected, result)

    def test_mask_borders_horizontal(self):
        se = structuring_element(self.mask_horizontal)
        self.assertEquals(0, se.border[0])
        self.assertEquals(1, se.border[1])

    def test_mask_borders_square(self):
        se = structuring_element(self.mask_square)
        self.assertEquals(1, se.border[0])
        self.assertEquals(1, se.border[1])

    def test_mask_borders_cross(self):
        se = structuring_element(self.mask_cross)
        self.assertEquals(1, se.border[0])
        self.assertEquals(1, se.border[1])

    def test_initiate_empty(self):
        psi = w_operator()
        psi.struct_elem = structuring_element(self.mask_square)
        psi.add_training_example(self.img, self.img)
        self.assertEquals(pattern_hash(psi.se.mask), pattern_hash(self.mask_square))
        self.assertTrue(len(psi.freqtable) > 0)

    def test_initiate_mask(self):
        psi = w_operator(self.mask_horizontal)
        psi.add_training_example(self.img, self.img)
        self.assertEquals(pattern_hash(psi.se.mask), pattern_hash(self.mask_horizontal))
        self.assertTrue(len(psi.freqtable) > 0)

    def test_initiate_mask_and_data(self):
        trainingdata = [(self.img, self.img), (self.img.T, self.img.T)]
        psi = w_operator(self.mask_cross, trainingdata)
        self.assertEquals(pattern_hash(psi.se.mask), pattern_hash(self.mask_cross))
        self.assertTrue(len(psi.freqtable) == 0)

    def test_slide_window_horizontal(self):
        psi = w_operator(self.mask_horizontal)
        result = psi.slide_window(self.img, 0, 1)
        expected = np.array([[True, False, True]])
        self.assertEquals(pattern_hash(expected), pattern_hash(result))

    def test_slide_window_square(self):
        psi = w_operator(self.mask_square)
        result = psi.slide_window(self.img, 1, 3)
        expected = np.array([[True, False, True],
                             [True, False, True],
                             [True, False, True]])
        self.assertEquals(pattern_hash(expected), pattern_hash(result))

    def test_slide_window_cross(self):
        psi = w_operator(self.mask_cross)
        result = psi.slide_window(self.img, 2, 2)
        expected = np.array([[False, True, False],
                            [False, True, False],
                            [False, True, False]])
        self.assertEquals(pattern_hash(expected), pattern_hash(result))

    def test_train(self):
        psi = w_operator(self.mask_horizontal, self.trainingdata)
        psi.train()
        v111 = (True,True,True)
        v010 = (False,True,False)
        v000 = (False,False,False)
        self.assertTrue(psi.freqtable[v111][True] > 0)
        self.assertEquals(psi.freqtable[v111][False], 0)
        self.assertTrue(psi.freqtable[v010][True] > 0)
        self.assertEquals(psi.freqtable[v010][False], 0)
        self.assertTrue(psi.freqtable[v000][False] > 0)
        self.assertEquals(psi.freqtable[v000][True], 0)

    def test_learn(self):
        psi = w_operator(self.mask_horizontal, self.trainingdata)
        psi.learn()
        self.assertTrue ((True,True,True) in psi.operator)
        self.assertTrue ((False,True,False) in psi.operator)
        self.assertFalse ((False,False,False) in psi.operator)

    def test_img_dist(self):
        a = np.array([[1,1,1],[2,2,2],[3,3,3],[4,4,4]])
        b = np.array([[1,0,1],[0,2,2],[3,3,0],[4,4,4]])
        self.assertAlmostEquals(0.25, img_dist(a,b))

if __name__ == '__main__':
    unittest.main()
