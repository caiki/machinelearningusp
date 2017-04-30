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

    def test_initiate_class(self):
        psi = w_operator(self.mask_cross)
        psi.add_training_example(self.img, self.img)

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

    def test_build_freqtable(self):
        psi = w_operator(self.mask_horizontal)
        psi.add_training_example(self.img, self.img)
        psi.add_training_example(self.img.T, self.img.T)
        psi.build_pattern_freqs()
        v111 = (True,True,True)
        v010 = (False,True,False)
        v000 = (False,False,False)
        self.assertTrue(psi.freqtable[v111][True] > 0)
        self.assertEquals(psi.freqtable[v111][False], 0)
        self.assertTrue(psi.freqtable[v010][True] > 0)
        self.assertEquals(psi.freqtable[v010][False], 0)
        self.assertTrue(psi.freqtable[v000][False] > 0)
        self.assertEquals(psi.freqtable[v000][True], 0)

    def test_generate_operator(self):
        psi = w_operator(self.mask_horizontal)
        psi.add_training_example(self.img, self.img)
        psi.add_training_example(self.img.T, self.img.T)
        psi.build_pattern_freqs()
        psi.generate_operator()
        self.assertTrue ((True,True,True) in psi.operator)
        self.assertTrue ((False,True,False) in psi.operator)
        self.assertFalse ((False,False,False) in psi.operator)


if __name__ == '__main__':
    unittest.main()
