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

    def test_p_hash(self):
        result = p_hash(np.ones((3,3), dtype=bool))
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
        self.assertEquals(p_hash(psi.se.mask), p_hash(self.mask_square))
        self.assertTrue(len(psi.freqtable) > 0)

    def test_initiate_mask(self):
        psi = w_operator(self.mask_horizontal)
        psi.add_training_example(self.img, self.img)
        self.assertEquals(p_hash(psi.se.mask), p_hash(self.mask_horizontal))
        self.assertTrue(len(psi.freqtable) > 0)

    def test_initiate_mask_and_data(self):
        trainingdata = [(self.img, self.img), (self.img.T, self.img.T)]
        psi = w_operator(self.mask_cross, trainingdata)
        self.assertEquals(p_hash(psi.se.mask), p_hash(self.mask_cross))
        self.assertTrue(len(psi.freqtable) > 0)

    def test_slide_window_horizontal(self):
        psi = w_operator(self.mask_horizontal)
        result = psi.slide_window(self.img, 0, 1)
        expected = np.array([[True, False, True]])
        self.assertEquals(p_hash(expected), p_hash(result))

    def test_slide_window_square(self):
        psi = w_operator(self.mask_square)
        result = psi.slide_window(self.img, 1, 3)
        expected = np.array([[True, False, True],
                             [True, False, True],
                             [True, False, True]])
        self.assertEquals(p_hash(expected), p_hash(result))

    def test_slide_window_cross(self):
        psi = w_operator(self.mask_cross)
        result = psi.slide_window(self.img, 2, 2)
        expected = np.array([[False, True, False],
                            [False, True, False],
                            [False, True, False]])
        self.assertEquals(p_hash(expected), p_hash(result))

    def test_train(self):
        psi = w_operator(self.mask_horizontal, self.trainingdata)
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
        self.assertTrue(psi.operator[(True,True,True)])
        self.assertTrue(psi.operator[(False,True,False)])
        self.assertFalse(psi.operator[(False,False,False)])
        self.assertFalse(psi.operator.get((False,False,True))) # not seen = false


    def test_img_dist(self):
        a = np.array([[1,1,1],[2,2,2],[3,3,3],[4,4,4]])
        b = np.array([[1,0,1],[0,2,2],[3,3,0],[4,4,4]])
        self.assertAlmostEquals(0.25, img_dist(a,b))

    def test_init_multires(self):
        m = multiresolution([self.mask_square, self.mask_horizontal], self.trainingdata)
        self.assertEquals(p_hash(m.operators[0].se.mask), p_hash(self.mask_square))
        self.assertEquals(p_hash(m.operators[1].se.mask), p_hash(self.mask_horizontal))
        p33 = p_hash(self.img[0:3,0:3])
        p13 = p_hash((self.img[0,0:3]))
        self.assertTrue(m.operators[0].operator.has_key(p33))
        self.assertFalse(m.operators[0].operator.has_key(p13))
        self.assertTrue(m.operators[1].operator.has_key(p13))
        self.assertFalse(m.operators[1].operator.has_key(p33))

    def test_pyramid_match(self):
        m = multiresolution([self.mask_square, self.mask_horizontal], self.trainingdata)
        self.assertFalse(m.pyramid_match(self.img, 1,1))
        self.assertTrue(m.pyramid_match(self.img, 2,2))

if __name__ == '__main__':
    unittest.main()
