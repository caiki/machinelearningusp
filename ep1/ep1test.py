# -*- coding: utf-8 -*-

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

    def test_mask_borders_horizontal(self):
        borders = get_mask_borders(self.mask_horizontal.shape)
        self.assertEquals(0, borders[0])
        self.assertEquals(1, borders[1])

    def test_mask_borders_square(self):
        borders = get_mask_borders(self.mask_square.shape)
        self.assertEquals(1, borders[0])
        self.assertEquals(1, borders[1])

    def test_mask_borders_cross(self):
        borders = get_mask_borders(self.mask_cross.shape)
        self.assertEquals(1, borders[0])
        self.assertEquals(1, borders[1])

    def test_get_pattern_horizontal(self):
        result = get_pattern(self.img, self.mask_horizontal, 0, 1)
        expected = np.array([[True, False, True]])
        self.assertEquals(expected.all(), result.all())

    def test_get_pattern_square(self):
        result = get_pattern(self.img, self.mask_horizontal, 1, 3)
        expected = np.array([[True, False, True],
                             [True, False, True],
                             [True, False, True]])
        self.assertEquals(expected.all(), result.all())

    def test_get_pattern_cross(self):
        result = get_pattern(self.img, self.mask_cross, 2, 2)
        expected = np.array([[False, True, False],
                            [False, True, False],
                            [False, True, False]])
        self.assertEquals(expected.all(), result.all())


    def test_pattern_hash(self):
        result = pattern_hash(np.ones((3,3), dtype=bool))
        expected = tuple([True for i in range(9)])
        self.assertEquals(expected, result)


if __name__ == '__main__':
    unittest.main()
