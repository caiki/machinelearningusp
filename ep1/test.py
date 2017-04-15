import matplotlib.pyplot as plt
import numpy as np
import PIL
from mac0460_5832.utils import *
from mac0460_5832.answer import *

src1 = read_img('images/q1/1_src.png')
dest1 = read_img('images/q1/1_dest.png')

#draw_img_pair(src1, dest1)

f = estimatePatternResults(src1, dest1)
for pattern in f:
    print pattern, predictPatternOutput(pattern, f)
