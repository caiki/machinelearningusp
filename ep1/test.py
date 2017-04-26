import matplotlib.pyplot as plt
import numpy as np
import PIL

from mac0460_5832.utils import *
from mac0460_5832.ep1 import *
import mac0460_5832.q1 as q1
import mac0460_5832.q2 as q2


### Q1

src1 = read_img('images/q1/1_src.png')
dest1 = read_img('images/q1/1_dest.png')

trainingdata = [(src1,dest1)]
operator = generate_operator(q1.build_pattern_freqs(trainingdata))
for pattern in operator:
    print pattern

test1_1 = read_img('images/q1/1_test1.png')
test_output1 = q1.apply_operator(test1_1, operator)
draw_img_pair(test1_1, test_output1)

test1_2 = read_img('images/q1/1_test2.png')
test_output2 = q1.apply_operator(test1_2, operator)
draw_img_pair(test1_2, test_output2)

### Q2

src2_1 = read_img_v2('images/q2/q2_src1.png')
src2_2 = read_img_v2('images/q2/q2_src2.png')
src2_3 = read_img_v2('images/q2/q2_src3.png')
dest2_1 = read_img_v2('images/q2/q2_dest1.png')
dest2_2 = read_img_v2('images/q2/q2_dest2.png')
dest2_3 = read_img_v2('images/q2/q2_dest3.png')

trainingdata = [(src2_1, dest2_1),(src2_2, dest2_2),(src2_3, dest2_3)]
operator = generate_operator(q2.build_pattern_freqs(trainingdata))
for pattern in operator:
    print np.array(pattern).reshape(3,3)

imgt1 = read_img("images/q2/q2_test.png")
t4 = q2.apply_operator(imgt1, operator)
draw_img_pair(imgt1, t4)

imgt2 = read_img("images/q2/q2_test2.png")
t5 = q2.apply_operator(imgt2, operator)
draw_img_pair(imgt2, t5)
