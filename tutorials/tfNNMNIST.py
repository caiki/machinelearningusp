import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data as mdata
mnist = mdata.read_data_sets("data", one_hot=True,
                             reshape=False, validation_size=0)


# placeholder for images
X = tf.placeholder(tf.float32, [None, 28, 28, 1])
# placeholder for labels
Y_ = tf.placeholder(tf.float32, [None, 10])

# weights that will be determined by training
W = tf.Variable(tf.zeros([784, 10]))
# bias values that will be determined by training
b = tf.Variable(tf.zeros([10]))

init = tf.initialize_all_variables()

# model using softmax as activation function
Y = tf.nn.softmax(tf.matmul(tf.reshape(X, [-1, 784]), W) + b)

# loss function
cross_entropy = -tf.reduce_sum(Y_ * tf.log(Y))

# optimization function
optimizer = tf.train.GradientDescentOptimizer(0.003)
train_step = optimizer.minimize(cross_entropy)

# % of correct answers found in batch
is_correct = tf.equal(tf.argmax(Y,1), tf.argmax(Y_,1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

## START COMPUTATION

sess = tf.Session()
sess.run(init)

test_data={X: mnist.test.images, Y_: mnist.test.labels}
e_in = []
e_out = []
entr_train = []
entr_test = []

for i in range(1000):
    # load a batch of images and correct answers
    batch_X, batch_Y = mnist.train.next_batch(100)
    train_data = {X: batch_X, Y_: batch_Y}
    sess.run(train_step, feed_dict=train_data)
    # accuracy in training data (E_in)

    if  not i%10:
        a, c = sess.run([accuracy, cross_entropy], feed_dict=train_data)
        e_in.append(a)
        entr_train.append(c)

        a,c = sess.run([accuracy, cross_entropy], feed_dict=test_data)
        e_out.append(a)
        entr_test.append(c)


plt.plot(range(1,101), e_in, 'b')
plt.plot(range(1,101), e_out, 'r')
plt.show()
