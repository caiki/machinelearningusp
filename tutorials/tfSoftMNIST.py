import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
# one_hot means a vector with a bunch of 0s and only one 1
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


### DEFINE THE MODEL ###

# input: 2-D tensor that represents any number of MNIST images,
# each flattened into a 784-dimensional vector
x = tf.placeholder(tf.float32, [None, 784])

# generally the model parameters are represented as Variables.
W = tf.Variable(tf.zeros([784, 10])) # 10-dim evidence for each digit class
b = tf.Variable(tf.zeros([10])) # bias nodes

y = tf.nn.softmax(tf.matmul(x, W) + b)

# defining the error/loss function
y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

# minimize cross entropy using gradient descent
train_step = tf.train.GradientDescentOptimizer(0.05).minimize(cross_entropy)


### RUN THE COMPUTATION ###

# init session and variables

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

# train
for _ in range(1000):
  # select batches from train dataset to use stochastic gradient descent
  batch_xs, batch_ys = mnist.train.next_batch(100)
  # feed_dict replaces the placeholder tensors
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})


### EVALUATE ###

# argmax gives the index of the highest entry in a tensor along some axis
# tf.argmax(y,1) is the label our model thinks is most likely for each input,
# while tf.argmax(y_,1) is the correct label.
# this will return a huge vector of booleans
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
# converts the booleans to 0 or 1 and finds the proportion of 1s
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# finds accuracy in test data
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
