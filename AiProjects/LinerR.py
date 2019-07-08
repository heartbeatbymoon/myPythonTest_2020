# encoding=utf-8

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

vectors_sets = []
for i in range(1000):
    x1 = np.random.normal(0.0, 0.55)
    y1 = x1 * 0.1 + 0.3 + np.random.normal(0.0, 0.03)
    vectors_sets.append([x1, y1])

x_data = [v[0] for v in vectors_sets]
y_data = [v[1] for v in vectors_sets]

plt.scatter(x_data, y_data, c="r")
plt.show()

W = tf.Variable(tf.random_uniform([1], -1.0, 1.0), name="W")
b = tf.Variable(tf.zeros([1]),name="b")
y = W * x_data +b
loss = tf.reduce_mean(tf.square(y-y_data),name="loss")
optimizer =  tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss,name="train")

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

print("W=",sess.run(W),"b = ",sess.run(b),"boss =",sess.run(loss))

for step in range(20):
    sess.run(train)
    print("W=", sess.run(W), "b = ", sess.run(b), "boss =", sess.run(loss))