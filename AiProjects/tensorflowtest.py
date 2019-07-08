import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf

state = tf.Variable(0)
new_value = tf.add(state,tf.constant(1))
update = tf.assign(state,new_value)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(state))
    for _ in range(3):
        sess.run(update)
        print(sess.run(state))

w = tf.Variable([[0.5,1.0]])
x = tf.Variable([[2.0],[1.0]])
y = tf.matmul(w,x)
init_op = tf.global_variables_initializer()
saver = tf.train.Saver
with tf.Session() as sess:
    sess.run(init_op)
