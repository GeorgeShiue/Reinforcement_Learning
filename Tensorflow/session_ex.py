import tensorflow as tf
tf.compat.v1.disable_eager_execution()

a = tf.multiply(2, 3)

with tf.compat.v1.Session() as sess:
    print(sess.run(a))