import tensorflow as tf

const = tf.constant([[[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [2, 4, 6, 8, ], [1, 3, 5, 7]]])
with tf.Session() as sess:
    sp1, sp2, sp3 = tf.split(const, [1, 2, 1], axis=1)
    print sess.run([sp1, sp2, sp3])
