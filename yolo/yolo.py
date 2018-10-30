import tensorflow as tf
import voc.voc_data as voc
import yolo1_net as net
import voc.config as config

dataset = voc.VocData(True)
yolo_net = net.Yolo_v1()

x = tf.placeholder(dtype=tf.float32, shape=[config.batch_size, 448, 448, 3], name='input')
y = tf.placeholder(dtype=tf.float32, shape=[config.batch_size, 7, 7, 25], name='label')

inf = yolo_net.inference(x)
loss = yolo_net.calu_loss(inf, y)
train_step = tf.train.GradientDescentOptimizer(0.0001).minimize(loss)

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    for i in range(10):
        input, label = dataset.get_next_batch()
        _, loss_value = sess.run([train_step, loss], feed_dict={x: input, y: label})
        print loss_value