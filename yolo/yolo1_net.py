# coding=utf-8

import voc.config as cfg
import tensorflow as tf


class Yolo_v1():
    def __init__(self):
        self.batch_size = cfg.batch_size

    def add_conv_layer(self, index, input, shape, stddev=0.0001, biases=0.0, strides=1, padding='SAME', action='relu'):
        with tf.variable_scope('layer{0}_conv'.format(index)) as scope:
            weight = tf.get_variable('weight', shape, initializer=tf.truncated_normal_initializer(stddev=stddev))
            biases = tf.get_variable('biases', shape[3], initializer=tf.constant_initializer(biases))

            conv = tf.nn.conv2d(input, weight, strides=[1, strides, strides, 1], padding=padding)
            with_biases = tf.nn.bias_add(conv, biases)
            if action == 'relu':
                relu = tf.nn.relu(with_biases)
                return relu
            else:
                return with_biases

    def add_pooling_layer(self, index, input, size=2, step=2, padding='SAME'):
        with tf.variable_scope('layer{0}_pool'.format(index)) as scope:
            return tf.nn.max_pool(input, ksize=[1, size, size, 1], strides=[1, step, step, 1], padding=padding)

    def add_fc_layer(self, index, input, shape, stddev=0.001, biases=0.0, action='relu'):
        with tf.variable_scope('layer{0}_fc'.format(index)) as scope:
            weight = tf.get_variable('weight', shape, initializer=tf.truncated_normal_initializer(stddev=stddev))
            biases = tf.get_variable('biases', shape[1], initializer=tf.constant_initializer(biases))

            matmul = tf.matmul(input, weight) + biases
            if action == 'relu':
                return tf.nn.relu(matmul)
            else:
                return matmul

    def inference(self, input):

        layer1 = self.add_conv_layer(1, input, [7, 7, 3, 64], strides=2, padding='VALID')
        layer2 = self.add_pooling_layer(2, layer1)
        layer3 = self.add_conv_layer(3, layer2, [3, 3, 64, 192])
        layer4 = self.add_pooling_layer(4, layer3)
        layer5 = self.add_conv_layer(5, layer4, [1, 1, 192, 128])
        layer6 = self.add_conv_layer(6, layer5, [3, 3, 128, 256])
        layer7 = self.add_conv_layer(7, layer6, [1, 1, 256, 256])
        layer8 = self.add_conv_layer(8, layer7, [3, 3, 256, 512])
        layer9 = self.add_pooling_layer(9, layer8)

        layer10 = self.add_conv_layer(10, layer9, [1, 1, 512, 256])
        layer11 = self.add_conv_layer(11, layer10, [3, 3, 256, 512])

        layer12 = self.add_conv_layer(12, layer9, [1, 1, 512, 256])
        layer13 = self.add_conv_layer(13, layer12, [3, 3, 256, 512])

        layer14 = self.add_conv_layer(14, layer9, [1, 1, 512, 256])
        layer15 = self.add_conv_layer(15, layer14, [3, 3, 256, 512])

        layer16 = self.add_conv_layer(16, layer9, [1, 1, 512, 256])
        layer17 = self.add_conv_layer(17, layer16, [3, 3, 256, 512])

        layer18 = tf.concat([layer11, layer13, layer15, layer17], axis=3)

        layer19 = self.add_conv_layer(19, layer18, [1, 1, 4 * 512, 512])
        layer20 = self.add_conv_layer(20, layer19, [3, 3, 512, 1024])

        layer21 = self.add_pooling_layer(21, layer20)

        layer22 = self.add_conv_layer(22, layer21, [1, 1, 1024, 512])
        layer23 = self.add_conv_layer(23, layer22, [3, 3, 512, 1024])

        layer24 = self.add_conv_layer(24, layer21, [1, 1, 1024, 512])
        layer25 = self.add_conv_layer(25, layer24, [3, 3, 512, 1024])

        layer26 = self.add_conv_layer(26, layer21, [1, 1, 1024, 512])
        layer27 = self.add_conv_layer(27, layer26, [3, 3, 512, 1024])

        layer28 = self.add_conv_layer(28, layer21, [1, 1, 1024, 512])
        layer29 = self.add_conv_layer(29, layer28, [3, 3, 512, 1024])

        layer30 = tf.concat([layer23, layer25, layer27, layer29], axis=3)

        layer31 = self.add_conv_layer(31, layer30, [3, 3, 1024 * 4, 1024])
        layer32 = self.add_conv_layer(32, layer31, [3, 3, 1024, 1024], strides=2)
        layer33 = self.add_conv_layer(33, layer32, [3, 3, 1024, 1024])
        layer34 = self.add_conv_layer(34, layer33, [3, 3, 1024, 1024])

        shape = layer34.get_shape().as_list()
        node = shape[1] * shape[2] * shape[3]
        reshaped = tf.reshape(layer34, [shape[0], node])

        layer35 = self.add_fc_layer(35, reshaped, [node, 4096])
        layer36 = self.add_fc_layer(36, layer35, [4096, 7 * 7 * 25], action='')

        return layer36

    def calu_loss(self, predict, label):
        """
        Calculate the step loss after the inference
        :param predict:
        :param label:
        :return:
        """

        # reshape the predict & label to [batch_size * cell_size * size , (pc_score,boxes,class)]
        predict_reshaped = tf.reshape(predict, [self.batch_size * 7 * 7, 25])
        label_reshaped = tf.reshape(label, [self.batch_size * 7 * 7, 25])

        p_split_score, p_split_box, p_spilt_class = tf.split(predict_reshaped, [1, 4, 20], axis=1)
        l_split_score, l_split_box, l_split_class = tf.split(label_reshaped, [1, 4, 20], axis=1)

        max_index = tf.image.non_max_suppression(
            self.convert_box_from_1dim_to_2dim(p_split_box), tf.reshape(p_split_score, [self.batch_size * 7 * 7]), 1)

        predict_boxes = tf.gather(p_split_box, max_index)
        label_boxes = tf.gather(l_split_box, max_index)

        # calculate the center point loss
        center_loss = 5 * tf.reduce_mean(
            tf.square(predict_boxes[..., 0] - label_boxes[..., 0]) + tf.square(predict_boxes[..., 1] - label_boxes[..., 1]))
        # calculate the boxes loss
        bounding_loss = 5 * tf.reduce_mean(tf.square(tf.sqrt(predict_boxes[..., 2]) - tf.sqrt(label_boxes[..., 2])) + tf.square(
            tf.sqrt(predict_boxes[..., 2]) - tf.sqrt(label_boxes[..., 2])))


        # get the eval of max-index
        # with tf.Session() as sess:
        #     max_index_arr = sess.run(max_index)
        #     for i in  max_index_arr:
        #         print i
        return center_loss + bounding_loss



    def convert_box_from_1dim_to_2dim(self, boxes):
        boxes_t = tf.stack([boxes[..., 0] - boxes[..., 2] / 2.0,
                            boxes[..., 1] - boxes[..., 3] / 2.0,
                            boxes[..., 0] + boxes[..., 2] / 2.0,
                            boxes[..., 1] + boxes[..., 3] / 2.0],
                           axis=-1)
        return boxes_t
