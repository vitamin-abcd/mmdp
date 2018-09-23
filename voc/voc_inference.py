# coding=utf-8

import tensorflow as tf

# mini-batch 大小
BATCH_SIZE = 128


def inference(input):
    """
    向前传播

    :param input:
    :return:
    """
    # 500 -> 248
    with tf.variable_scope('layer1:conv') as scope:
        layer1_weight = tf.get_variable('layer1_weight', [5, 5, 3, 64], dtype=tf.float64,
                                        initializer=tf.truncated_normal_initializer(stddev=0.000008))
        layer1_biases = tf.get_variable('layer1_biases', [64], dtype=tf.float64,
                                        initializer=tf.constant_initializer(0.0))

        layer1_conv = tf.nn.conv2d(input, layer1_weight, strides=[1, 2, 2, 1], padding='VALID')
        layer1_relu = tf.nn.relu(tf.nn.bias_add(layer1_conv, layer1_biases), name=scope.name)
    # 248 -> 124
    with tf.variable_scope('layer2_pool'):
        layer2_pool = tf.nn.max_pool(layer1_relu, strides=[1, 2, 2, 1], ksize=[1, 2, 2, 1], padding='SAME')

    with tf.variable_scope('layer3:conv') as scope:
        layer3_weight = tf.get_variable('layer3_weight', [3, 3, 64, 192], dtype=tf.float64,
                                        initializer=tf.truncated_normal_initializer(stddev=0.00013))
        layer3_biases = tf.get_variable('layer3_biases', [192], dtype=tf.float64,
                                        initializer=tf.constant_initializer(0.001))

        layer3_conv = tf.nn.conv2d(layer2_pool, layer3_weight, strides=[1, 1, 1, 1], padding='SAME')
        layer3_relu = tf.nn.relu(tf.nn.bias_add(layer3_conv, layer3_biases), name=scope.name)
    # 124 -> 62
    with tf.variable_scope('layer4_pool'):
        layer4_pool = tf.nn.max_pool(layer3_relu, strides=[1, 2, 2, 1], ksize=[1, 2, 2, 1], padding='SAME')

    with tf.variable_scope('layer5:conv') as scope:
        layer5_weight = tf.get_variable('layer5_weight', [1, 1, 192, 128], dtype=tf.float64,
                                        initializer=tf.truncated_normal_initializer(stddev=0.000512))
        layer5_biases = tf.get_variable('layer5_biases', [128], dtype=tf.float64,
                                        initializer=tf.constant_initializer(0.0))

        layer5_conv = tf.nn.conv2d(layer4_pool, layer5_weight, strides=[1, 1, 1, 1], padding='SAME')
        layer5_relu = tf.nn.relu(tf.nn.bias_add(layer5_conv, layer5_biases), name=scope.name)
    with tf.variable_scope('layer6:conv') as scope:
        layer6_weight = tf.get_variable('layer6_weight', [3, 3, 128, 256], dtype=tf.float64,
                                        initializer=tf.truncated_normal_initializer(stddev=0.000512))
        layer6_biases = tf.get_variable('layer6_biases', [256], dtype=tf.float64,
                                        initializer=tf.constant_initializer(0.0))

        layer6_conv = tf.nn.conv2d(layer5_relu, layer6_weight, strides=[1, 1, 1, 1], padding='SAME')
        layer6_relu = tf.nn.relu(tf.nn.bias_add(layer6_conv, layer6_biases), name=scope.name)
    with tf.variable_scope('layer7:conv') as scope:
        layer7_weight = tf.get_variable('layer7_weight', [1, 1, 256, 256], dtype=tf.float64,
                                        initializer=tf.truncated_normal_initializer(stddev=0.000512))
        layer7_biases = tf.get_variable('layer7_biases', [256], dtype=tf.float64,
                                        initializer=tf.constant_initializer(0.0))

        layer7_conv = tf.nn.conv2d(layer6_relu, layer7_weight, strides=[1, 1, 1, 1], padding='SAME')
        layer7_relu = tf.nn.relu(tf.nn.bias_add(layer7_conv, layer7_biases), name=scope.name)
    with tf.variable_scope('layer8:conv') as scope:
        layer8_weight = tf.get_variable('layer8_weight', [3, 3, 256, 512], dtype=tf.float64,
                                        initializer=tf.truncated_normal_initializer(stddev=0.000512))
        layer8_biases = tf.get_variable('layer8_biases', [512], dtype=tf.float64,
                                        initializer=tf.constant_initializer(0.0))

        layer8_conv = tf.nn.conv2d(layer7_relu, layer8_weight, strides=[1, 1, 1, 1], padding='SAME')
        layer8_relu = tf.nn.relu(tf.nn.bias_add(layer8_conv, layer8_biases), name=scope.name)
    # 62 -> 31
    with tf.variable_scope('layer9_pool'):
        layer9_pool = tf.nn.max_pool(layer8_relu, strides=[1, 2, 2, 1], ksize=[1, 2, 2, 1], padding='SAME')

    with tf.variable_scope('layer10:conv') as scope:
        layer10_weight = tf.get_variable('layer10_weight', [1, 1, 512, 256], dtype=tf.float64,
                                         initializer=tf.truncated_normal_initializer(stddev=0.001))
        layer10_biases = tf.get_variable('layer10_biases', [256], dtype=tf.float64,
                                         initializer=tf.constant_initializer(0.0))

        layer10_conv = tf.nn.conv2d(layer9_pool, layer10_weight, strides=[1, 1, 1, 1], padding='SAME')
        layer10_relu = tf.nn.relu(tf.nn.bias_add(layer10_conv, layer10_biases), name=scope.name)
    with tf.variable_scope('layer11:conv') as scope:
        layer11_weight = tf.get_variable('layer11_weight', [3, 3, 256, 512], dtype=tf.float64,
                                         initializer=tf.truncated_normal_initializer(stddev=0.001))
        layer11_biases = tf.get_variable('layer11_biases', [512], dtype=tf.float64,
                                         initializer=tf.constant_initializer(0.0))

        layer11_conv = tf.nn.conv2d(layer10_relu, layer11_weight, strides=[1, 1, 1, 1], padding='SAME')
        layer11_relu = tf.nn.relu(tf.nn.bias_add(layer11_conv, layer11_biases), name=scope.name)
    with tf.variable_scope('layer12:conv') as scope:
        layer12_weight = tf.get_variable('layer12_weight', [1, 1, 512, 512], dtype=tf.float64,
                                         initializer=tf.truncated_normal_initializer(stddev=0.000512))
        layer12_biases = tf.get_variable('layer12_biases', [512], dtype=tf.float64,
                                         initializer=tf.constant_initializer(0.0))

        layer12_conv = tf.nn.conv2d(layer11_relu, layer12_weight, strides=[1, 1, 1, 1], padding='SAME')
        layer12_relu = tf.nn.relu(tf.nn.bias_add(layer12_conv, layer12_biases), name=scope.name)
    with tf.variable_scope('layer13:conv') as scope:
        layer13_weight = tf.get_variable('layer13_weight', [3, 3, 512, 1024], dtype=tf.float64,
                                         initializer=tf.truncated_normal_initializer(stddev=0.000512))
        layer13_biases = tf.get_variable('layer13_biases', [1024], dtype=tf.float64,
                                         initializer=tf.constant_initializer(0.0))

        layer13_conv = tf.nn.conv2d(layer12_relu, layer13_weight, strides=[1, 1, 1, 1], padding='SAME')
        layer13_relu = tf.nn.relu(tf.nn.bias_add(layer13_conv, layer13_biases), name=scope.name)

        shape_list = layer13_relu.get_shape().as_list()
        nodes = shape_list[1] * shape_list[2] * shape_list[3]
        reshaped = tf.reshape(layer13_relu, [shape_list[0], nodes])

    with tf.variable_scope('layer14_fc') as scope:
        layer14_weight = tf.get_variable('layer14_weight', [nodes, 16384], dtype=tf.float64,
                                         initializer=tf.truncated_normal_initializer(stddev=0.000002))
        layer14_biases = tf.get_variable('layer14_biases', [16384], tf.constant_initializer(0.0))

        layer14_relu = tf.nn.relu(tf.matmul(reshaped, layer14_weight) + layer14_biases, name=scope.name)
    with tf.variable_scope('layer15_fc'):
        layer15_weight = tf.get_variable('layer15_weight', [16384, 10000], dtype=tf.float64,
                                         initializer=tf.truncated_normal_initializer(stddev=0.000012))
        layer15_biases = tf.get_variable('layer15_biases', [10000], tf.constant_initializer(0.0))

        layer15_result = tf.matmul(layer14_relu, layer15_weight) + layer15_biases

    result = tf.reshape(layer15_result, [400 * BATCH_SIZE, 25])

