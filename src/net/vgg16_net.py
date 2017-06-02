# _*_ coding:utf-8 _*_

import tensorflow as tf

from ..config import config

"""
@author jiangwenqiang
@date 2017/05/22
"""


def create_net(input_tensor, train=True):
    print 'start to create vgg16 network...'

    def_strides = [1, 1, 1, 1]

    with tf.name_scope('input_reshape'):
        image_shaped_input = tf.reshape(input_tensor, [-1, config.IMAGE_SIZE, config.IMAGE_SIZE, config.IMAGE_CHANNELS])
        tf.summary.image('input', image_shaped_input, 10)

    conv1_1 = conv_layer(input_tensor, [3, 3, config.IMAGE_CHANNELS, 64], def_strides, 'conv1_1')
    conv1_2 = conv_layer(conv1_1, [3, 3, 64, 64], def_strides, 'conv1_2')
    pool1 = max_pool(conv1_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], name='pool1')

    conv2_1 = conv_layer(pool1, [3, 3, 64, 128], def_strides, 'conv2_1')
    conv2_2 = conv_layer(conv2_1, [3, 3, 128, 128], def_strides, 'conv2_2')
    pool2 = max_pool(conv2_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], name='pool2')

    conv3_1 = conv_layer(pool2, [3, 3, 128, 256], def_strides, 'conv3_1')
    conv3_2 = conv_layer(conv3_1, [3, 3, 256, 256], def_strides, 'conv3_2')
    conv3_3 = conv_layer(conv3_2, [3, 3, 256, 256], def_strides, 'conv3_3')
    pool3 = max_pool(conv3_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], name='pool3')

    conv4_1 = conv_layer(pool3, [3, 3, 256, 512], def_strides, 'conv4_1')
    conv4_2 = conv_layer(conv4_1, [3, 3, 512, 512], def_strides, 'conv4_2')
    conv4_3 = conv_layer(conv4_2, [3, 3, 512, 512], def_strides, 'conv4_3')
    pool4 = max_pool(conv4_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], name='pool4')

    conv5_1 = conv_layer(pool4, [3, 3, 512, 512], def_strides, 'conv5_1')
    conv5_2 = conv_layer(conv5_1, [3, 3, 512, 512], def_strides, 'conv5_2')
    conv5_3 = conv_layer(conv5_2, [3, 3, 512, 512], def_strides, 'conv5_3')
    pool5 = max_pool(conv5_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], name='pool5')

    # conv4_1 = conv_layer(pool3, [3, 3, 256, 512], def_strides, 'conv4_1')
    # conv4_2 = conv_layer(conv4_1, [3, 1, 512, 512], def_strides, 'conv4_2')
    # conv4_3 = conv_layer(conv4_2, [1, 3, 512, 512], def_strides, 'conv4_3')
    # conv4_4 = conv_layer(conv4_3, [3, 1, 512, 512], def_strides, 'conv4_5')
    # conv4_5 = conv_layer(conv4_4, [1, 3, 512, 512], def_strides, 'conv4_4')
    # pool4 = max_pool(conv4_5, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], name='pool4')
    #
    # conv5_1 = conv_layer(pool4, [3, 1, 512, 512], def_strides, 'conv5_1')
    # conv5_2 = conv_layer(conv5_1, [1, 3, 512, 512], def_strides, 'conv5_2')
    # conv5_3 = conv_layer(conv5_2, [3, 1, 512, 512], def_strides, 'conv5_3')
    # conv5_4 = conv_layer(conv5_3, [1, 3, 512, 512], def_strides, 'conv5_4')
    # conv5_5 = conv_layer(conv5_4, [3, 3, 512, 512], def_strides, 'conv5_5')
    # pool5 = max_pool(conv5_5, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], name='pool5')

    pool_shape = pool5.get_shape().as_list()
    nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]
    pool5_reshaped = tf.reshape(pool5, [-1, nodes], name='reshape1')

    fc6 = fc_layer(pool5_reshaped, [nodes, 4096], "fc6")
    relu6 = tf.nn.relu(fc6)
    if train:
        relu6 = tf.nn.dropout(relu6, 0.5)

    fc7 = fc_layer(relu6, [4096, 4096], "fc7")
    relu7 = tf.nn.relu(fc7)
    if train:
        relu7 = tf.nn.dropout(relu7, 0.5)

    fc8 = fc_layer(relu7, [4096, config.NUM_LABELS], 'fc8')

    prob = tf.nn.softmax(fc8, name='prob')

    print 'vgg16 network created!'

    return fc8, prob, conv1_1


def conv_layer(bottom, weight_shape, strides, name):
    with tf.variable_scope(name):
        weight = get_weight(weight_shape)
        bias = get_biases(weight_shape[3])
        conv = tf.nn.conv2d(bottom, weight, strides=strides, padding='SAME')
        relu = tf.nn.relu(tf.nn.bias_add(conv, bias))

        variable_summaries(weight, name + '/weight')
        variable_summaries(bias, name + '/bias')
        # tf.summary.histogram(name + '/activations', relu)
        return relu


def get_weight(shape):
    conv_weight = tf.get_variable('weight', shape, initializer=tf.contrib.layers.xavier_initializer_conv2d())
    return conv_weight


def get_biases(kernel_deep):
    return tf.get_variable('bias', [kernel_deep], initializer=tf.constant_initializer(0.0))


def max_pool(bottom, ksize, strides, name):
    with tf.variable_scope(name):
        pool = tf.nn.max_pool(bottom, ksize=ksize, strides=strides, padding='SAME')

        variable_summaries(pool, name + '/pool')
        return pool


def batch_norm(input_tensor):
    epsilon = 1e-3
    batch_mean, batch_var = tf.nn.moments(input_tensor, [0])
    input_tensor = tf.nn.batch_normalization(input_tensor, mean=batch_mean, variance=batch_var,offset=None,
                                             scale=None, variance_epsilon=epsilon)

    return input_tensor


def fc_layer(bottom, weight_shape, name):
    with tf.variable_scope(name):
        fc_weight = tf.get_variable('weight', weight_shape, initializer=tf.truncated_normal_initializer(stddev=0.1))
        fc_bias = tf.get_variable('bias', weight_shape[1], initializer=tf.truncated_normal_initializer(stddev=0.1))
        fc = tf.nn.bias_add(tf.matmul(bottom, fc_weight), fc_bias)

        variable_summaries(fc_weight, name)
        variable_summaries(fc_bias, name)
        # tf.summary.histogram(name + '/fc', fc)
        return fc


def variable_summaries(var, name):

    with tf.name_scope('summaries'):
        # tf.summary.histogram(name, var)

        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean/' + name, mean)
        # stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        # tf.summary.scalar('stddev/' + name, stddev)


def cross_entropy_fn(fc8, ground_truth_input):

    with tf.name_scope('cross_entropy') as scope:
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=fc8, labels=ground_truth_input)
        cross_entropy_mean = tf.reduce_mean(cross_entropy)

        tf.summary.scalar(scope + '/loss', cross_entropy_mean)

    return cross_entropy_mean


def optimize(loss):

    with tf.name_scope('train'):
        train_step = tf.train.GradientDescentOptimizer(config.LEARNING_RATE_BASE).minimize(loss)

    return train_step


def evaluation(prob, ground_truth_input):

    with tf.name_scope('evaluation') as scope:
        correct_prediction = tf.equal(tf.argmax(prob, 1), tf.argmax(ground_truth_input, 1))

    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) * 100

    tf.summary.scalar(scope + '/accuracy', accuracy)

    return accuracy
