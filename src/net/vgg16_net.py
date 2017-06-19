# _*_ coding:utf-8 _*_

import tensorflow as tf

from ..config import config

"""
@author jiangwenqiang
@date 2017/05/22
"""


def create_net(input_tensor, train=True):
    """
    define network structure
    :param input_tensor: real date
    :param train: running stage
    :return: fc8, prob layer define
    """
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

    pool_shape = pool5.get_shape()
    flattened = pool_shape[1].value * pool_shape[2].value * pool_shape[3].value
    pool5_reshaped = tf.reshape(pool5, [-1, flattened], name='reshape1')

    fc6 = fc_layer(pool5_reshaped, [pool5_reshaped.get_shape()[-1].value, 4096], "fc6")
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

    return fc8, prob


def conv_layer(bottom, weight_shape, strides, name):
    """
    define convolution layer
    :param bottom: bottom layer
    :param weight_shape: weight shape
    :param strides: strides
    :param name: layer name
    :return: define of result layer
    """
    with tf.variable_scope(name):
        weight = get_weight(weight_shape)
        bias = get_biases(weight_shape[3])
        conv = tf.nn.conv2d(bottom, weight, strides=strides, padding='SAME')
        relu = tf.nn.relu(tf.nn.bias_add(conv, bias))
        print_activation(relu)

        # visitable weight, bias
        variable_summaries(weight, name + '/weight')
        variable_summaries(bias, name + '/bias')
        # tf.summary.histogram(name + '/activations', relu)
        return relu


def get_weight(shape):
    """
    define weight
    :param shape: weight shape
    :return: initialed weight
    """
    conv_weight = tf.get_variable('weight', shape, initializer=tf.contrib.layers.xavier_initializer_conv2d())
    return conv_weight


def get_biases(kernel_deep):
    """
    define biases
    :param kernel_deep: weight shape
    :return: initialed biases
    """
    return tf.get_variable('bias', [kernel_deep], initializer=tf.constant_initializer(0.0))


def max_pool(bottom, ksize, strides, name):
    """
    def max pool layer
    :param bottom: bottom layer
    :param ksize: kernel shape
    :param strides: strides
    :param name: layer name
    :return: max pool layer
    """
    with tf.variable_scope(name):
        pool = tf.nn.max_pool(bottom, ksize=ksize, strides=strides, padding='SAME')
        print_activation(pool)

        # visitable pool layer
        variable_summaries(pool, name + '/pool')
        return pool


def batch_norm(input_tensor):
    """
    define batch normalization layer
    :param input_tensor: input tensor
    :return: normalized batch
    """
    epsilon = 1e-3
    batch_mean, batch_var = tf.nn.moments(input_tensor, [0])
    input_tensor = tf.nn.batch_normalization(input_tensor, mean=batch_mean, variance=batch_var, offset=None,
                                             scale=None, variance_epsilon=epsilon)

    return input_tensor


def fc_layer(bottom, weight_shape, name):
    """
    define fully connected layer
    :param bottom: bottom layer
    :param weight_shape: weight shape
    :param name: layer name
    :return: fc layer
    """
    with tf.variable_scope(name):
        fc_weight = tf.get_variable('weight', weight_shape, initializer=tf.truncated_normal_initializer(stddev=0.1))
        fc_bias = tf.get_variable('bias', weight_shape[1], initializer=tf.truncated_normal_initializer(stddev=0.1))
        fc = tf.nn.bias_add(tf.matmul(bottom, fc_weight), fc_bias)
        print_activation(fc)

        # visitable fc_weight layer and fc_bias layer
        variable_summaries(fc_weight, name)
        variable_summaries(fc_bias, name)
        # tf.summary.histogram(name + '/fc', fc)
        return fc


def variable_summaries(var, name):
    """
    visitable with summaries
    :param var: value
    :param name: node name
    :return:
    """
    with tf.name_scope('summaries'):
        # tf.summary.histogram(name, var)

        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean/' + name, mean)
        # stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        # tf.summary.scalar('stddev/' + name, stddev)


def cross_entropy_fn(fc8, ground_truth_input):
    """
    define cross entropy layer
    :param fc8: last fc layer
    :param ground_truth_input: ground_truth_input
    :return: cross entropy layer
    """
    with tf.name_scope('cross_entropy') as scope:
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=fc8, labels=ground_truth_input)
        cross_entropy_mean = tf.reduce_mean(cross_entropy)

        tf.summary.scalar(scope + '/loss', cross_entropy_mean)

    return cross_entropy_mean


def optimize(loss):
    """
    define optimize method
    :param loss: loss layer
    :return:
    """
    with tf.name_scope('train'):
        train_step = tf.train.GradientDescentOptimizer(config.LEARNING_RATE_BASE).minimize(loss )
        # train_step = tf.train.MomentumOptimizer(config.LEARNING_RATE_BASE, momentum=0.1).minimize(loss)
    return train_step


def evaluation(prob, ground_truth_input):
    """
    evaluate model
    :param prob: prob layer
    :param ground_truth_input: ground_truth_input
    :return: accuracy
    """

    with tf.name_scope('evaluation') as scope:
        correct_prediction = tf.equal(tf.argmax(prob, 1), tf.argmax(ground_truth_input, 1))

    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) * 100

    tf.summary.scalar(scope + '/accuracy', accuracy)

    return accuracy


def print_activation(tensor):

    print tensor.op.name, ' ', tensor.get_shape().as_list()


