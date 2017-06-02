# _*_ coding:utf-8 _*_
import os

import tensorflow as tf

from src.config import config
from src.tools import tfrecord_process
from src.net.vgg16_net import VGG16


def read_and_decode(stage):

    record_path = config.TFRECORD_DIR + stage + '.tfrecords'
    reader = tf.TFRecordReader()
    filename_queue = tf.train.string_input_producer([record_path])
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={
            'label': tf.FixedLenFeature([], tf.string),
            'image_str': tf.FixedLenFeature([], tf.string)
        })

    image = tf.decode_raw(features['image_str'], tf.uint8)
    image = tf.reshape(image, [config.IMAGE_SIZE, config.IMAGE_SIZE, config.IMAGE_CHANNELS])
    image = tf.cast(image, tf.float32)

    label = tf.decode_raw(features['label'], tf.int32)
    label = tf.reshape(label, [config.NUM_LABELS])
    label = tf.cast(label, tf.float32)

    return image, label


if __name__ == '__main__':
    image, label = read_and_decode('train')

    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        init = tf.global_variables_initializer()
        sess.run(init)
        image, label = sess.run([image, label])
        print image, label