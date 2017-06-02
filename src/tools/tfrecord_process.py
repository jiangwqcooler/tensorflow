# _*_ coding:utf-8 _*_

"""
@author jiangwenqiang
@date 2017/05/23
"""
import os

import tensorflow as tf
import random
import Image
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from src.config import config
from src.tools import image_pre_process


def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def tfrecord_writer(data_set_name):
    """
    :param data_set_name: data_set_name
    :return:
    """
    image_list, training_index, validation_index, testing_index = dataset_divide(data_set_name)
    training_writer_filename = config.TFRECORD_DIR + data_set_name + '/' + 'train.tfrecords'
    validation_writer_filename = config.TFRECORD_DIR + data_set_name + '/' + 'validation.tfrecords'
    testing_writer_filename = config.TFRECORD_DIR + data_set_name + '/' + 'test.tfrecords'

    image_cache_dir = '../cache/' + data_set_name
    label_list = os.listdir(image_cache_dir)
    le = LabelEncoder()
    le.fit(label_list)
    num_babel_list = le.transform(label_list)

    # one_hot = OneHotEncoder()
    # one_hot.fit(num_babel_list)
    # ont_hot_encode = one_hot.transform(num_babel_list)
    training_writer = tf.python_io.TFRecordWriter(training_writer_filename)
    for item in training_index:
        zero_label = np.zeros(len(num_babel_list), dtype=np.int32)
        image_path = image_list[item]
        arr_dir = image_path.split('/')
        label = arr_dir[len(arr_dir) - 2]
        label_index = label_list.index(label)
        zero_label[label_index] = 1
        assert os.path.exists(image_path)
        image_data = Image.open(image_path)
        image_str = image_data.tobytes()
        zero_label_str = zero_label.tobytes()
        example = tf.train.Example(features=tf.train.Features(feature={
            'label': bytes_feature(zero_label_str),
            'image_str': bytes_feature(image_str)}))

        training_writer.write(example.SerializeToString())
    training_writer.close()

    validation_writer = tf.python_io.TFRecordWriter(validation_writer_filename)
    for item in validation_index:
        zero_label = np.zeros(len(num_babel_list), dtype=np.int32)
        image_path = image_list[item]
        arr_dir = image_path.split('/')
        label = arr_dir[len(arr_dir) - 2]
        label_index = label_list.index(label)
        zero_label[label_index] = 1
        assert os.path.exists(image_path)
        image_data = Image.open(image_path)
        image_str = image_data.tobytes()
        zero_label_str = zero_label.tobytes()
        example = tf.train.Example(features=tf.train.Features(feature={
            'label': bytes_feature(zero_label_str),
            'image_str': bytes_feature(image_str)}))

        validation_writer.write(example.SerializeToString())
    validation_writer.close()

    testing_writer = tf.python_io.TFRecordWriter(testing_writer_filename)
    for item in testing_index:
        zero_label = np.zeros(len(num_babel_list), dtype=np.int32)
        image_path = image_list[item]
        arr_dir = image_path.split('/')
        label = arr_dir[len(arr_dir) - 2]
        label_index = label_list.index(label)
        zero_label[label_index] = 1
        assert os.path.exists(image_path)
        image_data = Image.open(image_path)
        image_str = image_data.tobytes()
        zero_label_str = zero_label.tobytes()
        example = tf.train.Example(features=tf.train.Features(feature={
            'label': bytes_feature(zero_label_str),
            'image_str': bytes_feature(image_str)}))

        testing_writer.write(example.SerializeToString())
    testing_writer.close()

    return len(validation_index), len(testing_index)


def dataset_divide(data_set_name):
    """
    :return: training_dataset, validation_dataset, testing_dataset index
    """
    validation_percentage = config.VALIDATION_PERCENTAGE
    testing_percentage = config.TEST_PERCENTAGE
    image_list = image_pre_process.get_image_list(config.IMAGE_CACHE_DIR + data_set_name)

    num_image = len(image_list)
    num_validation = int(num_image * validation_percentage)
    num_testing = int(num_image * testing_percentage)
    num_training = num_image - num_validation - num_testing
    dataset_index = [item for item in range(0, num_image)]

    training_validation_index = random.sample(dataset_index, num_training + num_validation)
    validation_index = random.sample(training_validation_index, num_validation)
    tmp_set_training_validation_index = set(training_validation_index)
    training_index = list(tmp_set_training_validation_index - set(validation_index))
    testing_index = list(set(dataset_index) - tmp_set_training_validation_index)

    return image_list, training_index, validation_index, testing_index
