# _*_ coding:utf-8 _*_
import os
import time
import math
from datetime import datetime

import tensorflow as tf
from tensorflow.python import debug as tf_debug

from src.config import config
from src.tools import tfrecord_process
from src.net import vgg16_net


def read_and_decode(stage):
    """
    read data from tfrecord file
    :param stage: running stage
    :return: image file and category label
    """

    record_path = config.TFRECORD_DIR + config.DATASET_NAME + '/' + stage + '.tfrecords'
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


def read_batch_data(stage, num):
    """
    read a mini-batch data
    :param stage: running stage
    :param num: batch size
    :return: size equal batch_size data
    """

    min_after_dequeue = 1000
    capacity = min_after_dequeue + 3 * config.BATCH_SIZE

    with tf.name_scope(stage + '_batch'):
        train_image, train_label = read_and_decode(stage)
        image_batch, label_batch = tf.train.shuffle_batch(
            [train_image, train_label], batch_size=num, capacity=capacity, min_after_dequeue=min_after_dequeue)

    return image_batch, label_batch


def main(_):

    # define input placeholder
    with tf.name_scope('input'):
        input_tensor = tf.placeholder(
            tf.float32, [None, config.IMAGE_SIZE, config.IMAGE_SIZE, config.IMAGE_CHANNELS], name='input-tensor')
        ground_truth_input = tf.placeholder(tf.float32, [None, config.NUM_LABELS], name='GroundTruthInput')

    # if tfrecord data not existed need to write it
    data_set_name = config.DATASET_NAME
    tfrecord_dir = config.TFRECORD_DIR + '/' + data_set_name
    if os.path.exists(tfrecord_dir + 'train.tfrecords') is False or os.path.exists(tfrecord_dir + 'test.tfrecords') \
            is False or os.path.exists(tfrecord_dir + 'validation.tfrecords') is False:
        num_val, num_test = tfrecord_process.tfrecord_writer(data_set_name)

    # training/testing data real lable placeholder
    val_batch = tf.placeholder(tf.int32)
    test_batch = tf.placeholder(tf.int32)
    train_image_batch, train_label_batch = read_batch_data('train', config.BATCH_SIZE)  # read a batch training data
    val_image, val_label = read_batch_data('validation', val_batch)  # read a batch validation data
    test_image, test_label = read_batch_data('test', test_batch)  # read a batch testing data

    # init net and define loss, optimize strategy, accuracy
    fc8, prob = vgg16_net.create_net(input_tensor, True)
    cross_entropy_mean = vgg16_net.cross_entropy_fn(fc8, ground_truth_input)
    train_step = vgg16_net.optimize(cross_entropy_mean)
    accuracy = vgg16_net.evaluation(prob, ground_truth_input)

    # define model saver and tensorboard information
    saver = tf.train.Saver()
    merged = tf.summary.merge_all()

    # gpu using constrain
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.95)
    gpu_config = tf.ConfigProto(gpu_options=gpu_options)
    # gpu_config.gpu_options.allocator_type = 'BFC'
    gpu_config.gpu_options.allow_growth = True

    sess = tf.Session(config=gpu_config)

    # sess = tf_debug.LocalCLIDebugWrapperSession(sess)
    # sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)

    # def log writer
    summary_writer = tf.summary.FileWriter(config.LOG_DIR + config.DATASET_NAME, sess.graph)
    init = tf.global_variables_initializer()  # memory use more than 1000M
    sess.run(init)
    coord = tf.train.Coordinator()  # start mutil-thread
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    # if we want to finetune
    if config.FINETUNE:
        ckpt = tf.train.get_checkpoint_state(config.MODEL_DIR + data_set_name)
        if ckpt and ckpt.model_checkpoint_path:
            print "Continue training from the model {}".format(ckpt.model_checkpoint_path)
            saver.restore(sess, ckpt.ckpt.model_check_path)

    total_duration = 0.0
    total_duration_squared = 0.0
    print 'Starting training...'

    for i in range(config.TRAIN_STEPS):

        start_time_train = time.time()   # time element
        train_v, train_l = sess.run([train_image_batch, train_label_batch])  # first read a batch training data
        summary, loss_value, step = sess.run(
            [merged, cross_entropy_mean, train_step], feed_dict={input_tensor: train_v, ground_truth_input: train_l})
        duration_train = time.time() - start_time_train

        # total consume time and per image consume time
        total_duration += duration_train
        total_duration_squared += duration_train * duration_train
        ever_image_time = duration_train / config.BATCH_SIZE

        if i % 20 == 0:
            print '%s across %d step: mean cross entropy on random batch is : %.5f - cast time:%.3f sec/image ' \
                  % (datetime.now(), i, loss_value, ever_image_time)

        # training per 500 times evaluate model in validation dataset
        if (i % 500 == 0 and i != 0) or i + 1 == config.TRAIN_STEPS:

            # because it's not divisible, last batch is not enough batch_size, so there compute ever batch accuracy and
            # average in the end
            val_epoch = num_val / config.BATCH_SIZE
            val_last_batch = num_val % config.BATCH_SIZE
            val_accuracy_count = 0.0
            # evaluate 'val_epoch' batch
            for j in range(val_epoch):
                validation_v, validation_l = sess.run([val_image, val_label], feed_dict={val_batch: config.BATCH_SIZE})
                validation_accuracy = sess.run(
                    accuracy, feed_dict={input_tensor: validation_v, ground_truth_input: validation_l})
                val_accuracy_count += validation_accuracy

                print 'Step %d: Validation accuracy on random samples %d examples = %.3f%%' \
                      % (i, config.BATCH_SIZE, validation_accuracy)
            # evaluate rest batch
            validation_v, validation_l = sess.run([val_image, val_label], feed_dict={val_batch: val_last_batch})
            validation_accuracy = sess.run(
                accuracy, feed_dict={input_tensor: validation_v, ground_truth_input: validation_l})
            val_accuracy_count += validation_accuracy
            print 'Step %d: Validation accuracy on random samples %d examples = %.3f%%' \
                  % (i, val_last_batch, validation_accuracy)

            # evaluate mean accuracy
            val_accuracy_mean = val_accuracy_count / (val_epoch + 1)
            print 'Step %d: Validation mean accuracy on %d examples = %.3f%%' \
                  % (i, num_val, val_accuracy_mean)

            # per 500 times save model
            save_dir = config.MODEL_DIR + data_set_name + '/' + config.MODEL_NAME
            saver.save(sess, save_dir, global_step=i)

        summary_writer.add_summary(summary, i)

    # evaluate average time and total time
    mean_time = total_duration / config.TRAIN_STEPS
    mean_time_vr = total_duration_squared / config.TRAIN_STEPS - mean_time * mean_time
    mean_time_vr_sqrt = math.sqrt(mean_time_vr)
    print 'total training time %.3f average batch time %.3f +/- %.3f sec/batch' \
          % (total_duration, mean_time, mean_time_vr_sqrt)
    print 'Training End!'

    # test model perform in the end, and method same with above evaluate validation
    test_epoch = num_test / config.BATCH_SIZE
    test_last_batch = num_test % config.BATCH_SIZE
    test_accuracy_count = 0.0
    for k in range(test_epoch):

        test_v, test_l = sess.run([test_image, test_label], feed_dict={test_batch: config.BATCH_SIZE})
        test_accuracy = sess.run(accuracy, feed_dict={input_tensor: test_v, ground_truth_input: test_l})
        test_accuracy_count += test_accuracy

        print 'Final test accuracy on random samples %d examples = %.3f%%' % (config.BATCH_SIZE, test_accuracy)

    test_v, test_l = sess.run([test_image, test_label], feed_dict={test_batch: test_last_batch})
    test_accuracy = sess.run(accuracy, feed_dict={input_tensor: test_v, ground_truth_input: test_l})
    test_accuracy_count += test_accuracy
    print 'Final test accuracy on random samples %d examples = %.3f%%' % (test_last_batch, test_accuracy)

    test_accuracy_mean = test_accuracy_count / (test_epoch + 1)
    print 'Final test accuracy = %.3f%%' % test_accuracy_mean

    # close resource
    summary_writer.close()
    coord.request_stop()
    coord.join(threads)
    sess.close()

if __name__ == '__main__':
    tf.app.run()
