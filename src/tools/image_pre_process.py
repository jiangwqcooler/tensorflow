# _*_ coding:utf-8 _*_

"""
@author jiangwenqiang
@date 2017/07/23
"""
import os

from src.config import config
import tensorflow as tf


def image_resize(image_dir=None, output_dir=None):
    """
    :param image_dir: image saved path
    :param output_dir: resized image save path
    :return:
    """
    image_size = config.IMAGE_SIZE
    image_dir_list = get_image_list(image_dir)
    assert os.path.exists(image_dir)
    assert os.path.exists(output_dir)
    with tf.Session() as sess:
        for image_path in image_dir_list:
            image_raw_data = tf.gfile.FastGFile(image_path, 'r').read()
            file_path, file_ext = os.path.splitext(image_path)
            file_path_split = file_path.split('/')
            class_dir = file_path_split[-2]
            file_name = os.path.basename(image_path)
            if file_ext == '.jpg' or file_ext == '.jpeg':
                img_data = tf.image.decode_jpeg(image_raw_data, channels=3)
                img_data = tf.image.convert_image_dtype(img_data, dtype=tf.float32)
                resized_image = tf.image.resize_images(img_data, (image_size, image_size),
                                                       method=config.COMPRESSION_METHOD)
                encoded_image = tf.image.encode_jpeg(tf.cast(resized_image * 255, dtype=tf.uint8))

            elif file_ext == '.png':
                img_data = tf.image.decode_png(image_raw_data, channels=3)
                img_data = tf.image.convert_image_dtype(img_data, dtype=tf.float32)
                resized_image = tf.image.resize_images(img_data, (image_size, image_size),
                                                       method=config.COMPRESSION_METHOD)
                encoded_image = tf.image.encode_png(tf.cast(resized_image * 255, dtype=tf.uint8))

            resized_image_dir = output_dir + '/' + class_dir + '/'
            if os.path.exists(resized_image_dir) is False:
                os.mkdir(resized_image_dir)
            resized_image_name = resized_image_dir + file_name
            with tf.gfile.GFile(resized_image_name, 'wb') as f:
                f.write(encoded_image.eval())

    return image_dir_list


def get_image_list(image_dir=None):
    """
    :param image_dir:  image saved path
    :return: image set
    """
    image_dir_list = []
    image_dir = image_dir
    dirs_list = os.listdir(image_dir)
    for sub_dir in dirs_list:
        file_path = os.path.join(image_dir, sub_dir)
        assert os.path.isdir(file_path)
        images = os.listdir(file_path)
        for item in images:
            image_dir_list.append(file_path + '/' + item)
    return image_dir_list


def image_augment(data_set_name=None):

    image_dir = config.DATA_DIR + data_set_name
    image_dir_list = get_image_list(image_dir)

    with tf.Session() as sess:
        for image_path in image_dir_list:
            image_raw_data = tf.gfile.FastGFile(image_path, 'r').read()
            file_path, file_ext = os.path.splitext(image_path)

            # image transpose
            if config.IMAGE_TRANSPOSE:

                # according to image ext choice adapt method
                if file_ext == '.jpg' or file_ext == '.jpeg':
                    img_data = tf.image.decode_jpeg(image_raw_data, channels=3)
                    img_data = tf.image.convert_image_dtype(img_data, dtype=tf.float32)
                    flipped_ud = tf.image.flip_up_down(img_data)
                    flipped_lr = tf.image.flip_left_right(img_data)
                    transposed = tf.image.transpose_image(img_data)

                elif file_ext == '.png':
                    img_data = tf.image.decode_png(image_raw_data, channels=3)
                    img_data = tf.image.convert_image_dtype(img_data, dtype=tf.float32)
                    flipped_ud = tf.image.flip_up_down(img_data)
                    flipped_lr = tf.image.flip_left_right(img_data)
                    transposed = tf.image.transpose_image(img_data)

                flipped_ud_image_save_dir = file_path + 'flipped_ud' + file_ext
                flipped_lr_image_save_dir = file_path + 'flipped_lr' + file_ext
                transposed_image_save_dir = file_path + 'transposed' + file_ext

                with tf.gfile.GFile(flipped_ud_image_save_dir, 'wb') as f:
                    f.write(flipped_ud.eval())
                with tf.gfile.GFile(flipped_lr_image_save_dir, 'wb') as f:
                    f.write(flipped_lr.eval())
                with tf.gfile.GFile(transposed_image_save_dir, 'wb') as f:
                    f.write(transposed.eval())

            # image brightness
            if config.IMAGE_BRIGHTNESS:

                # according to image ext choice adapt method
                if file_ext == '.jpg' or file_ext == '.jpeg':
                    img_data = tf.image.decode_jpeg(image_raw_data, channels=3)
                    img_data = tf.image.convert_image_dtype(img_data, dtype=tf.float32)
                    brightness_reduce = tf.image.adjust_brightness(img_data, -0.3)
                    brightness_augment = tf.image.adjust_brightness(img_data, 0.3)
                elif file_ext == '.png':
                    img_data = tf.image.decode_png(image_raw_data, channels=3)
                    img_data = tf.image.convert_image_dtype(img_data, dtype=tf.float32)
                    brightness_reduce = tf.image.adjust_brightness(img_data, -0.3)
                    brightness_augment = tf.image.adjust_brightness(img_data, 0.3)

                brightness_reduce_image_save_dir = file_path + 'brightness_reduce' + file_ext
                brightness_augment_image_save_dir = file_path + 'brightness_augment' + file_ext

                with tf.gfile.GFile(brightness_reduce_image_save_dir, 'wb') as f:
                    f.write(brightness_reduce.eval())
                with tf.gfile.GFile(brightness_augment_image_save_dir, 'wb') as f:
                    f.write(brightness_augment.eval())

            # image hue
            if config.IMAGE_HUE:

                # according to image ext choice adapt method
                if file_ext == '.jpg' or file_ext == '.jpeg':
                    img_data = tf.image.decode_jpeg(image_raw_data, channels=3)
                    img_data = tf.image.convert_image_dtype(img_data, dtype=tf.float32)
                    hue_reduce = tf.image.adjust_hue(img_data, 0.1)
                    hue_augment = tf.image.adjust_hue(img_data, 0.6)
                elif file_ext == '.png':
                    img_data = tf.image.decode_png(image_raw_data, channels=3)
                    img_data = tf.image.convert_image_dtype(img_data, dtype=tf.float32)
                    hue_reduce = tf.image.adjust_hue(img_data, 0.1)
                    hue_augment = tf.image.adjust_hue(img_data, 0.6)

                hue_reduce_image_save_dir = file_path + 'hue_reduce' + file_ext
                hue_augment_image_save_dir = file_path + 'hue_augment' + file_ext

                with tf.gfile.GFile(hue_reduce_image_save_dir, 'wb') as f:
                    f.write(hue_reduce.eval())
                with tf.gfile.GFile(hue_augment_image_save_dir, 'wb') as f:
                    f.write(hue_augment.eval())

            # image saturation
            if config.IMAGE_SATURATION:

                # according to image ext choice adapt method
                if file_ext == '.jpg' or file_ext == '.jpeg':
                    img_data = tf.image.decode_jpeg(image_raw_data, channels=3)
                    img_data = tf.image.convert_image_dtype(img_data, dtype=tf.float32)
                    saturation_reduce = tf.image.adjust_saturation(img_data, -5)
                    saturation_augment = tf.image.adjust_saturation(img_data, 5)
                elif file_ext == '.png':
                    img_data = tf.image.decode_png(image_raw_data, channels=3)
                    img_data = tf.image.convert_image_dtype(img_data, dtype=tf.float32)
                    saturation_reduce = tf.image.adjust_saturation(img_data, -5)
                    saturation_augment = tf.image.adjust_saturation(img_data, 5)

                saturation_reduce_image_save_dir = file_path + 'saturation_reduce' + file_ext
                saturation_augment_image_save_dir = file_path + 'saturation_augment' + file_ext

                with tf.gfile.GFile(saturation_reduce_image_save_dir, 'wb') as f:
                    f.write(saturation_reduce.eval())
                with tf.gfile.GFile(saturation_augment_image_save_dir, 'wb') as f:
                    f.write(saturation_augment.eval())


if __name__ == '__main__':
    data_set_name = config.DATASET_NAME
    image_resize('../../data/' + data_set_name, '../../cache/' + data_set_name)






