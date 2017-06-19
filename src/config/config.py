# _*_ coding:utf-8 _*_

"""
@author jiangwenqiang
@date 2017/05/22
"""

"""
vgg net parameters config
"""

# training times
TRAIN_STEPS = 100000

# input image size
IMAGE_SIZE = 224

# image channels
IMAGE_CHANNELS = 3

# snapshoot step
STEP_MODEL_SAVE = 10000

# save image cache dir
IMAGE_CACHE_DIR = '../cache/'

# TFRecord dir
TFRECORD_DIR = '../record/'

# save model dir
MODEL_DIR = '../models/'

# data save dir
DATA_DIR = '../data/'

# log dir
LOG_DIR = '../logs/'

# percentage of dataset
VALIDATION_PERCENTAGE = 0.1

# percentage of dataset
TEST_PERCENTAGE = 0.1

# base lr
LEARNING_RATE_BASE = 0.0001

# batch_size
BATCH_SIZE = 64

# lr_decay
LEARNING_RATE_DECAY = 0.99

# regularization_rate
REGULARIZATION_RATE = 0.0001

# moving_average_decay
MOVING_AVERAGE_DECAY = 0.99

# if finetune
USE_MODEL = False

# weight regularization
WEIGHT_REGULARIZATION = True

# number of category
NUM_LABELS = 5

# compression method
COMPRESSION_METHOD = 0

# shuffle capacity
SHUFFLE_CAPACITY = 10000

# default model name
MODEL_NAME = 'vgg16-model.ckpt'

# if finetune
FINETUNE = False

IMAGE_TRANSPOSE = True

IMAGE_BRIGHTNESS = True

IMAGE_HUE = True

IMAGE_SATURATION = True

IMAGE_RANDOM_CROP = True

# dataset name
DATASET_NAME = '4d_dataset'




